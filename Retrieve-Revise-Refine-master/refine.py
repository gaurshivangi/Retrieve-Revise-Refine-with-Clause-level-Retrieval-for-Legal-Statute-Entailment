"""
Refine Stage: Final Optimization
Implements:
1. Clause-evidence Boosting: Articles with multiple entailing clauses get higher scores
2. Diversity-aware Selection: Use Maximal Marginal Relevance (MMR) to ensure diversity
3. Dynamic Cut-off: Adapt number of selected articles based on score distribution and confidence
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from clause_retriever import ClauseRetriever


def load_revise_output(revise_output_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load revise phase output (supports both basic and enhanced formats)
    Returns: Dictionary mapping query_id to list of article results
    """
    results = defaultdict(list)
    
    with open(revise_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                query_id = data['query_id']
                
                # Convert basic format to enhanced format if needed
                if 'final_decision' not in data:
                    # Basic format: extract decision from llm_response
                    llm_response = data.get('llm_response', '').lower()
                    if any(word in llm_response for word in ['yes', 'true', 'substantiated', 'valid', 'compliant', 'verified']):
                        data['final_decision'] = 'yes'
                        data['final_confidence'] = 0.7
                    elif any(word in llm_response for word in ['no', 'false', 'unsubstantiated', 'invalid']):
                        data['final_decision'] = 'no'
                        data['final_confidence'] = 0.3
                    else:
                        data['final_decision'] = 'unsure'
                        data['final_confidence'] = 0.5
                    
                    # Set clause_count from has_clauses
                    if 'clause_count' not in data:
                        data['clause_count'] = 3 if data.get('has_clauses', False) else 0
                
                results[query_id].append(data)
            except Exception as e:
                print(f"Error loading line: {e}")
                continue
    
    return dict(results)


class ClauseEvidenceBooster:
    """Boost articles with multiple entailing clauses"""
    
    def __init__(self, boost_factor: float = 1.5, min_clauses: int = 2):
        """
        Args:
            boost_factor: Factor to boost articles with multiple clauses
            min_clauses: Minimum number of clauses to trigger boosting
        """
        self.boost_factor = boost_factor
        self.min_clauses = min_clauses
    
    def boost_scores(self, query_id: str, articles: List[Dict[str, Any]], 
                     clause_retriever: ClauseRetriever = None) -> List[Dict[str, Any]]:
        """
        Boost scores for articles with multiple entailing clauses
        Args:
            query_id: Query ID
            articles: List of article results from revise phase
            clause_retriever: ClauseRetriever instance (optional)
        Returns:
            List of articles with boosted scores
        """
        boosted_articles = []
        
        for article in articles:
            article_id = article['article_id']
            base_score = article.get('final_confidence', 0.5)
            clause_count = article.get('clause_count', 0)
            
            # Boost if article has multiple clauses
            if clause_count >= self.min_clauses:
                # Calculate boost: more clauses = higher boost
                boost = 1.0 + (clause_count - 1) * self.boost_factor * 0.1
                boosted_score = base_score * boost
            else:
                boosted_score = base_score
                boost = 1.0
            
            article['boosted_score'] = boosted_score
            article['boost_factor'] = boost
            boosted_articles.append(article)
        
        return boosted_articles


class DiversitySelector:
    """Maximal Marginal Relevance (MMR) for diversity-aware selection"""
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Args:
            lambda_param: Lambda parameter for MMR (0 = pure relevance, 1 = pure diversity)
        """
        self.lambda_param = lambda_param
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute similarity matrix between texts"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except Exception as e:
            print(f"Error computing similarity: {e}")
            # Return identity matrix if computation fails
            return np.eye(len(texts))
    
    def mmr_select(self, articles: List[Dict[str, Any]], query_text: str, 
                   top_k: int, article_texts: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Select articles using Maximal Marginal Relevance
        Args:
            articles: List of article results with scores
            query_text: Query text
            top_k: Number of articles to select
            article_texts: Dictionary mapping article_id to article text
        Returns:
            Selected articles
        """
        if len(articles) <= top_k:
            return articles
        
        # Sort by score first
        sorted_articles = sorted(articles, key=lambda x: x.get('boosted_score', x.get('final_confidence', 0.5)), reverse=True)
        
        # Get article texts
        texts = [article_texts.get(article['article_id'], '') for article in sorted_articles]
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(texts)
        
        # MMR selection
        selected = []
        selected_indices = set()
        
        # Start with highest scoring article
        selected.append(sorted_articles[0])
        selected_indices.add(0)
        
        # Select remaining articles using MMR
        while len(selected) < min(top_k, len(sorted_articles)):
            best_score = -float('inf')
            best_idx = -1
            
            for i, article in enumerate(sorted_articles):
                if i in selected_indices:
                    continue
                
                # Relevance score (boosted score)
                relevance = article.get('boosted_score', article.get('final_confidence', 0.5))
                
                # Diversity penalty (max similarity to selected articles)
                max_similarity = 0.0
                for selected_idx in selected_indices:
                    similarity = similarity_matrix[i][selected_idx]
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(sorted_articles[best_idx])
                selected_indices.add(best_idx)
            else:
                break
        
        return selected


class DynamicCutoff:
    """Dynamic cut-off based on score distribution and confidence thresholds"""
    
    def __init__(self, min_confidence: float = 0.5, score_gap_threshold: float = 0.2, 
                 max_articles: int = 15, min_articles: int = 1):
        """
        Args:
            min_confidence: Minimum confidence threshold
            score_gap_threshold: Minimum score gap to include article
            max_articles: Maximum number of articles to select
            min_articles: Minimum number of articles to select
        """
        self.min_confidence = min_confidence
        self.score_gap_threshold = score_gap_threshold
        self.max_articles = max_articles
        self.min_articles = min_articles
    
    def select_with_dynamic_cutoff(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select articles using dynamic cut-off
        Args:
            articles: List of articles sorted by score
        Returns:
            Selected articles
        """
        if not articles:
            return []
        
        # Sort by boosted score
        sorted_articles = sorted(
            articles, 
            key=lambda x: x.get('boosted_score', x.get('final_confidence', 0.5)), 
            reverse=True
        )
        
        selected = []
        
        # Always include top article if it meets minimum confidence
        if sorted_articles:
            top_score = sorted_articles[0].get('boosted_score', sorted_articles[0].get('final_confidence', 0.5))
            if top_score >= self.min_confidence:
                selected.append(sorted_articles[0])
        
        # Add articles based on score gaps and confidence
        for i in range(1, min(len(sorted_articles), self.max_articles)):
            article = sorted_articles[i]
            score = article.get('boosted_score', article.get('final_confidence', 0.5))
            
            # Check minimum confidence
            if score < self.min_confidence:
                break
            
            # Check score gap
            if selected:
                prev_score = selected[-1].get('boosted_score', selected[-1].get('final_confidence', 0.5))
                score_gap = prev_score - score
                
                # Include if gap is small (similar confidence) or if we haven't reached minimum
                if score_gap <= self.score_gap_threshold or len(selected) < self.min_articles:
                    selected.append(article)
                else:
                    # Large gap indicates drop in quality
                    break
            else:
                selected.append(article)
        
        return selected


def get_civil_code(civil_code_file: str) -> Dict[str, str]:
    """Load civil code from JSON file"""
    with open(civil_code_file, 'r', encoding='utf-8') as fin:
        file_content = json.load(fin)
        
    civil_code = dict()
    for article_id in file_content:
        civil_code[article_id] = file_content[article_id]['content']
    return civil_code


def main(args):
    print("=" * 60)
    print("Refine Stage: Final Optimization")
    print("=" * 60)
    
    # Load revise phase output
    print(f"\nLoading revise phase output: {args.revise_output_file}")
    revise_results = load_revise_output(args.revise_output_file)
    print(f"[OK] Loaded results for {len(revise_results)} queries")
    
    # Load civil code for article texts
    print(f"\nLoading civil code: {args.civil_code_file}")
    civil_code = get_civil_code(args.civil_code_file)
    print(f"[OK] Loaded {len(civil_code)} articles")
    
    # Load gold data for query texts
    print(f"\nLoading gold data: {args.gold_file}")
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    print(f"[OK] Loaded {len(gold_data)} queries")
    
    # Initialize components
    booster = ClauseEvidenceBooster(
        boost_factor=args.boost_factor,
        min_clauses=args.min_clauses
    )
    print(f"[OK] Clause-evidence booster initialized (boost_factor: {args.boost_factor})")
    
    diversity_selector = DiversitySelector(lambda_param=args.mmr_lambda)
    print(f"[OK] Diversity selector initialized (MMR lambda: {args.mmr_lambda})")
    
    dynamic_cutoff = DynamicCutoff(
        min_confidence=args.min_confidence,
        score_gap_threshold=args.score_gap_threshold,
        max_articles=args.max_articles,
        min_articles=args.min_articles
    )
    print(f"[OK] Dynamic cut-off initialized (min_confidence: {args.min_confidence})")
    
    # Process each query
    print(f"\nProcessing queries...")
    print(f"{'='*60}")
    
    refined_results = {}
    total_articles_before = 0
    total_articles_after = 0
    
    for query_id, articles in revise_results.items():
        if query_id not in gold_data:
            continue
        
        query_text = gold_data[query_id]['query']
        total_articles_before += len(articles)
        
        # Step 1: Filter by decision (only keep 'yes' decisions)
        if args.filter_decision:
            articles = [a for a in articles if a.get('final_decision', 'unsure') == 'yes']
        
        # Step 2: Clause-evidence boosting
        if args.use_boosting:
            articles = booster.boost_scores(query_id, articles)
        
        # Step 3: Diversity-aware selection (MMR)
        if args.use_mmr:
            articles = diversity_selector.mmr_select(
                articles, query_text, args.top_k, civil_code
            )
        else:
            # Simple top-k selection
            articles = sorted(
                articles, 
                key=lambda x: x.get('boosted_score', x.get('final_confidence', 0.5)), 
                reverse=True
            )[:args.top_k]
        
        # Step 4: Dynamic cut-off
        if args.use_dynamic_cutoff:
            articles = dynamic_cutoff.select_with_dynamic_cutoff(articles)
        
        # Store results
        refined_results[query_id] = articles
        total_articles_after += len(articles)
    
    # Write output in TSV format (for evaluation)
    print(f"\nWriting output to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for query_id, articles in refined_results.items():
            for rank, article in enumerate(articles, 1):
                article_id = article['article_id']
                score = article.get('boosted_score', article.get('final_confidence', 0.5))
                f_out.write(f"{query_id} Q0 {article_id} {rank} {score:.6f} CAPTAIN\n")
    
    # Write detailed JSON output
    if args.detailed_output_file:
        print(f"Writing detailed output to: {args.detailed_output_file}")
        with open(args.detailed_output_file, 'w', encoding='utf-8') as f_out:
            json.dump(refined_results, f_out, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Refine Stage Complete!")
    print(f"[OK] Processed {len(refined_results)} queries")
    print(f"[OK] Articles before refinement: {total_articles_before}")
    print(f"[OK] Articles after refinement: {total_articles_after}")
    if total_articles_before > 0:
        print(f"[OK] Reduction: {total_articles_before - total_articles_after} articles ({(1 - total_articles_after/total_articles_before)*100:.1f}%)")
        print(f"[OK] Average articles per query: {total_articles_after/len(refined_results):.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Refine Stage: Final Optimization with Clause-Evidence Boosting, MMR, and Dynamic Cut-off"
    )
    parser.add_argument("--revise_output_file", type=str, required=True,
                       help="Path to revise phase output JSONL file")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to save refined results in TSV format")
    parser.add_argument("--detailed_output_file", type=str, default=None,
                       help="Path to save detailed JSON output (optional)")
    parser.add_argument("--civil_code_file", type=str, default="data/civil_code_en.json",
                       help="Path to civil code file")
    parser.add_argument("--gold_file", type=str, default="data/gold_R_series.json",
                       help="Path to gold standard JSON file")
    
    # Boosting parameters
    parser.add_argument("--use_boosting", action="store_true",
                       help="Use clause-evidence boosting")
    parser.add_argument("--boost_factor", type=float, default=1.5,
                       help="Boost factor for articles with multiple clauses")
    parser.add_argument("--min_clauses", type=int, default=2,
                       help="Minimum number of clauses to trigger boosting")
    
    # MMR parameters
    parser.add_argument("--use_mmr", action="store_true",
                       help="Use Maximal Marginal Relevance for diversity")
    parser.add_argument("--mmr_lambda", type=float, default=0.5,
                       help="MMR lambda parameter (0=relevance, 1=diversity)")
    
    # Dynamic cut-off parameters
    parser.add_argument("--use_dynamic_cutoff", action="store_true",
                       help="Use dynamic cut-off based on score distribution")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                       help="Minimum confidence threshold")
    parser.add_argument("--score_gap_threshold", type=float, default=0.2,
                       help="Minimum score gap to include article")
    parser.add_argument("--max_articles", type=int, default=15,
                       help="Maximum number of articles to select")
    parser.add_argument("--min_articles", type=int, default=1,
                       help="Minimum number of articles to select")
    
    # Other parameters
    parser.add_argument("--top_k", type=int, default=10,
                       help="Top-k articles to select (if not using dynamic cut-off)")
    parser.add_argument("--filter_decision", action="store_true",
                       help="Filter out articles with 'no' or 'unsure' decisions")
    
    args = parser.parse_args()
    main(args)


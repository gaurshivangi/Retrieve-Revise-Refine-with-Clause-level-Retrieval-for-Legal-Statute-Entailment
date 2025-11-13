"""
Clause-level retrieval system with hybrid dense+sparse retrieval and clause-aware aggregation
Implements the novel features from the IR_SOP document:
1. Clause Segmentation: Break statutes into clauses for fine-grained retrieval
2. Hybrid Retrieval: Combine dense embeddings with BM25 sparse retrieval
3. Clause-aware Aggregation: Boost articles with multiple entailing clauses
"""

import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import re
import logging
from data_utils.custom_rank_bm25 import BM25Okapi
import multiprocessing

logger = logging.getLogger(__name__)

class CUDASafeBM25(BM25Okapi):
    """CUDA-safe BM25 implementation that avoids multiprocessing issues"""
    
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        # Temporarily disable multiprocessing
        original_start_method = multiprocessing.get_start_method()
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Initialize without multiprocessing
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        self.q_freq_cache = {}
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        if tokenizer:
            # Process corpus without multiprocessing
            corpus = [self.tokenizer(doc) for doc in corpus]

        nd = self._initialize(corpus)
        self._calc_idf(nd)
        
        # Restore original start method
        try:
            multiprocessing.set_start_method(original_start_method, force=True)
        except RuntimeError:
            pass

class ClauseRetriever:
    """
    Main clause-level retrieval system implementing the novel features from IR_SOP
    """
    
    def __init__(self, 
                 clauses_db_path: str,
                 model_name: str = "bert-base-uncased",
                 dense_weight: float = 0.7,
                 boost_factor: float = 1.5,
                 top_k: int = 100):
        """
        Initialize the clause retriever
        
        Args:
            clauses_db_path: Path to clauses database CSV
            model_name: HuggingFace model name for dense retrieval
            dense_weight: Weight for dense retrieval (1-dense_weight for sparse)
            boost_factor: Factor to boost articles with multiple entailing clauses
            top_k: Number of top clauses to retrieve
        """
        self.clauses_db_path = clauses_db_path
        self.model_name = model_name
        self.dense_weight = dense_weight
        self.boost_factor = boost_factor
        self.top_k = top_k
        
        # Load clauses database
        self._load_clauses_database()
        
        # Initialize dense retriever
        self._initialize_dense_retriever()
        
        # Initialize sparse retriever (BM25)
        self._initialize_sparse_retriever()
        
        # Pre-compute embeddings and build indices
        self._build_retrieval_indices()
        
        # Initialize aggregator
        self.aggregator = ClauseAwareAggregator(boost_factor=boost_factor)
    
    def _load_clauses_database(self):
        """Load and process clauses database"""
        logger.info("Loading clauses database...")
        self.clauses_db = pd.read_csv(self.clauses_db_path)
        
        # Create mappings
        self.clause_id_to_text = dict(zip(self.clauses_db['clause_id'], self.clauses_db['text']))
        self.clause_id_to_article = dict(zip(self.clauses_db['clause_id'], self.clauses_db['article_number']))
        self.clause_id_to_full_text = dict(zip(self.clauses_db['clause_id'], self.clauses_db['full_text']))
        
        # Get unique articles and their clauses
        self.article_to_clauses = defaultdict(list)
        for clause_id, article_num in self.clause_id_to_article.items():
            self.article_to_clauses[article_num].append(clause_id)
        
        self.clause_texts = self.clauses_db['text'].tolist()
        self.clause_ids = self.clauses_db['clause_id'].tolist()
        self.article_numbers = self.clauses_db['article_number'].tolist()
        
        logger.info(f"Loaded {len(self.clause_ids)} clauses from {len(self.article_to_clauses)} articles")
    
    def _initialize_dense_retriever(self):
        """Initialize dense retrieval model"""
        logger.info(f"Initializing dense retriever with {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.dense_model = AutoModel.from_pretrained(self.model_name)
        self.dense_model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dense_model = self.dense_model.to(self.device)
        logger.info(f"Using device: {self.device}")
    
    def _initialize_sparse_retriever(self):
        """Initialize BM25 sparse retriever"""
        logger.info("Initializing BM25 sparse retriever...")
        # Use CUDA-safe BM25 to avoid multiprocessing issues
        self.bm25 = CUDASafeBM25(self.clause_texts, tokenizer=self._tokenize)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _build_retrieval_indices(self):
        """Pre-compute embeddings and build FAISS index"""
        logger.info("Computing dense embeddings and building retrieval indices...")
        
        # Compute clause embeddings
        self.clause_embeddings = self._compute_clause_embeddings()
        
        # Build FAISS index
        self._build_faiss_index()
        
        logger.info("Retrieval indices built successfully")
    
    def _compute_clause_embeddings(self) -> np.ndarray:
        """Pre-compute dense embeddings for all clauses"""
        embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(self.clause_texts), batch_size):
                batch_texts = self.clause_texts[i:i+batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to GPU if available
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.dense_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
                embeddings.append(batch_embeddings)
                
                if (i // batch_size) % 10 == 0:
                    logger.info(f"Processed {i}/{len(self.clause_texts)} clauses")
        
        embeddings = np.vstack(embeddings)
        logger.info(f"Computed embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search"""
        dimension = self.clause_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.clause_embeddings)
        self.faiss_index.add(self.clause_embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def retrieve_clauses(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, float, str]]:
        """
        Hybrid retrieval combining dense and sparse methods
        
        Args:
            query: Query text
            top_k: Number of top results to return (uses self.top_k if None)
        
        Returns:
            List of (clause_id, score, article_number) tuples
        """
        if top_k is None:
            top_k = self.top_k
        
        # Dense retrieval
        dense_scores, dense_indices = self._dense_retrieve(query, top_k)
        
        # Sparse retrieval (BM25)
        sparse_scores = self._sparse_retrieve(query)
        
        # Normalize scores to [0, 1]
        dense_scores = self._normalize_scores(dense_scores)
        sparse_scores = self._normalize_scores(sparse_scores)
        
        # Get sparse scores for the same indices as dense scores
        sparse_scores_subset = sparse_scores[dense_indices]
        
        # Combine scores
        combined_scores = self.dense_weight * dense_scores + (1 - self.dense_weight) * sparse_scores_subset
        
        # Get top results
        results = []
        for i, (idx, score) in enumerate(zip(dense_indices, combined_scores)):
            if i >= top_k:
                break
            clause_id = self.clause_ids[idx]
            article_number = self.article_numbers[idx]
            results.append((clause_id, float(score), article_number))
        
        return results
    
    def _dense_retrieve(self, query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Dense retrieval using FAISS"""
        query_embedding = self._get_query_embedding(query)
        scores, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            top_k
        )
        return scores[0], indices[0]
    
    def _sparse_retrieve(self, query: str) -> np.ndarray:
        """Sparse retrieval using BM25"""
        query_tokens = self._tokenize(query)
        return self.bm25.get_scores(query_tokens)
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get dense embedding for query"""
        with torch.no_grad():
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to GPU if available
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.dense_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            return embedding.cpu().numpy()
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score < 1e-8:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)
    
    def retrieve_articles(self, query: str, top_k: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve articles using clause-level retrieval with aggregation
        
        Args:
            query: Query text
            top_k: Number of top articles to return
        
        Returns:
            Dictionary mapping article_number to aggregated results
        """
        # Retrieve clauses
        clause_results = self.retrieve_clauses(query, top_k)
        
        # Aggregate to article level
        article_results = self.aggregator.aggregate(clause_results)
        
        # Sort by score and limit results
        if top_k is not None:
            sorted_articles = sorted(article_results.items(), key=lambda x: x[1]['score'], reverse=True)
            article_results = dict(sorted_articles[:top_k])
        
        return article_results
    
    def get_article_text(self, article_number: str) -> str:
        """Get full text of an article"""
        clauses = self.article_to_clauses.get(article_number, [])
        if not clauses:
            return ""
        
        # Get full text of first clause (contains article header)
        first_clause = clauses[0]
        return self.clause_id_to_full_text.get(first_clause, "")
    
    def get_top_clauses_for_article(self, article_number: str, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Get top clauses within a specific article for a query"""
        clauses = self.article_to_clauses.get(article_number, [])
        if not clauses:
            return []
        
        # Get clause texts
        clause_texts = [self.clause_id_to_text[clause_id] for clause_id in clauses]
        
        # Compute scores for these clauses
        query_embedding = self._get_query_embedding(query)
        clause_embeddings = []
        
        for clause_text in clause_texts:
            with torch.no_grad():
                inputs = self.tokenizer(
                    clause_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.dense_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
                clause_embeddings.append(embedding.cpu().numpy())
        
        clause_embeddings = np.array(clause_embeddings)
        faiss.normalize_L2(clause_embeddings)
        
        # Compute similarities
        similarities = np.dot(query_embedding, clause_embeddings.T)
        
        # Get top clauses
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            clause_id = clauses[idx]
            score = float(similarities[idx])
            results.append((clause_id, score, article_number))
        
        return results

class ClauseAwareAggregator:
    """
    Aggregate clause-level scores to article-level with boosting for multiple entailing clauses
    Implements the clause-aware aggregation novelty from IR_SOP
    """
    
    def __init__(self, boost_factor: float = 1.5):
        self.boost_factor = boost_factor
    
    def aggregate(self, clause_results: List[Tuple[str, float, str]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate clause-level scores to article-level with boosting
        
        Args:
            clause_results: List of (clause_id, score, article_number) tuples
        
        Returns:
            Dictionary mapping article_number to aggregated results
        """
        # Group by article
        article_clauses = defaultdict(list)
        for clause_id, score, article_number in clause_results:
            article_clauses[article_number].append((clause_id, score))
        
        # Aggregate scores for each article
        article_results = {}
        for article_number, clauses in article_clauses.items():
            scores = [score for _, score in clauses]
            clause_ids = [clause_id for clause_id, _ in clauses]
            
            # Calculate aggregated score with boosting for multiple clauses
            if len(scores) > 1:
                # Boost articles with multiple strong clauses
                max_score = max(scores)
                avg_score = np.mean(scores)
                # Novel aggregation: boost based on number of strong clauses
                aggregated_score = max_score + (avg_score * (len(scores) - 1) * self.boost_factor)
            else:
                aggregated_score = scores[0]
            
            article_results[article_number] = {
                'score': aggregated_score,
                'clause_count': len(clauses),
                'clause_ids': clause_ids,
                'clause_scores': scores,
                'max_clause_score': max(scores),
                'avg_clause_score': np.mean(scores),
                'boosted': len(scores) > 1
            }
        
        return article_results

class ClauseRetrievalEvaluator:
    """
    Evaluator for clause-level retrieval system
    """
    
    def __init__(self, retriever: ClauseRetriever):
        self.retriever = retriever
    
    def evaluate_on_dataset(self, test_data_path: str) -> Dict[str, float]:
        """
        Evaluate retrieval system on test dataset
        
        Args:
            test_data_path: Path to test dataset CSV
        
        Returns:
            Dictionary of evaluation metrics
        """
        test_data = pd.read_csv(test_data_path)
        
        # Group by query
        query_groups = test_data.groupby('query_id')
        
        total_queries = len(query_groups)
        total_retrieved = 0
        total_relevant = 0
        total_correct = 0
        
        f2_scores = []
        
        for query_id, group in query_groups:
            query_text = group['query'].iloc[0]
            relevant_articles = set(group[group['label'] == 'Y']['article_number'].tolist())
            
            # Retrieve articles
            retrieved_articles = self.retriever.retrieve_articles(query_text, top_k=20)
            retrieved_article_numbers = set(retrieved_articles.keys())
            
            # Calculate metrics
            correct = len(relevant_articles.intersection(retrieved_article_numbers))
            precision = correct / len(retrieved_article_numbers) if retrieved_article_numbers else 0
            recall = correct / len(relevant_articles) if relevant_articles else 0
            
            # F2 score (beta=2, emphasizes recall)
            if 4 * precision + recall == 0:
                f2 = 0
            else:
                f2 = (5 * precision * recall) / (4 * precision + recall)
            
            f2_scores.append(f2)
            total_correct += correct
            total_retrieved += len(retrieved_article_numbers)
            total_relevant += len(relevant_articles)
        
        # Calculate overall metrics
        overall_precision = total_correct / total_retrieved if total_retrieved > 0 else 0
        overall_recall = total_correct / total_relevant if total_relevant > 0 else 0
        overall_f2 = np.mean(f2_scores)
        
        return {
            'precision': overall_precision,
            'recall': overall_recall,
            'f2': overall_f2,
            'total_queries': total_queries,
            'total_retrieved': total_retrieved,
            'total_relevant': total_relevant,
            'total_correct': total_correct
        }

def create_clause_retriever_from_config(config_path: str) -> ClauseRetriever:
    """
    Create ClauseRetriever from configuration file
    
    Args:
        config_path: Path to configuration JSON file
    
    Returns:
        Configured ClauseRetriever instance
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return ClauseRetriever(
        clauses_db_path=config['clauses_db_path'],
        model_name=config.get('model_name', 'bert-base-uncased'),
        dense_weight=config.get('dense_weight', 0.7),
        boost_factor=config.get('boost_factor', 1.5),
        top_k=config.get('top_k', 100)
    )

def save_retriever(retriever: ClauseRetriever, save_path: str):
    """Save retriever to disk"""
    with open(save_path, 'wb') as f:
        pickle.dump(retriever, f)

def load_retriever(load_path: str) -> ClauseRetriever:
    """Load retriever from disk"""
    with open(load_path, 'rb') as f:
        return pickle.load(f)

"""
Enhanced Revise Phase with Self-Consistency and Entailment Verifier
Implements:
1. Clause-informed prompting with multiple prompt templates (self-consistency)
2. Voting mechanism to aggregate multiple LLM responses
3. Light entailment verifier to cross-check LLM decisions
"""

import argparse
import json
import os
import sys
import re
from typing import List, Dict, Tuple, Any
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import numpy as np

# Add parent directory to path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from clause_retriever import ClauseRetriever


def get_civil_code(civil_code_file):
    """Load civil code from JSON file"""
    with open(civil_code_file, 'r', encoding='utf-8') as fin:
        file_content = json.load(fin)
        
    civil_code = dict()
    for article_id in file_content:
        civil_code[article_id] = file_content[article_id]['content']
    return civil_code


class PromptTemplates:
    """Multiple prompt templates for self-consistency"""
    
    @staticmethod
    def prompt_template_1(query: str, clause_texts: List[str], article_text: str) -> str:
        """Template 1: Direct question format"""
        if clause_texts:
            prompt = f"Legal Statement: {query}\n\n"
            prompt += "Top Retrieved Clauses:\n"
            for i, clause in enumerate(clause_texts, 1):
                prompt += f"{i}. {clause}\n"
            prompt += f"\nFull Article Text:\n{article_text}\n\n"
            prompt += "Question: Is the legal statement supported by the provided clauses and article? Answer with 'Yes' or 'No' and provide a brief explanation."
        else:
            prompt = f"Legal Statement: {query}\n\n"
            prompt += f"Full Article Text:\n{article_text}\n\n"
            prompt += "Question: Is the legal statement supported by the article? Answer with 'Yes' or 'No' and provide a brief explanation."
        return prompt
    
    @staticmethod
    def prompt_template_2(query: str, clause_texts: List[str], article_text: str) -> str:
        """Template 2: Assessment format"""
        if clause_texts:
            prompt = f"Assessment of Legal Claim:\n\n"
            prompt += "Relevant Clauses:\n"
            for i, clause in enumerate(clause_texts, 1):
                prompt += f"{i}. {clause}\n"
            prompt += f"\nLegal Article: {article_text}\n\n"
            prompt += f"Claim to Verify: {query}\n\n"
            prompt += "Is the claim substantiated by the clauses and article? Choose: 'Claim substantiated', 'Claim unsubstantiated', or 'More information required'. Provide reasoning first, then answer."
        else:
            prompt = f"Assessment of Legal Claim:\n\n"
            prompt += f"Legal Article: {article_text}\n\n"
            prompt += f"Claim to Verify: {query}\n\n"
            prompt += "Is the claim substantiated by the article? Choose: 'Claim substantiated', 'Claim unsubstantiated', or 'More information required'. Provide reasoning first, then answer."
        return prompt
    
    @staticmethod
    def prompt_template_3(query: str, clause_texts: List[str], article_text: str) -> str:
        """Template 3: Conformity assessment format"""
        if clause_texts:
            prompt = f"Law Conformity Assessment:\n\n"
            prompt += "Statute Clauses Provided:\n"
            for i, clause in enumerate(clause_texts, 1):
                prompt += f"{i}. {clause}\n"
            prompt += f"\nFull Statute: {article_text}\n\n"
            prompt += f"Assertion: {query}\n\n"
            prompt += "Determine if the assertion is supported by the given statute(s). Respond with 'Assertion valid', 'Assertion invalid', or 'Insufficient legal context'. Start with reasoning, then conclusion."
        else:
            prompt = f"Law Conformity Assessment:\n\n"
            prompt += f"Statute: {article_text}\n\n"
            prompt += f"Assertion: {query}\n\n"
            prompt += "Determine if the assertion is supported by the given statute. Respond with 'Assertion valid', 'Assertion invalid', or 'Insufficient legal context'. Start with reasoning, then conclusion."
        return prompt
    
    @staticmethod
    def prompt_template_4(query: str, clause_texts: List[str], article_text: str) -> str:
        """Template 4: Hypothesis testing format"""
        if clause_texts:
            prompt = f"Statute Compliance Test:\n\n"
            prompt += "Legal Clauses:\n"
            for i, clause in enumerate(clause_texts, 1):
                prompt += f"{i}. {clause}\n"
            prompt += f"\nLegal Text: {article_text}\n\n"
            prompt += f"Hypothesis: {query}\n\n"
            prompt += "Assess if the hypothesis aligns with the legal text(s). Options: 'Hypothesis compliant', 'Hypothesis non-compliant', or 'Cannot determine compliance'. Explain reasoning, then conclusion."
        else:
            prompt = f"Statute Compliance Test:\n\n"
            prompt += f"Legal Text: {article_text}\n\n"
            prompt += f"Hypothesis: {query}\n\n"
            prompt += "Assess if the hypothesis aligns with the legal text. Options: 'Hypothesis compliant', 'Hypothesis non-compliant', or 'Cannot determine compliance'. Explain reasoning, then conclusion."
        return prompt
    
    @staticmethod
    def prompt_template_5(query: str, clause_texts: List[str], article_text: str) -> str:
        """Template 5: Verification format"""
        if clause_texts:
            prompt = f"Legal Verification:\n\n"
            prompt += "Legal Provisions (Clauses):\n"
            for i, clause in enumerate(clause_texts, 1):
                prompt += f"{i}. {clause}\n"
            prompt += f"\nFull Legal Provision: {article_text}\n\n"
            prompt += f"Conjecture: {query}\n\n"
            prompt += "Is the conjecture in accordance with the supplied legal provision(s)? Answer: 'Conjecture verified', 'Conjecture unverified', or 'Unable to verify'. Provide explanation, then conclusion."
        else:
            prompt = f"Legal Verification:\n\n"
            prompt += f"Legal Provision: {article_text}\n\n"
            prompt += f"Conjecture: {query}\n\n"
            prompt += "Is the conjecture in accordance with the supplied legal provision? Answer: 'Conjecture verified', 'Conjecture unverified', or 'Unable to verify'. Provide explanation, then conclusion."
        return prompt
    
    @classmethod
    def get_all_templates(cls):
        """Get all prompt template functions"""
        return [
            cls.prompt_template_1,
            cls.prompt_template_2,
            cls.prompt_template_3,
            cls.prompt_template_4,
            cls.prompt_template_5
        ]


class ResponseParser:
    """Parse LLM responses to extract Yes/No decisions"""
    
    @staticmethod
    def extract_decision(response: str) -> Tuple[str, float]:
        """
        Extract decision from LLM response
        Returns: (decision, confidence) where decision is 'yes', 'no', or 'unsure'
        """
        response_lower = response.lower()
        
        # Positive indicators
        positive_keywords = [
            'yes', 'true', 'substantiated', 'valid', 'compliant', 'verified',
            'supported', 'correct', 'accurate', 'entails', 'entailment'
        ]
        
        # Negative indicators
        negative_keywords = [
            'no', 'false', 'unsubstantiated', 'invalid', 'non-compliant', 'unverified',
            'not supported', 'incorrect', 'inaccurate', 'does not entail', 'no entailment'
        ]
        
        # Unsure indicators
        unsure_keywords = [
            'unsure', 'uncertain', 'cannot determine', 'insufficient', 'not enough information',
            'more information required', 'unable to verify', 'cannot verify'
        ]
        
        # Count matches
        positive_count = sum(1 for keyword in positive_keywords if keyword in response_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in response_lower)
        unsure_count = sum(1 for keyword in unsure_keywords if keyword in response_lower)
        
        # Determine decision
        if unsure_count > 0 and unsure_count >= max(positive_count, negative_count):
            return ('unsure', 0.5)
        elif positive_count > negative_count:
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
            return ('yes', confidence)
        elif negative_count > positive_count:
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
            return ('no', confidence)
        else:
            return ('unsure', 0.5)


class SelfConsistencyVoter:
    """Aggregate multiple LLM responses using voting"""
    
    def __init__(self, voting_strategy: str = 'majority'):
        """
        Args:
            voting_strategy: 'majority' or 'weighted' (by confidence)
        """
        self.voting_strategy = voting_strategy
        self.parser = ResponseParser()
    
    def vote(self, responses: List[str]) -> Dict[str, Any]:
        """
        Vote on multiple responses
        Returns: {'decision': 'yes'/'no'/'unsure', 'confidence': float, 'votes': dict}
        """
        decisions = []
        confidences = []
        
        for response in responses:
            decision, confidence = self.parser.extract_decision(response)
            decisions.append(decision)
            confidences.append(confidence)
        
        # Count votes
        vote_counts = Counter(decisions)
        
        if self.voting_strategy == 'majority':
            # Simple majority vote
            most_common = vote_counts.most_common(1)[0]
            decision = most_common[0]
            confidence = most_common[1] / len(decisions)
        else:  # weighted
            # Weight by confidence scores
            weighted_votes = {'yes': 0.0, 'no': 0.0, 'unsure': 0.0}
            for decision, conf in zip(decisions, confidences):
                weighted_votes[decision] += conf
            
            decision = max(weighted_votes, key=weighted_votes.get)
            total_weight = sum(weighted_votes.values())
            confidence = weighted_votes[decision] / total_weight if total_weight > 0 else 0.5
        
        return {
            'decision': decision,
            'confidence': confidence,
            'votes': dict(vote_counts),
            'all_decisions': decisions,
            'all_confidences': confidences
        }


class EntailmentVerifier:
    """Light entailment verifier using semantic similarity"""
    
    def __init__(self, device: str = "cpu", similarity_threshold: float = 0.3):
        """
        Initialize entailment verifier using simple similarity-based approach
        Args:
            device: Device to run on
            similarity_threshold: Minimum similarity threshold for verification
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.use_model = False
        
        try:
            # Try to load a model for semantic similarity
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = self.model.to(device)
            self.model.eval()
            self.use_model = True
            print("[OK] Loaded sentence transformer for entailment verification")
        except Exception as e:
            print(f"[WARN] Could not load entailment model: {e}")
            print("Using keyword-based verification instead")
            self.use_model = False
            self.tokenizer = None
            self.model = None
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if not self.use_model:
            # Fallback: simple keyword overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if len(words1) == 0 or len(words2) == 0:
                return 0.0
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return overlap / union if union > 0 else 0.0
        
        try:
            # Encode texts
            inputs1 = self.tokenizer(text1, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs2 = self.tokenizer(text2, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
            
            with torch.no_grad():
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
                # Mean pooling
                emb1 = outputs1.last_hidden_state.mean(dim=1)
                emb2 = outputs2.last_hidden_state.mean(dim=1)
                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
                return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            # Fallback to keyword overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if len(words1) == 0 or len(words2) == 0:
                return 0.0
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return overlap / union if union > 0 else 0.0
    
    def verify(self, query: str, article_text: str, llm_decision: str) -> Dict[str, Any]:
        """
        Verify LLM decision using semantic similarity
        Args:
            query: Legal statement
            article_text: Article text
            llm_decision: LLM's decision ('yes', 'no', 'unsure')
        Returns:
            {'verified': bool, 'entailment_score': float, 'contradiction_score': float}
        """
        # Compute similarity between query and article
        similarity = self._compute_similarity(query, article_text)
        
        # Normalize to [0, 1] as entailment score
        entailment_score = similarity
        contradiction_score = 1 - similarity
        
        # Verify decision based on similarity and LLM decision
        if llm_decision == 'yes':
            # For 'yes', we expect high similarity (entailment)
            verified = entailment_score >= self.similarity_threshold
        elif llm_decision == 'no':
            # For 'no', we might have low similarity (contradiction) or high similarity but wrong decision
            # Be more lenient for 'no' decisions
            verified = True  # Trust 'no' decisions more
        else:
            # For 'unsure', always verify
            verified = True
        
        return {
            'verified': verified,
            'entailment_score': float(entailment_score),
            'contradiction_score': float(contradiction_score),
            'similarity_threshold': self.similarity_threshold,
            'method': 'model_based' if self.use_model else 'keyword_based'
        }


def main(args):
    print("=" * 60)
    print("Enhanced Revise Phase with Self-Consistency and Entailment Verifier")
    print("=" * 60)
    
    print(f"\nLoading LLM model from: {args.model_name_or_path}")
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"Using device: {device}")
    
    # Load the LLM model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, 
                trust_remote_code=True, 
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, 
                trust_remote_code=True, 
                device_map="cpu"
            )
        model = model.eval()
        print("✓ LLM model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to GPT-2...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
        model = model.eval()
        print("✓ Using fallback model: GPT-2")
    
    # Load the civil code
    print(f"\nLoading civil code from: {args.civil_code_file}")
    civil_code = get_civil_code(args.civil_code_file)
    print(f"[OK] Loaded {len(civil_code)} articles")
    
    # Initialize the clause retriever
    print(f"\nInitializing clause retriever with database: {args.clauses_db_path}")
    retriever = None
    try:
        retriever = ClauseRetriever(
            clauses_db_path=args.clauses_db_path,
            model_name="bert-base-uncased"
        )
        print("[OK] Clause retriever initialized successfully")
    except Exception as e:
        print(f"[WARN] Could not initialize clause retriever: {e}")
        print("Continuing without clause retrieval...")
    
    # Load the gold standard data
    print(f"\nLoading gold data from: {args.gold_file}")
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    print(f"[OK] Loaded {len(gold_data)} queries")
    
    # Initialize self-consistency voter
    voter = SelfConsistencyVoter(voting_strategy=args.voting_strategy)
    print(f"[OK] Self-consistency voter initialized (strategy: {args.voting_strategy})")
    
    # Initialize entailment verifier
    verifier = EntailmentVerifier(device=device)
    print(f"[OK] Entailment verifier initialized")
    
    # Get prompt templates
    prompt_templates = PromptTemplates.get_all_templates()
    if args.num_prompts > 0:
        prompt_templates = prompt_templates[:args.num_prompts]
    print(f"[OK] Using {len(prompt_templates)} prompt template(s) for self-consistency")
    
    # Process retrieved file
    print(f"\nProcessing retrieved file: {args.retrieved_file}")
    print(f"Output will be saved to: {args.output_file}")
    print(f"\n{'='*60}")
    
    processed_count = 0
    error_count = 0
    verified_count = 0
    rejected_count = 0
    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        with open(args.retrieved_file, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc="Processing queries"):
                try:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    
                    query_id = parts[0]
                    article_id = parts[2]
                    
                    # Get query text
                    if query_id not in gold_data:
                        error_count += 1
                        continue
                    
                    query = gold_data[query_id]['query']
                    article_text = civil_code.get(article_id, "")
                    if not article_text:
                        error_count += 1
                        continue
                    
                    # Get top clauses for the article (clause-informed prompting)
                    clause_texts = []
                    if retriever:
                        try:
                            top_clauses = retriever.get_top_clauses_for_article(article_id, query, top_k=3)
                            clause_texts = [retriever.clause_id_to_text[clause_id] for clause_id, _, _ in top_clauses if clause_id in retriever.clause_id_to_text]
                        except Exception as e:
                            pass  # Continue without clauses
                    
                    # Generate responses with multiple prompts (self-consistency)
                    responses = []
                    for template_fn in prompt_templates:
                        prompt_text = template_fn(query, clause_texts, article_text)
                        
                        # Generate response
                        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
                        if device == "cuda" and next(model.parameters()).is_cuda:
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=args.do_sample,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                            )
                        
                        input_length = inputs['input_ids'].shape[1]
                        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                        responses.append(response)
                    
                    # Vote on responses (self-consistency)
                    vote_result = voter.vote(responses)
                    final_decision = vote_result['decision']
                    final_confidence = vote_result['confidence']
                    
                    # Verify decision with entailment model
                    verification = verifier.verify(query, article_text, final_decision)
                    
                    # Filter based on verification
                    if args.use_verifier and not verification['verified']:
                        rejected_count += 1
                        if args.filter_rejected:
                            continue  # Skip rejected articles
                    else:
                        verified_count += 1
                    
                    # Save the result
                    result = {
                        "query_id": query_id,
                        "article_id": article_id,
                        "llm_responses": responses,
                        "final_decision": final_decision,
                        "final_confidence": final_confidence,
                        "vote_details": vote_result,
                        "verification": verification,
                        "has_clauses": len(clause_texts) > 0,
                        "clause_count": len(clause_texts)
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if args.verbose:
                        print(f"Error processing line: {line.strip()[:50]}... Error: {e}")
                    continue
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"[OK] Successfully processed: {processed_count} queries")
    print(f"[OK] Verified by entailment model: {verified_count} queries")
    print(f"[OK] Rejected by entailment model: {rejected_count} queries")
    print(f"[ERR] Errors: {error_count} queries")
    print(f"[OK] Output saved to: {args.output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Revise Phase with Self-Consistency and Entailment Verifier"
    )
    parser.add_argument("--retrieved_file", type=str, required=True, 
                       help="Path to the TSV file from the retrieval phase")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Path to save the output of the revise phase")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-1_8B-Chat", 
                       help="LLM to use for the revise phase")
    parser.add_argument("--civil_code_file", type=str, default="data/civil_code_en.json", 
                       help="Path to the civil code file")
    parser.add_argument("--clauses_db_path", type=str, default="data/ir_dataset/clauses_database.csv", 
                       help="Path to the clauses database")
    parser.add_argument("--gold_file", type=str, default="data/gold_R_series.json", 
                       help="Path to the gold standard JSON file")
    parser.add_argument("--num_prompts", type=int, default=3, 
                       help="Number of prompt templates to use for self-consistency (1-5)")
    parser.add_argument("--voting_strategy", type=str, default="majority", 
                       choices=["majority", "weighted"],
                       help="Voting strategy for self-consistency")
    parser.add_argument("--use_verifier", action="store_true", 
                       help="Use entailment verifier to filter results")
    parser.add_argument("--filter_rejected", action="store_true", 
                       help="Filter out articles rejected by entailment verifier")
    parser.add_argument("--max_new_tokens", type=int, default=150, 
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", 
                       help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, 
                       help="Top-p for nucleus sampling")
    parser.add_argument("--force_cpu", action="store_true", 
                       help="Force CPU usage even if GPU is available")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print verbose error messages")
    args = parser.parse_args()
    main(args)


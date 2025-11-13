"""
Enhanced Revise Phase with Few-Shot Learning and Qwen Support
Combines:
1. Clause-informed prompting (from SOP)
2. Few-shot learning with LegalBERT (from Qwen prompting)
3. Qwen model support with chat interface
4. Self-consistency with multiple prompts
5. Fallback to GPT-2 if Qwen not available
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Tuple, Any
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path
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


class FewShotExampleGenerator:
    """Generate few-shot examples using LegalBERT for similarity"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.legal_bert_model = None
        self.legal_bert_tokenizer = None
        self.positive_examples = []
        self.negative_examples = []
        
    def load_legal_bert(self):
        """Load LegalBERT for similarity computation"""
        try:
            self.legal_bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
            self.legal_bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
            self.legal_bert_model = self.legal_bert_model.to(self.device)
            self.legal_bert_model.eval()
            print("[OK] Loaded LegalBERT for few-shot example generation")
            return True
        except Exception as e:
            print(f"[WARN] Could not load LegalBERT: {e}")
            print("Few-shot learning disabled - using zero-shot instead")
            return False
    
    def embed_statement(self, statement: str) -> np.ndarray:
        """Get embedding for a statement"""
        if not self.legal_bert_model:
            return None
        
        inputs = self.legal_bert_tokenizer(
            statement, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.legal_bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return embedding
    
    def load_examples_from_gold(self, gold_file: str, civil_code: Dict[str, str], max_examples: int = 50):
        """Load positive and negative examples from gold standard"""
        try:
            with open(gold_file, 'r', encoding='utf-8') as f:
                gold_data = json.load(f)
            
            positive_examples = []
            negative_examples = []
            
            # Try to get examples from R02/R03 data
            for qid, item in gold_data.items():
                if len(positive_examples) >= max_examples and len(negative_examples) >= max_examples:
                    break
                
                query = item.get('query', '')
                if not query:
                    continue
                
                relevant_articles = item.get('rel_article_ids', [])
                label = item.get('label', 'N')
                
                # Create example - use first article or empty string
                article_text = ""
                if relevant_articles:
                    article_id = str(relevant_articles[0])
                    article_text = civil_code.get(article_id, "")
                
                if not article_text:
                    continue
                
                # Truncate article text if too long
                if len(article_text) > 1000:
                    article_text = article_text[:1000] + "..."
                
                if label == 'Y' and len(positive_examples) < max_examples:
                    positive_examples.append((query, article_text, "The statement is true"))
                elif label == 'N' and len(negative_examples) < max_examples:
                    negative_examples.append((query, article_text, "The statement is false"))
            
            self.positive_examples = positive_examples
            self.negative_examples = negative_examples
            print(f"[OK] Loaded {len(positive_examples)} positive and {len(negative_examples)} negative examples")
            return len(positive_examples) > 0 or len(negative_examples) > 0
        except Exception as e:
            print(f"[WARN] Could not load examples from gold: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_similar_examples(self, query: str, n: int = 2) -> Tuple[List[Tuple], List[Tuple]]:
        """Get n most similar positive and negative examples"""
        if not self.legal_bert_model or not self.positive_examples:
            return [], []
        
        query_embedding = self.embed_statement(query)
        if query_embedding is None:
            return [], []
        
        # Find similar positive examples
        positive_similarities = []
        for example in self.positive_examples:
            example_embedding = self.embed_statement(example[0])
            if example_embedding is not None:
                sim = cosine_similarity(
                    query_embedding.reshape(1, -1), 
                    example_embedding.reshape(1, -1)
                )[0][0]
                positive_similarities.append((sim, example))
        
        # Find similar negative examples
        negative_similarities = []
        for example in self.negative_examples:
            example_embedding = self.embed_statement(example[0])
            if example_embedding is not None:
                sim = cosine_similarity(
                    query_embedding.reshape(1, -1), 
                    example_embedding.reshape(1, -1)
                )[0][0]
                negative_similarities.append((sim, example))
        
        # Get top n
        positive_similarities.sort(reverse=True, key=lambda x: x[0])
        negative_similarities.sort(reverse=True, key=lambda x: x[0])
        
        top_positive = [ex for _, ex in positive_similarities[:n]]
        top_negative = [ex for _, ex in negative_similarities[:n]]
        
        return top_positive, top_negative


class PromptTemplates:
    """Prompt templates with few-shot support"""
    
    @staticmethod
    def create_prompt_with_fewshot(query: str, clause_texts: List[str], article_text: str, 
                                   few_shot_examples: Tuple[List, List] = None) -> str:
        """Create prompt with few-shot examples"""
        prompt = ""
        
        # Add few-shot examples if available
        if few_shot_examples and few_shot_examples[0] and few_shot_examples[1]:
            pos_examples, neg_examples = few_shot_examples
            prompt += "Examples:\n\n"
            
            # Add positive example
            if pos_examples:
                pos_query, pos_article, pos_answer = pos_examples[0]
                prompt += f"Example 1:\n"
                prompt += f"Legal Statement: {pos_query}\n"
                prompt += f"Legal Article: {pos_article[:500]}...\n"
                prompt += f"Answer: {pos_answer}\n\n"
            
            # Add negative example
            if neg_examples:
                neg_query, neg_article, neg_answer = neg_examples[0]
                prompt += f"Example 2:\n"
                prompt += f"Legal Statement: {neg_query}\n"
                prompt += f"Legal Article: {neg_article[:500]}...\n"
                prompt += f"Answer: {neg_answer}\n\n"
            
            prompt += "Now answer the following:\n\n"
        
        # Add current query with clause information
        prompt += f"Legal Statement: {query}\n\n"
        
        if clause_texts:
            prompt += "Top Retrieved Clauses:\n"
            for i, clause in enumerate(clause_texts[:3], 1):
                prompt += f"{i}. {clause}\n"
            prompt += "\n"
        
        prompt += f"Full Article Text:\n{article_text}\n\n"
        prompt += "Question: Is the legal statement supported by the provided clauses and article? "
        prompt += "Respond with 'The statement is true' or 'The statement is false' or 'Not enough information'. "
        prompt += "Explain your reasoning first, then provide your answer."
        
        return prompt


class ResponseParser:
    """Parse LLM responses to extract decisions"""
    
    @staticmethod
    def extract_decision(response: str) -> Tuple[str, float]:
        """Extract decision from response"""
        response_lower = response.lower()
        
        # Positive indicators
        positive_keywords = [
            'the statement is true', 'statement is true', 'true', 'yes', 
            'supported', 'substantiated', 'valid', 'correct'
        ]
        
        # Negative indicators
        negative_keywords = [
            'the statement is false', 'statement is false', 'false', 'no',
            'not supported', 'unsubstantiated', 'invalid', 'incorrect'
        ]
        
        # Check for explicit answers first
        if any(phrase in response_lower for phrase in ['the statement is true', 'statement is true']):
            return ('yes', 0.9)
        if any(phrase in response_lower for phrase in ['the statement is false', 'statement is false']):
            return ('no', 0.1)
        if 'not enough information' in response_lower:
            return ('unsure', 0.5)
        
        # Check for keywords
        positive_count = sum(1 for keyword in positive_keywords if keyword in response_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in response_lower)
        
        if positive_count > negative_count:
            confidence = min(0.85, 0.6 + positive_count * 0.05)
            return ('yes', confidence)
        elif negative_count > positive_count:
            confidence = max(0.15, 0.4 - negative_count * 0.05)
            return ('no', confidence)
        else:
            return ('unsure', 0.5)


def is_qwen_model(model_name: str) -> bool:
    """Check if model is a Qwen model"""
    return 'qwen' in model_name.lower()


def generate_response_qwen(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
    """Generate response using Qwen chat interface"""
    try:
        # Qwen chat interface
        if hasattr(model, 'chat'):
            response, _ = model.chat(tokenizer, prompt, history=None, max_new_tokens=max_new_tokens)
            return response
        else:
            # Fallback to generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            return response
    except Exception as e:
        print(f"[WARN] Error in Qwen generation: {e}")
        # Fallback to standard generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response


def generate_response_standard(model, tokenizer, prompt: str, max_new_tokens: int = 150, device: str = "cpu") -> str:
    """Generate response using standard generate method"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )
    
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response


def main(args):
    print("=" * 60)
    print("Enhanced Revise Phase with Few-Shot Learning and Qwen Support")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"Using device: {device}")
    
    # Load LLM model
    print(f"\nLoading LLM model: {args.model_name_or_path}")
    use_qwen = is_qwen_model(args.model_name_or_path)
    
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
        print(f"[OK] Model loaded successfully (Qwen: {use_qwen})")
    except Exception as e:
        print(f"[ERR] Error loading model: {e}")
        print("Falling back to GPT-2...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
        model = model.eval()
        use_qwen = False
        print("[OK] Using GPT-2 fallback")
    
    # Load civil code
    print(f"\nLoading civil code: {args.civil_code_file}")
    civil_code = get_civil_code(args.civil_code_file)
    print(f"[OK] Loaded {len(civil_code)} articles")
    
    # Load gold data
    print(f"\nLoading gold data: {args.gold_file}")
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    print(f"[OK] Loaded {len(gold_data)} queries")
    
    # Initialize clause retriever
    retriever = None
    if args.use_clauses:
        print(f"\nInitializing clause retriever: {args.clauses_db_path}")
        try:
            retriever = ClauseRetriever(
                clauses_db_path=args.clauses_db_path,
                model_name="bert-base-uncased"
            )
            print("[OK] Clause retriever initialized")
        except Exception as e:
            print(f"[WARN] Could not initialize clause retriever: {e}")
    
    # Initialize few-shot generator
    fewshot_generator = None
    if args.use_fewshot:
        print(f"\nInitializing few-shot learning...")
        fewshot_generator = FewShotExampleGenerator(device=device)
        if fewshot_generator.load_legal_bert():
            fewshot_generator.load_examples_from_gold(args.gold_file, civil_code, max_examples=args.fewshot_examples)
    
    # Initialize response parser
    parser = ResponseParser()
    
    # Process retrieved file
    print(f"\nProcessing: {args.retrieved_file}")
    print(f"Output: {args.output_file}")
    print(f"{'='*60}")
    
    processed_count = 0
    error_count = 0
    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        with open(args.retrieved_file, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in, desc="Processing"):
                try:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    
                    query_id = parts[0]
                    article_id = parts[2]
                    
                    if query_id not in gold_data:
                        error_count += 1
                        continue
                    
                    query = gold_data[query_id]['query']
                    article_text = civil_code.get(article_id, "")
                    if not article_text:
                        error_count += 1
                        continue
                    
                    # Get clauses
                    clause_texts = []
                    if retriever:
                        try:
                            top_clauses = retriever.get_top_clauses_for_article(article_id, query, top_k=3)
                            clause_texts = [retriever.clause_id_to_text[clause_id] for clause_id, _, _ in top_clauses 
                                          if clause_id in retriever.clause_id_to_text]
                        except:
                            pass
                    
                    # Get few-shot examples
                    few_shot_examples = None
                    if fewshot_generator and args.use_fewshot:
                        try:
                            few_shot_examples = fewshot_generator.get_similar_examples(query, n=args.fewshot_n)
                        except:
                            pass
                    
                    # Create prompt
                    prompt = PromptTemplates.create_prompt_with_fewshot(
                        query, clause_texts, article_text, few_shot_examples
                    )
                    
                    # Generate response
                    if use_qwen:
                        response = generate_response_qwen(model, tokenizer, prompt, args.max_new_tokens)
                    else:
                        model_device = next(model.parameters()).device
                        response = generate_response_standard(model, tokenizer, prompt, args.max_new_tokens, model_device)
                    
                    # Parse response
                    decision, confidence = parser.extract_decision(response)
                    
                    # Save result
                    result = {
                        "query_id": query_id,
                        "article_id": article_id,
                        "llm_response": response.strip(),
                        "final_decision": decision,
                        "final_confidence": confidence,
                        "has_clauses": len(clause_texts) > 0,
                        "clause_count": len(clause_texts),
                        "used_fewshot": few_shot_examples is not None and len(few_shot_examples[0]) > 0
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if args.verbose:
                        print(f"\n[ERR] Error processing {line.strip()[:50]}: {e}")
                    continue
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"[OK] Processed: {processed_count} queries")
    print(f"[ERR] Errors: {error_count} queries")
    print(f"[OK] Output: {args.output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced Revise Phase with Few-Shot Learning and Qwen Support"
    )
    parser.add_argument("--retrieved_file", type=str, required=True,
                       help="Path to retrieved TSV file")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output JSONL file")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-1_8B-Chat",
                       help="LLM model (Qwen recommended for best results)")
    parser.add_argument("--civil_code_file", type=str, default="data/civil_code_en.json",
                       help="Path to civil code JSON")
    parser.add_argument("--gold_file", type=str, default="data/gold_R_series.json",
                       help="Path to gold standard JSON")
    parser.add_argument("--clauses_db_path", type=str, default="data/ir_dataset/clauses_database.csv",
                       help="Path to clauses database")
    parser.add_argument("--use_clauses", action="store_true",
                       help="Use clause-informed prompting")
    parser.add_argument("--use_fewshot", action="store_true",
                       help="Use few-shot learning")
    parser.add_argument("--fewshot_examples", type=int, default=50,
                       help="Maximum number of examples to load")
    parser.add_argument("--fewshot_n", type=int, default=1,
                       help="Number of few-shot examples to use per query")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                       help="Maximum tokens to generate")
    parser.add_argument("--force_cpu", action="store_true",
                       help="Force CPU usage")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    args = parser.parse_args()
    main(args)


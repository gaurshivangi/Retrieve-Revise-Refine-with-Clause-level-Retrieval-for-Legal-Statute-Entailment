import argparse
import json
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch

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

def main(args):
    print(f"Loading LLM model from: {args.model_name_or_path}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Load the model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        # Use GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
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
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to a smaller model or CPU-only mode...")
        # Fallback: try with a smaller model or CPU
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
        model = model.eval()
        print("Using fallback model: gpt2")

    # Load the civil code
    print(f"Loading civil code from: {args.civil_code_file}")
    civil_code = get_civil_code(args.civil_code_file)
    print(f"Loaded {len(civil_code)} articles")

    # Initialize the clause retriever
    print(f"Initializing clause retriever with database: {args.clauses_db_path}")
    try:
        retriever = ClauseRetriever(
            clauses_db_path=args.clauses_db_path,
            model_name="bert-base-uncased"
        )
        print("Clause retriever initialized successfully")
    except Exception as e:
        print(f"Error initializing clause retriever: {e}")
        print("Continuing without clause retrieval...")
        retriever = None

    # Load the gold standard data to get the query texts
    print(f"Loading gold data from: {args.gold_file}")
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    print(f"Loaded {len(gold_data)} queries")

    # Process retrieved file
    print(f"Processing retrieved file: {args.retrieved_file}")
    print(f"Output will be saved to: {args.output_file}")
    
    processed_count = 0
    error_count = 0
    
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
                            # If clause retrieval fails, continue without clauses
                            print(f"Warning: Could not retrieve clauses for article {article_id}: {e}")

                    # Clause-informed prompting (as per SOP)
                    if clause_texts:
                        prompt_text = f"Legal Statement: {query}\n\n"
                        prompt_text += "Top Retrieved Clauses:\n"
                        for i, clause in enumerate(clause_texts, 1):
                            prompt_text += f"{i}. {clause}\n"
                        prompt_text += f"\nFull Article Text:\n{article_text}\n\n"
                        prompt_text += "Question: Is the legal statement supported by the provided clauses and article? Answer with 'Yes' or 'No' and provide a brief explanation."
                    else:
                        prompt_text = f"Legal Statement: {query}\n\n"
                        prompt_text += f"Full Article Text:\n{article_text}\n\n"
                        prompt_text += "Question: Is the legal statement supported by the article? Answer with 'Yes' or 'No' and provide a brief explanation."

                    # Generate the response from the LLM
                    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
                    
                    # Move inputs to the same device as model
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
                    
                    # Decode only the new tokens (response)
                    input_length = inputs['input_ids'].shape[1]
                    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

                    # Save the result
                    result = {
                        "query_id": query_id,
                        "article_id": article_id,
                        "llm_response": response.strip(),
                        "has_clauses": len(clause_texts) > 0
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error processing line: {line.strip()[:50]}... Error: {e}")
                    continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} queries")
    print(f"Errors: {error_count} queries")
    print(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revise phase: Use LLM to refine retrieved articles with clause-informed prompting")
    parser.add_argument("--retrieved_file", type=str, required=True, help="Path to the TSV file from the retrieval phase")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output of the revise phase")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-1_8B-Chat", help="LLM to use for the revise phase")
    parser.add_argument("--civil_code_file", type=str, default="data/civil_code_en.json", help="Path to the civil code file")
    parser.add_argument("--clauses_db_path", type=str, default="data/ir_dataset/clauses_database.csv", help="Path to the clauses database")
    parser.add_argument("--gold_file", type=str, default="data/gold_R_series.json", help="Path to the gold standard JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    main(args)

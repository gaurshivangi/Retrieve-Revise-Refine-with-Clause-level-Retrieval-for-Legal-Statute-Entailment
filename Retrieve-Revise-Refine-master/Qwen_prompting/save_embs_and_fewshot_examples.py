import json
import os
import traceback

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def embed_statement(statement, legal_bert_tokenizer, legal_bert_model):
    inputs = legal_bert_tokenizer(statement, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = legal_bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def run_qwen(prompt, qwen_tokenizer, qwen_model):
    # Wrapper for calling a local Qwen model. If the model isn't available
    # the caller should handle None and avoid calling this function.
    response, _ = qwen_model.chat(qwen_tokenizer, prompt, history=None, max_new_tokens=512)
    return response

def get_extracted_answer(llm_response, prompt_type):
    if prompt_type == 'prompt_llm_support_1':
        validity = 'Not enough information'.lower() not in llm_response.lower()
    elif prompt_type == 'prompt_llm_support_2':
        validity = llm_response.strip().lower().startswith('Claim substantiated'.lower()) or llm_response.strip().lower().startswith('Claim unsubstantiated'.lower())
    elif prompt_type == 'prompt_llm_support_3':
        validity = 'Insufficient legal context'.lower() not in llm_response.lower()
    elif prompt_type == 'prompt_llm_support_4':
        validity = 'Cannot determine compliance'.lower() not in llm_response.lower()
    elif prompt_type == 'prompt_llm_support_5':
        validity = 'Conjecture verified'.lower() in llm_response.lower() or 'Conjecture unverified'.lower() in llm_response.lower()
    else:
        validity = 'y' in llm_response
        
    return validity

def process_data(prompt_template, data, legal_bert_tokenizer, legal_bert_model, qwen_tokenizer, qwen_model):
    embeddings = {}
    positive_storage = []
    negative_storage = []
    
    for query_id, item in data.items():
        statement = item['statement']
        relevant_articles = item['relevant_articles']
        gold_label = item['gold_label']
        
        # Embed statement
        embeddings[query_id] = embed_statement(statement, legal_bert_tokenizer, legal_bert_model)
        
        # Optionally prepare prompt and run Qwen if tokenizer/model provided.
        llm_response = ''
        if qwen_tokenizer is not None and qwen_model is not None:
            prompt = prompt_template("\n".join(relevant_articles), statement)
            try:
                llm_response = run_qwen(prompt, qwen_tokenizer, qwen_model)
            except Exception:
                # If running Qwen fails, continue — we'll still create embeddings
                llm_response = ''

        # If we don't have an LLM response, fall back to gold label to populate
        # positive/negative storages. This creates few-shot examples from the
        # labeled data and embeddings without requiring a local large LLM.
        if 'y' in gold_label.lower():
            positive_storage.append((statement, relevant_articles, llm_response))
        else:
            negative_storage.append((statement, relevant_articles, llm_response))
    
    # Save embeddings, positive_storage, and negative_storage to current working directory
    out_dir = os.getcwd()
    emb_path = os.path.join(out_dir, 'embeddings.pt')
    pos_path = os.path.join(out_dir, 'positive_storage.json')
    neg_path = os.path.join(out_dir, 'negative_storage.json')
    torch.save(embeddings, emb_path)
    with open(pos_path, 'w', encoding='utf-8') as f:
        json.dump(positive_storage, f, ensure_ascii=False, indent=2)
    with open(neg_path, 'w', encoding='utf-8') as f:
        json.dump(negative_storage, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved positive_storage to: {pos_path}")
    print(f"Saved negative_storage to: {neg_path}")


if __name__ == '__main__':
    try:
        print("save_embs_and_fewshot_examples.py starting")
        print("cwd:", os.getcwd())
        # Load LegalBERT model and tokenizer
        legal_bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        legal_bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

        # Force skip Qwen (CPU-only, no LLM calls)
        qwen_tokenizer = None
        qwen_model = None
        print("Skipping Qwen load; proceeding without LLM to create storages and embeddings.")

        # Example usage (small sample if you don't have full data prepared)
        prompt_template = lambda content_of_articles, statement: f"""Given the following legal article(s) and legal statement:
\nLegal article(s): ```{content_of_articles}```
\nLegal statement: ```{statement}```
\nIs it possible to verify the accuracy of the legal statement using the provided legal article(s), or is the content of the legal article(s) insufficient?
\nPlease respond with either "The statement is true" or "The statement is false" or "Not enough information". Explain first then answer later."""

        data = {
            "query1": {
                "statement": "The right to freedom of speech is protected by law.",
                "relevant_articles": ["Article 19 of the Universal Declaration of Human Rights protects freedom of expression.",
                                    "The First Amendment to the US Constitution guarantees freedom of speech."],
                "gold_label": "yes"
            },
            "query2": {
                "statement": "All forms of speech are protected under law.",
                "relevant_articles": ["There are certain restrictions on freedom of speech, such as hate speech and defamation.",
                                    "The right to free speech is not absolute and can be limited in certain circumstances."],
                "gold_label": "no"
            }
        }

        # Process
        process_data(prompt_template, data, legal_bert_tokenizer, legal_bert_model, qwen_tokenizer, qwen_model)
        print("Processing complete")
    except Exception:
        print("Error running save_embs_and_fewshot_examples.py:")
        traceback.print_exc()
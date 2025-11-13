import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def embed_statement(statement, legal_bert_tokenizer, legal_bert_model):
    inputs = legal_bert_tokenizer(statement, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = legal_bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def run_qwen(prompt, qwen_tokenizer, qwen_model):
    response, _ = qwen_model.chat(qwen_tokenizer, prompt, history=None, max_new_tokens=512)
    return response

def get_top_similar_examples(query_embedding, storage, n):
    similarities = [cosine_similarity(query_embedding.reshape(1, -1), embed_statement(item[0]).reshape(1, -1))[0][0] for item in storage]
    top_indices = np.argsort(similarities)[-n:][::-1]
    return [storage[i] for i in top_indices]

def generate_fewshot_prompt(query_statement, relevant_articles, prompt_template, legal_bert_tokenizer, legal_bert_model, positive_storage, negative_storage, n=1):
    query_embedding = embed_statement(query_statement, legal_bert_tokenizer, legal_bert_model)
    
    top_positive = get_top_similar_examples(query_embedding, positive_storage, n)
    top_negative = get_top_similar_examples(query_embedding, negative_storage, n)
    
    few_shot_prompt = ""
    for pos, neg in zip(top_positive, top_negative):
        few_shot_prompt += prompt_template("\n".join(pos[1]), pos[0])
        few_shot_prompt += f"\nLLM Response: {pos[2]}\n\n"
        few_shot_prompt += prompt_template("\n".join(neg[1]), neg[0])
        few_shot_prompt += f"\nLLM Response: {neg[2]}\n\n"
    
    main_prompt = prompt_template("\n".join(relevant_articles), query_statement)
    full_prompt = few_shot_prompt + main_prompt
    return full_prompt
    
def few_shot_prompting(full_prompt, qwen_tokenizer, qwen_model):
    return run_qwen(full_prompt, qwen_tokenizer, qwen_model)


if __name__ == '__main__':
    # Load LegalBERT model and tokenizer
    legal_bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    legal_bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

    # Load Qwen model and tokenizer
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat", trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-Chat", device_map="auto", trust_remote_code=True).eval()
    
    # Example usage
    prompt_template = lambda content_of_articles, statement: f"""Given the following legal article(s) and legal statement:
\nLegal article(s): ```{content_of_articles}```
\nLegal statement: ```{statement}```
\nIs it possible to verify the accuracy of the legal statement using the provided legal article(s), or is the content of the legal article(s) insufficient?
\nPlease respond with either "The statement is true" or "The statement is false" or "Not enough information". Explain first then answer later."""

    query_statement = "Freedom of speech includes the right to use hate speech."
    relevant_articles = [
        "Many countries have laws against hate speech, which is not protected under freedom of speech.",
        "The United Nations advocates for restrictions on speech that incites discrimination, hostility, or violence."
    ]
    
    # Load stored data
    with open('positive_storage.json', 'r') as f:
        positive_storage = json.load(f)
    with open('negative_storage.json', 'r') as f:
        negative_storage = json.load(f)

    full_prompt = generate_fewshot_prompt(query_statement, relevant_articles, prompt_template, legal_bert_tokenizer, legal_bert_model, positive_storage, negative_storage)
    result = few_shot_prompting(full_prompt, qwen_tokenizer, qwen_model)
    print(f"Query Statement: {query_statement}")
    print(f"LLM Response: {result}")

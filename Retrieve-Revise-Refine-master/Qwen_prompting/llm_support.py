from datetime import datetime
from itertools import combinations
import json
import os
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import get_fewshot_prompt
import utils_prompt_llm


def get_civil_code(civil_code_file):
    with open(civil_code_file) as fin:
        file_content = json.load(fin)
        
    civil_code = dict()
    for article_id in file_content:
        foo = file_content[article_id]['content'].split(f"Article {article_id}  ")  # double spaces in the end
        assert len(foo) == 2
        civil_code[article_id] = f"Article {article_id}: " + foo[1]
    return civil_code


def preprocess_text(text):
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()


def get_top_k_preds(top_100_file, top_k):
    rel_preds = dict()
    with open(top_100_file) as fin:
        for line in fin:
            foo = line.split()
            qid = foo[0]
            related = foo[2]
            if qid not in rel_preds:
                rel_preds[qid] = []
            if related not in rel_preds[qid] and len(rel_preds[qid]) < top_k:
                rel_preds[qid].append(related)
    return rel_preds


def main(args):
    # Process args
    if args.output_file == 'auto':
        output_dir = './llm_outputs'
        args.output_file = os.path.join(output_dir, f"{args.prompt}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json")
    print(args)

    # Read gold file
    with open(args.gold_file) as fin:
        gold = json.load(fin)

    # Read Civil Code
    articles = get_civil_code(args.civil_code_file)

    # Get prompt template
    prompt_template = getattr(utils_prompt_llm, args.prompt, None)
    if prompt_template is None:
        raise Exception(f"The --prompt {args.prompt} does not exist!")
    
    # Load LegalBERT model and tokenizer
    legal_bert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    legal_bert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

    # Load Qwen model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Get top-k predictions from the top-100 file
    top_k_preds = get_top_k_preds(args.top_100_file, args.top_k)
    
    # Load stored data
    with open('positive_storage.json', 'r') as f:
        positive_storage = json.load(f)
    with open('negative_storage.json', 'r') as f:
        negative_storage = json.load(f)

    # Prompt LLM
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    fout = open(args.output_file, 'w', encoding='utf-8')
    for qid in gold:
        statement = gold[qid]['query']

        # Prompt with r articles from top-k predictions
        for r in range(1, 6):  # combination of 1 article, 2 articles, ..., 5 articles
            for combination in combinations(top_k_preds[qid], r):
                print(f"{qid}\t{combination}")
                article_contents = [articles[article_id] for article_id in combination]
                content_of_articles = '\n\n'.join(article_contents)
                user_message = prompt_template(content_of_articles, statement)
                user_message = get_fewshot_prompt.generate_fewshot_prompt(statement, content_of_articles, prompt_template, legal_bert_tokenizer, legal_bert_model, positive_storage, negative_storage, 2)

                if args.max_new_tokens:
                    answer, _ = model.chat(tokenizer, user_message, history=None, max_new_tokens=args.max_new_tokens)
                else:
                    answer, _ = model.chat(tokenizer, user_message, history=None)

                to_write = {
                    'qid': qid,
                    'article_ids': list(combination),
                    'llm_answer': answer,
                    'user_message': user_message,
                }
                print(json.dumps(to_write, ensure_ascii=False), file=fout, flush=True)
    
    fout.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gold_file', type=str, default='./data/gold_task3_task4.json')
    p.add_argument('--civil_code_file', type=str, default='./data/civil_code_en.json')
    p.add_argument('--prompt', type=str, default='prompt_llm_support_1')
    p.add_argument('--model_name_or_path', type=str, default='./models/huggingface_models/Qwen-14B-Chat')
    p.add_argument('--output_file', type=str, default='auto')
    p.add_argument('--top_100_file', type=str, default='./retrived/CAPTAIN.bjpAll.R04-L.tsv')
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--max_new_tokens', type=int, default=512)
    args = p.parse_args()

    main(args)

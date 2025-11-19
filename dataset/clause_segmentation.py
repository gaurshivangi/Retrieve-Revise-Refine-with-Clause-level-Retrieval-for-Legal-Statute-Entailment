#!/usr/bin/env python3
"""
COLIEE 2025 Task 3 - IR Model Training Data Creator

This script creates training data specifically formatted for Information Retrieval models
that can be used with popular IR frameworks like BM25, Dense Retrieval, etc.

The script creates:
1. Query-document pairs for training
2. Negative sampling for contrastive learning
3. Evaluation sets for model validation
4. Format compatible with popular IR libraries
"""

import os
import json
import pandas as pd
import random
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse


class IRTrainingDataCreator:
    """Creates training data for IR models"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.clauses = []
        self.query_pairs = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load the processed dataset"""
        # Load clauses database
        with open(os.path.join(self.dataset_dir, 'clauses_database.json'), 'r', encoding='utf-8') as f:
            self.clauses = json.load(f)
        
        # Load query pairs
        with open(os.path.join(self.dataset_dir, 'query_article_mappings.json'), 'r', encoding='utf-8') as f:
            self.query_pairs = json.load(f)
        
        print(f"Loaded {len(self.clauses)} clauses and {len(self.query_pairs)} query pairs")
    
    def create_bm25_training_data(self, output_dir: str):
        """Create training data for BM25-based IR models"""
        print("Creating BM25 training data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create document collection
        documents = []
        doc_id_to_text = {}
        
        for clause in self.clauses:
            doc_id = clause['clause_id']
            text = clause['full_text']
            documents.append({
                'doc_id': doc_id,
                'text': text,
                'article_number': clause['article_number'],
                'article_title': clause['article_title']
            })
            doc_id_to_text[doc_id] = text
        
        # Create queries with relevant documents
        queries = []
        qrels = []  # Query relevance judgments
        
        for pair in self.query_pairs:
            query_id = pair['query_id']
            query_text = pair['query']
            
            queries.append({
                'query_id': query_id,
                'query': query_text,
                'label': pair['label']
            })
            
            # Add relevance judgments for relevant articles
            for article_num in pair['relevant_articles']:
                # Find all clauses for this article
                relevant_clauses = [c for c in self.clauses if c['article_number'] == article_num]
                for clause in relevant_clauses:
                    qrels.append({
                        'query_id': query_id,
                        'doc_id': clause['clause_id'],
                        'relevance': 1
                    })
        
        # Save BM25 training data
        with open(os.path.join(output_dir, 'bm25_documents.json'), 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, 'bm25_queries.json'), 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, 'bm25_qrels.json'), 'w', encoding='utf-8') as f:
            json.dump(qrels, f, indent=2, ensure_ascii=False)
        
        print(f"Created BM25 training data: {len(documents)} docs, {len(queries)} queries, {len(qrels)} relevance judgments")
    
    def create_dense_retrieval_data(self, output_dir: str):
        """Create training data for dense retrieval models"""
        print("Creating dense retrieval training data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create positive pairs
        positive_pairs = []
        for pair in self.query_pairs:
            query_text = pair['query']
            for article_num in pair['relevant_articles']:
                relevant_clauses = [c for c in self.clauses if c['article_number'] == article_num]
                for clause in relevant_clauses:
                    positive_pairs.append({
                        'query': query_text,
                        'passage': clause['text'],
                        'passage_id': clause['clause_id'],
                        'article_number': clause['article_number'],
                        'label': pair['label']
                    })
        
        # Create negative pairs (random sampling)
        negative_pairs = []
        random.seed(42)
        
        for pair in self.query_pairs:
            query_text = pair['query']
            relevant_articles = set(pair['relevant_articles'])
            
            # Sample negative articles (not relevant to this query)
            all_articles = set(c['article_number'] for c in self.clauses)
            negative_articles = list(all_articles - relevant_articles)
            
            # Sample up to 3 negative examples per query
            num_negatives = min(3, len(negative_articles))
            sampled_negatives = random.sample(negative_articles, num_negatives)
            
            for article_num in sampled_negatives:
                negative_clauses = [c for c in self.clauses if c['article_number'] == article_num]
                if negative_clauses:
                    clause = random.choice(negative_clauses)
                    negative_pairs.append({
                        'query': query_text,
                        'passage': clause['text'],
                        'passage_id': clause['clause_id'],
                        'article_number': clause['article_number'],
                        'label': 'negative'
                    })
        
        # Combine positive and negative pairs
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Split into train/validation
        split_idx = int(0.8 * len(all_pairs))
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]
        
        # Save dense retrieval data
        with open(os.path.join(output_dir, 'dense_train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_pairs, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, 'dense_val.json'), 'w', encoding='utf-8') as f:
            json.dump(val_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"Created dense retrieval data: {len(train_pairs)} train, {len(val_pairs)} val pairs")
    
    def create_evaluation_sets(self, output_dir: str):
        """Create evaluation sets for model testing"""
        print("Creating evaluation sets...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Split queries into train/test
        random.seed(42)
        random.shuffle(self.query_pairs)
        
        split_idx = int(0.8 * len(self.query_pairs))
        train_queries = self.query_pairs[:split_idx]
        test_queries = self.query_pairs[split_idx:]
        
        # Create evaluation format similar to TREC
        eval_data = []
        
        for query_set, split_name in [(train_queries, 'train'), (test_queries, 'test')]:
            for pair in query_set:
                eval_data.append({
                    'split': split_name,
                    'query_id': pair['query_id'],
                    'query': pair['query'],
                    'relevant_articles': pair['relevant_articles'],
                    'label': pair['label']
                })
        
        # Save evaluation data
        with open(os.path.join(output_dir, 'evaluation_data.json'), 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        # Create TREC-style format
        trec_format = []
        for item in eval_data:
            for article_num in item['relevant_articles']:
                trec_format.append(f"{item['query_id']} Q0 {article_num} 1 1.0 IR_MODEL")
        
        with open(os.path.join(output_dir, 'trec_format.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(trec_format))
        
        print(f"Created evaluation sets: {len(train_queries)} train, {len(test_queries)} test queries")
    
    def create_corpus_for_indexing(self, output_dir: str):
        """Create corpus file for indexing with IR libraries"""
        print("Creating corpus for indexing...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        corpus_data = []
        for clause in self.clauses:
            corpus_data.append({
                'id': clause['clause_id'],
                'text': clause['full_text'],
                'article_number': clause['article_number'],
                'article_title': clause['article_title']
            })
        
        # Save in various formats
        with open(os.path.join(output_dir, 'corpus.json'), 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, indent=2, ensure_ascii=False)
        
        # Create text file for simple indexing
        with open(os.path.join(output_dir, 'corpus.txt'), 'w', encoding='utf-8') as f:
            for doc in corpus_data:
                f.write(f"{doc['id']}\t{doc['text']}\n")
        
        print(f"Created corpus with {len(corpus_data)} documents")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create IR model training data')
    parser.add_argument('--dataset_dir', default='ir_dataset',
                       help='Directory containing processed dataset')
    parser.add_argument('--output_dir', default='ir_training_data',
                       help='Output directory for training data')
    
    args = parser.parse_args()
    
    # Create IR training data
    creator = IRTrainingDataCreator(args.dataset_dir)
    
    # Create different types of training data
    creator.create_bm25_training_data(os.path.join(args.output_dir, 'bm25'))
    creator.create_dense_retrieval_data(os.path.join(args.output_dir, 'dense'))
    creator.create_evaluation_sets(os.path.join(args.output_dir, 'evaluation'))
    creator.create_corpus_for_indexing(os.path.join(args.output_dir, 'corpus'))
    
    print("IR training data creation completed!")


if __name__ == "__main__":
    main()




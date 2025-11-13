#!/usr/bin/env python3
"""
Demonstration script for clause-level retrieval system
Shows the novel features implemented according to IR_SOP document
"""

import sys
import os
import json
import pandas as pd
from clause_retriever import ClauseRetriever, ClauseAwareAggregator, ClauseRetrievalEvaluator

def main():
    print("Clause-Level Retrieval System Demo")
    print("==================================")
    print("Implementing novel features from IR_SOP document:")
    print("1. Clause Segmentation: Break statutes into clauses for fine-grained retrieval")
    print("2. Hybrid Retrieval: Combine dense embeddings with BM25 sparse retrieval")
    print("3. Clause-aware Aggregation: Boost articles with multiple entailing clauses")
    print()
    
    # Initialize retriever
    print("Initializing clause retriever...")
    clauses_db_path = "../data/ir_dataset/clauses_database.csv"
    
    if not os.path.exists(clauses_db_path):
        print(f"Error: Clauses database not found at {clauses_db_path}")
        print("Please ensure the dataset is properly set up.")
        return
    
    retriever = ClauseRetriever(
        clauses_db_path=clauses_db_path,
        model_name="bert-base-uncased",
        dense_weight=0.7,
        boost_factor=1.5,
        top_k=50
    )
    
    print(f"✓ Loaded {len(retriever.clause_ids)} clauses from {len(retriever.article_to_clauses)} articles")
    print()
    
    # Test queries from the dataset
    test_queries = [
        "A minor can freely dispose of assets the statutory agent permits the disposition without specifying the purpose.",
        "If the other party of a manifestation of intention was a minor when the manifestation of intention was received, that manifestation of intention does not take effect.",
        "A person who has reached the age of majority has full capacity to act."
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test Query {i}: {query}")
        print("-" * 80)
        
        # Retrieve clauses
        print("1. Clause-level Retrieval (Hybrid Dense + Sparse):")
        clause_results = retriever.retrieve_clauses(query, top_k=10)
        
        for j, (clause_id, score, article_num) in enumerate(clause_results[:5], 1):
            clause_text = retriever.clause_id_to_text[clause_id]
            print(f"   {j}. Article {article_num}, Clause {clause_id} (Score: {score:.3f})")
            print(f"      Text: {clause_text[:100]}...")
            print()
        
        # Aggregate to article level
        print("2. Clause-aware Aggregation (with boosting for multiple clauses):")
        article_results = retriever.retrieve_articles(query, top_k=5)
        
        for j, (article_num, result) in enumerate(article_results.items(), 1):
            boosted = " (BOOSTED)" if result['boosted'] else ""
            print(f"   {j}. Article {article_num} (Score: {result['score']:.3f}){boosted}")
            print(f"      Clauses: {result['clause_count']}, Max: {result['max_clause_score']:.3f}, Avg: {result['avg_clause_score']:.3f}")
            
            # Show top clauses for this article
            top_clauses = retriever.get_top_clauses_for_article(article_num, query, top_k=3)
            for clause_id, score, _ in top_clauses:
                clause_text = retriever.clause_id_to_text[clause_id]
                print(f"         - {clause_id}: {clause_text[:60]}... (Score: {score:.3f})")
            print()
        
        print("=" * 80)
        print()
    
    # Show statistics
    print("System Statistics:")
    print(f"- Total clauses: {len(retriever.clause_ids)}")
    print(f"- Total articles: {len(retriever.article_to_clauses)}")
    print(f"- Average clauses per article: {sum(len(clauses) for clauses in retriever.article_to_clauses.values()) / len(retriever.article_to_clauses):.1f}")
    print()
    
    # Show boosting effect
    print("Clause-aware Aggregation Boosting:")
    print("- Articles with multiple strong clauses get boosted scores")
    print(f"- Boost factor: {retriever.boost_factor}x")
    print("- This helps surface articles with multiple entailing clauses")
    print()
    
    print("Hybrid Retrieval:")
    print(f"- Dense weight: {retriever.dense_weight} (BERT embeddings)")
    print(f"- Sparse weight: {1 - retriever.dense_weight} (BM25)")
    print("- Combines semantic similarity with exact keyword matching")
    print()
    
    print("Demo completed! The system is ready for training.")

if __name__ == "__main__":
    main()

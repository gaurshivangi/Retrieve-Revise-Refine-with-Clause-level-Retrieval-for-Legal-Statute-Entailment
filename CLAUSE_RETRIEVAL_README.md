# Clause-Level Retrieval System for Legal Statute Entailment

This implementation adds novel clause-level retrieval features to the Retrieve-Revise-Refine (RRR) framework for the COLIEE competition, as described in the IR_SOP document.

## Novel Features Implemented

### 1. Clause Segmentation
- **Purpose**: Break statutes into clauses for fine-grained retrieval instead of using entire articles
- **Implementation**: Uses pre-segmented clause database (`clauses_database.csv`)
- **Benefit**: Reduces noise from irrelevant clauses and improves precision

### 2. Hybrid Retrieval
- **Dense Retrieval**: BERT-based semantic embeddings for semantic similarity
- **Sparse Retrieval**: BM25 for exact keyword matching
- **Combination**: Weighted combination (default: 70% dense, 30% sparse)
- **Benefit**: Captures both semantic meaning and exact keyword matches

### 3. Clause-aware Aggregation
- **Purpose**: Aggregate clause-level scores to article-level with boosting
- **Boosting Strategy**: Articles with multiple strong clauses get boosted scores
- **Formula**: `score = max_clause_score + (avg_score * (clause_count - 1) * boost_factor)`
- **Benefit**: Surfaces articles with multiple entailing clauses

## File Structure

```
src/
├── clause_retriever.py              # Core clause retrieval system
├── train_task3_clause_retrieval.py  # Integrated training for Task 3
├── demo_clause_retrieval.py         # Demonstration script
└── clause_retrieval_config.json     # Configuration file

data/ir_dataset/
├── clauses_database.csv             # Pre-segmented clauses
├── training_data.csv               # Training data with clause labels
└── query_article_mappings.csv      # Query-article mappings

scripts/
└── train_clause_retrieval.sh        # Training script
```

## Installation

1. Install additional dependencies:
```bash
pip install -r requirements_clause_retrieval.txt
```

2. Ensure the dataset is properly set up in `data/ir_dataset/`

## Usage

### Quick Demo
```bash
cd src
python demo_clause_retrieval.py
```

### Training the Model


#### Integrated Training (Task 3)
```bash
python train_task3_clause_retrieval.py \
    --data_dir ../data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03 \
    --clauses_db_path ../data/ir_dataset/clauses_database.csv \
    --model_name_or_path bert-base-uncased \
    --use_clause_retrieval \
    --dense_weight 0.7 \
    --boost_factor 1.5
```

#### Option 3: Using the Script
```bash
chmod +x scripts/train_clause_retrieval.sh
./scripts/train_clause_retrieval.sh
```

## Configuration

Key parameters in `clause_retrieval_config.json`:

- `dense_weight`: Weight for dense retrieval (0.0-1.0)
- `boost_factor`: Multiplier for articles with multiple clauses
- `top_k`: Number of top results to retrieve
- `model_name`: HuggingFace model for dense retrieval

## Architecture

### ClauseRetriever Class
- **Hybrid Retrieval**: Combines BERT embeddings with BM25
- **FAISS Index**: Efficient similarity search for dense retrieval
- **Clause Database**: Manages clause-to-article mappings

### ClauseAwareAggregator Class
- **Score Aggregation**: Combines clause scores into article scores
- **Boosting Logic**: Enhances scores for multi-clause articles
- **Flexible Weighting**: Configurable boost factors

### Integration with Existing Pipeline
- **Enhanced Preprocessor**: Incorporates clause information into input
- **Modified Classifier**: Uses clause-aware features
- **Backward Compatibility**: Works with existing evaluation pipeline

## Evaluation

The system maintains compatibility with existing COLIEE evaluation metrics:
- **Precision**: Correctly retrieved articles / Total retrieved
- **Recall**: Correctly retrieved articles / Total relevant
- **F2 Score**: Weighted harmonic mean (emphasizes recall)

## Performance Improvements

### Expected Benefits:
1. **Higher Recall**: Fine-grained clause retrieval finds more relevant content
2. **Better Precision**: Clause-level filtering reduces noise
3. **Improved F2**: Better balance of precision and recall
4. **Interpretability**: Can point to specific entailing clauses

### Novel Aggregation Benefits:
- Articles with multiple entailing clauses get higher scores
- Reduces false negatives for complex legal questions
- Better handles cases where multiple clauses are needed

## Example Usage

```python
from clause_retriever import ClauseRetriever

# Initialize retriever
retriever = ClauseRetriever(
    clauses_db_path="data/ir_dataset/clauses_database.csv",
    model_name="bert-base-uncased",
    dense_weight=0.7,
    boost_factor=1.5
)

# Retrieve clauses
query = "A minor can freely dispose of assets..."
clause_results = retriever.retrieve_clauses(query, top_k=20)

# Aggregate to articles
article_results = retriever.retrieve_articles(query, top_k=10)

# Get top clauses for specific article
top_clauses = retriever.get_top_clauses_for_article("5", query, top_k=3)
```


```

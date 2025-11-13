#!/bin/bash

# Script to train the clause-level retrieval system for Task 3
# Implements the novel features from IR_SOP document

echo "Starting Clause-Level Retrieval Training for Task 3"
echo "=================================================="

# Set paths
DATA_DIR="../data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03"
CLAUSES_DB_PATH="../data/ir_dataset/clauses_database.csv"
LOG_DIR="../settings/clause_retrieval_bert_base"
MODEL_NAME="bert-base-uncased"

# Create log directory
mkdir -p $LOG_DIR

# Training arguments
python ../src/train_task3_clause_retrieval.py \
    --data_dir $DATA_DIR \
    --clauses_db_path $CLAUSES_DB_PATH \
    --log_dir $LOG_DIR \
    --model_name_or_path $MODEL_NAME \
    --max_epochs 10 \
    --batch_size 32 \
    --max_seq_length 512 \
    --learning_rate 2e-5 \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --dense_weight 0.7 \
    --boost_factor 1.5 \
    --retrieval_top_k 100 \
    --use_clause_retrieval \
    --file_output_id "clauseRetrieval" \
    --gpus 0

echo "Training completed!"
echo "Results saved in: $LOG_DIR"

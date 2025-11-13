#!/usr/bin/bash
#
# Training script for English Task 3 (Statute Law Retrieval)
# This script trains the original retrieve-revise-refine baseline model

MODEL_NAME="bert-base-uncased"
ROOT_DIR="."
DATA_DIR="${ROOT_DIR}/data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03"
MAX_EP=5
MAX_SEQ=512
LR=2e-5
SETTING_NAME="bert-base-uncased_en_task3_${MAX_EP}ep_${MAX_SEQ}seq_L${LR}"
SETTING_DIR="${ROOT_DIR}/settings/${SETTING_NAME}/"

mkdir -p $SETTING_DIR

echo "Starting training for Task 3 (English)"
echo "Model: $MODEL_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $SETTING_DIR"
echo "Max epochs: $MAX_EP"
echo "Max sequence length: $MAX_SEQ"
echo "Learning rate: $LR"

cd src/
python train.py \
  --data_dir $DATA_DIR \
  --model_name_or_path $MODEL_NAME \
  --log_dir $SETTING_DIR \
  --max_epochs $MAX_EP \
  --batch_size 16 \
  --max_keep_ckpt 1 \
  --lr $LR \
  --gpus 0 \
  --max_seq_length $MAX_SEQ \
  --file_output_id allEnss \
  --civi_code_path "" \
  > ${SETTING_DIR}/train.log 2>&1

echo "Training completed. Check ${SETTING_DIR}/train.log for details."


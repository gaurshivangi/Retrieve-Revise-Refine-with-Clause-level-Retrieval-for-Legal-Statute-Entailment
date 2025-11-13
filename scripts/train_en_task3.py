#!/usr/bin/env python
"""
Training script for English Task 3 (Statute Law Retrieval)
This script trains the original retrieve-revise-refine baseline model
"""
import os
import sys
import subprocess

# Configuration
MODEL_NAME = "bert-base-uncased"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03")
# Allow overriding via environment variables so wrapper can be reused without editing.
MAX_EP = int(os.environ.get("TRAIN_MAX_EPOCHS", "10"))
MAX_SEQ = 512
LR = 2e-5
SETTING_NAME = f"bert-base-uncased_en_task3_{MAX_EP}ep_{MAX_SEQ}seq_L{LR}"
SETTING_DIR = os.path.join(ROOT_DIR, "settings", SETTING_NAME)

# Create settings directory
os.makedirs(SETTING_DIR, exist_ok=True)

print("=" * 80)
print("Starting training for Task 3 (English)")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {SETTING_DIR}")
print(f"Max epochs: {MAX_EP}")
print(f"Max sequence length: {MAX_SEQ}")
print(f"Learning rate: {LR}")
print("=" * 80)

# Change to src directory
src_dir = os.path.join(ROOT_DIR, "src")
os.chdir(src_dir)

# Build command
cmd = [
    sys.executable,
    "train.py",
    "--data_dir", DATA_DIR,
    "--model_name_or_path", MODEL_NAME,
    "--log_dir", SETTING_DIR,
    "--max_epochs", str(MAX_EP),
    "--batch_size", "8",
    "--max_keep_ckpt", "1",
    "--lr", str(LR),
    "--gpus", "0",
    "--max_seq_length", str(MAX_SEQ),
    "--file_output_id", "allEnss",
]

# Allow setting validation check interval via TRAIN_VAL_CHECK env var (e.g. 0.5 for 50% of an epoch)
val_check = os.environ.get("TRAIN_VAL_CHECK", None)
if val_check is not None:
    cmd += ["--val_check_interval", str(val_check)]

# Run training
print("\nExecuting training command...")
print(f"Command: {' '.join(cmd)}")
print("\n" + "=" * 80 + "\n")

import sys
with open(os.path.join(SETTING_DIR, "train.log"), "w", buffering=1) as log_file:
    result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, bufsize=0)

if result.returncode == 0:
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Check {SETTING_DIR}/train.log for details.")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print(f"Training failed with exit code {result.returncode}")
    print(f"Check {SETTING_DIR}/train.log for error details.")
    print("=" * 80)
    sys.exit(1)


#!/usr/bin/env python3
"""
Quick Start Script for Google Colab
Run this in a Colab cell to set up and train the clause-level retrieval system
"""

# Cell 1: Install Dependencies
print("Installing dependencies...")
import subprocess
import sys

packages = [
    "torch", "transformers", "pytorch-lightning", "scikit-learn", 
    "pandas", "numpy", "faiss-cpu", "rank-bm25", "tqdm"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("✅ Dependencies installed!")

# Cell 2: Setup Environment and Fix CUDA Issues
import os
import torch
import multiprocessing

# Fix CUDA multiprocessing issues
try:
    multiprocessing.set_start_method('spawn', force=True)
    print("✅ Fixed CUDA multiprocessing")
except RuntimeError:
    print("⚠️  Multiprocessing already configured")

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()
    device = "cuda"
    batch_size = 8  # Reduced for stability
else:
    print("❌ No GPU - Enable GPU in Runtime > Change runtime type")
    device = "cpu"
    batch_size = 8

print("✅ Environment configured!")

# Cell 3: Upload Data (Manual step)
print("""
📁 Please upload these files to your Colab environment:
1. data/ir_dataset/clauses_database.csv
2. data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03/train.csv
3. data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03/dev.csv
4. data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03/test.csv
5. src/ folder (all Python files)

Use the file upload button in Colab or drag and drop the files.
""")

# Cell 4: Run Demo
#print("Running demo...")
#import subprocess
#result = subprocess.run(["python", "src/demo_clause_retrieval.py"], 
       #                capture_output=True, text=True, cwd=".")
#print(result.stdout)
#if result.stderr:
 #   print("Errors:", result.stderr)

# Cell 5: Train Model
print("Starting training...")
save_dir = "/kaggle/working/settings/clause_retrieval_gpu"
# Create log directory
os.makedirs("settings/clause_retrieval_gpu", exist_ok=True)
# Enable checkpoint saving every 1000 steps
# (handled internally in your Lightning script)
ckpt_path = os.path.join(save_dir, "last.ckpt")

# If a checkpoint exists, resume from it
resume_args = []
if os.path.exists(ckpt_path):
    print(f"🟡 Resuming from existing checkpoint: {ckpt_path}")
    resume_args = ["--pretrained_checkpoint", ckpt_path]
else:
    print("🟢 Starting fresh training run...")

# Training command
cmd = [
    "python", "src/train_task3_clause_retrieval.py",
    "--data_dir", "data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03",
    "--clauses_db_path", "data/ir_dataset/clauses_database.csv",
    "--log_dir", "settings/clause_retrieval_gpu",
    "--model_name_or_path", "bert-base-uncased",
    "--max_epochs", "1",
    "--batch_size", str(batch_size),
    "--max_seq_length", "256",
    "--lr", "2e-5",
    "--dropout", "0.1",
    "--weight_decay", "0.01",
    "--dense_weight", "0.7",
    "--boost_factor", "1.5",
    "--retrieval_top_k", "150",
    "--use_clause_retrieval",
    "--file_output_id", "clauseRetrievalGPU"
] + resume_args

print(f"Running: {' '.join(cmd)}")

# Run training
try:
    result = subprocess.run(cmd, cwd=".")
    if result.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
except Exception as e:
    print(f"❌ Error during training: {e}")

# Cell 6: Check Results
print("Checking results...")
import os
if os.path.exists("settings/clause_retrieval_gpu"):
    files = os.listdir("settings/clause_retrieval_gpu")
    print(f"✅ Results saved in settings/clause_retrieval_gpu/")
    print(f"   Files: {files}")
    
    # Check for model checkpoints
    ckpt_files = [f for f in files if f.endswith('.ckpt')]
    if ckpt_files:
        print(f"   Model checkpoints: {ckpt_files}")
    else:
        print("   No model checkpoints found")
else:
    print("❌ No results directory found")

print("""
🎉 Setup complete! 

To download results:
1. Run: !zip -r results.zip settings/clause_retrieval_gpu/
2. Run: from google.colab import files; files.download('results.zip')

To monitor training:
1. Run: !tail -f settings/clause_retrieval_gpu/run.log
2. Run: !nvidia-smi (to check GPU usage)
""")

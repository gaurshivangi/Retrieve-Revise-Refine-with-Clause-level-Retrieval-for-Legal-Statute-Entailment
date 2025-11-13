# Retrieve-Revise-Refine with Clause-level Retrieval for Legal Statute Entailment

This is our Information Retrieval (DS501) course project where we extend the Retrieve-Revise-Refine (RRR) framework for legal statute retrieval. We focus on improving legal question answering by breaking down statutes into clauses and retrieving them at a finer level.

## Project Overview

The goal is to answer legal bar exam questions by finding the right set of Civil Code articles that entail the answer. Instead of using whole articles, we break them into clauses and retrieve at clause-level, which helps us find more relevant information and improve accuracy.

## Problem Statement

Legal articles are usually long and contain multiple clauses, but only a few clauses might be relevant to a question. Using whole articles causes:
- *Dilution of meaning*: Irrelevant clauses add noise
- *Loss of recall*: Important clauses get hidden in long articles

Our solution: Retrieve at clause-level, then aggregate to article-level with boosting for articles that have multiple relevant clauses.

## Methodology

Our approach has three main stages:

### 1. Retrieve (Clause-level Retrieval)
- Break statutes into clauses/sentences
- Use hybrid retrieval: dense embeddings (LegalBERT) + sparse methods (BM25)
- Aggregate clause scores to article-level with boosting for multiple strong clauses

### 2. Revise (LLM-based Refinement)
- Use clause-informed prompting: pass both top clauses and full article to LLM
- Support for Qwen models with few-shot learning
- Optional self-consistency with multiple prompts
- Light entailment verifier to filter out irrelevant statutes

### 3. Refine (Final Optimization)
- Clause-evidence boosting: articles with multiple entailing clauses get higher scores
- Diversity-aware selection using MMR (Maximal Marginal Relevance)
- Dynamic cut-off based on confidence thresholds instead of fixed top-k

## Dataset

We use the *COLIEE* (Competition on Legal Information Extraction and Entailment) dataset:
- Legal bar exam questions in English (translated from Japanese)
- Japanese Civil Code articles (724 articles)
- Task 3: Retrieve relevant articles for each question

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Installation

1. Clone the repository:
bash
git clone <repository-url>
cd CAPTAIN-COLIEE2023-CAPTAIN-COLIEE2023


2. Create a virtual environment:
bash 
conda create -n env_coliee python=3.8
conda activate env_coliee


3. Install dependencies:
bash
pip install -r requirements.txt
pip install -r requirements_clause_retrieval.txt


4. Install additional dependencies for Qwen support:
bash
pip install torch transformers scikit-learn pandas faiss-cpu tiktoken einops transformers_stream_generator accelerate


5. Prepare data files:
            bash
# Create civil code JSON
python scripts/create_civil_code_json.py

# Create gold standard JSON (if needed)
python scripts/generate_gold_file.py


## Usage

### Complete Pipeline (Retrieve → Revise → Refine)

#### Option 1: Using PowerShell Script (Windows)
powershell
.\run_complete_pipeline.ps1


#### Option 2: Using Bash Script (Linux/Mac)
bash
bash run_complete_pipeline.sh


#### Option 3: Manual Step-by-Step

*Step 1: Retrieve Phase*
            bash
# This generates clause-level retrieval results
# (Implementation depends on your retrieval method)


*Step 2: Revise Phase (with Qwen + Few-shot)*
bash
python Retrieve-Revise-Refine-master/revise_fewshot_qwen.py \
  --retrieved_file "settings/clause_retrieval_gpu/CAPTAIN.clauseRetrieval.R02.tsv" \
  --output_file "settings/clause_retrieval_gpu/revise_fewshot_R02_output.jsonl" \
  --model_name_or_path "Qwen/Qwen-1_8B-Chat" \
  --use_clauses \
  --use_fewshot \
  --fewshot_n 1 \
  --fewshot_examples 50 \
  --max_new_tokens 150


*Step 3: Refine Phase*
bash
python Retrieve-Revise-Refine-master/refine.py \
  --revise_output_file "settings/clause_retrieval_gpu/revise_fewshot_R02_output.jsonl" \
  --output_file "settings/clause_retrieval_gpu/refine_R02_output.tsv" \
  --detailed_output_file "settings/clause_retrieval_gpu/refine_R02_output.json" \
  --use_boosting \
  --use_mmr \
  --use_dynamic_cutoff \
  --min_confidence 0.3 \
  --score_gap_threshold 0.4 \
  --max_articles 25 \
  --min_articles 1


*Step 4: Evaluation*
bash
python Retrieve-Revise-Refine-master/eval_2023_predictions.py \
  "settings/clause_retrieval_gpu/refine_R02_output.tsv" \
  "COLIEE2025statute_data-English/train/riteval_R02_en.xml"


## Project Structure


.
├── Retrieve-Revise-Refine-master/    # Main pipeline code
│   ├── revise_fewshot_qwen.py        # Revise phase with Qwen support
│   ├── refine.py                      # Refine phase
│   ├── eval_2023_predictions.py      # Evaluation script
│   └── Qwen_prompting/               # Qwen-specific utilities
├── src/                               # Source code for retrieval
│   ├── clause_retriever.py           # Clause-level retriever
│   └── data_utils/                   # Data processing utilities
├── scripts/                          # Utility scripts
│   ├── create_civil_code_json.py     # Create civil code JSON
│   └── generate_gold_file.py         # Generate gold standard
├── data/                             # Processed data files
├── logs/                             # Log files
├── settings/                         # Output directory and model checkpoints
├── COLIEE2025statute_data-English/  # Dataset files
├── IR_SOP (1).txt                   # Statement of Purpose
├── run_complete_pipeline.ps1        # Windows pipeline script
├── run_complete_pipeline.sh          # Linux/Mac pipeline script
└── requirements*.txt                 # Dependencies


## Key Features

- ✅ *Clause-level retrieval* for fine-grained matching
- ✅ *Hybrid retrieval* (dense + sparse) for better coverage
- ✅ *Qwen model support* with few-shot learning
- ✅ *Clause-informed prompting* for better LLM decisions
- ✅ *Clause-evidence boosting* in refinement
- ✅ *Diversity-aware selection* using MMR
- ✅ *Dynamic cut-off* based on confidence

## Team

*LexQuest Team* - DS501 Information Retrieval Project
- Member 1: Works on Retrieve stage
- Member 2: Works on Revise stage  
- Member 3: Works on Refine stage

## References

- COLIEE Competition: https://sites.ualberta.ca/~rabelo/COLIEE2023/
- Original RRR Framework: https://github.com/G-AOARD/Retrieve-Revise-Refine
- CAPTAIN at COLIEE 2023 Paper


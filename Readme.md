# Retrieve → Revise → Refine Pipeline Documentation

This document provides a comprehensive guide for executing the complete pipeline for COLIEE statute law retrieval and entailment tasks, covering the Retrieve, Revise, and Refine stages.

---

## 0. Prerequisites

### System Requirements
- **Operating System**: Windows (PowerShell)
- **Python**: Version 3.10 or higher
- **Hardware**: GPU with CUDA support (optional but recommended)
- **Working Directory**: `C:\Users\bhoom\Downloads\CAPTAIN-COLIEE2023-CAPTAIN-COLIEE2023_TRAINED`

### Environment Setup
1. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Data Requirements
- COLIEE 2025 competition datasets must be extracted under `COLIEE2025statute_data-English/`
- Training data files: `train/riteval_R02_en.xml` and `text/civil_code_en-1to724-2.txt`

---

## 1. Retrieve Stage

The Retrieve stage trains a neural retrieval model to produce ranked candidate articles for each query.

### 1.1 Data Preparation

Execute the data preparation script to generate training splits:

```powershell
python scripts/prepare_revise_data.py `
  --xml_file COLIEE2025statute_data-English/train/riteval_R02_en.xml `
  --text_file COLIEE2025statute_data-English/text/civil_code_en-1to724-2.txt `
  --output_dir data/COLIEE2025statute_data-English/data_en_topk_150_r02_r03
```

**Expected Output**: The target directory should contain `train.csv`, `dev.csv`, and `test.csv` files.

### 1.2 Model Training

Train the retrieval model using the prepared data:

```powershell
python scripts/train_en_task3.py
```

**Configuration**: The number of training epochs is controlled by the `TRAIN_MAX_EPOCHS` environment variable (default: 10).

**Output Location**: `settings/bert-base-uncased_en_task3_{MAX_EP}ep_512seq_L2e-05/`

**Key Artifacts**:
- `CAPTAIN.allEnss.R02-L.tsv`: Top-100 ranked predictions (required for Revise stage)
- `CAPTAIN.allEnss.R02.tsv`: Full ranked predictions
- `*.ckpt`: PyTorch Lightning checkpoint files
- `train.log`: Training log with validation metrics

**Note**: If the TSV files already exist from a previous run, the training step may be omitted and the existing files reused.

---

## 2. Revise Stage

The Revise stage employs a Large Language Model (LLM) to filter and refine the retrieved candidates.

### 2.1 Data Preparation for Revise Stage

Prepare the data files required for LLM processing:

```powershell
python scripts/prepare_revise_data.py `
  --xml_file COLIEE2025statute_data-English/train/riteval_R02_en.xml `
  --text_file COLIEE2025statute_data-English/text/civil_code_en-1to724-2.txt `
  --output_dir Retrieve-Revise-Refine-master/data
```

**Verification**: Confirm that `Retrieve-Revise-Refine-master/data/` contains `gold_task3_task4.json` and `civil_code_en.json`.

### 2.2 Dependency Installation

Install additional dependencies required for the Revise stage:

```powershell
pip install scikit-learn numpy
```

### 2.3 Embedding Generation (Optional)

Generate embeddings and few-shot examples for enhanced LLM prompting:

```powershell
cd Retrieve-Revise-Refine-master\Qwen_prompting
python save_embs_and_fewshot_examples.py
cd ..\..
```

**Output Files**:
- `embeddings.pt`: Article embeddings
- `positive_storage.json`: Positive few-shot examples
- `negative_storage.json`: Negative few-shot examples

**Note**: This step is optional but recommended for improved LLM performance. The generated files are cached and need not be regenerated unless the dataset changes.

### 2.4 LLM Prompting

Execute the LLM-based filtering process:

```powershell
cd Retrieve-Revise-Refine-master\Qwen_prompting
python llm_support.py `
  --gold_file ../data/gold_task3_task4.json `
  --civil_code_file ../data/civil_code_en.json `
  --prompt prompt_llm_support_1 `
  --model_name_or_path Qwen/Qwen-1_8B-Chat `
  --top_100_file ../../settings/bert-base-uncased_en_task3_5ep_512seq_L2e-05/CAPTAIN.allEnss.R02-L.tsv `
  --top_k 5 `
  --max_new_tokens 512
cd ..\..
```

**Parameters**:
- `--model_name_or_path`: Hugging Face model identifier or local checkpoint path
- `--top_k`: Number of top candidates to consider per query (default: 5)
- `--max_new_tokens`: Maximum tokens for LLM generation (default: 512)

**Output**: Timestamped JSONL file in `Retrieve-Revise-Refine-master/Qwen_prompting/llm_outputs/`

**Fallback Mode**: If GPU/LLM access is unavailable, add `--no_llm` flag to generate placeholder outputs.

### 2.5 LLM Output Processing

Process the LLM outputs to generate the revised predictions:

```powershell
$llmFile = Get-ChildItem "Retrieve-Revise-Refine-master/Qwen_prompting/llm_outputs/*.json" `
  | Sort-Object LastWriteTime -Descending `
  | Select-Object -First 1

python scripts/process_llm_outputs.py `
  --llm_output_file $llmFile.FullName `
  --prompt_type prompt_llm_support_1 `
  --output_file Retrieve-Revise-Refine-master/revised/COLIEE_2025_Qwen_R02.txt `
  --top_100_file settings/bert-base-uncased_en_task3_5ep_512seq_L2e-05/CAPTAIN.allEnss.R02-L.tsv `
  --top_k 5
```

**Fallback Mechanism**: The processing script includes fallback logic that utilizes top-k retrieval predictions when LLM responses indicate insufficient information.

**Output**: TREC-format file at `Retrieve-Revise-Refine-master/revised/COLIEE_2025_Qwen_R02.txt`

---

## 3. Refine Stage

The Refine stage performs ensemble aggregation of multiple prediction sources.

### 3.1 Ensemble Execution

Execute the ensemble script:

```powershell
cd Retrieve-Revise-Refine-master
python ensemble.py 6 join-cons 2025
cd ..
```

**Output**: Final predictions file at `Retrieve-Revise-Refine-master/revised/tmp_ensembled.txt`

**Note**: The script may attempt to call `eval_2025_predictions.py` for automatic evaluation. If this file is absent, a warning is displayed but execution continues.

---

## 4. Evaluation

### 4.1 Performance Metrics

Evaluate the final predictions against the gold standard:

```powershell
cd Retrieve-Revise-Refine-master
python eval_2023_predictions.py revised/tmp_ensembled.txt ..\COLIEE2025statute_data-English\train\riteval_R02_en.xml
cd ..
```

**Metrics Computed**:
- Macro-average Precision
- Macro-average Recall
- Macro-average F2-score
- Total retrieved articles and correct predictions

**Customization**: To modify exclusion lists or evaluate different datasets, create a custom evaluation script based on `eval_2023_predictions.py`.

---

## 5. Output Artifacts

| Stage | Output File | Location |
|-------|-------------|----------|
| **Retrieve** | Top-100 predictions | `settings/bert-base-uncased_en_task3_{EP}ep_512seq_L2e-05/CAPTAIN.allEnss.R02-L.tsv` |
| **Revise** | Filtered predictions | `Retrieve-Revise-Refine-master/revised/COLIEE_2025_Qwen_R02.txt` |
| **Refine** | Final ensemble predictions | `Retrieve-Revise-Refine-master/revised/tmp_ensembled.txt` |

**Repository Guidelines**: Large artifacts (checkpoints, logs, generated predictions) are reproducible and should not be committed to version control unless explicitly required.

---

## 6. Experimental Results

The following metrics were obtained from evaluation against the `riteval_R02_en.xml` gold standard:

| Stage | Precision | Recall | F2-Score | Additional Metrics |
|-------|-----------|--------|----------|-------------------|
| **Retrieve** | 0.6772 | 0.8011 | 0.7420 | Retrieved: 98, Missed queries: 0 |
| **Revise**  | 0.6113 | 0.7302 | 0.6655 |  120 predictions, 66 correct |
| **Refine**  | 0.6214 | 0.7407 | 0.6886 | Identical to Revise (fallback mechanism) |

**Evaluation Command**:
```powershell
cd Retrieve-Revise-Refine-master
python eval_2023_predictions.py <answer_file> ..\COLIEE2025statute_data-English\train\riteval_R02_en.xml
```


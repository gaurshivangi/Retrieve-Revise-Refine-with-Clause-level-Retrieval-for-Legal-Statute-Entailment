# Legal Article Set Retrieval

## Stage 1. Retrieve

- First, clone the following repository

``` bash
git clone https://github.com/Nguyen2015/CAPTAIN-COLIEE2023/tree/coliee2023
```

- Then, follow the instruction in this repository to produce the top-100 prediction file (it is an ensemble of predictions from many finetuned models, with duplicated article ID allowed but different scores) for each question in each year's dataset.

## Stage 2. Revise

In this stage, we will use large language models (LLMs) to refine the top-k predictions retrieved in the first stage.

We could use many LLMs of our choice. For example, Qwen, Orca, Mistral, LLaMA, Flan-Alpaca, etc.

Below we give example when promting with Qwen LLM.

### Qwen LLM

#### Save statement embeddings and generate few-shot data

- Run the following program:

``` bash
cd Qwen_prompting

python save_embs_and_fewshot_examples.py
```

#### Prompt Qwen LLM

- First, clone git Qwen repository

``` bash
cd ..

git clone https://github.com/QwenLM/Qwen.git
```

- Then install the required libraries
- Next, move the files from folder `Qwen_prompting` to folder `Qwen` just cloned.

``` bash
mv Qwen_prompting/* Qwen
```

- Run Qwen LLM on each prompt template with each value of top-k, for example:

``` bash
cd Qwen

python llm_support.py \
--gold_file ./data/gold_task3_task4.json \
--civil_code_file ./data/civil_code_en.json \
--prompt prompt_llm_support_1 \
--model_name_or_path ./models/huggingface_models/Qwen-14B-Chat \
--output_file auto \
--top_100_file ./retrived/CAPTAIN.bjpAll.R04-L.tsv \
--top_k 5 \
--max_new_tokens 512
```

## Stage 3. Refine

- Now we could perform the Refine stage through an customized ensemble with many options:

``` bash
python ensemble.py $TOPL $JOIN_CONSTRAINT_TYPE $YEAR
```

- We could run a batch to observe many results of different settings (e.g., for "join-cons"):

``` bash
YEAR=2023; for TOPL in {1..15} ; do echo "top_L = ${TOPL}: "; python ensemble.py $TOPL join-cons $YEAR ; echo "-------" ; done
```

- After that, we could use evaluation program to evaluate the final prediction file, for example:

``` bash
python eval_2023_predictions.py /path/to/prediction/file /path/to/the/gold/xml/file
```
## Citation
Chau Nguyen, Phuong Nguyen, Le-Minh Nguyen,
Retrieve–Revise–Refine: A novel framework for retrieval of concise entailing legal article set,
Information Processing & Management,
Volume 62, Issue 1,
2025,
103949,
ISSN 0306-4573,
https://doi.org/10.1016/j.ipm.2024.103949.
(https://www.sciencedirect.com/science/article/pii/S030645732400308X)
Abstract: The retrieval of entailing legal article sets aims to identify a concise set of legal articles that holds an entailment relationship with a legal query or its negation. Unlike traditional information retrieval that focuses on relevance ranking, this task demands conciseness. However, prior research has inadequately addressed this need by employing traditional methods. To bridge this gap, we propose a three-stage Retrieve–Revise–Refine framework which explicitly addresses the need for conciseness by utilizing both small and large language models (LMs) in distinct yet complementary roles. Empirical evaluations on the COLIEE 2022 and 2023 datasets demonstrate that our framework significantly enhances performance, achieving absolute increases in the macro F2 score by 3.17% and 4.24% over previous state-of-the-art methods, respectively. Specifically, our Retrieve stage, employing various tailored fine-tuning strategies for small LMs, achieved a recall rate exceeding 0.90 in the top-5 results alone—ensuring comprehensive coverage of entailing articles. In the subsequent Revise stage, large LMs narrow this set, improving precision while sacrificing minimal coverage. The Refine stage further enhances precision by leveraging specialized insights from small LMs, resulting in a relative improvement of up to 19.15% in the number of concise article sets retrieved compared to previous methods. Our framework offers a promising direction for further research on specialized methods for retrieving concise sets of entailing legal articles, thereby more effectively meeting the task’s demands.
Keywords: Retrieval–Revise–Refine framework; Legal article set retrieval; Information retrieval; COLIEE competition; Large language models

import sys

from extract_gold_task3_task4 import load_samples_from_file


answer_file = sys.argv[1]
xml_file_path = sys.argv[2]

excluded_qids = []
print(f"qids to exclude: {excluded_qids}")


def calculate_f2_score(y_true, y_pred):
    # Convert lists to sets for easier calculation
    true_set = set(y_true)
    pred_set = set(y_pred)

    # Calculate precision and recall (handle the zero division case)
    true_positives = len(true_set & pred_set)
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(true_set) if len(true_set) > 0 else 0

    # Calculate F2 score with beta=2
    beta_squared = 2**2
    f2_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f2_score, true_positives

# Function to compute the macro-average F2-score over all classes
def compute_macro_avg_f2_score(golds, preds):
    precisions = []
    recalls = []
    f2_scores = []
    cnt_correct = 0

    for key in golds:
        y_true = golds[key]
        y_pred = preds.get(key, [])
        precision, recall, f2_score, correct = calculate_f2_score(y_true, y_pred)
        precisions.append(precision)
        recalls.append(recall)
        f2_scores.append(f2_score)
        cnt_correct += correct

    # Calculate the macro-average precision, recall, F2-score
    macro_avg_precision = sum(precisions) / len(precisions) if precisions else 0
    macro_avg_recall = sum(recalls) / len(recalls) if recalls else 0
    macro_avg_f2_score = sum(f2_scores) / len(f2_scores) if f2_scores else 0

    return macro_avg_precision, macro_avg_recall, macro_avg_f2_score, cnt_correct


# Extract the predictions from submission
preds = dict()
with open(answer_file) as fin:
    for line in fin:
        foo = line.strip().split(' ')
        qid = foo[0]
        rel_article = foo[2]
        if qid not in preds:
            preds[qid] = []
        preds[qid].append(rel_article)
for qid in excluded_qids:
    if qid in preds:
        del preds[qid]
tmp_value = sum(len(values) for values in preds.values())
print(f"After excluding, preds has {len(preds)} samples with {tmp_value} relevant articles.")

# Read XML file and get gold answers so that we can calculate accuracy
samples = load_samples_from_file(xml_file_path)
golds = dict()
for sid in samples:
    golds[sid] = samples[sid]['rel_article_ids']
for qid in excluded_qids:
    if qid in golds:
        del golds[qid]
print(f"After excluding, golds has {len(golds)} samples with {sum(len(values) for values in golds.values())} relevant articles.")

# Calculate the macro-average F2-score
macro_avg_precision, macro_avg_recall, macro_avg_f2_score, cnt_correct = compute_macro_avg_f2_score(golds, preds)
print(f"Macro-average Precision: {round(macro_avg_precision, 4)}")
print(f"Macro-average Recall: {round(macro_avg_recall, 4)}")
print(f"Macro-average F2-score: {round(macro_avg_f2_score, 4)}")
print(f"preds has {tmp_value} relevant articles; {cnt_correct} of them are correct")
if len(preds) != len(golds):
    print(f'*** WARNING: len(preds) != len(golds): {len(preds)} != {len(golds)}')
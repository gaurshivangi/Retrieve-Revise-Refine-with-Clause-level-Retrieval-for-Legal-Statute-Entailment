# Command:
# python ensemble.py $TOPL $JOIN_CONSTRAINT_TYPE $YEAR
# YEAR=2023; for TOPL in {1..15} ; do echo "top_L = ${TOPL}: "; python IPM_ensemble.py $TOPL join-cons $YEAR ; echo "-------" ; done
# YEAR=2023; python IPM_ensemble.py 6 join-cons $YEAR ; echo "-------"


import os
import sys


################################## DEFINE ##################################
top_L = int(sys.argv[1])
join_constraint_type = sys.argv[2]  # 'join' or 'join-cons'
year = int(sys.argv[3])

assert join_constraint_type in ['join', 'join-cons']
# Extend supported years to include 2025 (maps to R02 series in our dataset)
years = [2021, 2022, 2023, 2025]
years_jp = ['R02', 'R03', 'R04', 'R02']

# Default combination files (can be overridden per-year below)
combination = (f"./revised/COLIEE_{year}_Orca.txt", f"./revised/COLIEE_{year}_Qwen.txt", f"./revised/COLIEE_{year}_Mistral.txt")
gold_file = f"./data/gold_COLIEE_{year}.txt"
submission_L_file = f"./retrieved/CAPTAIN.bjpAll.{years_jp[years.index(year)]}-L.tsv"
output_submission_file = './revised/tmp_ensembled.txt'

# Special-case settings for 2025 using our generated files/paths
if year == 2025:
    # Use only the Qwen revised output we generated for R02
    combination = ("./revised/COLIEE_2025_Qwen_R02.txt",)
    # Point to the retrieved top-100 TSV placed under ./retrieved/
    submission_L_file = "./retrieved/CAPTAIN.bjpAll.R02-L.tsv"
################################## end ##################################

# Get top L from file submission-L
submission_top_L = dict()
with open(submission_L_file) as fin:
    for line in fin:
        foo = line.split()
        if foo[0] not in submission_top_L:
            submission_top_L[foo[0]] = []
        # if foo[2] not in submission_top_L[foo[0]]:
        submission_top_L[foo[0]].append(foo[2])
for qid in submission_top_L:
    submission_top_L[qid] = submission_top_L[qid][:top_L]

# Join
submissions = dict()
for file in combination:
    with open(file) as fin:
        for line in fin:
            if line:
                foo = line.split()
                if foo[0] not in submissions:
                    submissions[foo[0]] = []
                if foo[2] not in submissions[foo[0]]:
                    submissions[foo[0]].append(foo[2])

# Constrain
if join_constraint_type == 'join-cons':
    # Constraint with submission-L
    for qid in submissions:
        submissions[qid] = list(set(submissions[qid]) & set(submission_top_L[qid]))

# Write ensembled to file
with open(output_submission_file, 'w') as fout:
    for qid in submissions:
        for related in submissions[qid]:
            print(f"{qid} Q0 {related}", file=fout)

# Evaluate ensembled
eval_script = f"eval_{year}_predictions.py"
if os.path.exists(eval_script):
    command = f"python {eval_script} {output_submission_file}"
    os.system(command)
else:
    print(f"[Info] Evaluation script '{eval_script}' not found. Skipping evaluation step.")
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import editdistance
from collections import defaultdict

from utils import Tools

def compute_EM(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    EM_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        if len(target_lines) != len(prediction_lines):
            EM_scores.append(0)
            continue
        if target_lines == prediction_lines:
            EM_scores.append(1)
            continue
        EM_scores.append(0)
    return any(EM_scores)

def compute_ES(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = '\n'.join(target_lines)
    ES_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        prediction_str = '\n'.join(prediction_lines)
        ES_scores.append(
            1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
        )
    return max(ES_scores)

def compute_score_by_repo_with_metadata(repos, lines, stype, passk=1):
    scores = defaultdict(list)
    for line in lines:
        repo = line['metadata']['task_id'].split('/')[0]
        if repo not in repos:
            continue
        samples = [line['choices'][i]['text'] for i in range(len(line['choices']))]
        if stype == 'EM':
            score = compute_EM(line['metadata']['ground_truth'], samples, passk)
        elif stype == 'ES':
            score = compute_ES(line['metadata']['ground_truth'], samples, passk)
        scores[repo].append(score)
    avg_scores = {repo: round(sum(scores[repo]) / len(scores[repo]), 4) for repo in scores}
    repo_count = {repo: len(scores[repo]) for repo in scores}
    print(stype)
    for repo in avg_scores.keys():
        print(f'{avg_scores[repo]}\t{repo_count[repo]}\t{repo}')

import csv
from collections import defaultdict
import csv
from collections import defaultdict

def compute_score_by_repo_with_metadata(repos, lines, stype, passk=1, csv_filename='results.csv'):
    scores = defaultdict(list)
    total_score_sum = 0
    total_correct = 0
    total_evaluated = 0

    for line in lines:
        repo = line['metadata']['task_id'].split('/')[0]
        if repo not in repos:
            continue

        samples = [line['choices'][i]['text'] for i in range(len(line['choices']))]

        if stype == 'EM':
            score = compute_EM(line['metadata']['ground_truth'], samples, passk)
        elif stype == 'ES':
            score = compute_ES(line['metadata']['ground_truth'], samples, passk)

        scores[repo].append(score)
        if score:
            total_correct += 1
        total_evaluated += 1
        total_score_sum += score

    avg_scores = {repo: round(sum(scores[repo]) / len(scores[repo]), 4) for repo in scores}
    repo_count = {repo: len(scores[repo]) for repo in scores}
    overall_average_score = round(total_score_sum / total_evaluated, 4) if total_evaluated > 0 else 0

    # Log the overall score and scores by repository to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile,
                                    fieldnames=["Repository", "ScoreType", "AverageScore", "ProblemCount",
                                                "TotalEvaluated", "TotalCorrect"])
        # Check if file is empty to write headers
        csvfile.seek(0, 2)  # Seek to the end of the file
        if csvfile.tell() == 0:  # If file is empty, write header
            csv_writer.writeheader()

        for repo, avg_score in avg_scores.items():
            data_row = {
                "Repository": repo,
                "ScoreType": stype,
                "AverageScore": avg_score,
                "ProblemCount": repo_count[repo],
                "TotalEvaluated": total_evaluated,
                "TotalCorrect": total_correct
            }
            csv_writer.writerow(data_row)

    return overall_average_score, avg_scores, repo_count

if __name__ == '__main__':
    repos = [
        'huggingface_diffusers',
        'nerfstudio-project_nerfstudio',
        'awslabs_fortuna',
        'huggingface_evaluate',
        'google_vizier',
        'alibaba_FederatedScope',
        'pytorch_rl',
        'opendilab_ACE',
    ]
    '''compute single prediction'''
    relative_path = "processed_generations\line_level_completion_2k_context_codegen.test_0.1_generations.jsonl"
    base_file = relative_path.split("\\")[-1].replace(".jsonl", "")
    file_path = 'processed_generations/' + base_file + '.jsonl'
    compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(file_path), 'EM', passk=1, csv_filename="results/" + base_file + ".csv")
    compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(file_path), 'ES', passk=1, csv_filename="results/" + base_file + ".csv")

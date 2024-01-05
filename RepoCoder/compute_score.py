# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import csv
import json
import os
import editdistance
from collections import defaultdict

from utils import Tools


# Function to preprocess the completion string
def preprocess_completion(completion):
    # Extract content within ``
    if '```python' in completion:
        start = completion.find('```python') + 9  # Start index of content inside ``
        end = completion.rfind('```')  # End index of content inside ``
        completion = completion[start:end]
    elif '```' in completion:
        start = completion.find('```') + 3  # Start index of content inside ``
        end = completion.rfind('```')  # End index of content inside ``
        completion = completion[start:end]
    elif '`' in completion:
        start = completion.find('`') + 1  # Start index of content inside ``
        end = completion.rfind('`')  # End index of content inside ``
        completion = completion[start:end]

    # Remove lines starting with '#'
    lines = [line.split('#', 1)[0].rstrip() for line in completion.split('\n')]
    # Save only the first non-empty line
    for line in lines:
        if line.strip():
            return " ".join(line.split()).replace("( ", "(").replace(" )", ")")
    return ""  # Return empty string if no non-empty lines are found

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

def calculate_similarities_and_export(repo_list, jsonl_lines, run_name):
    # Initialize data structure to store the metrics
    metrics = {
        "run_name": run_name,
        'average_es': 0,
        'percent_exact_matches': 0,
        'repo_metrics': {repo: {'es': 0, 'em': 0, 'count': 0} for repo in repo_list}
    }
    
    # Process each JSONL line
    for line in jsonl_lines:
        data = line
        ground_truth = data['metadata']['ground_truth']
        generated_completion = data['choices'][0]["text"]
        repo_name = data['metadata']['task_id'].split('/')[0]

        # Update overall metrics
        metrics['average_es'] += compute_ES(ground_truth, [generated_completion], 1)
        metrics['percent_exact_matches'] += compute_EM(ground_truth, [generated_completion], 1)

        # Update per-repository metrics
        repo_metric = metrics['repo_metrics'].get(repo_name)
        if repo_metric:
            repo_metric['es'] += compute_ES(ground_truth, [generated_completion], 1)
            repo_metric['em'] += compute_EM(ground_truth, [generated_completion], 1)
            repo_metric['count'] += 1

    # Calculate averages
    num_lines = len(jsonl_lines)
    metrics['average_es'] = round(metrics['average_es'] / num_lines, 2)
    metrics['percent_exact_matches'] = round(metrics['percent_exact_matches'] / num_lines, 2)

    for repo, repo_metric in metrics['repo_metrics'].items():
        if repo_metric['count'] > 0:
            repo_metric['es'] = round( repo_metric['es'] / repo_metric['count'], 2)
            repo_metric['em'] = round( repo_metric['em'] / repo_metric['count'], 2)

    # Write results to CSV
    csv_filename = 'results4.csv'
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ["run_name",'overall_average_es', 'overall_percent_exact_matches'] + \
                     [f'{repo}_es' for repo in repo_list] + \
                     [f'{repo}_em' for repo in repo_list]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # if not file_exists:
        writer.writeheader()

        writer.writerow({
            "run_name": metrics['run_name'],
            'overall_average_es': metrics['average_es'],
            'overall_percent_exact_matches': metrics['percent_exact_matches'],
            **{f'{repo}_es': repo_metric['es'] for repo, repo_metric in metrics['repo_metrics'].items()},
            **{f'{repo}_em': repo_metric['em'] for repo, repo_metric in metrics['repo_metrics'].items()}
        })


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
base_file = "rg-one-gram-ws-20-ss-2-one-line_0.1_instruct_gpt-3.5-0301_generations_processed"
file_path = 'processed_generations/' + base_file + '.jsonl'

compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(file_path), "EM", passk=1)
compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(file_path), 'ES', passk=1)

calculate_similarities_and_export(repos, Tools.load_jsonl(file_path), base_file)

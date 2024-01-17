# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import csv
import json
import sys
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

# def calculate_similarities_and_export(repo_list, jsonl_lines, run_name):
#     # Initialize data structure to store the metrics
#     metrics = {
#         "run_name": run_name,
#         'average_es': 0,
#         'percent_exact_matches': 0,
#         'repo_metrics': {repo: {'es': 0, 'em': 0, 'count': 0} for repo in repo_list}
#     }
    
#     # Process each JSONL line
#     for line in jsonl_lines:
#         data = line
#         ground_truth = data['metadata']['ground_truth']
#         generated_completion = data['choices'][0]["text"]
#         repo_name = data['metadata']['task_id'].split('/')[0]

#         # Update overall metrics
#         metrics['average_es'] += compute_ES(ground_truth, [generated_completion], 1)
#         metrics['percent_exact_matches'] += compute_EM(ground_truth, [generated_completion], 1)

#         # Update per-repository metrics
#         repo_metric = metrics['repo_metrics'].get(repo_name)
#         if repo_metric:
#             repo_metric['es'] += compute_ES(ground_truth, [generated_completion], 1)
#             repo_metric['em'] += compute_EM(ground_truth, [generated_completion], 1)
#             repo_metric['count'] += 1

#     # Calculate averages
#     num_lines = len(jsonl_lines)
#     metrics['average_es'] = round(metrics['average_es'] / num_lines, 2)
#     metrics['percent_exact_matches'] = round(metrics['percent_exact_matches'] / num_lines, 2)

#     for repo, repo_metric in metrics['repo_metrics'].items():
#         if repo_metric['count'] > 0:
#             repo_metric['es'] = round( repo_metric['es'] / repo_metric['count'], 2)
#             repo_metric['em'] = round( repo_metric['em'] / repo_metric['count'], 2)

#     # Write results to CSV
#     csv_filename = 'results.csv'
#     file_exists = os.path.isfile(csv_filename)

#     with open(csv_filename, 'a', newline='') as csvfile:
#         fieldnames = ["run_name",'overall_average_es', 'overall_percent_exact_matches'] + \
#                      [f'{repo}_es' for repo in repo_list] + \
#                      [f'{repo}_em' for repo in repo_list]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         if not file_exists:
#             writer.writeheader()

#         writer.writerow({
#             "run_name": metrics['run_name'],
#             'overall_average_es': metrics['average_es'],
#             'overall_percent_exact_matches': metrics['percent_exact_matches'],
#             **{f'{repo}_es': repo_metric['es'] for repo, repo_metric in metrics['repo_metrics'].items()},
#             **{f'{repo}_em': repo_metric['em'] for repo, repo_metric in metrics['repo_metrics'].items()}
#         })


def calculate_similarities_and_export(repo_list, jsonl_lines, run_name):
    # Initialize data structures to store metrics
    metrics = {
        "run_name": run_name,
        'average_es': 0,
        'percent_exact_matches': 0,
        'repo_metrics': {repo: {'es': 0, 'em': 0, 'count': 0} for repo in repo_list},
        'best_task_variants': {}
    }

    # Process each JSONL line
    for line in jsonl_lines:
        data = line
        ground_truth = data['metadata']['ground_truth']
        generated_completion = data['choices'][0]["text"]
        task_id_components = data['metadata']['task_id'].split('/')
        repo_name = task_id_components[0]
        base_task_id = '/'.join(task_id_components[:-1]) if '_' in task_id_components[-1] else data['metadata']['task_id']
        
        current_es = compute_ES(ground_truth, [generated_completion], 1)
        current_em = compute_EM(ground_truth, [generated_completion], 1)

        best_variant = metrics['best_task_variants'].get(base_task_id, {'es': 0, 'em': 0})
        if current_es > best_variant['es']:
            if best_variant['es'] > 0:
                print("found better!")
            best_variant = {'es': current_es, 'em': current_em}
        metrics['best_task_variants'][base_task_id] = best_variant

        # Update per-repository metrics
        repo_metric = metrics['repo_metrics'][repo_name]
        repo_metric['count'] += 1

    # Calculate averages and update per-repository metrics
    best_variants_count = len(metrics['best_task_variants'])
    print(len(metrics['best_task_variants']))
    for base_task_id, best_scores in metrics['best_task_variants'].items():
        metrics['average_es'] += best_scores['es']
        metrics['percent_exact_matches'] += best_scores['em']
        repo_name = base_task_id.split('/')[0]
        repo_metric = metrics['repo_metrics'][repo_name]
        repo_metric['es'] += best_scores['es']
        repo_metric['em'] += best_scores['em']

    metrics['average_es'] = round(metrics['average_es'] / best_variants_count, 2)
    metrics['percent_exact_matches'] = round(metrics['percent_exact_matches'] / best_variants_count, 2)
    for repo_metric in metrics['repo_metrics'].values():
        if repo_metric['count'] > 0:
            repo_metric['es'] = round(repo_metric['es'] / repo_metric['count'], 2)
            repo_metric['em'] = round(repo_metric['em'] / repo_metric['count'], 2)

    # Write results to CSV
    csv_filename = 'temp_results.csv'
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = ["run_name", 'overall_average_es', 'overall_percent_exact_matches'] + \
                     [f'{repo}_es' for repo in repo_list] + [f'{repo}_em' for repo in repo_list]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
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

# Check if the command line argument is given (1 argument besides the script name)
if len(sys.argv) < 2:
    print("Usage: python compute_score.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(file_path), "EM", passk=1)
compute_score_by_repo_with_metadata(repos, Tools.load_jsonl(file_path), 'ES', passk=1)

base_file_path = file_path.split("/")[-1].replace(".jsonl", "")

calculate_similarities_and_export(repos, Tools.load_jsonl(file_path), base_file_path)
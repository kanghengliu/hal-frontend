import json
from pathlib import Path
import sqlite3
import pickle
from functools import lru_cache
import threading
import pandas as pd
import ast
from scipy import stats
import yaml
import numpy as np
import re
import argparse


# Define column schemas
PARSED_RESULTS_COLUMNS = {
    'benchmark_name': 'TEXT',
    'agent_name': 'TEXT', 
    'date': 'TEXT',
    'run_id': 'TEXT',
    'successful_tasks': 'TEXT',
    'failed_tasks': 'TEXT',
    'total_cost': 'REAL',
    'accuracy': 'REAL',
    'average_score': 'REAL', #assistantbench
    'exact_matches': 'REAL', #assistantbench
    'answer_rate': 'REAL', #assistantbench
    'total_tasks': 'REAL', #mind2web
    'accuracy_easy': 'REAL', #mind2web
    'total_easy': 'REAL', #mind2web
    'accuracy_medium': 'REAL', #mind2web
    'total_medium': 'REAL', #mind2web
    'accuracy_hard': 'REAL', #mind2web
    'total_hard': 'REAL', #mind2web
    'average_correctness': 'REAL', #colbench_backend_frontend
    'subtask_accuracy': 'REAL', #scicode
    'codebert_score': 'REAL', #scienceagentbench
    'success_rate': 'REAL', #scienceagentbench
    'valid_program_rate': 'REAL', #scienceagentbench
    'trace_stem': 'TEXT', # Stores file.stem from the JSON
    'precision': 'REAL',
    'recall': 'REAL',
    'f1_score': 'REAL',
    'auc': 'REAL',
    'overall_score': 'REAL',
    'vectorization_score': 'REAL',
    'fathomnet_score': 'REAL',
    'feedback_score': 'REAL',
    'house_price_score': 'REAL',
    'spaceship_titanic_score': 'REAL',
    'amp_parkinsons_disease_progression_prediction_score': 'REAL',
    'cifar10_score': 'REAL',
    'imdb_score': 'REAL',
    'level_1_accuracy': 'REAL',
    'level_2_accuracy': 'REAL',
    'level_3_accuracy': 'REAL',
    'task_goal_completion': 'REAL',  # New column
    'scenario_goal_completion': 'REAL',  # New column
    'combined_scorer_inspect_evals_avg_refusals': 'REAL',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'REAL',
    'accuracy_ci': 'TEXT',  # Using TEXT since it stores formatted strings like "-0.123/+0.456"
    'cost_ci': 'TEXT',
    'model_name': 'TEXT',
}

# Define which columns should be included in aggregation and how
AGGREGATION_RULES = {
    'date': 'first',
    'total_cost': 'mean',
    'accuracy': 'mean',
    'average_score': 'mean',
    'exact_matches': 'mean',
    'answer_rate': 'mean',
    'total_tasks': 'mean',
    'accuracy_easy': 'mean',
    'total_easy': 'mean',
    'accuracy_medium': 'mean',
    'total_medium': 'mean',
    'accuracy_hard': 'mean',
    'total_hard': 'mean',
    'average_correctness': 'mean',
    'subtask_accuracy': 'mean',
    'codebert_score': 'mean',
    'success_rate': 'mean',
    'valid_program_rate': 'mean',
    'precision': 'mean',
    'recall': 'mean',
    'f1_score': 'mean',
    'auc': 'mean',
    'overall_score': 'mean',
    'vectorization_score': 'mean',
    'fathomnet_score': 'mean',
    'feedback_score': 'mean',
    'house_price_score': 'mean',
    'spaceship_titanic_score': 'mean',
    'amp_parkinsons_disease_progression_prediction_score': 'mean',
    'cifar10_score': 'mean',
    'imdb_score': 'mean',
    'level_1_accuracy': 'mean',
    'level_2_accuracy': 'mean',
    'level_3_accuracy': 'mean',
    'task_goal_completion': 'mean',
    'scenario_goal_completion': 'mean',
    'combined_scorer_inspect_evals_avg_refusals': 'mean',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'mean',
    'Verified': 'first',
    'Runs': 'first',
    'Traces': 'first',
    'accuracy_ci': 'first',
    'cost_ci': 'first',
    'run_id': 'first',
    'trace_stem': 'first',
    'model_name': 'first',
}

# Define column display names
COLUMN_DISPLAY_NAMES = {
    'agent_name': 'Agent Name',
    'url': 'URL',
    'date': 'Date',
    'total_cost': 'Total Cost',
    'accuracy': 'Accuracy',
    'average_score': 'Average Score',
    'exact_matches': 'Exact Matches',
    'answer_rate': 'Answer Rate',
    'total_tasks': 'Total Tasks',
    'accuracy_easy': 'Accuracy (Easy)',
    'total_easy': 'Total (Easy)',
    'accuracy_medium': 'Accuracy (Medium)',
    'total_medium': 'Total (Medium)',
    'accuracy_hard': 'Accuracy (Hard)',
    'total_hard': 'Total (Hard)',
    'average_correctness': 'Average Correctness',
    'subtask_accuracy': 'Subtask Accuracy',
    'codebert_score': 'CodeBERT Score',
    'success_rate': 'Success Rate',
    'valid_program_rate': 'Valid Program Rate',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1 Score',
    'auc': 'AUC',
    'overall_score': 'Overall Score',
    'vectorization_score': 'Vectorization Score',
    'fathomnet_score': 'Fathomnet Score',
    'feedback_score': 'Feedback Score',
    'house_price_score': 'House Price Score',
    'spaceship_titanic_score': 'Spaceship Titanic Score',
    'amp_parkinsons_disease_progression_prediction_score': 'AMP Parkinsons Disease Progression Prediction Score',
    'cifar10_score': 'CIFAR10 Score',
    'imdb_score': 'IMDB Score',
    'level_1_accuracy': 'Level 1 Accuracy',
    'level_2_accuracy': 'Level 2 Accuracy',
    'level_3_accuracy': 'Level 3 Accuracy',
    'task_goal_completion': 'Task Goal Completion',
    'scenario_goal_completion': 'Scenario Goal Completion',
    'accuracy_ci': 'Accuracy CI',
    'cost_ci': 'Total Cost CI',
    'combined_scorer_inspect_evals_avg_refusals': 'Refusals',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'Non-Refusal Harm Score',
    'trace_stem': 'Trace Stem',
    'model_name': 'Model Name',
}

# DEFAULT_PRICING = {
#     "text-embedding-3-small": {"prompt_tokens": 0.02, "completion_tokens": 0},
#     "text-embedding-3-large": {"prompt_tokens": 0.13, "completion_tokens": 0},
#     "gpt-4o-2024-05-13": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "gpt-4o-2024-08-06": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "gpt-4o-2024-11-20": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "gpt-3.5-turbo-0125": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
#     "gpt-3.5-turbo": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
#     "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10, "completion_tokens": 30},
#     "gpt-4-turbo": {"prompt_tokens": 10, "completion_tokens": 30},
#     "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
#     "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10, "completion_tokens": 30},
#     "o1-2024-12-17": {"prompt_tokens": 15, "completion_tokens": 60},
#     "meta-llama/Meta-Llama-3.1-8B-Instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18},
#     "meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "meta-llama/Meta-Llama-3.1-405B-Instruct": {"prompt_tokens": 5, "completion_tokens": 15},
#     "meta-llama/Llama-3-70b-chat-hf": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "deepseek-ai/deepseek-coder-33b-instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18}, # these are set to the llama 8n prices
#     "gpt-4o": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75, "completion_tokens": 150},
#     "o1-mini-2024-09-12": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "o1-preview-2024-09-12": {"prompt_tokens": 15, "completion_tokens": 60},
#     "o3-mini-2025-01-14": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "o3-mini-2025-01-31": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "claude-3-5-sonnet-20240620": {"prompt_tokens": 3, "completion_tokens": 15},
#     "claude-3-5-sonnet-20241022": {"prompt_tokens": 3, "completion_tokens": 15},
#     "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt_tokens": 3, "completion_tokens": 15},
#     "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"prompt_tokens": 3, "completion_tokens": 15},
#     "claude-3-5-haiku-20241022": {"prompt_tokens": 0.8, "completion_tokens": 4},
#     "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"prompt_tokens": 0.8, "completion_tokens": 4},
#     "openai/gpt-4o-2024-11-20": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "openai/gpt-4o-2024-08-06": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "openai/gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
#     "openai/gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75, "completion_tokens": 150},
#     "openai/o1-mini-2024-09-12": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "openai/o1-preview-2024-09-12": {"prompt_tokens": 15, "completion_tokens": 60},
#     "openai/o1-2024-12-17": {"prompt_tokens": 15, "completion_tokens": 60},
#     "openai/o3-mini-2025-01-14": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "openai/o3-mini-2025-01-31": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "anthropic/claude-3-5-sonnet-20240620": {"prompt_tokens": 3, "completion_tokens": 15},
#     "anthropic/claude-3-5-sonnet-20241022": {"prompt_tokens": 3, "completion_tokens": 15},
#     "google/gemini-1.5-pro": {"prompt_tokens": 1.25, "completion_tokens": 5},
#     "google/gemini-1.5-flash": {"prompt_tokens": 0.075, "completion_tokens": 0.3},
#     "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt_tokens": 3.5, "completion_tokens": 3.5},
#     "Meta-Llama-3.3-70B-Instruct-Turbo": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "us.meta.llama3-3-70b-instruct-v1:0": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt_tokens": 3.5, "completion_tokens": 3.5},
#     "together/meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "claude-3-7-sonnet-20250219": {"prompt_tokens": 3, "completion_tokens": 15},
#     "anthropic/claude-3-7-sonnet-20250219": {"prompt_tokens": 3, "completion_tokens": 15},
#     "o4-mini-2025-04-16": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "o3-2025-04-16":{"prompt_tokens": 2, "completion_tokens": 8},
#     "gpt-4.1-2025-04-14":{"prompt_tokens": 2, "completion_tokens": 8},
#     "claude-3-7-sonnet-2025-02-19": {"prompt_tokens": 3, "completion_tokens": 15},
#     "deepseek-ai/DeepSeek-V3": {"prompt_tokens": 0.27, "completion_tokens": 1.1},
#     "gemini-2.0-flash": {"prompt_tokens": 0.1, "completion_tokens": 0.4},
#     "deepseek-ai/DeepSeek-R1":{"prompt_tokens": 0.55, "completion_tokens": 2.19},
# }

DEFAULT_PRICING = {
    "Text-Embedding-3 Small": {"prompt_tokens": 0.02, "completion_tokens": 0},
    "text-embedding-3-large": {"prompt_tokens": 0.13, "completion_tokens": 0},
    "GPT-4o (May 2024)": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "GPT-4o (August 2024)": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "GPT-4o (November 2024)": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "gpt-3.5-turbo-0125": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
    "gpt-3.5-turbo": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
    "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10, "completion_tokens": 30},
    "gpt-4-turbo": {"prompt_tokens": 10, "completion_tokens": 30},
    "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
    "o1 Medium (December 2024)": {"prompt_tokens": 15, "completion_tokens": 60},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18},
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {"prompt_tokens": 5, "completion_tokens": 15},
    "meta-llama/Llama-3-70b-chat-hf": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "deepseek-ai/deepseek-coder-33b-instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18},
    "o1-mini Medium (September 2024)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o1-preview-2024-09-12": {"prompt_tokens": 15, "completion_tokens": 60},
    "o3-mini Medium (January 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "claude-3-5-sonnet-20240620": {"prompt_tokens": 3, "completion_tokens": 15},
    "claude-3-5-sonnet-20241022": {"prompt_tokens": 3, "completion_tokens": 15},
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt_tokens": 3, "completion_tokens": 15},
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"prompt_tokens": 3, "completion_tokens": 15},
    "claude-3-5-haiku-20241022": {"prompt_tokens": 0.8, "completion_tokens": 4},
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"prompt_tokens": 0.8, "completion_tokens": 4},
    "o4-mini Medium (April 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o3 Medium (April 2025)": {"prompt_tokens": 2, "completion_tokens": 8},
    "GPT-4.1 (April 2025)": {"prompt_tokens": 2, "completion_tokens": 8},
    "Claude-3.7 Sonnet (February 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "DeepSeek V3 (March 2025)": {"prompt_tokens": 1.25, "completion_tokens": 1.25},
    "DeepSeek V3.1 (August 2025)": {"prompt_tokens": 0.60, "completion_tokens": 1.70},
    "DeepSeek R1 (January 2025)": {"prompt_tokens": 3, "completion_tokens": 7},
    "DeepSeek R1 (May 2025)": {"prompt_tokens": 0.55, "completion_tokens": 2.19},
    "Gemini 2.0 Flash (February 2025)": {"prompt_tokens": 0.1, "completion_tokens": 0.4},
    "o4-mini Low (April 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "Claude-3.7 Sonnet Low (February 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "o4-mini High (April 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o3 Low (April 2025)": {"prompt_tokens": 2, "completion_tokens": 8},
    "o3-mini Low (January 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o3-mini High (January 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "Gemini 2.5 Pro Preview (March 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "Gemini 2.5 Pro (June 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "Claude-3.7 Sonnet High (February 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75, "completion_tokens": 150},
    "Claude Opus 4 (May 2025)": {"prompt_tokens": 15, "completion_tokens": 75},
    "Claude Opus 4 High (May 2025)": {"prompt_tokens": 15, "completion_tokens": 75},
    "Claude Sonnet 4 High (May 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "Claude Sonnet 4 (May 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "Claude Opus 4.1 High (August 2025)": {"prompt_tokens": 15, "completion_tokens": 75},
    "Claude Opus 4.1 (August 2025)": {"prompt_tokens": 15, "completion_tokens": 75},
    "GPT-5 High (August 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "GPT-5 Medium (August 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "GPT-5 Low (August 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "GPT-OSS-120B (August 2025)": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
    "GPT-OSS-120B High (August 2025)": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
    "Claude Sonnet 4.5 (September 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "Claude Sonnet 4.5 High (September 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "Claude Haiku 4.5 (October 2025)": {"prompt_tokens": 1, "completion_tokens": 5},
    "Claude Haiku 4.5 High (October 2025)": {"prompt_tokens": 1, "completion_tokens": 5},
    "Claude Haiku 4.5 Low (October 2025)": {"prompt_tokens": 1, "completion_tokens": 5},
    "Claude Haiku 4.5 Medium (October 2025)": {"prompt_tokens": 1, "completion_tokens": 5},
    "Claude Opus 4.5 High (November 2025)": {"prompt_tokens": 5, "completion_tokens": 25},
    "Claude Opus 4.5 Medium (November 2025)": {"prompt_tokens": 5, "completion_tokens": 25},
    "Claude Opus 4.5 Low (November 2025)": {"prompt_tokens": 5, "completion_tokens": 25},
    "Claude Opus 4.5 (November 2025)": {"prompt_tokens": 5, "completion_tokens": 25},

    "Gemini 3 Pro Preview High (November 2025)": {"prompt_tokens": 2, "completion_tokens": 12},
    "Gemini 3 Pro Preview Medium (November 2025)": {"prompt_tokens": 2, "completion_tokens": 12},
    "Gemini 3 Pro Preview Low (November 2025)": {"prompt_tokens": 2, "completion_tokens": 12},
    "Gemini 3 Pro Preview (November 2025)": {"prompt_tokens": 2, "completion_tokens": 12},

    "Gemini 2.5 Pro High (June 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "Gemini 2.5 Pro Medium (June 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "Gemini 2.5 Pro Low (June 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "Gemini 2.5 Pro (June 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},

    "Gemini 2.5 Pro Preview High (March 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "Gemini 2.5 Pro Preview Medium (March 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "Gemini 2.5 Pro Preview Low (March 2025)": {"prompt_tokens": 1.25, "completion_tokens": 10},
}

# Cache token pricing overrides (prices per 1M tokens)
CACHED_PRICE_OVERRIDES = {
    "o4-mini Medium (April 2025)": 0.275,
    "o3-mini Medium (January 2025)": 0.55,
    "Claude-3.7 Sonnet (February 2025)": 0.30,
    "Claude-3.7 Sonnet Low (February 2025)": 0.30,
    "Claude-3.7 Sonnet High (February 2025)": 0.30,
    "Claude Opus 4 (May 2025)": 1.50,
    "Claude Opus 4 High (May 2025)": 1.50,
    "Claude Opus 4.1 (August 2025)": 1.50,
    "Claude Opus 4.1 High (August 2025)": 1.50,
    "GPT-4.1 (April 2025)": 0.50,
    "GPT-5 High (August 2025)": 0.125,
    "GPT-5 Medium (August 2025)": 0.125,
    "GPT-5 Low (August 2025)": 0.125,
    "o3 Medium (April 2025)": 0.5,
    "o3 Low (April 2025)": 0.5,
}

# Model name mappings: (input name, display name, canonical name)
MODEL_MAPPING = [
    ("o4-mini-2025-04-16", "o4-mini Medium (April 2025)", "o4-mini-2025-04-16"),
    ("openai/o4-mini-2025-04-16", "o4-mini Medium (April 2025)", "o4-mini-2025-04-16"),
    ("gpt-4.1-2025-04-14", "GPT-4.1 (April 2025)", "gpt-4.1-2025-04-14"),
    ("o4-mini-2025-04-16 low", "o4-mini Low (April 2025)", "o4-mini-2025-04-16"),
    ("claude-3-7-sonnet-20250219 low", "Claude-3.7 Sonnet Low (February 2025)", "claude-3-7-sonnet-20250219"),
    ("anthropic/claude-3-7-sonnet-20250219 low", "Claude-3.7 Sonnet Low (February 2025)", "claude-3-7-sonnet-20250219"),
    ("o4-mini-2025-04-16 high", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-V3"),
    ("claude-3-7-sonnet-20250219", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("claude-opus-4-20250514", "Claude Opus 4 (May 2025)", "claude-opus-4-20250514"),
    ("Claude-Opus 4 High (May 2025)", "Claude Opus 4 High (May 2025)", "claude-opus-4-20250514"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash (February 2025)", "gemini-2.0-flash"),
    ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash (February 2025)", "google/gemini-2.0-flash-001"),
    ("models/gemini-2.0-flash", "Gemini 2.0 Flash (February 2025)", "models/gemini-2.0-flash"),
    ("google/gemini-2.0-flash-001", "Gemini 2.0 Flash (February 2025)", "google/gemini-2.0-flash-001"),
    ("google/gemini-2.0-flash-001 high", "Gemini 2.0 Flash High (February 2025)", "google/gemini-2.0-flash-001-high"),
    ("gemini-2.0-flash-001", "Gemini 2.0 Flash (February 2025)", "google/gemini-2.0-flash-001"),
    ("gemini-2.0-flash-001 high", "Gemini 2.0 Flash High (February 2025)", "google/gemini-2.0-flash-001-high"),
    ("claude-3-7-sonnet-20250219 high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("anthropic/claude-3.7-sonnet", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("anthropic/claude-3.7-sonnet high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("claude-3.7-sonnet", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("anthropic/claude-3.7-sonnet high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("o3-2025-04-16", "o3 Medium (April 2025)", "o3-2025-04-16"),
    ("claude-3-7-sonnet-2025-02-19", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("claude-opus-4-20250514 high", "Claude Opus 4 High (May 2025)", "claude-opus-4-20250514"),
    ("openrouter/anthropic/claude-opus-4.1", "Claude Opus 4.1 (August 2025)", "openrouter/anthropic/claude-opus-4.1"),
    ("deepseek-ai/DeepSeek-R1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("DeepSeek-R1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("deepseek/deepseek-r1-0528", "DeepSeek R1 (May 2025)", "deepseek-ai/DeepSeek-R1"),
    ("deepseek/deepseek-r1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("deepseek-R1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("deepseek/deepseek-r1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("deepseek-r1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("Deepseek-R1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("openrouter/deepseek/deepseek-r1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("claude-3-7-sonnet-2025-02-19 low", "Claude-3.7 Sonnet Low (February 2025)", "claude-3-7-sonnet-20250219"),
    ("DeepSeek-V3", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-V3"),
    ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-Chat-V3"),
    ("deepseek-chat-v3-0324", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-Chat-V3"),
    ("deepseek/deepseek-chat", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-Chat-V3"),
    ("deepseek-chat", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-Chat-V3"),
    ("openrouter/deepseek/deepseek-chat-v3-0324", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-Chat-V3"),
    ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-Chat-V3"),
    ("deepseek-r1-0528", "DeepSeek R1 (May 2025)", "deepseek-ai/DeepSeek-R1"),
    ("deepseek-chat-v3.1", "DeepSeek V3.1 (August 2025)", "deepseek-ai/DeepSeek-Chat-V3.1"),
    ("deepseek/deepseek-chat-v3.1", "DeepSeek V3.1 (August 2025)", "deepseek-ai/DeepSeek-Chat-V3.1"),
    ("deepseek-chat-v3-0324", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-Chat-V3"),
    ("together_ai/deepseek-ai/DeepSeek-V3", "DeepSeek V3 (March 2025)", "deepseek-ai/DeepSeek-V3"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek V3 (March 2025)", "openai/gpt-oss-120b"),
    ("together_ai/deepseek-ai/DeepSeek-R1", "DeepSeek R1 (January 2025)", "deepseek-ai/DeepSeek-R1"),
    ("openrouter/anthropic/claude-opus-4.1 high", "Claude Opus 4.1 High (August 2025)", "openrouter/anthropic/claude-opus-4.1-high"),
    ("Sonnet3.7", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("claude-sonnet-4-20250514 high", "Claude Sonnet 4 High (May 2025)", "claude-sonnet-4-20250514-high"),
    ("claude-3.7-sonnet:thinking high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219-thinking-high"),
    ("anthropic/claude-3.7-sonnet:thinking", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219-thinking"),
    ("anthropic/claude-3.7-sonnet:thinking:high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219-thinking-high"),
    ("anthropic/claude-3.7-sonnet:thinking high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219-thinking"),
    ("claude-3.7-sonnet", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("claude-3.7-sonnet:thinking:high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219-thinking-high"),
    ("anthropic/claude-3.7-sonnet", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("claude-3.7-sonnet high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("O4-mini-high", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-high", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("GPT4.1", "GPT-4.1 (April 2025)", "gpt-4.1-2025-04-14"),
    ("O3-low", "o3 Low (April 2025)", "o3-2025-04-16"),
    ("o3-low", "o3 Low (April 2025)", "o3-2025-04-16"),
    ("Sonnet 3.7", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("o4-mini-low", "o4-mini Low (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-2025-04-16 medium", "o4-mini Medium (April 2025)", "o4-mini-2025-04-16"),
    ("o3-mini-2025-01-31 low", "o3-mini Low (January 2025)", "o3-mini-2025-01-31"),
    ("o3-mini-2025-01-31 medium", "o3-mini Medium (January 2025)", "o3-mini-2025-01-31"),
    ("claude-3-7-sonnet-2025-02-19 high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("gemini/gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),
    ("o3-mini-2025-01-31 high", "o3-mini High (January 2025)", "o3-mini-2025-01-31"),
    ("claude-3-7-sonnet-20250219_thinking_high_4096", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),
    ("o4-mini-2025-04-16_high_reasoning_effort", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-2025-04-16_low_reasoning_effort", "o4-mini Low (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-2025-04-16:high", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-2025-04-16:low", "o4-mini Low (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-2025-04-16:medium", "o4-mini Medium (April 2025)", "o4-mini-2025-04-16"),
    ("gemini-2.5-pro-preview", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),
    ("o3-mini", "o3-mini Medium (January 2025)", "o3-mini-2025-01-31"),
    ("gpt-4o-2024-11-20", "GPT-4o (November 2024)", "gpt-4o-2024-11-20"),
    ("gpt-4o", "GPT-4o (August 2024)", "gpt-4o-2024-08-06"),
    ("o1", "o1 Medium (December 2024)", "o1-2024-12-17"),
    ("gpt-4.1", "GPT-4.1 (April 2025)", "gpt-4.1-2025-04-14"),
    ("o3-mini-2025-01-31", "o3-mini Medium (January 2025)", "o3-mini-2025-01-31"),
    ("o3-2025-04-03", "o3 Medium (April 2025)", "o3-2025-04-03"),
    ("o3-2025-04-03:medium", "o3 Medium (April 2025)", "o3-2025-04-03"),
    ("o3-2025-04-16 medium", "o3 Medium (April 2025)", "o3-2025-04-16"),
    ("o3-2025-04-16:medium", "o3 Medium (April 2025)", "o3-2025-04-16"),
    ("o3-2025-04-16 low", "o3 Low (April 2025)", "o3-2025-04-16"),
    ("openai/o3-2025-04-16 medium", "o3 Medium (April 2025)", "o3-2025-04-16"),
    ("o3-mini low", "o3-mini Low (January 2025)", "o3-mini-2025-01-31"),
    ("o3-mini high", "o3-mini High (January 2025)", "o3-mini-2025-01-31"),
    ("gpt-4o-2024-08-06", "GPT-4o (August 2024)", "gpt-4o-2024-08-06"),
    ("o1-2024-12-17", "o1 Medium (December 2024)", "o1-2024-12-17"),
    ("text-embedding-3-small", "Text-Embedding-3 Small", "text-embedding-3-small"),
    ("gpt-5 high", "GPT-5 Medium (August 2025)", "gpt-5-high"), # Temporary hack because of errors in taubench
    ("gpt-5 minimal", "GPT-5 Minimal (August 2025)", "gpt-5-high"), # Temporary hack because of errors in taubench
    ("gpt-5", "GPT-5 Medium (August 2025)", "gpt-5"),
    ("gpt-5 low", "GPT-5 Low (August 2025)", "gpt-5-low"),
    ("gpt-5 medium", "GPT-5 Medium (August 2025)", "gpt-5-medium"),
    ("gemini/gemini-2.0-flash", "Gemini 2.0 Flash (February 2025)", "gemini/gemini-2.0-flash"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash (February 2025)", "gemini-2.0-flash"),
    ("openai/o3-mini-2025-01-31 low", "o3-mini Low (January 2025)", "o3-mini-2025-01-31"),
    ("openai/gpt-5-2025-08-07", "GPT-5 Medium (August 2025)", "gpt-5"),
    ("openrouter/openai/gpt-oss-120b high", "GPT-OSS-120B High (August 2025)", "openrouter/openai/gpt-oss-120b"),
    ("openrouter/openai/gpt-oss-120b", "GPT-OSS-120B (August 2025)", "openrouter/openai/gpt-oss-120b"),
    ("openai/gpt-oss-120b high", "GPT-OSS-120B High (August 2025)", "openai/gpt-oss-120b"),
    ("openai/gpt-oss-120b", "GPT-OSS-120B (August 2025)", "openai/gpt-oss-120b"),
    ("gpt-oss-120b high", "GPT-OSS-120B High (August 2025)", "gpt-oss-120b"),
    ("gpt-oss-120b", "GPT-OSS-120B (August 2025)", "gpt-oss-120b"),
    ("claude-opus-4-1-20250805", "Claude Opus 4.1 (August 2025)", "claude-opus-4-1-20250805"),
    ("claude-opus-4-1-20250514", "Claude Opus 4.1 (August 2025)", "claude-opus-4.1-20250514"),
    ("claude-opus-4-1-20250514 high", "Claude Opus 4.1 High (August 2025)", "claude-opus-4.1-20250514"),
    ("claude-opus-4", "Claude Opus 4 (May 2025)", "claude-opus-4"),
    ("anthropic/claude-opus-4.1", "Claude Opus 4.1 (August 2025)", "anthropic/claude-opus-4.1"),
    ("anthropic/claude-opus-4", "Claude Opus 4 (May 2025)", "anthropic/claude-opus-4"),
    ("gpt-5-2025-08-07", "GPT-5 Medium (August 2025)", "gpt-5"),
    ("claude-opus-4.1-20250514 high", "Claude Opus 4.1 High (August 2025)", "claude-opus-4.1-20250514"),
    ("claude-opus-4.1-20250514", "Claude Opus 4.1 (August 2025)", "claude-opus-4.1-20250514"),
    ("gpt-5-2025-08-07", "GPT-5 Medium (August 2025)", "gpt-5-2025-08-07"),
    ("claude-opus-4-1-20250805", "Claude Opus 4.1 (August 2025)", "claude-opus-4-1-20250805"),
    ("gpt-5-2025-08-07 high", "GPT-5 High (August 2025)", "gpt-5-2025-08-07"),
    ("gpt-5-2025-08-07 medium", "GPT-5 Medium (August 2025)", "gpt-5-2025-08-07"),
    ("gpt-5-2025-08-07_minimal_reasoning_effort", "GPT-5 Minimal (August 2025)", "gpt-5-2025-08-07_minimal_reasoning_effort"),
    ("gpt-5-2025-08-07_medium_reasoning_effort", "GPT-5 Medium (August 2025)", "gpt-5-2025-08-07_medium_reasoning_effort"),
    ("gpt-5-2025-08-07_high_reasoning_effort", "GPT-5 High (August 2025)", "gpt-5-2025-08-07_high_reasoning_effort"),
    ("gpt-5:openai:medium", "GPT-5 Medium (August 2025)", "gpt-5"),
    ("claude-opus-4-1-20250805 high", "Claude Opus 4.1 High (August 2025)", "claude-opus-4-1-20250805"),
    ("claude-opus-4.1", "Claude Opus 4.1 (August 2025)", "claude-opus-4.1"),
    ("claude-opus-4-1", "Claude Opus 4.1 (August 2025)", "claude-opus-4-1"),
    ("claude-opus-4.1 high", "Claude Opus 4.1 High (August 2025)", "claude-opus-4.1-high"),
    ("claude-sonnet-4-20250514_thinking_high_4096", "Claude Sonnet 4 High (May 2025)", "claude-sonnet-4-20250514_thinking_high_4096"),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4 (May 2025)", "claude-sonnet-4-20250514"),
    ("anthropic/claude-sonnet-4", "Claude Sonnet 4 (May 2025)", "anthropic/claude-sonnet-4"),
    ("anthropic/claude-sonnet-4 high", "Claude Sonnet 4 High (May 2025)", "anthropic/claude-sonnet-4-high"),
    ("openrouter/anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5 (September 2025)", "openrouter/anthropic/claude-sonnet-4.5"),
    ("openrouter/anthropic/claude-sonnet-4.5 high", "Claude Sonnet 4.5 High (September 2025)", "openrouter/anthropic/claude-sonnet-4.5-high"),
    ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5 (September 2025)", "anthropic/claude-sonnet-4.5"),
    ("anthropic/claude-sonnet-4.5 high", "Claude Sonnet 4.5 High (September 2025)", "anthropic/claude-sonnet-4.5-high"),
    ("claude-sonnet-4.5", "Claude Sonnet 4.5 (September 2025)", "claude-sonnet-4.5"),
    ("claude-sonnet-4.5 high", "Claude Sonnet 4.5 High (September 2025)", "claude-sonnet-4.5-high"),
    ("claude-sonnet-4-5", "Claude Sonnet 4.5 (September 2025)", "claude-sonnet-4-5"),
    ("claude-sonnet-4-5 high", "Claude Sonnet 4.5 High (September 2025)", "claude-sonnet-4-5-high"),
    ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (September 2025)", "claude-sonnet-4-5-20250929"),
    ("claude-sonnet-4-5-20250929 high", "Claude Sonnet 4.5 High (September 2025)", "claude-sonnet-4-5-20250929-high"),
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5 (October 2025)", "claude-haiku-4-5-20251001"),
    ("claude-haiku-4-5-20251001 high", "Claude Haiku 4.5 High (October 2025)", "claude-haiku-4-5-20251001-high"),
    ("claude-haiku-4-5-20251001 low", "Claude Haiku 4.5 Low (October 2025)", "claude-haiku-4-5-20251001-low"),
    ("claude-haiku-4-5-20251001 medium", "Claude Haiku 4.5 Medium (October 2025)", "claude-haiku-4-5-20251001-medium"),
    ("anthropic/claude-haiku-4-5", "Claude Haiku 4.5 (October 2025)", "anthropic/claude-haiku-4.5"),
    ("anthropic/claude-haiku-4-5 high", "Claude Haiku 4.5 High (October 2025)", "anthropic/claude-haiku-4-5-high"),
    ("anthropic/claude-haiku-4-5 low", "Claude Haiku 4.5 Low (October 2025)", "anthropic/claude-haiku-4-5-low"),
    ("anthropic/claude-haiku-4-5 medium", "Claude Haiku 4.5 Medium (October 2025)", "anthropic/claude-haiku-4-5-medium"),
    ("claude-haiku-4-5", "Claude Haiku 4.5 (October 2025)", "claude-haiku-4-5"),
    ("claude-haiku-4-5 high", "Claude Haiku 4.5 High (October 2025)", "claude-haiku-4-5-high"),
    ("claude-haiku-4-5 low", "Claude Haiku 4.5 Low (October 2025)", "claude-haiku-4-5-low"),
    ("claude-haiku-4-5 medium", "Claude Haiku 4.5 Medium (October 2025)", "claude-haiku-4-5-medium"),
    ("anthropic/claude-haiku-4-5", "Claude Haiku 4.5 (October 2025)", "anthropic/claude-haiku-4-5"),
    ("anthropic/claude-haiku-4-5 high", "Claude Haiku 4.5 High (October 2025)", "anthropic/claude-haiku-4-5-high"),
    ("anthropic/claude-haiku-4-5 low", "Claude Haiku 4.5 Low (October 2025)", "anthropic/claude-haiku-4-5-low"),
    ("anthropic/claude-haiku-4-5 medium", "Claude Haiku 4.5 Medium (October 2025)", "anthropic/claude-haiku-4-5-medium"),
    ("Claude-Haiku-4-5", "Claude Haiku 4.5 (October 2025)", "claude-haiku-4-5"),
    ("Claude-Haiku-4-5 High", "Claude Haiku 4.5 High (October 2025)", "claude-haiku-4-5-high"),
    ("Claude-Haiku-4-5 Low", "Claude Haiku 4.5 Low (October 2025)", "claude-haiku-4-5-low"),
    ("Claude-Haiku-4-5 Medium", "Claude Haiku 4.5 Medium (October 2025)", "claude-haiku-4-5-medium"),
    ("claude-opus-4-5-20251101", "Claude Opus 4.5 (November 2025)", "claude-opus-4-5-20251101"),
    ("claude-opus-4-5-20251101 high", "Claude Opus 4.5 High (November 2025)", "claude-opus-4-5-20251101"),
    ("claude-opus-4-5-20251101 low", "Claude Opus 4.5 Low (November 2025)", "claude-opus-4-5-20251101"),
    ("claude-opus-4-5-20251101 medium", "Claude Opus 4.5 Medium (November 2025)", "claude-opus-4-5-20251101"),
    ("claude-opus-4-5-20251101", "Claude Opus 4.5 (November 2025)", "claude-opus-4-5-20251101"),
    ("claude-opus-4-5-20251101 high", "Claude Opus 4.5 High (November 2025)", "claude-opus-4-5-20251101"),
    ("claude-opus-4-5-20251101 medium", "Claude Opus 4.5 Medium (November 2025)", "claude-opus-4-5-20251101"),
    ("claude-opus-4-5-20251101 low", "Claude Opus 4.5 Low (November 2025)", "claude-opus-4-5-20251101"),
    ("anthropic/claude-opus-4-5-20251101", "Claude Opus 4.5 (November 2025)", "claude-opus-4-5-20251101"),
    ("anthropic/claude-opus-4-5-20251101 high", "Claude Opus 4.5 High (November 2025)", "claude-opus-4-5-20251101"),
    ("anthropic/claude-opus-4-5-20251101 medium", "Claude Opus 4.5 Medium (November 2025)", "claude-opus-4-5-20251101"),
    ("anthropic/claude-opus-4-5-20251101 low", "Claude Opus 4.5 Low (November 2025)", "claude-opus-4-5-20251101"),

    ("gemini-3-pro-preview", "Gemini 3 Pro Preview (November 2025)", "gemini-3-pro-preview"),
    ("gemini-3-pro-preview high", "Gemini 3 Pro Preview High (November 2025)", "gemini-3-pro-preview"),
    ("gemini-3-pro-preview medium", "Gemini 3 Pro Preview Medium (November 2025)", "gemini-3-pro-preview"),
    ("gemini-3-pro-preview low", "Gemini 3 Pro Preview Low (November 2025)", "gemini-3-pro-preview"),
    ("google/gemini-3-pro-preview", "Gemini 3 Pro Preview (November 2025)", "gemini-3-pro-preview"),
    ("google/gemini-3-pro-preview high", "Gemini 3 Pro Preview High (November 2025)", "gemini-3-pro-preview"),
    ("google/gemini-3-pro-preview medium", "Gemini 3 Pro Preview Medium (November 2025)", "gemini-3-pro-preview"),
    ("google/gemini-3-pro-preview low", "Gemini 3 Pro Preview Low (November 2025)", "gemini-3-pro-preview"),

    ("models/gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),
    ("gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),
    ("google/gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),

    ("models/gemini-2.5-pro", "Gemini 2.5 Pro (June 2025)", "models/gemini-2.5-pro"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro (June 2025)", "models/gemini-2.5-pro"),
    ("google/gemini-2.5-pro", "Gemini 2.5 Pro (June 2025)", "models/gemini-2.5-pro"),
    ("gemini-2.5-pro high", "Gemini 2.5 Pro High (June 2025)", "models/gemini-2.5-pro"),
    ("gemini-2.5-pro medium", "Gemini 2.5 Pro Medium (June 2025)", "models/gemini-2.5-pro"),
    ("gemini-2.5-pro low", "Gemini 2.5 Pro Low (June 2025)", "models/gemini-2.5-pro"),
]

MODELS_TO_SKIP = [
'Gemini 2.5 Pro Preview (March 2025)',
'o1 Medium (December 2024)',
'o3-mini Low (January 2025)',
'o3-mini Medium (January 2025)',
'o3-mini High (January 2025)',
'GPT-4o (November 2024)',
'GPT-4o (August 2024)',
'o3 Low (April 2025)',
'Claude-3.7 Sonnet Low (February 2025)',
'o4-mini Medium (April 2025)',
'GPT-5 High (August 2025)',
'GPT-5 Minimal (August 2025)',
]

RUNIDS_TO_SKIP = [
    'swebench_verified_mini_my_agento320250416_1745453708',
    'assistantbench_hal_generalist_agent_claude37sonnet20250219_1746223066',
    'assistantbench_hal_generalist_agent_claude37sonnet20250219_high_1748661923',
    'assistantbench_hal_generalist_agent_claude37sonnet20250219_low_1748660152',
    'assistantbench_hal_generalist_agent_claudeopus41_high_1754936482',
    'assistantbench_hal_generalist_agent_claudeopus4120250514_1754545969',
    'assistantbench_hal_generalist_agent_claudeopus4120250514_high_1754543029',
    'assistantbench_hal_generalist_agent_deepseekaideepseekr1_1748900721',
    'assistantbench_hal_generalist_agent_deepseekaideepseekv3_1746221486',
    'assistantbench_hal_generalist_agent_gemini20flash_1746220438',
    'assistantbench_hal_generalist_agent_gpt4120250414_1746216493',
    'assistantbench_hal_generalist_agent_o4mini20250416_1746216959',
    'assistantbench_hal_generalist_agent_o4mini20250416_high_1746217844',
    'assistantbench_hal_generalist_agent_o4mini20250416_low_1746217634',
    'assistantbench_hal_generalist_agent_o320250416_1746219524',
]

class TracePreprocessor:
    def __init__(self, db_dir='preprocessed_traces'):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)
        self.local = threading.local()
        self.connections = {}
    
    @staticmethod
    def get_fallback_accuracy(results):
        if 'accuracy' in results and results['accuracy'] is not None:
            return results['accuracy']
        elif 'average_correctness' in results and results['average_correctness'] is not None:
            return results['average_correctness']
        elif 'success_rate' in results and results['success_rate'] is not None:
            return results['success_rate']
        elif 'average_score' in results and results['average_score'] is not None:
            return results['average_score']
        else:
            return None
    
    @staticmethod
    def get_model_show_name(model_name):
        for mapping in MODEL_MAPPING:
            if model_name == mapping[0]:
                return mapping[1]
        return model_name
    
    def get_all_runs(self):
        records = []
        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            with self.get_conn(benchmark_name) as conn:
                # Try to get all runs from parsed_results
                df = pd.read_sql_query(
                    "SELECT benchmark_name, agent_name, model_name, run_id FROM parsed_results",
                    conn
                )
                if not df.empty:
                    records.append(df)
        if not records:
            return pd.DataFrame(columns=['benchmark_name', 'agent_name', 'model_name', 'run_id'])
        all_df = pd.concat(records, ignore_index=True)
        return all_df
    
    def get_model_benchmark_accuracies(self):
        EXCLUDE_BENCHMARKS = [
            'colbench_backend_programming',
            'colbench_frontend_design',
        ]

        BENCHMARK_ALIAS = {
            "usaco":                 "USACO",
            "taubench_airline":      "TAU-bench Airline",
            "swebench_verified_mini":"SWE-bench Verified Mini",
            "scicode":               "Scicode",
            "online_mind2web":       "Online Mind2Web",
            "gaia":                  "GAIA",
            "corebench_hard":        "CORE-Bench Hard",
        }

        MODEL_ALIAS = {
            "Claude-3.7 Sonnet High (February 2025)":   "Claude-3.7 High Feb 25",
            "Claude-3.7 Sonnet (February 2025)":   "Claude-3.7 Feb 25",
            "DeepSeek R1":                             "DeepSeek R1",
            "DeepSeek V3":                             "DeepSeek V3",
            "GPT-4.1 (April 2025)":                    "GPT-4.1 Apr 25",
            "GPT-4o (August 2024)":                    "GPT-4o Aug 24",
            "GPT-4o (November 2024)":                  "GPT-4o Nov 24",
            "Gemini 2.0 Flash":                        "Gemini 2.0 Flash",
            "Gemini 2.5 Pro Preview (March 2025)":     "Gemini 2.5 Mar 5",
            "o1 Medium (December 2024)":               "o1 Med Dec 24",
            "o3 Medium (April 2025)":                  "o3 Med Apr 25",
            "o3-mini Low (January 2025)":              "o3-mini Low Jan 25",
            "o4-mini Medium (April 2025)":             "o4-mini Med Apr 25",
            "o4-mini Low (April 2025)":             "o4-mini Low Apr 25",
            "o4-mini High (April 2025)":             "o4-mini High Apr 25",
            "Claude Opus 4 High (May 2025)":         "Claude Opus 4 High May 25",
            "Claude Opus 4 (May 2025)":              "Claude Opus 4 May 25",
            "Claude Opus 4.1 (August 2025)":         "Claude Opus 4.1 Aug 25",
            "Claude Opus 4.1 High (August 2025)":    "Claude Opus 4.1 High Aug 25",
        }

        # MODEL_ALIAS = {
        #     "Claude-3.7 Sonnet Low (February 2025)":   "Claude-3.7 Sonnet Low (February 2025)",
        #     "DeepSeek R1":                             "DeepSeek R1",
        #     "DeepSeek V3":                             "DeepSeek V3",
        #     "GPT-4.1 (April 2025)":                    "GPT-4.1 (April 2025)",
        #     "GPT-4o (August 2024)":                    "GPT-4o (August 2024)",
        #     "GPT-4o (November 2024)":                  "GPT-4o (November 2024)",
        #     "Gemini 2.0 Flash":                        "Gemini 2.0 Flash",
        #     "Gemini 2.5 Pro Preview (March 2025)":     "Gemini 2.5 Pro Preview (March 2025)",
        #     "o1 Medium (December 2024)":               "o1 Medium (December 2024)",
        #     "o3 Medium (April 2025)":                  "o3 Medium (April 2025)",
        #     "o3-mini Low (January 2025)":              "o3-mini Low (January 2025)",
        #     "o4-mini Medium (April 2025)":             "o4-mini Medium (April 2025)",
        # }

        records = []
        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            if benchmark_name in EXCLUDE_BENCHMARKS:
                continue
            try:
                with self.get_conn(benchmark_name) as conn:
                    # Only select rows with model_name and accuracy
                    df = pd.read_sql_query(
                        "SELECT model_name, accuracy FROM parsed_results",
                        conn
                    )
                    if df.empty:
                        continue
                    # Group by model_name and compute mean accuracy
                    grouped = df.groupby('model_name')['accuracy'].mean().reset_index()
                    grouped['benchmark_name'] = benchmark_name
                    records.append(grouped)
            except Exception as e:
                print(f"Error processing {db_file}: {e}")
                continue
        if not records:
            return pd.DataFrame(columns=['benchmark_name', 'model_name', 'accuracy'])
        all_df = pd.concat(records, ignore_index=True)
        # Clean up names if needed
        all_df = all_df[['benchmark_name', 'model_name', 'accuracy']]
        all_df["benchmark_name"] = all_df["benchmark_name"].replace(BENCHMARK_ALIAS)
        all_df["model_name"] = all_df["model_name"].replace(MODEL_ALIAS)
        return all_df
        
    def get_conn(self, benchmark_name):
        # Sanitize benchmark name for filename
        safe_name = benchmark_name.replace('/', '_').replace('\\', '_')
        db_path = self.db_dir / f"{safe_name}.db"
        
        # Get thread-specific connections dictionary
        if not hasattr(self.local, 'connections'):
            self.local.connections = {}
            
        # Create new connection if not exists for this benchmark
        if safe_name not in self.local.connections:
            self.local.connections[safe_name] = sqlite3.connect(db_path)
            
        return self.local.connections[safe_name]

    def create_tables(self, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            # Create parsed_results table dynamically from schema
            columns = [f"{col} {dtype}" for col, dtype in PARSED_RESULTS_COLUMNS.items()]
            create_parsed_results = f'''
                CREATE TABLE IF NOT EXISTS parsed_results (
                    {', '.join(columns)},
                    PRIMARY KEY (benchmark_name, agent_name, run_id)
                )
            '''
            conn.execute(create_parsed_results)
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS preprocessed_traces (
                    benchmark_name TEXT,
                    agent_name TEXT,
                    date TEXT,
                    run_id TEXT,
                    raw_logging_results BLOB,
                    PRIMARY KEY (benchmark_name, agent_name, run_id)
                )
            ''')
            # conn.execute('''
            #     CREATE TABLE IF NOT EXISTS failure_reports (
            #         benchmark_name TEXT,
            #         agent_name TEXT,
            #         date TEXT,
            #         run_id TEXT,
            #         failure_report BLOB,
            #         PRIMARY KEY (benchmark_name, agent_name, run_id)
            #     )
            # ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS token_usage (
                    benchmark_name TEXT,
                    agent_name TEXT,
                    run_id TEXT,
                    model_name TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    input_tokens_cache_write INTEGER,
                    input_tokens_cache_read INTEGER,
                    is_primary INTEGER DEFAULT 0,
                    PRIMARY KEY (benchmark_name, agent_name, run_id, model_name)
                )
            ''')

    def preprocess_traces(self, processed_dir="evals_live", skip_existing=True):
        processed_dir = Path(processed_dir)
        
        # Track processed files to avoid reprocessing
        processed_files_log = self.db_dir / "processed_files.txt"
        processed_files = set()
        if skip_existing and processed_files_log.exists():
            with open(processed_files_log, 'r') as f:
                processed_files = set(line.strip() for line in f)

        for file in processed_dir.glob('*.json'):
            # Skip if file already processed
            if skip_existing and str(file) in processed_files:
                print(f"Skipping already processed file: {file}")
                continue
            print(f"Processing {file}")
            primary_model_name = None
            show_primary_model_name = None
            agent_name_with_model = None
            model_show_name = None

            with open(file, 'r') as f:
                data = json.load(f)
                stem = file.stem

                config = data['config']
                run_id = config.get('run_id')
                if run_id in RUNIDS_TO_SKIP:
                    continue

                agent_name = config['agent_name']
                benchmark_name = config['benchmark_name']
                if "inspect" in benchmark_name:
                    benchmark_name = benchmark_name.split("/")[-1]
                date = config['date']



            # Create tables for this benchmark if they don't exist
            self.create_tables(benchmark_name)

            # try:
            #     failure_report = pickle.dumps(data['failure_report'])
            #     with self.get_conn(benchmark_name) as conn:
            #         conn.execute('''
            #             INSERT INTO failure_reports 
            #             (benchmark_name, agent_name, date, run_id, failure_report)
            #             VALUES (?, ?, ?, ?, ?)
            #         ''', (benchmark_name, agent_name, date, config['run_id'], failure_report))
            # except Exception as e:
            #     print(f"Error preprocessing failure_report in {file}: {e}")

            try:
                total_usage = data.get('total_usage', {})
                print(f"Total usage is: {total_usage}")

                # get reasoning effort if any - try both old and new key formats
                reasoning_effort = (data['config']['agent_args'].get('reasoning_effort') or
                                    data['config']['agent_args'].get('agent.model.reasoning_effort'))

                if benchmark_name in ['corebench_hard', 'swebench_verified_mini', 'taubench_airline', 'scienceagentbench'] or '(' not in agent_name:
                    # Try both old and new key formats to cover all possible cases
                    primary_model_name = (data['config']['agent_args'].get('model_name') or 
                                        data['config']['agent_args'].get('agent.model.name'))
                else:
                    # Find primary model from agent_name knowing Agent name is in the format "AgentName (ModelName)"
                    primary_model_name = agent_name.split('(')[-1].strip(' )') if '(' in agent_name else None

                
                # Find the primary model based on total tokens
                if primary_model_name is None:
                    max_tokens = -1
                    for model_name, usage in total_usage.items():
                        completion_tokens = usage.get('completion_tokens', 0)
                        if completion_tokens > max_tokens:
                            max_tokens = completion_tokens
                            primary_model_name = model_name

                # Add reasoning effort if it exists and isn't already in the model name
                if reasoning_effort and primary_model_name and reasoning_effort.lower() not in primary_model_name.lower():
                    primary_model_name = f"{primary_model_name} {reasoning_effort}"
                
                # if primary model has a provider in will be in the form "anthropic/claude-3-7-sonnet-20250219"
                # split on "/" and just take the model name
                if primary_model_name and "/" in primary_model_name:
                    primary_model_name = primary_model_name.split("/")[-1]

                show_primary_model_name = self.get_model_show_name(primary_model_name) if primary_model_name else primary_model_name

                # save in csv for debugging
                with open('primary_model.csv', 'a') as f:
                    f.write(f"{benchmark_name},{agent_name},{primary_model_name},{show_primary_model_name}\n")

                # If show_primary_model_name is part of models to  skip, skip this agent
                if show_primary_model_name in MODELS_TO_SKIP:
                    # Do not skip 'Gemini 2.5 Pro Preview (March 2025)' if it's part of corebench
                    if benchmark_name == 'corebench_hard' and show_primary_model_name == 'Gemini 2.5 Pro Preview (March 2025)':
                        print(f"Not skipping agent {agent_name} for benchmark {benchmark_name} despite primary model {show_primary_model_name} being in MODELS_TO_SKIP")
                    else:
                        print(f"Skipping agent {agent_name} for benchmark {benchmark_name} due to primary model {show_primary_model_name} being in MODELS_TO_SKIP")
                        continue # This will skip this agent for this benchmark and continue to the next file

                # Rename agent_name with primary model show name (only once)
                base_agent_name = re.sub(r'\s*\(.*?\)$', '', agent_name)

                # Simple string replacements
                simple_replacements = [
                    ('Browser-Use_test', 'Browser-Use'),
                    ('hal', 'HAL'),
                    ('Hal', 'HAL'),
                    ('HAl', 'HAL'),
                ]
                
                # Apply simple replacements
                for old, new in simple_replacements:
                    base_agent_name = base_agent_name.replace(old, new)
                
                # Pattern-based mappings with case-insensitive matching
                # HAL Generalist patterns
                if 'hal' in base_agent_name.lower() and 'generalist' in base_agent_name.lower():
                    base_agent_name = 'HAL Generalist Agent'
                
                # Self-Debug patterns
                elif 'self-debug' in base_agent_name.lower() or 'selfdebug' in base_agent_name.lower():
                    base_agent_name = 'SAB Self-Debug'
                
                # TAU-bench patterns
                elif any(pattern in base_agent_name.lower() for pattern in ['few shot', 'fewshot']):
                    base_agent_name = 'TAU-bench Few Shot'
                
                elif ('tool calling' in base_agent_name.lower() or 'toolcalling' in base_agent_name.lower()) and 'tau' in base_agent_name.lower():
                    base_agent_name = 'TAU-bench Tool Calling'
                
                # USACO patterns
                elif 'usaco' in base_agent_name.lower():
                    if 'episodic' in base_agent_name.lower() and 'semantic' in base_agent_name.lower():
                        base_agent_name = 'USACO Episodic + Semantic'
                    else:
                        base_agent_name = 'USACO Agent'
                
                # Browser/Assistant patterns
                elif any(pattern in base_agent_name.lower() for pattern in ['browser', 'assistantbench']):
                    base_agent_name = 'Browser-Use'
                
                # CORE-Agent patterns
                elif 'coreagent' in base_agent_name.lower() or 'core-agent' in base_agent_name.lower() or 'Core Agent' in base_agent_name.lower():
                    base_agent_name = 'CORE-Agent'
                
                # Col-bench patterns
                elif 'colbench' in base_agent_name.lower():
                        base_agent_name = 'Col-bench Text'
                
                # SWE-Agent patterns
                elif any(pattern in base_agent_name.lower() for pattern in ['my_agent', 'my agent', 'sweagent', 'swe-agent']):
                    base_agent_name = 'SWE-Agent'
                
                # SciCode patterns
                
                # HF Open Deep Research patterns
                elif 'hf_open_deep_research' in base_agent_name.lower() or 'hf open deep research' in base_agent_name.lower():
                    base_agent_name = 'HF Open Deep Research'
                
                # SeeAct patterns
                elif 'seeact' in base_agent_name.lower():
                    base_agent_name = 'SeeAct'
                
                # If no pattern matches, use exact mappings as fallback
                else:
                    exact_mappings = {
                        'HAL Generalist': 'HAL Generalist Agent',
                        'HAL Generalist High Reasoning': 'HAL Generalist Agent',
                        'HAL Generalist No Reasoning': 'HAL Generalist Agent',
                        'HAL Generalist Minimal Reasoning': 'HAL Generalist Agent',
                        'TauBench Few Shot High Reasoning': 'TAU-bench Few Shot',
                        'TauBench Few Shot': 'TAU-bench Few Shot',
                        'TauBench Few-Shot High Reasoning': 'TAU-bench Few Shot',
                        'TauBench Few-Shot Minimal Reasoning': 'TAU-bench Few Shot',
                        'TAU-bench Few-shot No Reasoning': 'TAU-bench Few Shot',
                        'TAU-bench FewShot No Reasoning': 'TAU-bench Few Shot',
                        'TAU-bench FewShot': 'TAU-bench Few Shot',
                        'Taubench FewShot High Reasoning': 'TAU-bench Few Shot',
                        'Taubench FewShot No Reasoning': 'TAU-bench Few Shot',
                        'TAU-bench Few Shot High Reasoning': 'TAU-bench Few Shot',
                        'Taubench ToolCalling': 'TAU-bench Tool Calling',
                        'Assistantbench Browser Agent': 'Browser-Use',
                        'Browser Agent': 'Browser-Use',
                        'coreagent': 'CORE-Agent',
                        'CORE-Agent': 'CORE-Agent',
                        'colbench_text_sonnet37': 'Col-bench Text',
                        'SAB Self-Debug Claude-3-7 low': 'SAB Self-Debug',
                        'My Agent': 'SWE-Agent',
                        'SAB Example Agent': 'SAB Self-Debug',
                        'SciCode Tool Calling Agent': 'Scicode Tool Calling Agent',
                        'colbench_backend_programming colbench_example_agent': 'Col-bench Text',
                        'Core Agent Opus 4.5': 'CORE-Agent',
                        'Core Agent Opus 4.5 high': 'CORE-Agent',
                        'Core Agent': 'CORE-Agent',
                    }
                    
                    # Apply exact mappings
                    if base_agent_name in exact_mappings:
                        base_agent_name = exact_mappings[base_agent_name]

                agent_name_with_model = f"{base_agent_name} ({show_primary_model_name})" if show_primary_model_name else base_agent_name

                for model_name, usage in total_usage.items():
                    model_show_name = self.get_model_show_name(model_name)
                    with self.get_conn(benchmark_name) as conn:
                        conn.execute('''
                            INSERT OR REPLACE INTO token_usage 
                            (benchmark_name, agent_name, run_id, model_name, 
                            prompt_tokens, completion_tokens, input_tokens, output_tokens, total_tokens,
                            input_tokens_cache_write, input_tokens_cache_read, is_primary)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            benchmark_name,
                            agent_name_with_model,
                            config['run_id'],
                            model_show_name,
                            usage.get('prompt_tokens', 0),
                            usage.get('completion_tokens', 0),
                            usage.get('input_tokens', 0),
                            usage.get('output_tokens', 0),
                            usage.get('total_tokens', 0),
                            usage.get('cache_creation_input_tokens', 0),
                            usage.get('cache_read_input_tokens', 0),
                            1 if model_name == primary_model_name else 0
                        ))
                        print(f"{benchmark_name + agent_name + config['run_id'] + model_name}")
            except Exception as e:
                print(f"Error preprocessing token usage in {file}: {e}")
                print(f"{benchmark_name + agent_name + config['run_id'] + model_show_name}")
            

            try:
                # raw_logging_results = pickle.dumps(data['raw_logging_results'])
                with self.get_conn(benchmark_name) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO preprocessed_traces 
                        (benchmark_name, agent_name, date, run_id) 
                        VALUES (?, ?, ?, ?)
                    ''', (benchmark_name, agent_name_with_model, date, config['run_id']))
            except Exception as e:
                print(f"Error preprocessing raw_logging_results in {file}: {e}")

            try:
                results = data['results']

                # Ensure 'accuracy' key exists with a fallback
                if 'accuracy' not in results or results['accuracy'] is None:
                    fallback = self.get_fallback_accuracy(results)
                    if fallback is not None:
                        results['accuracy'] = fallback

                results['model_name'] = show_primary_model_name
                results['trace_stem'] = stem

                with self.get_conn(benchmark_name) as conn:
                    columns = [col for col in PARSED_RESULTS_COLUMNS.keys() 
                            if col not in ['benchmark_name', 'agent_name', 'date', 'run_id']]
                    placeholders = ','.join(['?'] * (len(columns) + 4)) # +4 for benchmark_name, agent_name, date, run_id
                    values = [
                        benchmark_name,
                        agent_name_with_model,
                        config['date'],
                        config['run_id']
                    ] + [str(results.get(col)) if col in ['successful_tasks', 'failed_tasks'] 
                        else results.get(col) for col in columns]

                    query = f'''
                        INSERT OR REPLACE INTO parsed_results  
                        ({', '.join(PARSED_RESULTS_COLUMNS.keys())})
                        VALUES ({placeholders})
                    '''
                    conn.execute(query, values)
                    
                    # After successful processing, create a reduced version of the JSON file
                    # Keep only the essential keys used during preprocessing
                    reduced_data = {
                        'config': data['config'],
                        'results': data['results'],
                        'total_usage': data.get('total_usage', {}),
                        # Keep any other small essential keys if they exist
                        'metadata': {
                            'processed': True,
                            'original_file_size': len(json.dumps(data)),
                            'processed_date': date
                        }
                    }
                    
                    # Overwrite the original file with the reduced version
                    try:
                        with open(file, 'w') as f:
                            json.dump(reduced_data, f, indent=2)
                        print(f"Reduced file size for {file}")
                    except Exception as e:
                        print(f"Error saving reduced file {file}: {e}")
                    
                    # Mark file as processed
                    if skip_existing:
                        with open(processed_files_log, 'a') as f:
                            f.write(f"{file}\n")
            except Exception as e:
                print(f"Error preprocessing parsed results in {file}: {e}")

    @lru_cache(maxsize=100)
    def get_analyzed_traces(self, agent_name, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, raw_logging_results, date FROM preprocessed_traces 
                WHERE benchmark_name = ? AND agent_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name, agent_name))

        # check for each row if raw_logging_results is not None
        df = df[df['raw_logging_results'].apply(lambda x: pickle.loads(x) is not None and x != 'None')]

        if len(df) == 0:
            return None

        # select latest run
        df = df.sort_values('date', ascending=False).groupby('agent_name').first().reset_index()

        return pickle.loads(df['raw_logging_results'][0])

    @lru_cache(maxsize=100)
    def get_failure_report(self, agent_name, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, date, failure_report FROM failure_reports 
                WHERE benchmark_name = ? AND agent_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name, agent_name))

        df = df[df['failure_report'].apply(lambda x: pickle.loads(x) is not None and x != 'None')]

        if len(df) == 0:
            return None

        df = df.sort_values('date', ascending=False).groupby('agent_name').first().reset_index()

        return pickle.loads(df['failure_report'][0])

    def _filter_invalid_agents(self, df):
        """Filter out agents with known data leakage issues"""
        # Filter out TAU-bench Few Shot agents due to data leakage issue
        # See Appendix A5 of HAL paper for detailed explanation
        return df[~df['agent_name'].str.contains('TAU-bench Few Shot', case=False, na=False)]

    def _calculate_ci(self, data, confidence=0.95, type='minmax'):
        data = data[np.isfinite(data)]

        if len(data) < 2:
            return '', '', '' # No CI for less than 2 samples
        n = len(data)

        mean = np.mean(data)

        if type == 't':
            sem = stats.sem(data)
            ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)

        elif type == 'minmax':
            min = np.min(data)
            max = np.max(data)
            ci = (min, max)
        return mean, ci[0], ci[1]

    
    def get_parsed_results(self, benchmark_name, aggregate=True):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT * FROM parsed_results 
                WHERE benchmark_name = ?
                ORDER BY accuracy DESC
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))

        # Filter out TAU-bench Few Shot agents due to data leakage issue
        # See Appendix A5 of HAL paper for detailed explanation
        df = df[~df['agent_name'].str.contains('TAU-bench Few Shot', case=False, na=False)]

        # Load metadata
        with open('agents_metadata.yaml', 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Create URL mapping
        url_mapping = {}
        if benchmark_name in metadata:
            for agent in metadata[benchmark_name]:
                if 'url' in agent and agent['url']:  # Only add if URL exists and is not empty
                    url_mapping[agent['agent_name']] = agent['url']

        # Add 'Verified' column
        # verified_agents = self.load_verified_agents()
        # Temporary hack TO DO: Restore logic with yaml file later 
        df['Verified'] = '✓'
        # df['Verified'] = df.apply(lambda row: '✓' if (benchmark_name, row['agent_name']) in verified_agents else '', axis=1)

        # Add URLs to agent names if they exist
        df['url'] = df['agent_name'].apply(lambda x: url_mapping.get(x, ''))

        # Add column for how many times an agent_name appears in the DataFrame
        df['Runs'] = df.groupby('agent_name')['agent_name'].transform('count')

        # Compute the 95% confidence interval for accuracy and cost for agents that have been run more than once
        df['accuracy_ci'] = None
        df['cost_ci'] = None

        # Before dropping run_id, create new column from it with download link
        # First create a temporary dataframe with agent_name and max accuracy
        max_accuracy_df = df.groupby('agent_name')['accuracy'].transform('max')
        # Create mask for rows with max accuracy in their group
        max_accuracy_mask = df['accuracy'] == max_accuracy_df
        # Create the Traces column, only setting values for max accuracy rows
        df['Traces'] = ''
        df.loc[max_accuracy_mask, 'Traces'] = df.loc[max_accuracy_mask, 'trace_stem'].apply(
            lambda x: f'https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/{x}.zip?download=true' if x else ''
        )
        # df['Traces'] = ''
        # df.loc[max_accuracy_mask, 'Traces'] = df.loc[max_accuracy_mask, 'run_id'].apply(
        #     lambda x: f'https://huggingface.co/datasets/agent-evals/agent_traces/resolve/main/{x}.zip?download=true'
        #     if x else ''
        # )
        
        df = df.drop(columns=['successful_tasks', 'failed_tasks'], axis=1)
        
        if aggregate:
            df = df.groupby('agent_name').agg(AGGREGATION_RULES).reset_index()
            
        # Rename columns using the display names mapping
        df = df.rename(columns=COLUMN_DISPLAY_NAMES)
        
        # Multiply accuracy by 100
        df['Accuracy'] = df['Accuracy'] * 100
        df['Scenario Goal Completion'] = df['Scenario Goal Completion'] * 100
        df['Task Goal Completion'] = df['Task Goal Completion'] * 100
        df['Level 1 Accuracy'] = df['Level 1 Accuracy'] * 100
        df['Level 2 Accuracy'] = df['Level 2 Accuracy'] * 100
        df['Level 3 Accuracy'] = df['Level 3 Accuracy'] * 100
        df['Refusals'] = df['Refusals'] * 100
        df['Non-Refusal Harm Score'] = df['Non-Refusal Harm Score'] * 100
        return df
    
    def get_task_success_data(self, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, accuracy, successful_tasks, failed_tasks
                FROM parsed_results 
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        
        # Filter out invalid agents
        df = self._filter_invalid_agents(df)
        
        # Get all unique task IDs
        task_ids = set()
        for tasks in df['successful_tasks']:
            if ast.literal_eval(tasks) is not None:
                task_ids.update(ast.literal_eval(tasks))
        for tasks in df['failed_tasks']:
            if ast.literal_eval(tasks) is not None:
                task_ids.update(ast.literal_eval(tasks))

        # Create a DataFrame with agent_name, task_ids, and success rates
        data_list = []
        for task_id in task_ids:
            for agent_name in df['agent_name'].unique():
                agent_runs = df[df['agent_name'] == agent_name]
                # Count how many times this task was successful across all runs
                successes = sum(1 for _, row in agent_runs.iterrows() 
                              if ast.literal_eval(row['successful_tasks']) is not None 
                              and task_id in ast.literal_eval(row['successful_tasks']))
                total_runs = len(agent_runs)
                success_rate = successes / total_runs if total_runs > 0 else 0
                
                data_list.append({
                    'agent_name': agent_name,
                    'task_id': task_id,
                    'success': success_rate
                })
        df = pd.DataFrame(data_list)

        df = df.rename(columns={
            'agent_name': 'Agent Name',
            'task_id': 'Task ID',
            'success': 'Success'
        })

        return df
    
    def load_verified_agents(self, file_path='agents_metadata.yaml'):
        with open(file_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        verified_agents = set()
        for benchmark, agents in metadata.items():
            for agent in agents:
                if 'verification_date' in agent:  # Only add if verified
                    verified_agents.add((benchmark, agent['agent_name']))
        
        return verified_agents

    def _normalize_usage(self, token_data):
        """Normalize token usage data from different API formats"""
        # Handle different token naming conventions
        if "prompt_tokens" in token_data or "completion_tokens" in token_data:
            # OpenAI-style
            prompt_tokens = token_data.get("prompt_tokens", 0)
            cached_input = token_data.get("prompt_tokens_details", {}).get("cached_tokens", 0) if isinstance(token_data.get("prompt_tokens_details"), dict) else 0
            cache_creation = 0  # OpenAI doesn't report cache writes separately
            
        elif "input_tokens" in token_data or "output_tokens" in token_data:
            # Anthropic-style
            fresh_input = token_data.get("input_tokens", 0)
            cached_input = token_data.get("cache_read_input_tokens", token_data.get("input_tokens_cache_read", 0))
            cache_creation = token_data.get("cache_creation_input_tokens", token_data.get("input_tokens_cache_write", 0))
            prompt_tokens = fresh_input + cached_input
            
        elif "inputTokens" in token_data or "outputTokens" in token_data:
            # Bedrock-style
            prompt_tokens = token_data.get("inputTokens", 0)
            cached_input = token_data.get("cacheReadInputTokens", token_data.get("input_tokens_cache_read", 0))
            cache_creation = token_data.get("cacheWriteInputTokens", token_data.get("input_tokens_cache_write", 0))
            
        else:
            prompt_tokens = 0
            cached_input = 0
            cache_creation = 0
        
        completion = (
            token_data.get("completion_tokens", 0)
            + token_data.get("output_tokens", 0)
            + token_data.get("outputTokens", 0)
        )
        
        return prompt_tokens, cached_input, cache_creation, completion

    def get_token_usage_with_costs(self, benchmark_name, pricing_config=None, ignore_caching=True):
        """Get token usage data with configurable pricing
        
        Args:
            benchmark_name: Name of the benchmark
            pricing_config: Custom pricing configuration
            ignore_caching: If True, treats all tokens as regular prompt/completion tokens (old logic)
        """
        if pricing_config is None:
            pricing_config = DEFAULT_PRICING

        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, model_name, run_id,
                SUM(prompt_tokens) as prompt_tokens,
                SUM(completion_tokens) as completion_tokens,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(input_tokens_cache_write) as input_tokens_cache_write,
                SUM(input_tokens_cache_read) as input_tokens_cache_read
                FROM token_usage
                WHERE benchmark_name = ?
                GROUP BY agent_name, model_name, run_id
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        
        # Filter out invalid agents
        df = self._filter_invalid_agents(df)
                    
        # Calculate costs based on pricing config
        df['total_cost'] = 0.0
        for model, prices in pricing_config.items():
            mask = df['model_name'] == model
            if mask.any():
                if ignore_caching:
                    # Old logic: treat all tokens as regular prompt/completion tokens
                    df.loc[mask, 'total_cost'] = (
                        df.loc[mask, 'input_tokens'].fillna(0) * prices['prompt_tokens'] / 1e6 +
                        df.loc[mask, 'output_tokens'].fillna(0) * prices['completion_tokens'] / 1e6 +
                        df.loc[mask, 'prompt_tokens'].fillna(0) * prices['prompt_tokens'] / 1e6 +
                        df.loc[mask, 'completion_tokens'].fillna(0) * prices['completion_tokens'] / 1e6
                    )
                else:
                    # New logic: use cache-specific pricing
                    cache_create_price = CACHED_PRICE_OVERRIDES.get(model, prices.get("prompt_tokens", 0))
                    cache_read_price = CACHED_PRICE_OVERRIDES.get(model, prices.get("prompt_tokens", 0))
                    
                    # Calculate fresh input tokens (total input - cached reads)
                    fresh_input_tokens = (
                        df.loc[mask, 'input_tokens'].fillna(0) + 
                        df.loc[mask, 'prompt_tokens'].fillna(0) - 
                        df.loc[mask, 'input_tokens_cache_read'].fillna(0)
                    )
                    
                    # Calculate total cost with proper cache pricing
                    df.loc[mask, 'total_cost'] = (
                        fresh_input_tokens * prices['prompt_tokens'] / 1e6 +
                        df.loc[mask, 'input_tokens_cache_write'].fillna(0) * cache_create_price / 1e6 +
                        df.loc[mask, 'input_tokens_cache_read'].fillna(0) * cache_read_price / 1e6 +
                        (df.loc[mask, 'output_tokens'].fillna(0) + df.loc[mask, 'completion_tokens'].fillna(0)) * prices['completion_tokens'] / 1e6
                    )
            
        # Sum total_cost for each run_id (if agents use multiple models, this will be the total cost for that run)
        df_temp = df.groupby('run_id')['total_cost'].sum().reset_index()
        df_temp = df_temp.rename(columns={'total_cost': 'total_cost_temp'})
        df = df.merge(df_temp, on='run_id', how='left')
        df['total_cost'] = df['total_cost_temp']
        df = df.drop('total_cost_temp', axis=1)
                                
        return df

    def get_parsed_results_with_costs(self, benchmark_name, pricing_config=None, aggregate=False, ignore_caching=True):
        """Get parsed results with recalculated costs based on token usage"""
        # Get base results with URLs
        results_df = self.get_parsed_results(benchmark_name, aggregate=False)
        benchmark_name = results_df['benchmark_name'].iloc[0]
        
        # Get token usage with new costs
        token_costs = self.get_token_usage_with_costs(benchmark_name, pricing_config, ignore_caching=ignore_caching)
        # import pdb; pdb.set_trace()

        for agent_name in results_df['Agent Name'].unique():
            agent_df = results_df[results_df['Agent Name'] == agent_name]
            
            if agent_name not in token_costs['agent_name'].unique():
                token_costs_df = results_df[results_df['Agent Name'] == agent_name]
            else:
                token_costs_df = token_costs[token_costs['agent_name'] == agent_name]
                
            if len(agent_df) > 1:
                accuracy_mean, accuracy_lower, accuracy_upper = self._calculate_ci(agent_df['Accuracy'], type='minmax')
                if agent_name not in token_costs['agent_name'].unique():
                    cost_mean, cost_lower, cost_upper = self._calculate_ci(token_costs_df['Total Cost'], type='minmax')
                else:
                    cost_mean, cost_lower, cost_upper = self._calculate_ci(token_costs_df['total_cost'], type='minmax')
                
                # Round CI values to 2 decimals
                accuracy_ci = f"-{abs(accuracy_mean - accuracy_lower):.2f}/+{abs(accuracy_mean - accuracy_upper):.2f}"
                cost_ci = f"-{abs(cost_mean - cost_lower):.2f}/+{abs(cost_mean - cost_upper):.2f}"
                
                results_df.loc[results_df['Agent Name'] == agent_name, 'Accuracy CI'] = accuracy_ci
                results_df.loc[results_df['Agent Name'] == agent_name, 'Total Cost CI'] = cost_ci
            
            if agent_name == 'Inspect ReAct Agent (o3-mini-2025-01-14)':
                results_df.loc[results_df['Agent Name'] == agent_name, 'Total Cost CI'] = "lower bound: $16.04*"
    
        # Group token costs by agent
        agent_costs = token_costs.groupby('agent_name')['total_cost'].mean().reset_index()

        agent_costs = agent_costs.rename(columns={
            'agent_name': 'agent_name_temp',
            'total_cost': 'Total Cost'
        })
                        
        # Drop existing Total Cost column if it exists
        if 'Total Cost' in results_df.columns:
            results_df['total_cost_temp'] = results_df['Total Cost']
            results_df = results_df.drop('Total Cost', axis=1)
            
        # Create temp column for matching, preserving the original Agent Name with URL
        results_df['agent_name_temp'] = results_df['Agent Name'].apply(lambda x: x.split('[')[1].split(']')[0] if '[' in x else x)
        
        # Update costs in results
        results_df = results_df.merge(agent_costs, on='agent_name_temp', how='left')
                
        # if Total Cost is NaN, set it to the value from total_cost_temp if it exists
        results_df['Total Cost'] = results_df['Total Cost'].fillna(results_df['total_cost_temp'])
        
        # If there is no token usage data, set Total Cost to total_cost from results key
        if len(token_costs) < 1:
            results_df['Total Cost'] = results_df['Total Cost'].fillna(results_df['total_cost_temp'])
                
        # Drop temp column
        results_df = results_df.drop('agent_name_temp', axis=1)
        results_df = results_df.drop('total_cost_temp', axis=1)
        
                    
        if aggregate:
            # Aggregate results while preserving URLs in Agent Name
            results_df = results_df.groupby('Agent Name', as_index=False).agg({
                'Date': 'first',
                'Total Cost': 'mean',
                'Accuracy': 'mean',
                'Precision': 'mean',
                'Recall': 'mean',
                'F1 Score': 'mean',
                'AUC': 'mean',
                'Overall Score': 'mean',
                'Vectorization Score': 'mean',
                'Fathomnet Score': 'mean',
                'Feedback Score': 'mean',
                'House Price Score': 'mean',
                'Spaceship Titanic Score': 'mean',
                'AMP Parkinsons Disease Progression Prediction Score': 'mean',
                'CIFAR10 Score': 'mean',
                'IMDB Score': 'mean',
                'Scenario Goal Completion': 'mean',
                'Task Goal Completion': 'mean',
                'Level 1 Accuracy': 'mean',
                'Level 2 Accuracy': 'mean',
                'Level 3 Accuracy': 'mean',
                'Verified': 'first',
                'Traces': 'first',
                'Runs': 'first',
                'Accuracy CI': 'first',
                'Total Cost CI': 'first',
                'URL': 'first',
                'Refusals': 'mean',
                'Non-Refusal Harm Score': 'mean',  # Preserve URL
                'Model Name': 'first',
            })
        
        # Round float columns to 2 decimal places
        float_columns = [
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score',
            'AUC',
            'Overall Score',
            'Vectorization Score',
            'Fathomnet Score',
            'Feedback Score',
            'House Price Score',
            'Spaceship Titanic Score',
            'AMP Parkinsons Disease Progression Prediction Score',
            'CIFAR10 Score',
            'IMDB Score',
            'Level 1 Accuracy',
            'Level 2 Accuracy',
            'Level 3 Accuracy',
            'Total Cost'
        ]
        
        for column in float_columns:
            if column in results_df.columns:
                try:
                    results_df[column] = results_df[column].round(2)
                except Exception as e:
                    print(f"Error rounding {column}: {e}")
        
        return results_df

    def check_token_usage_data(self, benchmark_name):
        """Debug helper to check token usage data"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT * FROM token_usage
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df

    def get_models_for_benchmark(self, benchmark_name):
        """Get list of unique model names used in a benchmark"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT DISTINCT model_name
                FROM token_usage
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df['model_name'].tolist()

    def get_all_agents(self, benchmark_name):
        """Get list of all agent names for a benchmark"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT DISTINCT agent_name
                FROM parsed_results
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df['agent_name'].tolist()

    def get_total_benchmarks(self):
        """Get the total number of unique benchmarks in the database"""
        benchmarks = set()
        for db_file in self.db_dir.glob('*.db'):
            benchmarks.add(db_file.stem.replace('_', '/'))
        return len(benchmarks) - 2 # TODO hardcoded -2 because of benchmarks not added for now

    def get_total_agents(self):
        """Get the total number of unique agents across all benchmarks"""
        total_agents = set()
        # Use the parsed_results table since it's guaranteed to have all benchmark-agent pairs
        for db_file in self.db_dir.glob('*.db'):
            # skip colbench, scienceagentbench
            if db_file.stem in ['colbench_backend_programming', 'colbench_frontend_design']:
                continue # TODO remove hardcoded skip once these benchmarks are added
            benchmark_name = db_file.stem.replace('_', '/')
            with self.get_conn(benchmark_name) as conn:
                query = '''
                    SELECT DISTINCT benchmark_name, agent_name 
                    FROM parsed_results
                '''
                
                results = conn.execute(query).fetchall()
                # Add each benchmark-agent pair to the set
                total_agents.update(results)
        return len(total_agents)
    
    def get_total_agent_runs(self):
        """Get the total number of agent runs across all benchmarks"""
        total_runs = 0
        for db_file in self.db_dir.glob('*.db'):
            # skip colbench, scienceagentbench
            if db_file.stem in ['colbench_backend_programming', 'colbench_frontend_design']:
                continue # TODO remove hardcoded skip once these benchmarks are added
            benchmark_name = db_file.stem.replace('_', '/')
            with self.get_conn(benchmark_name) as conn:
                query = '''
                    SELECT COUNT(*) FROM parsed_results
                '''
                count = conn.execute(query).fetchone()[0]
                total_runs += count
        return total_runs
    
    def get_total_evaluations(self):
        """Get the total number of evaluations (rollouts) across all benchmarks"""
        # Use the same logic as analyze_benchmark_evaluations.py
        
        # Known task counts for each benchmark (same as in the script)
        known_task_counts = {
            'gaia': 165,
            'scicode': 65,
            'usaco': 307,
            'assistantbench': 33,
            'corebench_hard': 45,
            'online_mind2web': 300,
            'taubench_airline': 50,
            'scienceagentbench': 102,
            'swebench_verified_mini': 50,
        }
        
        # List of available benchmarks (same as script)
        available_benchmarks = [
            'gaia',
            'scicode', 
            'usaco',
            'assistantbench',
            'corebench_hard',
            'online_mind2web',
            'taubench_airline',
            'scienceagentbench',
            'swebench_verified_mini',
        ]
        
        total_evaluations = 0
        
        for benchmark_name in available_benchmarks:
            try:
                # Get parsed results with costs which contains detailed run information
                df = self.get_parsed_results_with_costs(benchmark_name, aggregate=False)
                
                if df.empty:
                    continue
                
                # Count unique tasks and runs from the parsed results
                num_tasks = 0
                num_runs = 0
                
                # Method 1: Use known benchmark task counts (most reliable for common benchmarks)
                if benchmark_name in known_task_counts:
                    num_tasks = known_task_counts[benchmark_name]
                
                # Method 2: Try to get tasks from 'Total Tasks' column (override known values if available)
                if 'Total Tasks' in df.columns:
                    # Take the first non-null value or the max if there are variations
                    total_tasks_values = df['Total Tasks'].dropna()
                    if len(total_tasks_values) > 0:
                        actual_count = int(total_tasks_values.iloc[0])
                        if actual_count > 0:  # Only override if we get a positive value
                            num_tasks = actual_count
                
                # Method 3: Try to determine number of tasks from successful/failed task lists
                if num_tasks == 0 and 'successful_tasks' in df.columns and 'failed_tasks' in df.columns:
                    import ast
                    all_task_ids = set()
                    
                    for _, row in df.iterrows():
                        try:
                            # Parse successful tasks
                            if pd.notna(row['successful_tasks']) and row['successful_tasks']:
                                successful = ast.literal_eval(row['successful_tasks'])
                                if successful:
                                    all_task_ids.update(successful)
                            
                            # Parse failed tasks  
                            if pd.notna(row['failed_tasks']) and row['failed_tasks']:
                                failed = ast.literal_eval(row['failed_tasks'])
                                if failed:
                                    all_task_ids.update(failed)
                        except:
                            continue
                    
                    num_tasks = len(all_task_ids)
                
                # Method 4: Look for other task-related columns
                if num_tasks == 0:
                    task_cols = [col for col in df.columns if 'task' in col.lower() and col != 'Total Tasks']
                    for col in task_cols:
                        try:
                            unique_count = df[col].nunique()
                            if unique_count > num_tasks:
                                num_tasks = unique_count
                        except:
                            continue
                
                # Count runs - each row in parsed_results represents one agent run
                num_runs = len(df)
                
                # Calculate total evaluations (tasks × runs)
                benchmark_evaluations = num_tasks * num_runs if num_tasks and num_runs else 0
                total_evaluations += benchmark_evaluations
                
            except Exception as e:
                # Skip benchmarks that cause errors
                continue
        
        return total_evaluations

    def get_agent_url(self, agent_name, benchmark_name):
        """Get the URL for an agent from the metadata file."""
        try:
            with open('agents_metadata.yaml', 'r') as f:
                metadata = yaml.safe_load(f)
                if benchmark_name in metadata:
                    for agent in metadata[benchmark_name]:
                        if agent['agent_name'] == agent_name:
                            return agent.get('url', '')
        except Exception as e:
            print(f"Error getting agent URL: {e}")
        return ''

    def get_highlight_results(self, limit_per_benchmark=3):
        """Get highlight results organized by benchmark and agent for the landing page"""
        EXCLUDE_BENCHMARKS = [
            'colbench_backend_programming',
            'colbench_frontend_design',
        ]
        
        BENCHMARK_DISPLAY_NAMES = {
            "usaco": "USACO",
            "taubench_airline": "TAU-bench Airline",
            "swebench_verified_mini": "SWE-bench Verified Mini",
            "scicode": "Scicode",
            "online_mind2web": "Online Mind2Web",
            "gaia": "GAIA",
            "corebench_hard": "CORE-Bench Hard",
            "assistantbench": "AssistantBench",
            "scienceagentbench": "ScienceAgentBench"
        }
        
        BENCHMARK_CATEGORIES = {
            "usaco": "Programming",
            "taubench_airline": "Customer Service",
            "swebench_verified_mini": "Software Engineering",
            "scicode": "Scientific Programming",
            "online_mind2web": "Web Assistance",
            "gaia": "Web Assistance",
            "corebench_hard": "Scientific Programming",
            "assistantbench": "Web Assistance",
            "scienceagentbench": "Scientific Programming",
        }
        
        highlights = []
        
        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            if benchmark_name in EXCLUDE_BENCHMARKS:
                continue
                
            display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name.replace('_', ' ').title())
            category = BENCHMARK_CATEGORIES.get(benchmark_name, "Other")
            
            try:
                # Use get_parsed_results_with_costs to ensure consistent cost calculations
                df = self.get_parsed_results_with_costs(benchmark_name, aggregate=False)
                # Additional filtering for highlights to ensure no invalid agents
                if not df.empty:
                    df = df[~df['Agent Name'].str.contains('TAU-bench Few Shot', case=False, na=False)]
                
                if df.empty:
                    continue
                
                # Sort by accuracy and take top results
                df_sorted = df.sort_values('Accuracy', ascending=False).head(limit_per_benchmark)
                
                # Convert to list of agent-model combinations
                top_agents = []
                for _, row in df_sorted.iterrows():
                    # Extract base agent name (without model info and URL formatting)
                    agent_name = row['Agent Name']
                    # Remove URL formatting if present
                    if '[' in agent_name and ']' in agent_name:
                        base_agent = agent_name.split('[')[1].split(']')[0]
                    else:
                        base_agent = re.sub(r'\s*\(.*?\)$', '', agent_name).strip()

                    # Get model name - try from Model Name column first, then extract from agent name
                    model_name = row.get('Model Name', '')
                    if not model_name and '(' in agent_name and ')' in agent_name:
                        model_name = agent_name.split('(')[-1].split(')')[0]

                    top_agents.append({
                        'agent_name': agent_name,
                        'base_agent': base_agent,
                        'model_name': model_name,
                        'accuracy': row['Accuracy'],  # Already in percentage form
                        'total_cost': row['Total Cost'] if pd.notna(row['Total Cost']) else 0,
                    })
                
                if top_agents:
                    highlights.append({
                        'benchmark': display_name,
                        'benchmark_key': benchmark_name,
                        'category': category,
                        'agents': top_agents
                    })
                        
            except Exception as e:
                print(f"Error processing highlights for {benchmark_name}: {e}")
                continue
        
        # Sort highlights alphabetically by benchmark display name
        highlights.sort(key=lambda x: x['benchmark'])
        
        return highlights

    def get_model_data_across_benchmarks(self, model_name):
        """Get data for a specific model across all benchmarks"""
        all_data = []
        
        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            try:
                # Use get_parsed_results_with_costs to get correct token-based costs
                df = self.get_parsed_results_with_costs(benchmark_name, aggregate=False)
                
                if not df.empty:
                    # Filter for the specific model
                    df = df[df['Model Name'] == model_name]
                    
                    if not df.empty:
                        # Add benchmark name for reference
                        df['benchmark_name'] = benchmark_name
                        all_data.append(df)
                        
            except Exception as e:
                print(f"Error processing model {model_name} for {benchmark_name}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
            
        # Combine all benchmark data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # The data already has the correct column names from get_parsed_results_with_costs
        # and includes URL information
        
        return combined_df

    def get_agent_data_across_benchmarks(self, agent_name):
        """Get data for a specific agent across all benchmarks
        Note: agent_name should be the base name without model in parentheses"""
        all_data = []
        
        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            try:
                # Use get_parsed_results_with_costs to get correct token-based costs
                df = self.get_parsed_results_with_costs(benchmark_name, aggregate=False)
                
                if not df.empty:
                    # Search for agents that start with the base agent name
                    # This handles cases like "AgentName (ModelName)"
                    mask = df['Agent Name'].str.startswith(agent_name, na=False)
                    df = df[mask]
                    
                    if not df.empty:
                        # Add benchmark name for reference
                        df['benchmark_name'] = benchmark_name
                        all_data.append(df)
                        
            except Exception as e:
                print(f"Error processing agent {agent_name} for {benchmark_name}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
            
        # Combine all benchmark data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # The data already has the correct column names from get_parsed_results_with_costs
        # and includes URL information
        
        return combined_df

    def get_model_performance_data(self, model_name):
        """Get performance data for a specific model with Pareto flag based on agent performance"""
        data = self.get_model_data_across_benchmarks(model_name)
        
        if data.empty:
            return {
                'model_name': model_name,
                'benchmarks': [],
                'is_pareto': False,
                'pricing': self.get_model_pricing(model_name)
            }
        
        # Check if any agent using this model is Pareto optimal
        is_pareto = False
        if 'Is Pareto' in data.columns:
            is_pareto = data['Is Pareto'].any()
        
        # Group by benchmark for easier display
        benchmark_data = []
        for benchmark_name in data['benchmark_name'].unique():
            benchmark_df = data[data['benchmark_name'] == benchmark_name]
            
            benchmark_info = {
                'name': benchmark_name,
                'agents': []
            }
            
            for _, row in benchmark_df.iterrows():
                agent_info = {
                    'agent_name': row.get('Agent Name', ''),
                    'accuracy': row.get('Accuracy', 0),
                    'total_cost': row.get('Total Cost', 0),
                    'url': row.get('URL', ''),
                    'is_pareto': row.get('Is Pareto', False),
                    'date': row.get('Date', '')
                }
                benchmark_info['agents'].append(agent_info)
            
            benchmark_data.append(benchmark_info)
        
        return {
            'model_name': model_name,
            'benchmarks': benchmark_data,
            'is_pareto': is_pareto,
            'pricing': self.get_model_pricing(model_name)
        }
    
    def get_agent_performance_data(self, agent_name):
        """Get performance data for a specific agent"""
        data = self.get_agent_data_across_benchmarks(agent_name)
        
        if data.empty:
            return {
                'agent_name': agent_name,
                'benchmarks': [],
                'is_pareto': False
            }
        
        # Check if this agent is Pareto optimal in any benchmark
        is_pareto = False
        if 'Is Pareto' in data.columns:
            is_pareto = data['Is Pareto'].any()
        
        # Group by benchmark
        benchmark_data = []
        for benchmark_name in data['benchmark_name'].unique():
            benchmark_df = data[data['benchmark_name'] == benchmark_name]
            
            benchmark_info = {
                'name': benchmark_name,
                'runs': []
            }
            
            for _, row in benchmark_df.iterrows():
                run_info = {
                    'model_name': row.get('Model Name', ''),
                    'accuracy': row.get('Accuracy', 0),
                    'total_cost': row.get('Total Cost', 0),
                    'url': row.get('URL', ''),
                    'is_pareto': row.get('Is Pareto', False),
                    'date': row.get('Date', '')
                }
                benchmark_info['runs'].append(run_info)
            
            benchmark_data.append(benchmark_info)
        
        return {
            'agent_name': agent_name,
            'benchmarks': benchmark_data,
            'is_pareto': is_pareto
        }
    
    def calculate_cost_with_cache_tokens(self, benchmark_name, run_id=None, pricing_config=None, ignore_caching=True):
        """
        Calculate costs with proper cache token handling similar to Weave's approach.
        
        Args:
            benchmark_name: Name of the benchmark
            run_id: Optional specific run ID to calculate costs for
            pricing_config: Optional custom pricing configuration
            ignore_caching: If True, treats all tokens as regular prompt/completion tokens (old logic)
            
        Returns:
            dict: Contains total_cost, token_usage breakdown, and number of entries
        """
        if pricing_config is None:
            pricing_config = DEFAULT_PRICING
            
        with self.get_conn(benchmark_name) as conn:
            if run_id:
                query = '''
                    SELECT model_name, 
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(input_tokens_cache_write) as input_tokens_cache_write,
                    SUM(input_tokens_cache_read) as input_tokens_cache_read
                    FROM token_usage
                    WHERE benchmark_name = ? AND run_id = ?
                    GROUP BY model_name
                '''
                df = pd.read_sql_query(query, conn, params=(benchmark_name, run_id))
            else:
                query = '''
                    SELECT model_name, 
                    SUM(prompt_tokens) as prompt_tokens,
                    SUM(completion_tokens) as completion_tokens,
                    SUM(input_tokens) as input_tokens,
                    SUM(output_tokens) as output_tokens,
                    SUM(input_tokens_cache_write) as input_tokens_cache_write,
                    SUM(input_tokens_cache_read) as input_tokens_cache_read
                    FROM token_usage
                    WHERE benchmark_name = ?
                    GROUP BY model_name
                '''
                df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        
        total_cost = 0
        token_usage = {}
        
        for _, row in df.iterrows():
            model_name = row['model_name']
            
            if model_name not in pricing_config:
                print(f"Warning: Model '{model_name}' not found in pricing config. Skipping cost calculation.")
                continue
                
            # Initialize token usage tracking
            if model_name not in token_usage:
                token_usage[model_name] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
            
            # Normalize token counts (handle NaN values)
            prompt_tokens = (row.get('prompt_tokens', 0) or 0) + (row.get('input_tokens', 0) or 0)
            completion_tokens = (row.get('completion_tokens', 0) or 0) + (row.get('output_tokens', 0) or 0)
            cache_creation = row.get('input_tokens_cache_write', 0) or 0
            cache_read = row.get('input_tokens_cache_read', 0) or 0
            
            # Update token usage
            token_usage[model_name]["prompt_tokens"] += prompt_tokens
            token_usage[model_name]["completion_tokens"] += completion_tokens
            token_usage[model_name]["cache_creation_input_tokens"] += cache_creation
            token_usage[model_name]["cache_read_input_tokens"] += cache_read
            
            # Get pricing information
            prices = pricing_config[model_name]
            
            if ignore_caching:
                # Old logic: treat all tokens as regular prompt/completion tokens
                model_cost = (
                    prompt_tokens * prices.get("prompt_tokens", 0) / 1e6 +
                    completion_tokens * prices.get("completion_tokens", 0) / 1e6
                )
            else:
                # New logic: use cache-specific pricing
                cache_create_price = CACHED_PRICE_OVERRIDES.get(model_name, prices.get("prompt_tokens", 0))
                cache_read_price = CACHED_PRICE_OVERRIDES.get(model_name, prices.get("prompt_tokens", 0))
                
                # Calculate fresh input tokens (total input - cached reads)
                fresh_input_tokens = prompt_tokens - cache_read
                
                # Calculate model cost with proper cache pricing
                model_cost = (
                    fresh_input_tokens * prices.get("prompt_tokens", 0) / 1e6 +
                    cache_creation * cache_create_price / 1e6 +
                    cache_read * cache_read_price / 1e6 +
                    completion_tokens * prices.get("completion_tokens", 0) / 1e6
                )
            
            total_cost += model_cost
        
        return {
            "total_cost": total_cost,
            "token_usage": token_usage,
            "num_models": len(df)
        }

    def get_model_pricing(self, model_name):
        """Get pricing information for a model from DEFAULT_PRICING with cache token support"""
        # First try exact match
        if model_name in DEFAULT_PRICING:
            pricing_info = DEFAULT_PRICING[model_name]
            cache_price = CACHED_PRICE_OVERRIDES.get(model_name, pricing_info.get('prompt_tokens', 0))
            return {
                'prompt_tokens': pricing_info.get('prompt_tokens', 0),
                'completion_tokens': pricing_info.get('completion_tokens', 0),
                'cache_creation_tokens': cache_price,
                'cache_read_tokens': cache_price,
                'unit': 'per 1M tokens'
            }
        
        # Try fuzzy matching for slight variations
        clean_model = model_name.lower().replace('-', '').replace('_', '').replace(' ', '')
        
        for pricing_key, pricing_info in DEFAULT_PRICING.items():
            clean_key = pricing_key.lower().replace('-', '').replace('_', '').replace(' ', '')
            if clean_model in clean_key or clean_key in clean_model:
                cache_price = CACHED_PRICE_OVERRIDES.get(pricing_key, pricing_info.get('prompt_tokens', 0))
                return {
                    'prompt_tokens': pricing_info.get('prompt_tokens', 0),
                    'completion_tokens': pricing_info.get('completion_tokens', 0),
                    'cache_creation_tokens': cache_price,
                    'cache_read_tokens': cache_price,
                    'unit': 'per 1M tokens'
                }
        
        return {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cache_creation_tokens': 0,
            'cache_read_tokens': 0,
            'unit': 'per 1M tokens'
        }

    def get_model_pareto_per_benchmark(self, model_name):
        """Return {benchmark: {is_pareto: bool, pareto_agents: [..]}} for a model.
        A model is Pareto on a benchmark if at least one Pareto agent on that
        benchmark uses this model (membership in the 'Models' column)."""
        pareto_data = {}

        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            try:
                from utils.viz import create_leaderboard
                full_benchmark_df = self.get_parsed_results(benchmark_name, aggregate=True)
                leaderboard = create_leaderboard(full_benchmark_df, benchmark_name)

                def model_used(cell):
                    # Handle list or string in 'Models' column
                    if isinstance(cell, list):
                        return any(model_name in str(m) for m in cell)
                    if isinstance(cell, str):
                        return model_name in cell
                    return False

                if 'Models' in leaderboard.columns:
                    mask = leaderboard['Models'].apply(model_used)
                elif 'Model Name' in leaderboard.columns:
                    mask = leaderboard['Model Name'].astype(str).str.contains(model_name, na=False)
                else:
                    mask = pd.Series(False, index=leaderboard.index)

                model_rows = leaderboard[mask]
                is_pareto = False
                pareto_agents = []
                if not model_rows.empty and 'Is Pareto' in model_rows.columns:
                    pareto_mask = model_rows['Is Pareto'] == True
                    is_pareto = pareto_mask.any()
                    if 'Agent Name' in model_rows.columns:
                        pareto_agents = list(model_rows.loc[pareto_mask, 'Agent Name'])

                pareto_data[benchmark_name] = {
                    'is_pareto': bool(is_pareto),
                    'pareto_agents': pareto_agents,
                }

            except Exception as e:
                print(f"Error calculating Pareto for {model_name} in {benchmark_name}: {e}")
                pareto_data[benchmark_name] = {'is_pareto': False, 'pareto_agents': []}

        return pareto_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Database and cost calculation utilities')
    parser.add_argument('--with_caching', action='store_true',
                        help='Use new cost calculation logic with cache token considerations')
    args = parser.parse_args()
    
    # Set global flag for ignore_caching (reversed logic - True by default, False when --with_caching is used)
    IGNORE_CACHING = not args.with_caching
    
    if IGNORE_CACHING:
        print("Using old cost calculation logic (ignoring cache token pricing)")
    else:
        print("Using new cost calculation logic (with cache token pricing)")
    
    preprocessor = TracePreprocessor()
    # Run the preprocessing
    preprocessor.preprocess_traces()
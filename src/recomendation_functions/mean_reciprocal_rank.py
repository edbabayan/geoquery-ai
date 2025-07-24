from typing import List

import numpy as np


def calculate_mrr(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for RAG validation

    Args:
        predictions: List of ranked prediction lists for each query
        ground_truth: List of correct answers for each query

    Returns:
        MRR score (0-1, higher is better)
    """
    reciprocal_ranks = []

    for prediction_list, correct_answer in zip(predictions, ground_truth):
        rank = None
        for i, prediction in enumerate(prediction_list):
            if prediction.strip().lower() == correct_answer.strip().lower():
                rank = i + 1  # rank starts from 1
                break

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)
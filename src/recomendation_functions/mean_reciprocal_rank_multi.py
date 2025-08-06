from typing import List, Set, Tuple
import numpy as np


def parse_multi_table_ground_truth(ground_truth: str) -> Set[str]:
    """
    Parse ground truth that may contain multiple tables separated by commas
    
    Args:
        ground_truth: String that may contain one or more tables (e.g., "well, well_reservoir")
    
    Returns:
        Set of individual table names
    """
    # Split by comma and clean up whitespace
    tables = [table.strip().lower() for table in ground_truth.split(',')]
    return set(tables)


def calculate_mrr_multi_table(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for RAG validation with multi-table support
    
    For multi-table ground truth:
    - Rank is determined by the position where ALL required tables have been seen
    - Example: If ground truth is "table1, table2" and predictions are ["table1", "other", "table2"],
      the rank is 3 (position where both tables have been retrieved)
    
    Args:
        predictions: List of ranked prediction lists for each query
        ground_truth: List of correct answers for each query (may contain comma-separated tables)
    
    Returns:
        MRR score (0-1, higher is better)
    """
    reciprocal_ranks = []
    
    for prediction_list, correct_answer in zip(predictions, ground_truth):
        required_tables = parse_multi_table_ground_truth(correct_answer)
        
        # Track which required tables we've found
        found_tables = set()
        rank = None
        
        # Go through predictions in order
        for i, prediction in enumerate(prediction_list):
            pred_table = prediction.strip().lower()
            
            # Check if this prediction matches any required table
            if pred_table in required_tables:
                found_tables.add(pred_table)
                
                # If we've found all required tables, record the rank
                if found_tables == required_tables:
                    rank = i + 1  # rank starts from 1
                    break
        
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def calculate_accuracy_multi_table(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """
    Calculate accuracy - whether ALL ground truth tables were retrieved at any rank
    
    Args:
        predictions: List of ranked prediction lists for each query
        ground_truth: List of correct answers for each query (may contain comma-separated tables)
        
    Returns:
        Accuracy score (0-1, higher is better)
    """
    correct_predictions = 0
    
    for prediction_list, correct_answer in zip(predictions, ground_truth):
        required_tables = parse_multi_table_ground_truth(correct_answer)
        
        # Get all predicted tables (lowercased)
        predicted_tables = {pred.strip().lower() for pred in prediction_list}
        
        # Check if all required tables are found
        if required_tables.issubset(predicted_tables):
            correct_predictions += 1
    
    return correct_predictions / len(ground_truth) if ground_truth else 0.0


def calculate_partial_accuracy_multi_table(predictions: List[List[str]], ground_truth: List[str]) -> Tuple[float, float]:
    """
    Calculate partial accuracy metrics for multi-table queries
    
    Returns:
        Tuple of (partial_accuracy, average_recall)
        - partial_accuracy: Fraction of queries where at least one required table was found
        - average_recall: Average fraction of required tables found per query
    """
    partial_correct = 0
    recall_scores = []
    
    for prediction_list, correct_answer in zip(predictions, ground_truth):
        required_tables = parse_multi_table_ground_truth(correct_answer)
        predicted_tables = {pred.strip().lower() for pred in prediction_list}
        
        # Calculate how many required tables were found
        found_tables = required_tables.intersection(predicted_tables)
        
        if found_tables:
            partial_correct += 1
        
        # Calculate recall for this query
        recall = len(found_tables) / len(required_tables) if required_tables else 0.0
        recall_scores.append(recall)
    
    partial_accuracy = partial_correct / len(ground_truth) if ground_truth else 0.0
    average_recall = np.mean(recall_scores)
    
    return partial_accuracy, average_recall


# Backward compatibility - keep old function names that redirect to new ones
def calculate_mrr(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """Backward compatible wrapper for calculate_mrr_multi_table"""
    return calculate_mrr_multi_table(predictions, ground_truth)


def calculate_accuracy(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """Backward compatible wrapper for calculate_accuracy_multi_table"""
    return calculate_accuracy_multi_table(predictions, ground_truth)
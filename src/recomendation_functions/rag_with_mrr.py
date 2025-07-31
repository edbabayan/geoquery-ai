from typing import Any, Dict, List
from tqdm import tqdm

from src.recomendation_functions.mean_reciprocal_rank import calculate_mrr


def evaluate_rag_with_mrr(rag_system, test_queries: List[str],
                          ground_truth: List[str], top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate RAG system using MRR metric

    Args:
        rag_system: Your RAG system with an invoke method
        test_queries: List of test questions
        ground_truth: List of expected answers (table names)
        top_k: Number of top predictions to consider

    Returns:
        Dictionary with MRR score and detailed results
    """
    all_predictions = []

    for query in tqdm(test_queries, desc="Processing queries", unit="query"):
        # Get top-k predictions from your RAG system using invoke
        retrieved_docs = rag_system.invoke(query)[:top_k]
        # Extract table names from retrieved documents
        predictions = [doc.metadata.get('table_name', '') for doc in retrieved_docs]
        all_predictions.append(predictions)

    mrr_score = calculate_mrr(all_predictions, ground_truth)

    return {
        'mrr_score': mrr_score,
        'predictions': all_predictions,
        'ground_truth': ground_truth,
        'num_queries': len(test_queries)
    }


if __name__ == '__main__':
    # Example usage:
    rag_system = None  # Replace with your RAG system instance

    test_queries = ["What is the capital of France?", "Who wrote 1984?"]
    ground_truth = ["Paris", "George Orwell"]
    results = evaluate_rag_with_mrr(rag_system, test_queries, ground_truth)
    print(f"MRR Score: {results['mrr_score']:.3f}")
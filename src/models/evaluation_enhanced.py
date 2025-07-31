import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_retrieval_models import retriever_enhanced, ground_truth_renamed
from retrieval_models import retriever_default, retriever_artem_v1, retriever_artem_v2, retriever_artem_v3
from advanced_retrieval_models import retriever_advanced_reranked, retriever_advanced_no_rerank
from src.recomendation_functions.rag_with_mrr import evaluate_rag_with_mrr
import pandas as pd
import time

# Use actual queries from the ground truth data
test_data = ground_truth_renamed.to_dict('records')
test_queries = [item['question'] for item in test_data]
expected_tables = [item['tables'].split('.')[-1] if '.' in item['tables'] else item['tables'] 
                   for item in test_data]

# Models to evaluate
models = [
    ("Default Retriever", retriever_default),
    ("Artem V1 Retriever", retriever_artem_v1),
    ("Artem V2 Retriever", retriever_artem_v2),
    ("Artem V3 Retriever", retriever_artem_v3),
    ("Enhanced Retriever", retriever_enhanced),
    ("Advanced Retriever (No Rerank)", retriever_advanced_no_rerank),
    ("Advanced Retriever (With Rerank)", retriever_advanced_reranked)
]

print(f"Evaluating {len(models)} retrieval models on {len(test_queries)} queries...\n")

# Store results
results = []

for model_name, model in models:
    print(f"Evaluating model: {model_name}")
    
    start_time = time.time()
    evaluation = evaluate_rag_with_mrr(model, test_queries=test_queries, ground_truth=expected_tables)
    end_time = time.time()
    
    eval_time = end_time - start_time
    mrr_score = evaluation['mrr_score']
    
    results.append({
        'Model Name': model_name,
        'MRR Score': mrr_score,
        'Evaluation Time (seconds)': eval_time
    })
    
    print(f"MRR Score: {mrr_score:.3f}")
    print(f"Evaluation Time: {eval_time:.2f} seconds\n")

# Save to Excel
df = pd.DataFrame(results)
df.to_excel('retriever_evaluation_results.xlsx', index=False)

print("Results saved to retriever_evaluation_results.xlsx")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_retrieval_models import retriever_enhanced, ground_truth_renamed
from retrieval_models import retriever_default, retriever_artem_v1, retriever_artem_v2, retriever_artem_v3
from advanced_retrieval_models import retriever_advanced_reranked, retriever_advanced_no_rerank
from query_decomposition import default_composition
from src.recomendation_functions.rag_with_mrr import evaluate_rag_with_mrr
import pandas as pd
import time
from tqdm import tqdm

# Load existing results to check for evaluated models
RESULTS_FILE = 'retriever_evaluation_results.xlsx'

def load_existing_results():
    """Load existing evaluation results if file exists"""
    try:
        return pd.read_excel(RESULTS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=['Model Name', 'MRR Score', 'Evaluation Time (seconds)'])


def model_already_evaluated(model_name, existing_df):
    """Check if model has already been evaluated"""
    return model_name in existing_df['Model Name'].values

# Use actual queries from the ground truth data
test_data = ground_truth_renamed.to_dict('records')
test_queries = [item['question'] for item in test_data]
expected_tables = [item['tables'].split('.')[-1] if '.' in item['tables'] else item['tables'] 
                   for item in test_data]

# Create retriever with query decomposition
retriever_default_with_decomposition = default_composition(retriever_default)

# Models to evaluate
models = [
    ("Default Retriever", retriever_default),
    ("Default Retriever with Decomposition", retriever_default_with_decomposition),
    ("Artem V1 Retriever", retriever_artem_v1),
    ("Artem V2 Retriever", retriever_artem_v2),
    ("Artem V3 Retriever", retriever_artem_v3),
    ("Enhanced Retriever", retriever_enhanced),
    ("Advanced Retriever (No Rerank)", retriever_advanced_no_rerank),
    ("Advanced Retriever (With Rerank)", retriever_advanced_reranked)
]

# Load existing results
existing_results = load_existing_results()
print(f"Loaded {len(existing_results)} existing evaluation results")

print(f"Evaluating {len(models)} retrieval models on {len(test_queries)} queries...\n")

# Store results (start with existing results)
results = existing_results.to_dict('records') if not existing_results.empty else []

for model_name, model in models:
    # Check if model already exists in Excel
    if model_already_evaluated(model_name, existing_results):
        print(f"Skipping {model_name} - already evaluated")
        continue
        
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

# Save to Excel (combining existing and new results)
df = pd.DataFrame(results)
df.to_excel(RESULTS_FILE, index=False)

print(f"Results saved to {RESULTS_FILE}")
print(f"Total models evaluated: {len(df)}")
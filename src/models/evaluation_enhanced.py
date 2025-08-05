import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_retrieval_models import retriever_enhanced, ground_truth_renamed
from retrieval_models import retriever_default, retriever_artem_v1, retriever_artem_v2, retriever_artem_v3
from advanced_retrieval_models import retriever_advanced_reranked, retriever_advanced_no_rerank
# from advanced_retrieval_models_v2 import retriever_advanced_v2_reranked, retriever_advanced_v2_no_rerank
from column_enhanced_retrieval_models import retriever_columns_basic, retriever_columns_bm25_emphasis, retriever_columns_vector_emphasis, retriever_columns_reranked
from default_with_columns_retrieval import retriever_default_with_columns_bm25, retriever_default_with_columns_balanced, retriever_default_with_columns_vector
from advanced_with_columns_retrieval import retriever_advanced_with_columns_reranked, retriever_advanced_with_columns_no_rerank
from query_decomposition import default_composition
from default_with_reranker import retriever_default_with_reranker
from knowledge_graph_retrieval_models import retriever_kg_basic, retriever_kg_gpt4, retriever_kg_enhanced
from questions_enhanced_retrieval import (
    retriever_default_questions_balanced,
    retriever_default_questions_bm25,
    retriever_default_questions_vector,
    retriever_default_columns_questions_balanced,
    retriever_default_columns_questions_bm25,
    retriever_default_columns_questions_vector
)
from graph_retrieval_variations import (
    retriever_kg_default,
    retriever_kg_artem_v1,
    retriever_kg_artem_v2,
    retriever_kg_artem_v3,
    retriever_kg_artem_v4,
    retriever_kg_default_bm25,
    retriever_kg_default_vector
)
from recomendation_functions.rag_with_mrr import evaluate_rag_with_mrr
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
        return pd.DataFrame(columns=['Model Name', 'MRR Score', 'Accuracy Score', 'Evaluation Time (seconds)'])


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
    ("Default Retriever with Reranker", retriever_default_with_reranker),
    ("Artem V1 Retriever", retriever_artem_v1),
    ("Artem V2 Retriever", retriever_artem_v2),
    ("Artem V3 Retriever", retriever_artem_v3),
    ("Enhanced Retriever", retriever_enhanced),
    ("Advanced Retriever (No Rerank)", retriever_advanced_no_rerank),
    ("Advanced Retriever (With Rerank)", retriever_advanced_reranked),
    # Column-enhanced models
    ("Column Enhanced Basic", retriever_columns_basic),
    ("Column Enhanced BM25 Emphasis", retriever_columns_bm25_emphasis),
    ("Column Enhanced Vector Emphasis", retriever_columns_vector_emphasis),
    ("Column Enhanced with Reranking", retriever_columns_reranked),
    # Default with columns models
    ("Default with Columns (BM25 0.7-0.3)", retriever_default_with_columns_bm25),
    ("Default with Columns (Balanced 0.5-0.5)", retriever_default_with_columns_balanced),
    ("Default with Columns (Vector 0.3-0.7)", retriever_default_with_columns_vector),
    # Advanced with columns models
    ("Advanced with Columns (No Rerank)", retriever_advanced_with_columns_no_rerank),
    ("Advanced with Columns (With Rerank)", retriever_advanced_with_columns_reranked),
    # Knowledge Graph models
    ("Knowledge Graph Basic", retriever_kg_basic),
    ("Knowledge Graph GPT-4", retriever_kg_gpt4),
    ("Knowledge Graph Enhanced", retriever_kg_enhanced),
    # Default + Questions models
    ("Default + Questions (Balanced 0.5-0.5)", retriever_default_questions_balanced),
    ("Default + Questions (BM25 0.7-0.3)", retriever_default_questions_bm25),
    ("Default + Questions (Vector 0.3-0.7)", retriever_default_questions_vector),
    # Default + Columns + Questions models
    ("Default + Columns + Questions (Balanced 0.5-0.5)", retriever_default_columns_questions_balanced),
    ("Default + Columns + Questions (BM25 0.7-0.3)", retriever_default_columns_questions_bm25),
    ("Default + Columns + Questions (Vector 0.3-0.7)", retriever_default_columns_questions_vector),
    # Graph-based variations matching original retrievers
    ("KG Default (Balanced 0.5-0.5)", retriever_kg_default),
    ("KG Artem V1 Full (Balanced 0.5-0.5)", retriever_kg_artem_v1),
    ("KG Artem V2 Purpose+Insights (Balanced 0.5-0.5)", retriever_kg_artem_v2),
    ("KG Artem V3 Full (Balanced 0.5-0.5)", retriever_kg_artem_v3),
    ("KG Artem V4 Minimal (Balanced 0.5-0.5)", retriever_kg_artem_v4),
    ("KG Default (BM25 0.7-0.3)", retriever_kg_default_bm25),
    ("KG Default (Vector 0.3-0.7)", retriever_kg_default_vector),
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
    accuracy_score = evaluation['accuracy_score']
    
    results.append({
        'Model Name': model_name,
        'MRR Score': mrr_score,
        'Accuracy Score': accuracy_score,
        'Evaluation Time (seconds)': eval_time
    })
    
    print(f"MRR Score: {mrr_score:.3f}")
    print(f"Accuracy Score: {accuracy_score:.3f}")
    print(f"Evaluation Time: {eval_time:.2f} seconds\n")

# Save to Excel (combining existing and new results)
df = pd.DataFrame(results)
df.to_excel(RESULTS_FILE, index=False)

print(f"Results saved to {RESULTS_FILE}")
print(f"Total models evaluated: {len(df)}")
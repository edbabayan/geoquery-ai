"""
Evaluate weight combinations both with and without reranker to see true impact
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import pandas as pd
import time
from dotenv import load_dotenv
from config import CFG
from recomendation_functions.rag_with_mrr import evaluate_rag_with_mrr
from advanced_retrieval_models import (
    documents_advanced, 
    ground_truth_renamed,
)

# Load environment variables
load_dotenv(CFG.env_file)

# Initialize embeddings
embed = OpenAIEmbeddings(model="text-embedding-3-large")

# Create base components
print("Setting up components...")
vectorstore = FAISS.from_documents(documents_advanced, embed)

# Initialize cross-encoder for reranking
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=5)

# Prepare test data
test_data = ground_truth_renamed.to_dict('records')
test_queries = [item['question'] for item in test_data]
expected_tables = [item['tables'].split('.')[-1] if '.' in item['tables'] else item['tables'] 
                   for item in test_data]

# Define weight combinations
weight_combinations = [
    (0.0, 1.0),   # Pure vector
    (0.2, 0.8),
    (0.4, 0.6),
    (0.5, 0.5),   # Balanced
    (0.6, 0.4),   # Current setting
    (0.8, 0.2),
    (1.0, 0.0),   # Pure BM25
]

# Store results
results_no_rerank = []
results_with_rerank = []

print(f"\nEvaluating {len(weight_combinations)} weight combinations on {len(test_queries)} queries...")
print("="*60)

for idx, (bm25_weight, vector_weight) in enumerate(weight_combinations):
    print(f"\n[{idx+1}/{len(weight_combinations)}] Testing weights - BM25: {bm25_weight:.1f}, Vector: {vector_weight:.1f}")
    
    # Create fresh retrievers for each iteration
    bm25_retriever = BM25Retriever.from_documents(documents_advanced)
    bm25_retriever.k = 5  # For no-rerank version
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Test 1: Without reranker (k=5)
    print("  Evaluating WITHOUT reranker...")
    ensemble_no_rerank = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[bm25_weight, vector_weight]
    )
    
    start_time = time.time()
    eval_no_rerank = evaluate_rag_with_mrr(
        ensemble_no_rerank, 
        test_queries=test_queries, 
        ground_truth=expected_tables
    )
    time_no_rerank = time.time() - start_time
    
    # Test 2: With reranker (k=10 -> 5)
    print("  Evaluating WITH reranker...")
    # Create new retrievers with k=10 for reranking
    bm25_retriever_k10 = BM25Retriever.from_documents(documents_advanced)
    bm25_retriever_k10.k = 10
    
    vector_retriever_k10 = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    ensemble_with_rerank = EnsembleRetriever(
        retrievers=[bm25_retriever_k10, vector_retriever_k10], 
        weights=[bm25_weight, vector_weight]
    )
    
    retriever_with_reranker = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_with_rerank
    )
    
    start_time = time.time()
    eval_with_rerank = evaluate_rag_with_mrr(
        retriever_with_reranker, 
        test_queries=test_queries, 
        ground_truth=expected_tables
    )
    time_with_rerank = time.time() - start_time
    
    # Store results
    results_no_rerank.append({
        'BM25_Weight': bm25_weight,
        'Vector_Weight': vector_weight,
        'MRR_Score': eval_no_rerank['mrr_score'],
        'Time_seconds': time_no_rerank
    })
    
    results_with_rerank.append({
        'BM25_Weight': bm25_weight,
        'Vector_Weight': vector_weight,
        'MRR_Score': eval_with_rerank['mrr_score'],
        'Time_seconds': time_with_rerank
    })
    
    print(f"  Results:")
    print(f"    Without reranker: MRR = {eval_no_rerank['mrr_score']:.4f}")
    print(f"    With reranker:    MRR = {eval_with_rerank['mrr_score']:.4f}")
    print(f"    Improvement:      {((eval_with_rerank['mrr_score'] - eval_no_rerank['mrr_score']) / eval_no_rerank['mrr_score'] * 100):+.1f}%")

# Create DataFrames
df_no_rerank = pd.DataFrame(results_no_rerank)
df_with_rerank = pd.DataFrame(results_with_rerank)

# Create comparison DataFrame
df_comparison = df_no_rerank.merge(
    df_with_rerank, 
    on=['BM25_Weight', 'Vector_Weight'], 
    suffixes=('_no_rerank', '_with_rerank')
)
df_comparison['MRR_Improvement'] = (
    (df_comparison['MRR_Score_with_rerank'] - df_comparison['MRR_Score_no_rerank']) / 
    df_comparison['MRR_Score_no_rerank'] * 100
)

# Sort by best performance without reranker
df_comparison = df_comparison.sort_values('MRR_Score_no_rerank', ascending=False)

# Save results
output_file = 'weight_evaluation_comparison.xlsx'
with pd.ExcelWriter(output_file) as writer:
    df_comparison.to_excel(writer, sheet_name='Comparison', index=False)
    df_no_rerank.to_excel(writer, sheet_name='Without_Reranker', index=False)
    df_with_rerank.to_excel(writer, sheet_name='With_Reranker', index=False)

print("\n" + "="*60)
print("SUMMARY - Impact of Weights on Retrieval Performance")
print("="*60)

print("\n1. WITHOUT RERANKER (showing true impact of weights):")
print(df_no_rerank.sort_values('MRR_Score', ascending=False).to_string(index=False))

print("\n2. WITH RERANKER (showing normalized results):")
print(df_with_rerank.sort_values('MRR_Score', ascending=False).to_string(index=False))

print("\n3. KEY INSIGHTS:")
# Calculate variance to show impact
var_no_rerank = df_no_rerank['MRR_Score'].var()
var_with_rerank = df_with_rerank['MRR_Score'].var()
print(f"   - MRR variance without reranker: {var_no_rerank:.6f}")
print(f"   - MRR variance with reranker:    {var_with_rerank:.6f}")
print(f"   - Variance reduction:             {((var_no_rerank - var_with_rerank) / var_no_rerank * 100):.1f}%")

# Find best weights for each scenario
best_no_rerank = df_no_rerank.loc[df_no_rerank['MRR_Score'].idxmax()]
best_with_rerank = df_with_rerank.loc[df_with_rerank['MRR_Score'].idxmax()]

print(f"\n   Best weights WITHOUT reranker: BM25={best_no_rerank['BM25_Weight']:.1f}, Vector={best_no_rerank['Vector_Weight']:.1f} (MRR={best_no_rerank['MRR_Score']:.4f})")
print(f"   Best weights WITH reranker:    BM25={best_with_rerank['BM25_Weight']:.1f}, Vector={best_with_rerank['Vector_Weight']:.1f} (MRR={best_with_rerank['MRR_Score']:.4f})")

print(f"\nResults saved to {output_file}")
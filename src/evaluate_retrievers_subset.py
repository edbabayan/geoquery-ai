#!/usr/bin/env python3
"""Quick evaluation of retrievers on a subset of queries"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from enhanced_retrieval import EnhancedTableRetriever
from config import CFG
from default_descriptions import table_description_list_default
from recomendation_functions.mean_reciprocal_rank import calculate_mrr
import json

# Load environment variables
load_dotenv(CFG.env_file)

def evaluate_retriever_mrr(retriever, queries_df, retriever_name, top_k=5):
    """Evaluate a retriever using MRR"""
    print(f"\nEvaluating {retriever_name}...")
    
    all_predictions = []
    all_ground_truth = []
    
    for idx, row in queries_df.iterrows():
        query = row['NL_QUERY']
        ground_truth_tables = row['TABLES'].split(',') if pd.notna(row['TABLES']) else []
        ground_truth_tables = [t.strip() for t in ground_truth_tables]
        
        try:
            if hasattr(retriever, 'retrieve'):
                docs = retriever.retrieve(query, top_k=top_k)
            else:
                docs = retriever.invoke(query)[:top_k]
            
            retrieved_tables = [doc.metadata.get('table_name', '') for doc in docs]
        except Exception as e:
            print(f"  Error: {str(e)}")
            retrieved_tables = []
        
        # For multiple ground truth, take the first one
        if ground_truth_tables:
            all_predictions.append(retrieved_tables)
            all_ground_truth.append(ground_truth_tables[0])
    
    # Calculate MRR
    mrr = calculate_mrr(all_predictions, all_ground_truth)
    
    return {
        'retriever_name': retriever_name,
        'mrr': mrr,
        'queries_evaluated': len(all_predictions)
    }

def prepare_documents():
    """Prepare documents from default descriptions"""
    documents = []
    
    for table_info in table_description_list_default:
        full_table_name = table_info["table_name"]
        table_description = table_info["table_description"]
        short_table_name = full_table_name.split(".")[-1]
        
        description = f"Table name is {short_table_name}. Description: {table_description}"
        
        doc = Document(
            page_content=description,
            metadata={"table_name": short_table_name}
        )
        documents.append(doc)
    
    return documents

def main():
    # Load ground truth data (first 20 queries only)
    print("Loading ground truth data...")
    ground_truth = pd.read_csv(CFG.golden_dataset)[['NL_QUERY', "TABLES"]].head(20)
    print(f"Using {len(ground_truth)} queries for quick evaluation")
    
    # Initialize
    embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
    documents = prepare_documents()
    
    results = []
    
    # 1. BM25 only
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    results.append(evaluate_retriever_mrr(bm25_retriever, ground_truth, "BM25 Only"))
    
    # 2. Vector only
    vectorstore = Chroma.from_documents(documents, embed_model, collection_name="quick_eval")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results.append(evaluate_retriever_mrr(vector_retriever, ground_truth, "Vector Only"))
    
    # 3. Ensemble (0.5, 0.5)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    results.append(evaluate_retriever_mrr(ensemble, ground_truth, "Ensemble (0.5, 0.5)"))
    
    # 4. Enhanced retriever (basic)
    enhanced_retriever = EnhancedTableRetriever(documents)
    
    class BasicEnhanced:
        def __init__(self, retriever):
            self.retriever = retriever
        
        def retrieve(self, query, top_k=5):
            return self.retriever.retrieve(query, top_k, use_query_expansion=False, use_reranking=False)
    
    basic_enhanced = BasicEnhanced(enhanced_retriever)
    results.append(evaluate_retriever_mrr(basic_enhanced, ground_truth, "Enhanced (Basic)"))
    
    # 5. Enhanced retriever (full)
    class FullEnhanced:
        def __init__(self, retriever):
            self.retriever = retriever
        
        def retrieve(self, query, top_k=5):
            return self.retriever.retrieve(query, top_k, use_query_expansion=True, use_reranking=True)
    
    full_enhanced = FullEnhanced(enhanced_retriever)
    results.append(evaluate_retriever_mrr(full_enhanced, ground_truth, "Enhanced (Full)"))
    
    # Print results
    print("\n" + "="*60)
    print("QUICK EVALUATION RESULTS (MRR)")
    print("="*60)
    
    # Sort by MRR
    results.sort(key=lambda x: x['mrr'], reverse=True)
    
    print(f"\n{'Retriever':<30} {'MRR':>10}")
    print("-" * 42)
    
    for result in results:
        print(f"{result['retriever_name']:<30} {result['mrr']:>10.4f}")
    
    print(f"\nBest performer: {results[0]['retriever_name']} (MRR: {results[0]['mrr']:.4f})")

if __name__ == "__main__":
    main()
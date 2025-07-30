#!/usr/bin/env python3
"""Evaluate retrievers separately with async support for better performance"""

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
from typing import List, Dict, Tuple
import time
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio

# Allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv(CFG.env_file)

class AsyncRetrieverWrapper:
    """Async wrapper for retrievers"""
    def __init__(self, retriever, is_enhanced=False):
        self.retriever = retriever
        self.is_enhanced = is_enhanced
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def retrieve_async(self, query, top_k=5):
        """Async retrieval using thread pool"""
        loop = asyncio.get_event_loop()
        if self.is_enhanced:
            return await loop.run_in_executor(
                self.executor,
                self.retriever.retrieve,
                query,
                top_k,
                True,  # use_query_expansion
                True,  # use_reranking
                0.3    # alpha (BM25 weight)
            )
        else:
            docs = await loop.run_in_executor(
                self.executor,
                self.retriever.invoke,
                query
            )
            return docs[:top_k]

async def evaluate_batch_async(retriever_wrapper, queries_batch, top_k=5):
    """Evaluate a batch of queries asynchronously"""
    tasks = []
    for idx, query, ground_truth in queries_batch:
        task = retriever_wrapper.retrieve_async(query, top_k)
        tasks.append((idx, query, ground_truth, task))
    
    results = []
    for idx, query, ground_truth, task in tasks:
        try:
            docs = await task
            retrieved_tables = [doc.metadata.get('table_name', '') for doc in docs]
        except Exception as e:
            print(f"  Error on query {idx}: {str(e)}")
            retrieved_tables = []
        results.append((idx, query, ground_truth, retrieved_tables))
    
    return results

async def evaluate_retriever_async(retriever, queries_df: pd.DataFrame, retriever_name: str, 
                                 top_k: int = 5, batch_size: int = 5, is_enhanced: bool = False) -> Dict:
    """
    Evaluate a retriever asynchronously with batching
    """
    print(f"\nEvaluating {retriever_name} (async mode)...")
    start_time = time.time()
    
    # Create async wrapper
    retriever_wrapper = AsyncRetrieverWrapper(retriever, is_enhanced)
    
    all_predictions = []
    all_ground_truth = []
    detailed_results = []
    
    # Process in batches
    for batch_start in range(0, len(queries_df), batch_size):
        batch_end = min(batch_start + batch_size, len(queries_df))
        batch_queries = []
        
        for idx in range(batch_start, batch_end):
            row = queries_df.iloc[idx]
            ground_truth_tables = row['TABLES'].split(',') if pd.notna(row['TABLES']) else []
            ground_truth_tables = [t.strip() for t in ground_truth_tables]
            batch_queries.append((idx, row['NL_QUERY'], ground_truth_tables))
        
        # Process batch asynchronously
        batch_results = await evaluate_batch_async(retriever_wrapper, batch_queries, top_k)
        
        # Process results
        for idx, query, ground_truth_tables, retrieved_tables in batch_results:
            if ground_truth_tables:
                best_mrr = 0.0
                best_match = None
                for gt_table in ground_truth_tables:
                    predictions = [retrieved_tables]
                    ground_truth = [gt_table]
                    mrr = calculate_mrr(predictions, ground_truth)
                    if mrr > best_mrr:
                        best_mrr = mrr
                        best_match = gt_table
                
                all_predictions.append(retrieved_tables)
                all_ground_truth.append(best_match if best_match else ground_truth_tables[0])
                
                detailed_results.append({
                    'query_idx': idx,
                    'query': query,
                    'ground_truth': ground_truth_tables,
                    'retrieved': retrieved_tables,
                    'best_match': best_match,
                    'mrr': best_mrr
                })
        
        # Progress update
        if batch_end % 10 == 0 or batch_end == len(queries_df):
            elapsed = time.time() - start_time
            print(f"  Processed {batch_end}/{len(queries_df)} queries... ({elapsed:.1f}s elapsed)")
    
    # Calculate overall metrics
    overall_mrr = calculate_mrr(all_predictions, all_ground_truth)
    hits_at_1 = sum(1 for r in detailed_results if r['mrr'] == 1.0) / len(detailed_results)
    hits_at_5 = sum(1 for r in detailed_results if r['mrr'] > 0) / len(detailed_results)
    
    elapsed_time = time.time() - start_time
    print(f"  Completed in {elapsed_time:.1f} seconds")
    
    return {
        'retriever_name': retriever_name,
        'mrr': overall_mrr,
        'hits_at_1': hits_at_1,
        'hits_at_5': hits_at_5,
        'total_queries': len(detailed_results),
        'failed_queries': len([r for r in detailed_results if r['mrr'] == 0]),
        'time_seconds': elapsed_time,
        'details': detailed_results
    }

def evaluate_retriever_mrr(retriever, queries_df: pd.DataFrame, retriever_name: str, top_k: int = 5) -> Dict:
    """
    Evaluate a retriever on the ground truth dataset using MRR
    
    Args:
        retriever: The retriever to evaluate
        queries_df: DataFrame with NL_QUERY and TABLES columns
        retriever_name: Name of the retriever for reporting
        top_k: Number of documents to retrieve
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"\nEvaluating {retriever_name}...")
    start_time = time.time()
    
    all_predictions = []
    all_ground_truth = []
    detailed_results = []
    
    for idx, row in queries_df.iterrows():
        query = row['NL_QUERY']
        ground_truth_tables = row['TABLES'].split(',') if pd.notna(row['TABLES']) else []
        
        # Clean ground truth tables
        ground_truth_tables = [t.strip() for t in ground_truth_tables]
        
        # Retrieve documents
        try:
            if hasattr(retriever, 'retrieve'):
                # Enhanced retriever
                docs = retriever.retrieve(query, top_k=top_k)
            else:
                # Standard retriever
                docs = retriever.invoke(query)[:top_k]
            
            retrieved_tables = [doc.metadata.get('table_name', '') for doc in docs]
        except Exception as e:
            print(f"  Error on query {idx}: {str(e)}")
            retrieved_tables = []
        
        # For MRR calculation with multiple ground truth tables
        # We'll calculate MRR for each ground truth table and take the best
        if ground_truth_tables:
            best_mrr = 0.0
            best_match = None
            for gt_table in ground_truth_tables:
                # Create predictions and ground truth lists for MRR calculation
                predictions = [retrieved_tables]
                ground_truth = [gt_table]
                mrr = calculate_mrr(predictions, ground_truth)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_match = gt_table
            
            all_predictions.append(retrieved_tables)
            all_ground_truth.append(best_match if best_match else ground_truth_tables[0])
            
            # Store detailed results
            detailed_results.append({
                'query_idx': idx,
                'query': query,
                'ground_truth': ground_truth_tables,
                'retrieved': retrieved_tables,
                'best_match': best_match,
                'mrr': best_mrr
            })
        
        # Progress indicator every 5 queries for better tracking
        if (idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {idx + 1}/{len(queries_df)} queries... ({elapsed:.1f}s elapsed)")
    
    # Calculate overall MRR
    overall_mrr = calculate_mrr(all_predictions, all_ground_truth)
    
    # Calculate additional metrics
    hits_at_1 = sum(1 for r in detailed_results if r['mrr'] == 1.0) / len(detailed_results)
    hits_at_5 = sum(1 for r in detailed_results if r['mrr'] > 0) / len(detailed_results)
    
    elapsed_time = time.time() - start_time
    print(f"  Completed in {elapsed_time:.1f} seconds")
    
    return {
        'retriever_name': retriever_name,
        'mrr': overall_mrr,
        'hits_at_1': hits_at_1,
        'hits_at_5': hits_at_5,
        'total_queries': len(detailed_results),
        'failed_queries': len([r for r in detailed_results if r['mrr'] == 0]),
        'time_seconds': elapsed_time,
        'details': detailed_results
    }

def create_ensemble_retriever_from_documents(documents, embed_model, num_docs_retrieved=5, weights=[0.5, 0.5], name=""):
    """Create an ensemble retriever matching the function from main.py"""
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = num_docs_retrieved
    
    vectorstore = Chroma.from_documents(documents, embed_model, collection_name=f"test_{name}")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": num_docs_retrieved})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=weights
    )
    
    return ensemble_retriever

def prepare_documents_artem_v1(df, result_full):
    """Prepare documents for artem_v1: full descriptions with questions"""
    documents = []
    for i in df.index:
        table_name = df.loc[i, 'name']
        table_name_short = table_name.split('.')[-1]
        
        # Get questions for this table
        questions = result_full.get(table_name_short, [])
        questions_text = ' '.join(questions[:5]) if questions else ""
        
        description = (
            f"Table name is {table_name}. "
            f"Industry terms are {df.loc[i, 'industry_terms']}. "
            f"Data granularity is {df.loc[i, 'data_granularity']}. "
            f"Main business purpose is {df.loc[i, 'main_business_purpose']}. "
            f"Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. "
            f"Unique insights are {df.loc[i, 'unique_insights']}. "
            f"Question examples are: {questions_text}"
        )
        
        doc = Document(
            page_content=description,
            metadata={"table_name": table_name_short}
        )
        documents.append(doc)
    
    return documents

def prepare_documents_artem_v2(df, result_full):
    """Prepare documents for artem_v2: simplified descriptions with questions"""
    documents = []
    for i in df.index:
        table_name = df.loc[i, 'name']
        table_name_short = table_name.split('.')[-1]
        
        # Get questions for this table
        questions = result_full.get(table_name_short, [])
        questions_text = ' '.join(questions[:5]) if questions else ""
        
        description = (
            f"Table name is {table_name}. "
            f"Main business purpose is {df.loc[i, 'main_business_purpose']}. "
            f"Unique insights are {df.loc[i, 'unique_insights']}. "
            f"Question examples are: {questions_text}"
        )
        
        doc = Document(
            page_content=description,
            metadata={"table_name": table_name_short}
        )
        documents.append(doc)
    
    return documents

def prepare_documents_artem_v3(df):
    """Prepare documents for artem_v3: descriptions without questions"""
    documents = []
    for i in df.index:
        table_name = df.loc[i, 'name']
        table_name_short = table_name.split('.')[-1]
        
        description = (
            f"Table name is {table_name}. "
            f"Industry terms are {df.loc[i, 'industry_terms']}. "
            f"Data granularity is {df.loc[i, 'data_granularity']}. "
            f"Main business purpose is {df.loc[i, 'main_business_purpose']}. "
            f"Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. "
            f"Unique insights are {df.loc[i, 'unique_insights']}."
        )
        
        doc = Document(
            page_content=description,
            metadata={"table_name": table_name_short}
        )
        documents.append(doc)
    
    return documents

def prepare_documents_default():
    """Prepare documents for default retriever: from table_description_list_default"""
    documents = []
    for table_info in table_description_list_default:
        table_name_short = table_info['table_name'].split('.')[-1]
        
        description = f"Table name is {table_name_short}. Description: {table_info['table_description']}"
        
        doc = Document(
            page_content=description,
            metadata={"table_name": table_name_short}
        )
        documents.append(doc)
    
    return documents

def prepare_documents_enhanced():
    """Prepare documents for enhanced retriever with questions"""
    documents = []
    
    # Load previously generated questions if available
    try:
        with open('src/result_full.json', 'r') as f:
            result_full = json.load(f)
    except:
        result_full = {}
    
    for table_info in table_description_list_default:
        full_table_name = table_info["table_name"]
        table_description = table_info["table_description"]
        short_table_name = full_table_name.split(".")[-1]
        
        # Get example questions
        questions = result_full.get(short_table_name, [])
        questions_text = ' '.join(questions[:5]) if questions else ""
        
        # Create description
        description = (
            f"Table name is {short_table_name}. "
            f"Description: {table_description} "
            f"Example questions: {questions_text}"
        )
        
        doc = Document(
            page_content=description,
            metadata={"table_name": short_table_name}
        )
        documents.append(doc)
    
    return documents

async def main_async():
    """Run evaluation of selected retrievers with async support"""
    
    if len(sys.argv) > 1:
        retriever_to_run = sys.argv[1]
    else:
        retriever_to_run = "all"
    
    # Load ground truth data
    print("Loading ground truth data...")
    ground_truth = pd.read_csv(CFG.golden_dataset)[['NL_QUERY', "TABLES"]]
    print(f"Loaded {len(ground_truth)} queries for evaluation")
    
    # Load table descriptions
    print("\nLoading table descriptions...")
    df = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/table_description 1.xlsx")
    
    # Load generated questions
    try:
        with open('src/result_full.json', 'r') as f:
            result_full = json.load(f)
        print("Loaded generated questions")
    except:
        result_full = {}
        print("Warning: Could not load generated questions")
    
    # Initialize embeddings
    embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Load existing results if any
    try:
        with open('selected_retrievers_evaluation_full.json', 'r') as f:
            all_results = json.load(f)
        completed_retrievers = {r['retriever_name'] for r in all_results}
        print(f"\nFound existing results for {len(completed_retrievers)} retrievers")
    except:
        all_results = []
        completed_retrievers = set()
    
    print("\n" + "="*80)
    print("EVALUATING SELECTED RETRIEVERS")
    print("="*80)
    
    # Define retrievers to evaluate
    retrievers_config = [
        ("artem_v1", "artem_v1 (Full descriptions with questions)", prepare_documents_artem_v1),
        ("artem_v2", "artem_v2 (Simplified descriptions with questions)", prepare_documents_artem_v2),
        ("artem_v3", "artem_v3 (Descriptions without questions)", prepare_documents_artem_v3),
        ("default", "default (Default table descriptions)", prepare_documents_default),
        ("enhanced", "Enhanced (Full, BM25=0.3)", prepare_documents_enhanced)
    ]
    
    for short_name, full_name, doc_prep_func in retrievers_config:
        # Skip if already evaluated or not requested
        if full_name in completed_retrievers:
            print(f"\nSkipping {full_name} (already evaluated)")
            continue
        
        if retriever_to_run != "all" and retriever_to_run != short_name:
            continue
        
        print(f"\nPreparing {short_name} documents...")
        
        if short_name == "enhanced":
            # Enhanced retriever with async evaluation
            docs_enhanced = doc_prep_func()
            enhanced_retriever = EnhancedTableRetriever(docs_enhanced)
            
            # Use async evaluation for Enhanced retriever
            results = await evaluate_retriever_async(
                enhanced_retriever, 
                ground_truth, 
                full_name,
                is_enhanced=True,
                batch_size=10  # Process 10 queries at a time
            )
        else:
            # Standard retrievers with sync evaluation
            if short_name in ["artem_v1", "artem_v2"]:
                docs = doc_prep_func(df, result_full)
            elif short_name == "artem_v3":
                docs = doc_prep_func(df)
            else:
                docs = doc_prep_func()
            
            retriever = create_ensemble_retriever_from_documents(
                docs, embed_model, name=short_name
            )
            
            # Use sync evaluation for standard retrievers
            results = evaluate_retriever_mrr(retriever, ground_truth, full_name)
        
        all_results.append(results)
        
        # Save intermediate results
        results_for_json = []
        for result in all_results:
            # Create a copy without the detailed results for the summary
            summary = {k: v for k, v in result.items() if k != 'details'}
            results_for_json.append(summary)
        
        with open('selected_retrievers_evaluation_full.json', 'w') as f:
            json.dump(results_for_json, f, indent=2)
        print(f"  Saved intermediate results ({len(all_results)} retrievers evaluated)")
    
    # Generate summary report
    print("\n" + "="*80)
    print("EVALUATION SUMMARY - MEAN RECIPROCAL RANK (MRR)")
    print("="*80)
    
    # Sort by MRR
    all_results.sort(key=lambda x: x['mrr'], reverse=True)
    
    # Print table header
    print(f"\n{'Rank':<5} {'Retriever':<65} {'MRR':>8} {'Hit@1':>8} {'Hit@5':>8} {'Failed':>8} {'Time(s)':>8}")
    print("-" * 108)
    
    # Print results
    for i, result in enumerate(all_results, 1):
        print(f"{i:<5} {result['retriever_name']:<65} "
              f"{result['mrr']:>8.4f} "
              f"{result['hits_at_1']:>8.2%} "
              f"{result['hits_at_5']:>8.2%} "
              f"{result['failed_queries']:>8} "
              f"{result.get('time_seconds', 0):>8.1f}")
    
    # Save results to Excel
    print("\nSaving results to Excel...")
    
    # Create DataFrame for summary results
    summary_data = []
    for i, result in enumerate(all_results, 1):
        summary_data.append({
            'Rank': i,
            'Retriever': result['retriever_name'],
            'MRR': result['mrr'],
            'Total Queries': result['total_queries'],
            'Failed Queries': result['failed_queries'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create detailed results DataFrame
    detailed_data = []
    for result in all_results:
        retriever_name = result['retriever_name']
        if 'details' in result:
            for detail in result['details']:
                detailed_data.append({
                    'Retriever': retriever_name,
                    'Query Index': detail['query_idx'],
                    'Query': detail['query'],
                    'Ground Truth': ', '.join(detail['ground_truth']),
                    'Retrieved Tables': ', '.join(detail['retrieved'][:5]),  # Top 5
                    'Best Match': detail.get('best_match', ''),
                    'MRR Score': detail['mrr']
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter('retriever_evaluation_results.xlsx', engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format the summary sheet
        worksheet = writer.sheets['Summary']
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        # Save detailed results if not too large
        if len(detailed_df) < 65000:  # Excel row limit
            detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        else:
            print(f"  Warning: Detailed results too large for Excel ({len(detailed_df)} rows), saving summary only")
    
    print(f"  Results saved to retriever_evaluation_results.xlsx")
    
    # Best performer analysis
    if all_results:
        best_result = all_results[0]
        print(f"\nBest performing retriever: {best_result['retriever_name']}")
        print(f"  MRR: {best_result['mrr']:.4f}")
        print(f"  Hit@1: {best_result['hits_at_1']:.2%}")
        print(f"  Hit@5: {best_result['hits_at_5']:.2%}")
    
    # Show performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    print("\nMRR Scores:")
    for result in all_results:
        print(f"  {result['retriever_name']}: {result['mrr']:.4f}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")

if __name__ == "__main__":
    asyncio.run(main_async())
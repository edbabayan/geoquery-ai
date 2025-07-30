#!/usr/bin/env python3
"""Evaluation of all retrieval methods using Mean Reciprocal Rank (MRR) on the first 5 queries"""

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
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(CFG.env_file)

def evaluate_retriever_mrr(retriever, queries_df: pd.DataFrame, retriever_name: str, top_k: int = 5) -> Dict:
    """
    Evaluate a retriever on the ground truth dataset using MRR with progress tracking
    
    Args:
        retriever: The retriever to evaluate
        queries_df: DataFrame with NL_QUERY and TABLES columns
        retriever_name: Name of the retriever for reporting
        top_k: Number of documents to retrieve
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Starting evaluation of {retriever_name}")
    start_time = time.time()
    
    all_predictions = []
    all_ground_truth = []
    detailed_results = []
    errors = []
    
    # Progress bar
    with tqdm(total=len(queries_df), desc=f"Evaluating {retriever_name}", ncols=100) as pbar:
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
                
                # Log sample retrievals every 10 queries
                if idx % 10 == 0:
                    logger.debug(f"Query {idx}: '{query[:50]}...' -> Retrieved: {retrieved_tables[:3]}")
                    
            except Exception as e:
                error_msg = f"Error on query {idx}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                retrieved_tables = []
            
            # For MRR calculation with multiple ground truth tables
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
            
            pbar.update(1)
    
    # Calculate overall MRR
    overall_mrr = calculate_mrr(all_predictions, all_ground_truth)
    
    # Calculate additional metrics
    hits_at_1 = sum(1 for r in detailed_results if r['mrr'] == 1.0) / len(detailed_results) if detailed_results else 0
    hits_at_5 = sum(1 for r in detailed_results if r['mrr'] > 0) / len(detailed_results) if detailed_results else 0
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed evaluation of {retriever_name} in {elapsed_time:.2f} seconds")
    logger.info(f"  MRR: {overall_mrr:.4f}, Hit@1: {hits_at_1:.2%}, Hit@5: {hits_at_5:.2%}")
    
    return {
        'retriever_name': retriever_name,
        'mrr': overall_mrr,
        'hits_at_1': hits_at_1,
        'hits_at_5': hits_at_5,
        'total_queries': len(detailed_results),
        'failed_queries': len([r for r in detailed_results if r['mrr'] == 0]),
        'errors': len(errors),
        'elapsed_time': elapsed_time,
        'details': detailed_results,
        'error_messages': errors
    }

def prepare_documents_enhanced():
    """Prepare documents with enhanced descriptions and questions"""
    logger.info("Preparing enhanced documents...")
    documents = []
    
    # Load previously generated questions if available
    try:
        with open('src/result_full.json', 'r') as f:
            result_full = json.load(f)
        logger.info(f"Loaded questions from result_full.json")
    except Exception as e:
        logger.warning(f"Could not load result_full.json: {e}")
        result_full = {}
    
    for table_info in table_description_list_default:
        full_table_name = table_info["table_name"]
        table_description = table_info["table_description"]
        short_table_name = full_table_name.split(".")[-1]
        
        # Get example questions
        questions = result_full.get(full_table_name, [])
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
    
    logger.info(f"Created {len(documents)} enhanced documents")
    return documents

def prepare_documents_simple():
    """Prepare documents with just table name and description"""
    logger.info("Preparing simple documents...")
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
    
    logger.info(f"Created {len(documents)} simple documents")
    return documents

def create_ensemble_retriever_from_documents(documents, embed_model, num_docs_retrieved=5, weights=[0.5, 0.5], name=""):
    """Create an ensemble retriever matching the function from main.py"""
    logger.debug(f"Creating ensemble retriever '{name}' with weights {weights}")
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = num_docs_retrieved
    
    vectorstore = Chroma.from_documents(documents, embed_model, collection_name=f"test_{name}")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": num_docs_retrieved})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=weights
    )
    
    return ensemble_retriever

def main():
    """Run comprehensive evaluation of all retrievers"""
    logger.info("="*80)
    logger.info("Starting comprehensive retriever evaluation with MRR")
    logger.info("="*80)
    
    # Load ground truth data
    logger.info("Loading ground truth data...")
    ground_truth = pd.read_csv(CFG.golden_dataset)[['NL_QUERY', "TABLES"]]
    logger.info(f"Loaded {len(ground_truth)} queries for evaluation")
    
    # Use first 5 queries to match test_with_ground_truth
    subset_size = 5
    ground_truth = ground_truth.head(subset_size)
    logger.info(f"Using first {len(ground_truth)} queries (same as test_with_ground_truth)")
    
    # Initialize embeddings
    logger.info("Initializing OpenAI embeddings...")
    embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Prepare different document versions
    docs_enhanced = prepare_documents_enhanced()
    docs_simple = prepare_documents_simple()
    
    # Results storage
    all_results = []
    
    # 1. Test basic retrievers
    logger.info("\n" + "="*80)
    logger.info("TESTING BASIC RETRIEVERS")
    logger.info("="*80)
    
    # BM25 only
    bm25_retriever = BM25Retriever.from_documents(docs_simple)
    bm25_retriever.k = 5
    results = evaluate_retriever_mrr(bm25_retriever, ground_truth, "BM25 Only")
    all_results.append(results)
    
    # Vector only
    vectorstore = Chroma.from_documents(docs_simple, embed_model, collection_name="eval_vector_only")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    results = evaluate_retriever_mrr(vector_retriever, ground_truth, "Vector Only (text-embedding-3-large)")
    all_results.append(results)
    
    # 2. Test ensemble retrievers with different weights
    logger.info("\n" + "="*80)
    logger.info("TESTING ENSEMBLE RETRIEVERS WITH DIFFERENT WEIGHTS")
    logger.info("="*80)
    
    weight_configs = [
        ([0.7, 0.3], "Ensemble (BM25=0.7, Vector=0.3)"),
        ([0.5, 0.5], "Ensemble (BM25=0.5, Vector=0.5)"),
        ([0.3, 0.7], "Ensemble (BM25=0.3, Vector=0.7)")
    ]
    
    for weights, name in weight_configs:
        ensemble = create_ensemble_retriever_from_documents(
            docs_simple, 
            embed_model, 
            weights=weights,
            name=f"ensemble_{weights[0]}"
        )
        results = evaluate_retriever_mrr(ensemble, ground_truth, name)
        all_results.append(results)
    
    # 3. Test enhanced retriever with different configurations
    logger.info("\n" + "="*80)
    logger.info("TESTING ENHANCED RETRIEVER (THIS MAY TAKE A WHILE)")
    logger.info("="*80)
    
    logger.info("Initializing enhanced retriever with cross-encoder...")
    enhanced_retriever = EnhancedTableRetriever(docs_enhanced)
    
    # Different configurations
    enhanced_configs = [
        {
            "use_query_expansion": False, 
            "use_reranking": False, 
            "alpha": 0.5, 
            "name": "Enhanced (No expansion, No reranking)"
        },
        {
            "use_query_expansion": True, 
            "use_reranking": False, 
            "alpha": 0.5,
            "name": "Enhanced (Query expansion, No reranking)"
        },
        {
            "use_query_expansion": False, 
            "use_reranking": True, 
            "alpha": 0.5,
            "name": "Enhanced (No expansion, Cross-encoder reranking)"
        },
        {
            "use_query_expansion": True, 
            "use_reranking": True, 
            "alpha": 0.5,
            "name": "Enhanced (Full: expansion + reranking)"
        },
        {
            "use_query_expansion": True, 
            "use_reranking": True, 
            "alpha": 0.7,
            "name": "Enhanced (Full, BM25=0.7)"
        },
        {
            "use_query_expansion": True, 
            "use_reranking": True, 
            "alpha": 0.3,
            "name": "Enhanced (Full, BM25=0.3)"
        }
    ]
    
    for config in enhanced_configs:
        name = config.pop("name")
        
        # Create a wrapper to pass config
        class ConfiguredRetriever:
            def __init__(self, retriever, config):
                self.retriever = retriever
                self.config = config
            
            def retrieve(self, query, top_k=5):
                return self.retriever.retrieve(query, top_k=top_k, **self.config)
        
        configured_retriever = ConfiguredRetriever(enhanced_retriever, config)
        results = evaluate_retriever_mrr(configured_retriever, ground_truth, name)
        all_results.append(results)
    
    # 4. Generate summary report
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY - MEAN RECIPROCAL RANK (MRR)")
    logger.info("="*80)
    
    # Sort by MRR
    all_results.sort(key=lambda x: x['mrr'], reverse=True)
    
    # Print table header
    print(f"\n{'Rank':<5} {'Retriever':<55} {'MRR':>8} {'Hit@1':>8} {'Hit@5':>8} {'Failed':>8} {'Time(s)':>10}")
    print("-" * 100)
    
    # Print results
    for i, result in enumerate(all_results, 1):
        print(f"{i:<5} {result['retriever_name']:<55} "
              f"{result['mrr']:>8.4f} "
              f"{result['hits_at_1']:>8.2%} "
              f"{result['hits_at_5']:>8.2%} "
              f"{result['failed_queries']:>8} "
              f"{result['elapsed_time']:>10.2f}")
    
    # Save detailed results
    logger.info("\nSaving results...")
    
    # Prepare results for JSON serialization (without details to keep file size manageable)
    results_summary = []
    for result in all_results:
        summary = {k: v for k, v in result.items() if k not in ['details', 'error_messages']}
        results_summary.append(summary)
    
    with open('evaluation_results_mrr.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Saved summary results to evaluation_results_mrr.json")
    
    # Save detailed results for best performer
    best_result = all_results[0]
    with open('best_retriever_detailed_results.json', 'w') as f:
        json.dump({
            'retriever_name': best_result['retriever_name'],
            'mrr': best_result['mrr'],
            'sample_failures': [d for d in best_result['details'] if d['mrr'] == 0][:10],
            'sample_successes': [d for d in best_result['details'] if d['mrr'] == 1.0][:10]
        }, f, indent=2)
    logger.info("Saved detailed results for best retriever to best_retriever_detailed_results.json")
    
    # Best and worst performers
    best_result = all_results[0]
    worst_result = all_results[-1]
    
    logger.info(f"\nBest performing retriever: {best_result['retriever_name']}")
    logger.info(f"  MRR: {best_result['mrr']:.4f}")
    logger.info(f"  Hit@1: {best_result['hits_at_1']:.2%}")
    logger.info(f"  Hit@5: {best_result['hits_at_5']:.2%}")
    logger.info(f"  Time: {best_result['elapsed_time']:.2f}s")
    
    logger.info(f"\nWorst performing retriever: {worst_result['retriever_name']}")
    logger.info(f"  MRR: {worst_result['mrr']:.4f}")
    logger.info(f"  Hit@1: {worst_result['hits_at_1']:.2%}")
    logger.info(f"  Hit@5: {worst_result['hits_at_5']:.2%}")
    
    # Show some failure cases from the best retriever
    logger.info("\n" + "="*80)
    logger.info("SAMPLE FAILURE CASES (Best Retriever)")
    logger.info("="*80)
    
    failures = [d for d in best_result['details'] if d['mrr'] == 0][:5]
    for i, failure in enumerate(failures, 1):
        logger.info(f"\n{i}. Query: {failure['query']}")
        logger.info(f"   Expected: {failure['ground_truth']}")
        logger.info(f"   Retrieved: {failure['retrieved'][:3]}...")
    
    logger.info("\n" + "="*80)
    logger.info("Evaluation complete!")
    logger.info(f"Total evaluation time: {sum(r['elapsed_time'] for r in all_results):.2f} seconds")
    
    # Show the specific queries used
    logger.info("\nQueries evaluated:")
    for idx, row in ground_truth.iterrows():
        logger.info(f"  {idx+1}. {row['NL_QUERY']}")
        logger.info(f"     Expected: {row['TABLES']}")

if __name__ == "__main__":
    main()
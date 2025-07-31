"""
Query decomposition module for enhanced retrieval with ADNOC oil and gas expertise
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv
import os
import sys
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

# Load environment variables
load_dotenv(CFG.env_file)

# Initialize LLM
llm = ChatOpenAI(model='gpt-4.1')

# Query decomposition prompt template
query_decomposition_prompt = PromptTemplate(
    template="""
    You are an oil and gas specialist from ADNOC. Your task is to decompose a complex query into 3 simpler, more focused sub-queries that will help retrieve relevant information from oil and gas production databases.

    Original Query: {query}

    Please generate exactly 3 sub-queries that:
    1. Break down different aspects of the original query
    2. Focus on specific oil and gas operations, production data, well information, or reservoir analysis
    3. Use ADNOC-specific terminology and concepts where appropriate
    4. Are more specific and targeted than the original query
    5. Together cover all aspects of the original query

    Format your response as a JSON array with exactly 3 strings, each being a sub-query.
    Example format: ["sub-query 1", "sub-query 2", "sub-query 3"]
    """,
    input_variables=["query"]
)

# Create query decomposition chain
query_decomposition_chain = query_decomposition_prompt | llm | JsonOutputParser()

def decompose_query(query: str) -> List[str]:
    """
    Decompose a query into 3 sub-queries using ADNOC oil and gas expertise
    
    Args:
        query: Original query string
        
    Returns:
        List of 3 sub-queries
    """
    try:
        sub_queries = query_decomposition_chain.invoke({"query": query})
        if isinstance(sub_queries, list) and len(sub_queries) == 3:
            return sub_queries
        else:
            # Fallback: return original query if decomposition fails
            return [query, query, query]
    except Exception as e:
        print(f"Query decomposition failed: {e}")
        # Fallback: return original query
        return [query, query, query]

def enhanced_retrieval_with_decomposition(retriever: EnsembleRetriever, query: str, num_docs_per_query: int = 3, top_k_final: int = 5) -> List[Dict[str, Any]]:
    """
    Perform enhanced retrieval using query decomposition
    
    Args:
        retriever: The ensemble retriever to use
        query: Original query
        num_docs_per_query: Number of documents to retrieve per sub-query
        top_k_final: Number of top documents to return after deduplication
        
    Returns:
        List of top-k unique documents retrieved from all sub-queries
    """
    # Decompose the query
    sub_queries = decompose_query(query)
    
    # Retrieve documents for each sub-query and track scores
    all_docs_with_scores = []
    seen_content = set()
    
    for sub_query_idx, sub_query in enumerate(sub_queries):
        try:
            docs = retriever.invoke(sub_query)[:num_docs_per_query]
            for doc_idx, doc in enumerate(docs):
                # Avoid duplicates based on content
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    # Calculate score: higher for earlier sub-queries and earlier positions
                    # Score ranges from 1.0 (best) to near 0 (worst)
                    score = 1.0 / ((sub_query_idx + 1) * (doc_idx + 1))
                    all_docs_with_scores.append((doc, score))
        except Exception as e:
            print(f"Retrieval failed for sub-query '{sub_query}': {e}")
            continue
    
    # Sort by score (descending) and take top-k
    all_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in all_docs_with_scores[:top_k_final]]
    
    return top_docs

class QueryDecompositionRetriever:
    """
    Retriever wrapper that uses query decomposition
    """
    def __init__(self, base_retriever: EnsembleRetriever):
        self.base_retriever = base_retriever
    
    def invoke(self, query: str, num_docs_per_query: int = 3):
        """
        Invoke method compatible with evaluation framework
        """
        return enhanced_retrieval_with_decomposition(self.base_retriever, query, num_docs_per_query)

def default_composition(base_retriever: EnsembleRetriever):
    """
    Create a retriever with query decomposition for default composition
    
    Args:
        base_retriever: Base ensemble retriever
        
    Returns:
        QueryDecompositionRetriever with invoke method
    """
    return QueryDecompositionRetriever(base_retriever)
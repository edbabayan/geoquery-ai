"""
Retrieval models enhanced with generated questions for tables and columns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from typing import List, Dict, Any

# Import base components
from retrieval_models import table_description_list_default, ground_truth_renamed
from questions_generation import generate_questions_for_tables, generate_questions_for_columns, result_full


# Initialize components
embed = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model='gpt-4o', temperature=0)

# Read additional metadata
try:
    df_metadata = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/table_description 1.xlsx")
except:
    df_metadata = None


def create_documents_with_questions(table_descriptions: List[Dict[str, Any]], 
                                  generated_questions: Dict[str, List[str]]) -> List[Document]:
    """Create documents with default descriptions + generated questions"""
    documents = []
    
    for table_desc in table_descriptions:
        table_name = table_desc["table_name"].split(".")[-1]
        
        # Get generated questions for this table
        table_questions = generated_questions.get(table_name, [])
        questions_text = " Example questions: " + " ".join(table_questions) if table_questions else ""
        
        # Combine description with questions
        content = f"{table_desc['table_description']}{questions_text}"
        
        doc = Document(
            page_content=content,
            metadata={"table_name": table_name}
        )
        documents.append(doc)
    
    return documents


def create_documents_with_columns_and_questions(table_descriptions: List[Dict[str, Any]], 
                                               df_metadata: pd.DataFrame,
                                               generated_questions: Dict[str, List[str]]) -> List[Document]:
    """Create documents with default descriptions + column info + generated questions"""
    documents = []
    
    for i, table_desc in enumerate(table_descriptions):
        table_name = table_desc["table_name"].split(".")[-1]
        
        # Get base description
        base_description = table_desc['table_description']
        
        # Get column information if available
        column_info = ""
        if df_metadata is not None and i < len(df_metadata):
            # Extract column-related metadata
            columns_text = []
            
            # Add main columns if available
            if 'main_columns' in df_metadata.columns:
                main_cols = df_metadata.loc[i, 'main_columns']
                if pd.notna(main_cols):
                    columns_text.append(f"Main columns: {main_cols}")
            
            # Add column descriptions if available
            if 'column_descriptions' in df_metadata.columns:
                col_desc = df_metadata.loc[i, 'column_descriptions']
                if pd.notna(col_desc):
                    columns_text.append(f"Column details: {col_desc}")
            
            # Add any other relevant column information
            relevant_cols = ['industry_terms', 'data_granularity', 'unique_insights']
            for col in relevant_cols:
                if col in df_metadata.columns:
                    value = df_metadata.loc[i, col]
                    if pd.notna(value):
                        columns_text.append(f"{col.replace('_', ' ').title()}: {value}")
            
            if columns_text:
                column_info = " " + " ".join(columns_text)
        
        # Get generated questions for this table
        table_questions = generated_questions.get(table_name, [])
        questions_text = " Example questions: " + " ".join(table_questions) if table_questions else ""
        
        # Combine all information
        content = f"{base_description}{column_info}{questions_text}"
        
        doc = Document(
            page_content=content,
            metadata={"table_name": table_name}
        )
        documents.append(doc)
    
    return documents


def create_ensemble_retriever_from_documents(documents: List[Document], 
                                           num_docs_retrieved: int = 5, 
                                           weights: List[float] = None, 
                                           name: str = "") -> EnsembleRetriever:
    """Create ensemble retriever from documents"""
    if weights is None:
        weights = [0.5, 0.5]
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = num_docs_retrieved
    
    # Vector retriever
    vectorstore = Chroma.from_documents(documents, embed, collection_name=f"questions_{name}")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": num_docs_retrieved})
    
    # Ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=weights
    )
    
    return ensemble_retriever


# Use the pre-generated questions from result_full
# This avoids making API calls during module import
generated_table_questions = result_full

# Generate questions for tables if not already available
if not generated_table_questions:
    print("Warning: No pre-generated questions found. Models will use empty questions.")
    generated_table_questions = {}

# Create documents with questions
documents_default_with_questions = create_documents_with_questions(
    table_description_list_default, 
    generated_table_questions
)

# Create documents with columns and questions
documents_default_columns_questions = create_documents_with_columns_and_questions(
    table_description_list_default,
    df_metadata,
    generated_table_questions
)

# Create retrievers with different weight configurations
# Default + Questions models
retriever_default_questions_balanced = create_ensemble_retriever_from_documents(
    documents_default_with_questions, 
    name="default_questions_balanced",
    weights=[0.5, 0.5]
)

retriever_default_questions_bm25 = create_ensemble_retriever_from_documents(
    documents_default_with_questions, 
    name="default_questions_bm25",
    weights=[0.7, 0.3]
)

retriever_default_questions_vector = create_ensemble_retriever_from_documents(
    documents_default_with_questions, 
    name="default_questions_vector",
    weights=[0.3, 0.7]
)

# Default + Columns + Questions models
retriever_default_columns_questions_balanced = create_ensemble_retriever_from_documents(
    documents_default_columns_questions, 
    name="default_columns_questions_balanced",
    weights=[0.5, 0.5]
)

retriever_default_columns_questions_bm25 = create_ensemble_retriever_from_documents(
    documents_default_columns_questions, 
    name="default_columns_questions_bm25",
    weights=[0.7, 0.3]
)

retriever_default_columns_questions_vector = create_ensemble_retriever_from_documents(
    documents_default_columns_questions, 
    name="default_columns_questions_vector",
    weights=[0.3, 0.7]
)

# Export retrievers
__all__ = [
    'retriever_default_questions_balanced',
    'retriever_default_questions_bm25',
    'retriever_default_questions_vector',
    'retriever_default_columns_questions_balanced',
    'retriever_default_columns_questions_bm25',
    'retriever_default_columns_questions_vector',
    'ground_truth_renamed'
]
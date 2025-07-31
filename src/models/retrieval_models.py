"""
Retrieval models extracted from mrr_test notebook
"""

# Loaders
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain.schema import Document

# Text Splitters
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# Embedding Support
from langchain_openai import OpenAIEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

# Vector Stores
from langchain_community.vectorstores import Chroma, FAISS

# Retrievers
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Storage
from langchain.storage import InMemoryStore

# Prompts and Parsing
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Additional Libraries
import os
from langchain_core.prompts import PromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_core.output_parsers.json import JsonOutputParser
from dotenv import load_dotenv
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

import numpy as np
from typing import List, Dict, Any
import pandas as pd

# Load environment variables
load_dotenv(CFG.env_file)

# Initialize LLM
llm = ChatOpenAI(model='gpt-4.1')

# MRR calculation functions
def calculate_mrr(predictions: List[List[str]], ground_truth: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for RAG validation
    
    Args:
        predictions: List of ranked prediction lists for each query
        ground_truth: List of correct answers for each query
    
    Returns:
        MRR score (0-1, higher is better)
    """
    reciprocal_ranks = []
    
    for pred_list, correct_answer in zip(predictions, ground_truth):
        rank = None
        for i, pred in enumerate(pred_list):
            if pred.strip().lower() == correct_answer.strip().lower():
                rank = i + 1  # rank starts from 1
                break
        
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)

def evaluate_rag_with_mrr(rag_system, test_queries: List[str], 
                         ground_truth: List[str], top_k: int = 5) -> Dict[str, Any]:
    """
    Evaluate RAG system using MRR metric
    
    Args:
        rag_system: Your RAG system with a query method
        test_queries: List of test questions
        ground_truth: List of expected answers
        top_k: Number of top predictions to consider
    
    Returns:
        Dictionary with MRR score and detailed results
    """
    all_predictions = []
    
    for query in test_queries:
        # Get top-k predictions from your RAG system
        # Adjust this based on your RAG system's API
        predictions = rag_system.query(query, top_k=top_k)
        all_predictions.append(predictions)
    
    mrr_score = calculate_mrr(all_predictions, ground_truth)
    
    return {
        'mrr_score': mrr_score,
        'predictions': all_predictions,
        'ground_truth': ground_truth,
        'num_queries': len(test_queries)
    }

# Table descriptions
table_description_list_default = [
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.daily_allocation",
        "table_description": "This table captures the result of allocating daily production and injection volumes to individual wells. It provides production and injection details such as flow direction (production or injection), material disposition codes (natural, gas-lift, ESP), material type (oil, water, gas), production date, flowing hours (duration). It provides detailed insights into production and injection volumes across multiple fields and reservoirs as well."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.inactive_string",
        "table_description": "This table provides inactive wells and strings on a monthly basis. It contains inactive reasons such as inactive category, problem name, string status (inactive, active, abandoned), string health (healthy, problematic, etc.). It also tracks downtime start dates, estimated action dates & expected rates across multiple assets, fields, and reservoirs."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.real_time_corporate_pi",
        "table_description": "This table captures time series real-time operations sensor data for oil production & injection wells. Attributes include pressure, temperature, choke size, valve status, injection, company name, well name, string type, and real-time data tag name for unique well identifiers (UWI)."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well_reservoir",
        "table_description": "Maps wells and strings to their associated reservoirs. Contains unique well identifiers, string name (LS, SS, ST/TB), associated field name. Each row represents a unique well (UWI) with its reservoir, field, and string designation."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.string",
        "table_description": "Same as well_reservoir: maps wells and strings to reservoirs. Contains unique well identifiers, string name (LS, SS, ST/TB), field name. Each row represents a unique well (UWI) with reservoir, field, and string designation."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.string_event",
        "table_description": "Captures well and string status (flowing, down, injecting, etc.) and reasons for status. Contains well and string identifiers, reservoir details, event date, reasons, descriptions, remarks, choke size, pressures, temperatures, flow rates, with timestamps for event start/end. Granularity is at string level within wells."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.unified_pressure_test",
        "table_description": "Consolidates pressure surveys/test data for categories like BHCIP, PBU, PFO, GRAD, BHFP for producers/injectors. Includes test date, reservoir pressure (datum, mean, gauge), BHP (bottom hole pressures), permeability, productivity index, reservoir, gauge, and service company details."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well",
        "table_description": "Master data for wells: company, project, field, UWI, well name, type (producer, observer, injector), operator, coordinates, elevation, depth, current/previous status, spud/completion dates, and unique well identifiers."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well_completion",
        "table_description": "Represents well completion downhole equipment data. Includes completion type, dimensions (OD, ID, length), installation/removal dates, inner/outer diameters, equipment lengths for tracking completion components within wells."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well_log_index",
        "table_description": "Represents logging services on wells. Provides data for services (GR, PLT, RST, etc.), hole conditions, mud properties, formation details, depth intervals, logging dates, service provider, fluid characteristics supporting subsurface analysis and well performance."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.wellbore",
        "table_description": "Detailed wellbore information: unique well identifiers, drilling metrics, geological targets, coordinates, operational details, bore status, borehole name, operator, depth, formation codes, rig details, spud/completion dates across multiple assets, fields, and reservoirs."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.well_allowable_limits",
        "table_description": "Well-level data on allowable production/injection rates, technical rates updated monthly. Includes allowable rates, technical rates, material types, field, reservoir, well identifiers, and production/injection limits (start/end dates)."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.field",
        "table_description": "Unified view of fields: field codes, names, associated company, and project codes."
    },
    {
        "table_name": "fws_aiq_enai_dp_silver_rag.ctdh_unified_data_rag.flow_test",
        "table_description": "Captures detailed flow test data for wells: test type, duration, rates (oil, gas, water), choke size, wellhead pressures, temperatures, chemical compositions. Granular at well and tubing string level for reservoir performance and production analysis over time."
    }
]

# Read table descriptions
df = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/table_description 1.xlsx")

# Read ground truth data
ground_truth = pd.read_csv("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/golden_dataset_export_20250716_095952 1.csv")[['NL_QUERY', "TABLES"]]
ground_truth_renamed = ground_truth.rename(columns={
    'NL_QUERY': 'question',
    'TABLES': 'tables'
})
json_data = ground_truth_renamed.to_dict('records')

# Prompt for generating questions
prompt = PromptTemplate(
    template="""
    You are tasked with generating example questions for given tables and formatting the output in JSON. 
    Your input will be a set of tables, and your output should be pairs of tables and corresponding example questions that a user might ask about the data in those tables.
    First, carefully examine the input tables provided in the {{TABLES}} variable. 
    Each table may contain different types of data, so pay attention to the column headers and the information presented.
    For each table:
    1. Generate 3-5 example questions that a user might ask about the tables based on the question examples provided to you. These questions should:
    - Be diverse and cover different aspects of the data
    - Range from simple to more complex queries
    - Not be exact copies of the actual questions, but similar in nature
    Format your output as a JSON array with objects containing two keys: "table" and "questions". 
    The "table" should be the name of the table, and the "questions" value should be an array of example questions.
    This is the desription and questions to the tables: 
    <table_description>     
    {TABLE_DESCRIPTION}
    </table_description>
    Ensure that your JSON is properly formatted and that each table is accurately represented as a string, including the header row and separators.
    Remember to create questions that are relevant to the this table based on examples, and try to showcase different types of queries that users might be interested in asking about the data.
""",
    input_variables=["TABLE_DESCRIPTION"]
)

chain = prompt | llm | JsonOutputParser()

# Generate questions (commented out to avoid API calls during module import)
# result1 = chain.invoke({"TABLE_DESCRIPTION": json_data})

# Create document variations
documents_default_desription = [Document(page_content=i["table_description"], metadata={"table_name": (i["table_name"]).split(".")[-1]}) for i in table_description_list_default]

# Create different description strings variations
# For now, we'll create variations without the generated questions to avoid API calls
# Version 1: Full descriptions (without questions for now)
description_strings_questions = []
for i in df.index:
    if i < len(table_description_list_default):  # Ensure we don't go out of bounds
        description_strings_questions.append(f"Table name is {df.loc[i, 'name']}. Industry terms are {df.loc[i, 'industry_terms']}. Data granularity is {df.loc[i, 'data_granularity']}. Main business purpose is {df.loc[i, 'main_business_purpose']}. Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}.")

# Version 2: Main purpose + unique insights  
description_strings_questions_v2 = []
for i in df.index:
    if i < len(table_description_list_default):  # Ensure we don't go out of bounds
        description_strings_questions_v2.append(f"Table name is {df.loc[i, 'name']}. Main business purpose is {df.loc[i, 'main_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}.")

# Version 3: Full descriptions without questions
description_strings_questions_v3 = []
for i in df.index:
    if i < len(table_description_list_default):  # Ensure we don't go out of bounds
        description_strings_questions_v3.append(f"Table name is {df.loc[i, 'name']}. Industry terms are {df.loc[i, 'industry_terms']}. Data granularity is {df.loc[i, 'data_granularity']}. Main business purpose is {df.loc[i, 'main_business_purpose']}. Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}.")

# Version 4: Main purpose + unique insights only
description_strings_questions_v4 = []
for i in df.index:
    if i < len(table_description_list_default):  # Ensure we don't go out of bounds
        description_strings_questions_v4.append(f"Table name is {df.loc[i, 'name']}. Main business purpose is {df.loc[i, 'main_business_purpose']} Unique insights are {df.loc[i, 'unique_insights']}")

# Initialize embeddings
embed = OpenAIEmbeddings(model="text-embedding-3-large")

# Create documents for each variation
documents_artem_description = [Document(page_content=desc, metadata={"table_name": (table_description_list_default[i]["table_name"]).split(".")[-1]}) for i, desc in enumerate(description_strings_questions)]
documents_artem_v2_desription = [Document(page_content=desc, metadata={"table_name": (table_description_list_default[i]["table_name"]).split(".")[-1]}) for i, desc in enumerate(description_strings_questions_v2)]
documents_artem_v3_desription = [Document(page_content=desc, metadata={"table_name": (table_description_list_default[i]["table_name"]).split(".")[-1]}) for i, desc in enumerate(description_strings_questions_v3)]
documents_artem_v4_desription = [Document(page_content=desc, metadata={"table_name": (table_description_list_default[i]["table_name"]).split(".")[-1]}) for i, desc in enumerate(description_strings_questions_v4)]

# Function to create ensemble retriever
def create_ensemble_retriever_from_documents(documents, num_docs_retrieved=5, weights=None, name=""):
    if weights is None:
        weights = [0.5, 0.5]
    bm25_retriever = BM25Retriever.from_documents(
        documents
    )
    bm25_retriever.k = num_docs_retrieved
    vectorestore = Chroma.from_documents(documents, embed, collection_name=f"test_{name}")  
    vector_retriever = vectorestore.as_retriever(search_kwargs={"k": num_docs_retrieved})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=weights
    )
    return ensemble_retriever

# Create retrievers
retriever_artem_v1 = create_ensemble_retriever_from_documents(documents_artem_description, name="artem_v1")
retriever_artem_v2 = create_ensemble_retriever_from_documents(documents_artem_v2_desription, name="artem_v2")
retriever_default = create_ensemble_retriever_from_documents(documents_default_desription, name="default_v1")
retriever_artem_v3 = create_ensemble_retriever_from_documents(documents_artem_v3_desription, name="artem_v3")
# retriever_artem_v4 would be created similarly with documents_artem_v4_desription

# Example query
query = "Can you plot the wellhead pressure vs. time for the top 5 producing wells over the last 30 days?"

# Test retrievers (uncomment to use)
# retriever_artem_v1.invoke(query)
# retriever_artem_v2.invoke(query)
# retriever_default.invoke(query)
# retriever_artem_v3.invoke(query)

# Export ground_truth_renamed for evaluation
__all__ = ['retriever_default', 'retriever_artem_v1', 'retriever_artem_v2', 'retriever_artem_v3', 'ground_truth_renamed']
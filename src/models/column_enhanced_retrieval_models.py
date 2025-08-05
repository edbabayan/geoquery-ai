"""
Column-enhanced retrieval model that explicitly stores and uses column descriptions
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

# Load environment variables
load_dotenv(CFG.env_file)

# Read table descriptions
df = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/table_description 1.xlsx")

# Read ground truth data
ground_truth = pd.read_csv("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/golden_dataset_export_20250716_095952 1.csv")[['NL_QUERY', "TABLES"]]
ground_truth_renamed = ground_truth.rename(columns={
    'NL_QUERY': 'question',
    'TABLES': 'tables'
})

# Table descriptions from original list
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

def load_column_descriptions():
    """Load column descriptions from the Excel file"""
    try:
        df_columns = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/columns_description_new.xlsx")
        print(f"Loaded {len(df_columns)} column descriptions")
        return df_columns
    except FileNotFoundError:
        print("Column descriptions file not found, proceeding without column details")
        return None

def get_formatted_column_info(table_name, df_columns):
    """Get formatted column information for a specific table"""
    if df_columns is None:
        return []
    
    # Filter columns for this table
    table_columns = df_columns[df_columns['table_name'].str.contains(table_name, case=False, na=False)]
    
    if table_columns.empty:
        return []
    
    # Format column information
    column_info = []
    for _, row in table_columns.iterrows():
        col_desc = f"{row['column_name']}: {row['description']}"
        if pd.notna(row['unit_of_measure']):
            col_desc += f" (Unit: {row['unit_of_measure']})"
        column_info.append(col_desc)
    
    return column_info

# Load column descriptions
print("Loading column descriptions...")
df_columns = load_column_descriptions()

# Create comprehensive descriptions with column details
column_enhanced_descriptions = []
column_metadata_list = []

for i in range(min(len(df), len(table_description_list_default))):
    table_info = table_description_list_default[i]
    table_short_name = table_info['table_name'].split('.')[-1]
    
    # Get column information
    column_details = get_formatted_column_info(table_short_name, df_columns)
    
    # Build comprehensive description with explicit column information
    desc_parts = [
        f"Table: {df.loc[i, 'name']}",
        f"Description: {table_info['table_description']}",
        f"Industry Terms: {df.loc[i, 'industry_terms']}",
        f"Data Granularity: {df.loc[i, 'data_granularity']}",
        f"Main Purpose: {df.loc[i, 'main_business_purpose']}",
        f"Alternative Purpose: {df.loc[i, 'alternative_business_purpose']}",
        f"Unique Insights: {df.loc[i, 'unique_insights']}"
    ]
    
    # Add column descriptions if available
    if column_details:
        desc_parts.append("\nColumn Descriptions:")
        desc_parts.extend([f"- {col}" for col in column_details])
    
    full_description = "\n".join(desc_parts)
    column_enhanced_descriptions.append(full_description)
    
    # Store column metadata separately for enriched retrieval
    column_metadata_list.append({
        "table_name": table_short_name,
        "full_table_name": table_info['table_name'],
        "num_columns": len(column_details),
        "has_column_descriptions": len(column_details) > 0,
        "column_names": [col.split(":")[0] for col in column_details] if column_details else []
    })

# Create documents with column-enhanced descriptions
documents_column_enhanced = [
    Document(
        page_content=desc,
        metadata=column_metadata_list[i]
    )
    for i, desc in enumerate(column_enhanced_descriptions)
]

# Initialize embeddings
embed = OpenAIEmbeddings(model="text-embedding-3-large")

# Create retrievers with different configurations

# 1. Basic column-enhanced retriever (BM25 + Vector with equal weights)
bm25_retriever_columns = BM25Retriever.from_documents(documents_column_enhanced)
bm25_retriever_columns.k = 5

vectorstore_columns = FAISS.from_documents(documents_column_enhanced, embed)
vector_retriever_columns = vectorstore_columns.as_retriever(search_kwargs={"k": 5})

retriever_columns_basic = EnsembleRetriever(
    retrievers=[bm25_retriever_columns, vector_retriever_columns],
    weights=[0.5, 0.5]
)

# 2. Column-enhanced retriever with BM25 emphasis (better for column name matching)
retriever_columns_bm25_emphasis = EnsembleRetriever(
    retrievers=[bm25_retriever_columns, vector_retriever_columns],
    weights=[0.7, 0.3]
)

# 3. Column-enhanced retriever with vector emphasis (better for semantic matching)
retriever_columns_vector_emphasis = EnsembleRetriever(
    retrievers=[bm25_retriever_columns, vector_retriever_columns],
    weights=[0.3, 0.7]
)

# 4. Column-enhanced retriever with reranking
# Create retrievers that fetch more documents for reranking
bm25_retriever_columns_rerank = BM25Retriever.from_documents(documents_column_enhanced)
bm25_retriever_columns_rerank.k = 10

vector_retriever_columns_rerank = vectorstore_columns.as_retriever(search_kwargs={"k": 10})

ensemble_retriever_columns_rerank = EnsembleRetriever(
    retrievers=[bm25_retriever_columns_rerank, vector_retriever_columns_rerank],
    weights=[0.6, 0.4]
)

# Initialize cross-encoder for reranking
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=5)

retriever_columns_reranked = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever_columns_rerank
)

# Export all retrievers for evaluation
__all__ = [
    'retriever_columns_basic',
    'retriever_columns_bm25_emphasis', 
    'retriever_columns_vector_emphasis',
    'retriever_columns_reranked',
    'ground_truth_renamed'
]
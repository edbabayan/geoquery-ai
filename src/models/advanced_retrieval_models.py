"""
Advanced retrieval models with enhanced metadata using column descriptions and hybrid optimization
"""

from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
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
from generate_table_metadata import generate_table_metadata_enhanced, load_metadata_from_file, save_metadata_to_file

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

# Load or generate enhanced metadata with column descriptions
print("Loading enhanced table metadata...")
# Try to load hybrid metadata first, fall back to generated
table_metadata_enhanced = load_metadata_from_file("hybrid_table_metadata.json")

if table_metadata_enhanced is None:
    table_metadata_enhanced = load_metadata_from_file("generated_table_metadata.json")
    
if table_metadata_enhanced is None:
    print("No existing metadata found. Generating new metadata...")
    table_metadata_enhanced = generate_table_metadata_enhanced()
    save_metadata_to_file(table_metadata_enhanced)
else:
    print("Loaded existing metadata from file.")

# Create advanced descriptions with all metadata
advanced_descriptions = []
for i in range(min(len(df), len(table_description_list_default))):
    table_info = table_description_list_default[i]
    table_short_name = table_info['table_name'].split('.')[-1]
    
    # Get enhanced metadata with safe access
    metadata = table_metadata_enhanced.get(table_short_name, {}) if table_metadata_enhanced else {}
    
    # Build comprehensive description
    advanced_desc = f"""
Table: {df.loc[i, 'name']}
Core Description: {table_info['table_description']}

Business Context:
- Main Purpose: {df.loc[i, 'main_business_purpose']}
- Alternative Uses: {df.loc[i, 'alternative_business_purpose']}
- Industry Terms: {df.loc[i, 'industry_terms']}
- Unique Insights: {df.loc[i, 'unique_insights']}

Technical Details:
- Key Columns: {', '.join(metadata.get('key_columns', []))}
- Data Granularity: {df.loc[i, 'data_granularity']}
- Temporal Granularity: {metadata.get('temporal_granularity', 'varies')}

Relationships & Usage:
- Table Relationships: {'; '.join(metadata.get('relationships', []))}
- Common Query Patterns: {'; '.join(metadata.get('common_queries', []))}
- Search Keywords: {', '.join(metadata.get('query_keywords', []))}
"""
    advanced_descriptions.append(advanced_desc)

# Create documents with advanced descriptions
documents_advanced = [
    Document(
        page_content=desc, 
        metadata={
            "table_name": table_description_list_default[i]["table_name"].split(".")[-1],
            "full_table_name": table_description_list_default[i]["table_name"],
            "temporal_granularity": table_metadata_enhanced.get(
                table_description_list_default[i]["table_name"].split(".")[-1], {}
            ).get('temporal_granularity', 'varies') if table_metadata_enhanced else 'varies',
            "key_columns": ', '.join(table_metadata_enhanced.get(
                table_description_list_default[i]["table_name"].split(".")[-1], {}
            ).get('key_columns', [])) if table_metadata_enhanced else ''
        }
    ) 
    for i, desc in enumerate(advanced_descriptions)
]

# Initialize embeddings
embed = OpenAIEmbeddings(model="text-embedding-3-large")

# Create BM25 retriever with advanced documents
bm25_retriever_advanced = BM25Retriever.from_documents(documents_advanced)
bm25_retriever_advanced.k = 10  # Retrieve more for reranking

# Create FAISS vector store for better performance
vectorstore_advanced = FAISS.from_documents(documents_advanced, embed)
vector_retriever_advanced = vectorstore_advanced.as_retriever(search_kwargs={"k": 10})

# Create ensemble retriever with dynamic weights
# Higher BM25 weight for keyword matching given the rich metadata
ensemble_retriever_advanced = EnsembleRetriever(
    retrievers=[bm25_retriever_advanced, vector_retriever_advanced], 
    weights=[0.6, 0.4]  # Favor BM25 for keyword-rich queries
)

# Initialize cross-encoder for reranking
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=5)

# Create compression retriever with reranking
retriever_advanced_reranked = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever_advanced
)

# Create separate retrievers for non-reranked version with k=5
bm25_retriever_advanced_k5 = BM25Retriever.from_documents(documents_advanced)
bm25_retriever_advanced_k5.k = 5

vector_retriever_advanced_k5 = vectorstore_advanced.as_retriever(search_kwargs={"k": 5})

# Also create a version without reranking for comparison
retriever_advanced_no_rerank = EnsembleRetriever(
    retrievers=[bm25_retriever_advanced_k5, vector_retriever_advanced_k5], 
    weights=[0.6, 0.4]
)

# Export for evaluation (maintaining backward compatibility)
__all__ = ['retriever_advanced_reranked', 'retriever_advanced_no_rerank', 'ground_truth_renamed']
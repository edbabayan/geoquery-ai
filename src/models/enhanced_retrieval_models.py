"""
Simple enhanced retrieval model compatible with evaluate_rag_with_mrr
"""

from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
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

# Create enhanced descriptions by combining table info with additional metadata
enhanced_descriptions = []
for i in range(min(len(df), len(table_description_list_default))):
    table_info = table_description_list_default[i]
    # Enhanced description includes original description plus additional context
    enhanced_desc = f"""
Table: {df.loc[i, 'name']}
Description: {table_info['table_description']}
Industry Terms: {df.loc[i, 'industry_terms']}
Data Granularity: {df.loc[i, 'data_granularity']}
Main Purpose: {df.loc[i, 'main_business_purpose']}
Alternative Purpose: {df.loc[i, 'alternative_business_purpose']}
Unique Insights: {df.loc[i, 'unique_insights']}
"""
    enhanced_descriptions.append(enhanced_desc)

# Create documents with enhanced descriptions
documents_enhanced = [
    Document(
        page_content=desc, 
        metadata={"table_name": table_description_list_default[i]["table_name"].split(".")[-1]}
    ) 
    for i, desc in enumerate(enhanced_descriptions)
]

# Initialize embeddings
embed = OpenAIEmbeddings(model="text-embedding-3-large")

# Create enhanced retriever with balanced weights
def create_enhanced_retriever(documents, num_docs_retrieved=5, weights=None):
    if weights is None:
        weights = [0.5, 0.5]  # Equal weight on BM25 and semantic search
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = num_docs_retrieved
    
    # Vector retriever
    vectorstore = Chroma.from_documents(documents, embed, collection_name="enhanced_retriever")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": num_docs_retrieved})
    
    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=weights
    )
    
    return ensemble_retriever

# Create the enhanced retriever
retriever_enhanced = create_enhanced_retriever(documents_enhanced, num_docs_retrieved=5)

# Export for evaluation
__all__ = ['retriever_enhanced', 'ground_truth_renamed']
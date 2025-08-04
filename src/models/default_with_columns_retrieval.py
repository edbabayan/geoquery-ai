"""
Default retriever enhanced with column descriptions
This model uses the same simple table descriptions as the default model,
but adds column information to improve retrieval accuracy.
Creates 3 versions with different BM25/Vector weight combinations.
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

# Read ground truth data
ground_truth = pd.read_csv("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/golden_dataset_export_20250716_095952 1.csv")[['NL_QUERY', "TABLES"]]
ground_truth_renamed = ground_truth.rename(columns={
    'NL_QUERY': 'question',
    'TABLES': 'tables'
})

# Table descriptions from original list (same as default)
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

def get_simple_column_list(table_name, df_columns):
    """Get a simple list of column names for a table"""
    if df_columns is None:
        return ""
    
    # Filter columns for this table
    table_columns = df_columns[df_columns['table_name'].str.contains(table_name, case=False, na=False)]
    
    if table_columns.empty:
        return ""
    
    # Get unique column names
    column_names = table_columns['column_name'].unique().tolist()
    
    # Return as a comma-separated list (limit to first 15 for readability)
    if len(column_names) > 15:
        return f"Columns: {', '.join(column_names[:15])}, ..."
    else:
        return f"Columns: {', '.join(column_names)}"

# Load column descriptions
print("Loading column descriptions...")
df_columns = load_column_descriptions()

# Create enhanced descriptions by appending column names to default descriptions
default_with_columns_descriptions = []

for table_info in table_description_list_default:
    table_short_name = table_info['table_name'].split('.')[-1]
    
    # Get column names
    column_list = get_simple_column_list(table_short_name, df_columns)
    
    # Combine default description with column list
    if column_list:
        enhanced_desc = f"{table_info['table_description']}\n\n{column_list}"
    else:
        enhanced_desc = table_info['table_description']
    
    default_with_columns_descriptions.append(enhanced_desc)

# Create documents with default + columns descriptions
documents_default_with_columns = [
    Document(
        page_content=desc,
        metadata={"table_name": table_description_list_default[i]["table_name"].split(".")[-1]}
    )
    for i, desc in enumerate(default_with_columns_descriptions)
]

# Initialize embeddings
embed = OpenAIEmbeddings(model="text-embedding-3-large")

# Create shared retrievers for reuse
bm25_retriever_shared = BM25Retriever.from_documents(documents_default_with_columns)
bm25_retriever_shared.k = 5

vectorstore_shared = Chroma.from_documents(documents_default_with_columns, embed, collection_name="default_with_columns")
vector_retriever_shared = vectorstore_shared.as_retriever(search_kwargs={"k": 5})

# Version 1: BM25 emphasis (0.7-0.3)
retriever_default_with_columns_bm25 = EnsembleRetriever(
    retrievers=[bm25_retriever_shared, vector_retriever_shared],
    weights=[0.7, 0.3]
)

# Version 2: Balanced (0.5-0.5) - same as original default
retriever_default_with_columns_balanced = EnsembleRetriever(
    retrievers=[bm25_retriever_shared, vector_retriever_shared],
    weights=[0.5, 0.5]
)

# Version 3: Vector emphasis (0.3-0.7)
retriever_default_with_columns_vector = EnsembleRetriever(
    retrievers=[bm25_retriever_shared, vector_retriever_shared],
    weights=[0.3, 0.7]
)

# Export all versions for evaluation
__all__ = [
    'retriever_default_with_columns_bm25',      # 0.7-0.3
    'retriever_default_with_columns_balanced',   # 0.5-0.5
    'retriever_default_with_columns_vector',     # 0.3-0.7
    'ground_truth_renamed'
]
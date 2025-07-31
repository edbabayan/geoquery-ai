"""
Generate enhanced table metadata using LLM for advanced retrieval models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
import pandas as pd
import json
from dotenv import load_dotenv
from config import CFG

# Load environment variables
load_dotenv(CFG.env_file)

# Initialize LLM
llm = ChatOpenAI(model='gpt-4o-mini')

def load_column_descriptions():
    """Load column descriptions from the Excel file"""
    try:
        df_columns = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/columns_description_new.xlsx")
        print(f"Loaded {len(df_columns)} column descriptions")
        return df_columns
    except FileNotFoundError:
        print("Column descriptions file not found, proceeding without column details")
        return None

def get_table_column_info(table_name, df_columns):
    """Get formatted column information for a specific table"""
    if df_columns is None:
        return "Column details not available"
    
    # Filter columns for this table
    table_columns = df_columns[df_columns['table_name'].str.contains(table_name, case=False, na=False)]
    
    if table_columns.empty:
        return "No column details found for this table"
    
    # Format column information
    column_info = []
    for _, row in table_columns.iterrows():
        col_desc = f"- {row['column_name']}: {row['description']}"
        if pd.notna(row['unit_of_measure']):
            col_desc += f" (Unit: {row['unit_of_measure']})"
        column_info.append(col_desc)
    
    return "\n".join(column_info)

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

# Enhanced prompt that uses column descriptions
metadata_generation_prompt = PromptTemplate(
    template="""
You are an oil and gas data expert from ADNOC. Generate comprehensive metadata for this table using both the table description and detailed column information.

Table Name: {table_name}
Table Description: {table_description}

Column Details:
{column_details}

CRITICAL INSTRUCTIONS:
- UWI means "Unique Well Identifier" - this is THE standard well identifier in oil & gas, NOT "well_id"
- Use EXACT column names from the column details above (usually in UPPERCASE)
- Focus on ADNOC-specific terminology and operations
- Leverage the column descriptions and units to understand data types and usage patterns

Generate a JSON object with the following structure:
{{
    "key_columns": ["list of 3-5 most important EXACT column names from the column details"],
    "relationships": ["how this table joins with others - most tables join via UWI"],
    "common_queries": ["3-5 concise, practical queries an oil & gas analyst would ask based on available columns"],
    "temporal_granularity": "one of: daily, monthly, real-time/hourly, event-based, static",
    "query_keywords": ["8-12 critical search terms including technical abbreviations from column descriptions"]
}}

Focus on:
- Key identifiers (UWI, STRING_NAME, FIELD_NAME)
- Date/time columns for temporal granularity
- Technical measurements and their units
- Status and category fields
- Production/injection related columns

Examples of domain keywords to include:
- Technical: ESP, BHP, BHCIP, PBU, choke, valve, pressure, temperature
- Operations: producer, injector, flowing, down, active, inactive
- Materials: oil, gas, water
- String types: LS, SS, ST/TB
- Measurements: rate, volume, pressure, temperature, depth

Return only the JSON object.
""",
    input_variables=["table_name", "table_description", "column_details"]
)

# Create chain
metadata_chain = metadata_generation_prompt | llm | JsonOutputParser()

def generate_table_metadata_enhanced():
    """Generate enhanced metadata for all tables using column descriptions"""
    table_metadata_enhanced = {}
    
    # Load column descriptions
    df_columns = load_column_descriptions()
    
    print("Generating enhanced metadata for tables...")
    
    for table_info in table_description_list_default:
        table_name = table_info["table_name"]
        table_short_name = table_name.split('.')[-1]
        table_description = table_info["table_description"]
        
        print(f"\nProcessing: {table_short_name}")
        
        # Get column information for this table
        column_details = get_table_column_info(table_short_name, df_columns)
        print(f"  Found {len(column_details.split(chr(10))) if column_details != 'No column details found for this table' else 0} columns")
        
        try:
            # Generate metadata using LLM with both table and column info
            metadata = metadata_chain.invoke({
                "table_name": table_short_name,
                "table_description": table_description,
                "column_details": column_details
            })
            
            table_metadata_enhanced[table_short_name] = metadata
            print(f"✓ Generated metadata for {table_short_name}")
            
        except Exception as e:
            print(f"✗ Error generating metadata for {table_short_name}: {e}")
            # Fallback metadata
            table_metadata_enhanced[table_short_name] = {
                "key_columns": ["UWI"],
                "relationships": ["general table in the database"],
                "common_queries": ["general queries about " + table_short_name],
                "temporal_granularity": "varies",
                "query_keywords": table_short_name.split('_')
            }
    
    return table_metadata_enhanced

def save_metadata_to_file(metadata, filename="generated_table_metadata.json"):
    """Save generated metadata to JSON file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {filepath}")
    return filepath

def load_metadata_from_file(filename="generated_table_metadata.json"):
    """Load metadata from JSON file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    # Generate metadata
    metadata = generate_table_metadata_enhanced()
    
    # Save to file
    save_metadata_to_file(metadata)
    
    # Display sample
    print("\nSample generated metadata:")
    for table_name, table_metadata in list(metadata.items())[:2]:
        print(f"\n{table_name}:")
        print(json.dumps(table_metadata, indent=2))
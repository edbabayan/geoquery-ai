"""
Generate hybrid metadata combining auto-generation with hand-crafted patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from generate_table_metadata import table_description_list_default, load_metadata_from_file
import json
from dotenv import load_dotenv
from config import CFG

# Load environment variables
load_dotenv(CFG.env_file)

# Initialize LLM
llm = ChatOpenAI(model='gpt-4o-mini')

# Hand-crafted patterns to guide the generation
hand_crafted_patterns = {
    "relationship_patterns": [
        "joins with {table} via {column}",
        "connects to {table} via {column}",
        "master mapping table",
        "time-series data for {entity}",
        "event history for {entity}"
    ],
    "query_patterns": [
        "{metric} by {dimension}",
        "top {entity}",
        "{comparison} analysis",
        "{entity} lookup",
        "{metric} trends",
        "{status} analysis"
    ],
    "keyword_categories": {
        "operations": ["production", "injection", "flowing", "down", "active", "inactive"],
        "technical": ["ESP", "BHP", "BHCIP", "PBU", "choke", "valve", "gas-lift"],
        "materials": ["oil", "gas", "water"],
        "temporal": ["daily", "monthly", "real-time", "hourly", "event-based"],
        "analysis": ["volume", "rate", "pressure", "temperature", "trend", "performance"]
    }
}

# Improved prompt that mimics hand-crafted patterns exactly
metadata_refinement_prompt = PromptTemplate(
    template="""
You are an ADNOC oil & gas data expert. Refine the metadata to match hand-crafted patterns that achieve 0.75+ MRR scores.

Table: {table_name}
Original Metadata: {original_metadata}

CRITICAL PATTERNS TO FOLLOW:

1. RELATIONSHIPS: Use EXACT table names when possible:
   - "joins with well_reservoir via UWI" 
   - "connects to well via UWI"
   - "master mapping table"
   - "time-series data for wells"

2. COMMON_QUERIES: Short concept phrases (2-4 words):
   - "production volumes by date" (time dimension)
   - "top producing wells" (ranking)
   - "injection vs production analysis" (comparison)
   - "downtime analysis" (status)
   - "pressure trends" (time series)

3. QUERY_KEYWORDS: Focus on CONCEPTS not columns (6-9 keywords):
   - Prefer: ["production", "volume", "allocation", "daily", "oil", "gas", "ESP"]
   - Avoid: ["UWI", "PRODUCTION_DATE", "MATERIAL_TYPE"] (too many columns)
   - Include: Technical terms (ESP, BHP, gas-lift) + operations (production, injection) + materials (oil, gas, water)

4. KEY_COLUMNS: Keep most important 3-4 columns only

EXAMPLES OF WINNING PATTERNS:
- Keywords: ["production", "volume", "allocation", "daily", "oil", "gas", "water", "ESP", "gas-lift"]
- Queries: ["production volumes by date", "top producing wells", "injection vs production analysis"]
- Relationships: ["joins with well_reservoir via UWI", "connects to well via UWI"]

Generate refined JSON following these EXACT patterns:
{{
    "key_columns": [3-4 most important columns only],
    "relationships": [2 concise relationships with specific table names if known],
    "common_queries": [3-4 concept phrases, not questions],
    "temporal_granularity": "{temporal_granularity}",
    "query_keywords": [6-9 concept/operation words, minimal column names]
}}

Return only the JSON.
""",
    input_variables=["table_name", "original_metadata", "temporal_granularity"]
)

# Create refinement chain
refinement_chain = metadata_refinement_prompt | llm | JsonOutputParser()

def refine_metadata(original_metadata_file="generated_table_metadata.json"):
    """Refine auto-generated metadata to be more concise and effective"""
    
    # Load original metadata
    print("Loading original metadata...")
    original_metadata = load_metadata_from_file(original_metadata_file)
    
    if not original_metadata:
        print("No original metadata found!")
        return None
    
    refined_metadata = {}
    
    print("\nRefining metadata for better performance...")
    
    for table_name, metadata in original_metadata.items():
        print(f"\nRefining: {table_name}")
        
        try:
            # Refine using LLM with patterns
            refined = refinement_chain.invoke({
                "table_name": table_name,
                "original_metadata": json.dumps(metadata, indent=2),
                "temporal_granularity": metadata.get('temporal_granularity', 'varies')
            })
            
            refined_metadata[table_name] = refined
            print(f"✓ Refined {table_name}")
            
        except Exception as e:
            print(f"✗ Error refining {table_name}: {e}")
            # Keep original if refinement fails
            refined_metadata[table_name] = metadata
    
    return refined_metadata

def save_refined_metadata(metadata, filename="hybrid_table_metadata.json"):
    """Save refined metadata to file"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nRefined metadata saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    # Refine the metadata
    refined = refine_metadata()
    
    if refined:
        # Save refined version
        save_refined_metadata(refined)
        
        # Show comparison for one table
        print("\n" + "="*60)
        print("COMPARISON - daily_allocation")
        print("="*60)
        
        original = load_metadata_from_file("generated_table_metadata.json")
        
        if original and 'daily_allocation' in original and 'daily_allocation' in refined:
            print("\nORIGINAL relationships:")
            print(original['daily_allocation']['relationships'][0])
            print("\nREFINED relationships:")
            print(refined['daily_allocation']['relationships'][0])
            
            print("\nORIGINAL queries:")
            print(original['daily_allocation']['common_queries'][:2])
            print("\nREFINED queries:")
            print(refined['daily_allocation']['common_queries'][:2])
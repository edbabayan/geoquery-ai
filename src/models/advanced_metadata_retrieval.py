"""
Advanced Metadata Retrieval Model
Uses enriched document structure with comprehensive metadata
"""

import json
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from generate_table_metadata import table_description_list_default

def create_advanced_metadata_documents() -> List[Document]:
    """Create documents with advanced metadata structure"""
    
    # Load the hybrid metadata (contains questions and other info)
    with open('hybrid_table_metadata.json', 'r') as f:
        hybrid_metadata = json.load(f)
    
    # Load the generated metadata (contains more detailed info)
    with open('generated_table_metadata.json', 'r') as f:
        generated_metadata = json.load(f)
    
    documents = []
    
    for table_desc in table_description_list_default:
        # Extract table name and description
        table_name = table_desc["table_name"].split(".")[-1]  # Get short name
        description = table_desc["table_description"]
        
        # Get data from metadata sources
        hybrid_info = hybrid_metadata.get(table_name, {})
        generated_info = generated_metadata.get(table_name, {})
        
        # Create full qualified table name (matching the example structure)
        full_table_name = f"fws_aiq_enai_dp_silver_structured_rag.wells.{table_name}"
        
        # Generate table alias
        parts = table_name.split('_')
        if len(parts) == 1:
            alias = parts[0][:3].upper()
        else:
            alias = ''.join([p[0].upper() for p in parts[:3]])
        
        # Determine data domain
        table_lower = table_name.lower()
        desc_lower = description.lower()
        
        if 'well' in table_lower or 'well' in desc_lower:
            data_domain = "Wells"
        elif 'field' in table_lower or 'field' in desc_lower:
            data_domain = "Fields"
        elif 'production' in table_lower or 'production' in desc_lower or 'allocation' in table_lower:
            data_domain = "Production"
        elif 'facility' in table_lower or 'facilities' in desc_lower:
            data_domain = "Facilities"
        elif 'operator' in table_lower or 'operator' in desc_lower:
            data_domain = "Operators"
        elif 'reservoir' in table_lower or 'reservoir' in desc_lower:
            data_domain = "Reservoirs"
        elif 'inactive' in table_lower:
            data_domain = "Operations"
        elif 'real_time' in table_lower or 'pi' in table_lower:
            data_domain = "Real-Time Data"
        else:
            data_domain = "General"
        
        # Get questions from generated metadata or create default ones
        questions = generated_info.get('common_queries', [])
        if not questions:
            questions = hybrid_info.get('common_queries', [])
        if not questions:
            questions = [
                f"What data is available in {table_name}?",
                f"How can I query {table_name} for recent records?",
                f"What are the key metrics in {table_name}?"
            ]
        
        # Format questions for page content
        formatted_questions = " - ".join([f"{q}" for q in questions[:5]])
        
        # Build enhanced page content with more keywords for better matching
        keywords_str = ", ".join(generated_info.get('query_keywords', hybrid_info.get('query_keywords', [])))
        
        # Add domain-specific keywords based on table name and description
        additional_keywords = []
        if 'allocation' in table_name:
            additional_keywords.extend(['oil production', 'gas production', 'well production', 'field production', 'top wells', 'production volumes'])
        if 'well' in table_name:
            additional_keywords.extend(['well data', 'well information', 'wellbore', 'drilling'])
        if 'reservoir' in table_name:
            additional_keywords.extend(['reservoir data', 'formation', 'pressure'])
        if 'real_time' in table_name or 'pi' in table_name:
            additional_keywords.extend(['real-time data', 'operational data', 'live data'])
        if 'inactive' in table_name:
            additional_keywords.extend(['inactive wells', 'shut-in wells', 'downtime'])
        
        all_keywords = keywords_str
        if additional_keywords:
            all_keywords += ", " + ", ".join(additional_keywords)
        
        page_content = f"Table {full_table_name} (aka {alias}). {description}. Keywords: {all_keywords}. Questions: {formatted_questions}"
        
        # Get column information for DDL (from metadata sources)
        key_columns = generated_info.get('key_columns', hybrid_info.get('key_columns', []))
        
        # Create simplified DDL
        ddl_parts = [f"CREATE TABLE [wells].[{table_name}] ("]
        
        # Add key columns first
        for col_name in key_columns[:10]:
            ddl_parts.append(f"    [{col_name}] nvarchar(255) NOT NULL,")
        
        ddl_parts.append("    ...")
        table_ddl = "\n".join(ddl_parts)
        
        # Create metadata (convert lists to strings for vector store compatibility)
        metadata = {
            "entity_type": "table",
            "table_name": table_name,  # Store short table name for evaluation compatibility
            "full_table_name": full_table_name,  # Keep full name in separate field
            "enhanced_descriptions": description,
            "data_domain": data_domain,
            "alias": alias,
            "related_questions": " | ".join(questions[:5]),  # Convert list to string
            "table_info": table_ddl,
            "key_columns": ", ".join(key_columns),  # Convert list to string
            "query_keywords": ", ".join(generated_info.get('query_keywords', hybrid_info.get('query_keywords', []))),  # Convert list to string
            "temporal_granularity": generated_info.get('temporal_granularity', hybrid_info.get('temporal_granularity', 'unknown'))
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    
    return documents

# Create retrievers with different weight configurations
def create_advanced_metadata_retriever(weights=(0.5, 0.5)):
    """Create ensemble retriever with advanced metadata documents"""
    
    # Create documents fresh each time to get latest content
    documents = create_advanced_metadata_documents()
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create vector store with unique collection name
    import time
    collection_name = f"advanced_metadata_v2_{weights[0]}_{weights[1]}_{int(time.time())}"
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name
    )
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    
    # Create vector retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=list(weights)
    )
    
    return ensemble_retriever

# Create different weight configurations
retriever_advanced_metadata_balanced = create_advanced_metadata_retriever((0.5, 0.5))
retriever_advanced_metadata_bm25 = create_advanced_metadata_retriever((0.7, 0.3))
retriever_advanced_metadata_vector = create_advanced_metadata_retriever((0.3, 0.7))
import pandas as pd
from dotenv import load_dotenv
from enhanced_retrieval import EnhancedTableRetriever
from config import CFG
from default_descriptions import table_description_list_default
from langchain_core.documents import Document
import json

# Load environment variables
load_dotenv(CFG.env_file)

# Load data
df = pd.read_excel(CFG.table_description)
ground_truth = pd.read_csv(CFG.golden_dataset)[['NL_QUERY', "TABLES"]]

# Prepare documents with enhanced descriptions
def prepare_enhanced_documents():
    """Prepare documents with table descriptions and metadata"""
    documents = []
    
    # Load previously generated questions if available
    try:
        with open('src/result_full.json', 'r') as f:
            result_full = json.load(f)
    except:
        result_full = {}
    
    for i in df.index:
        table_name = df.loc[i, "name"]
        
        # Get example questions for this table
        questions = result_full.get(table_name, [])
        questions_text = ' '.join(questions[:5]) if questions else ""
        
        # Create comprehensive description
        description = (
            f"Table name is {table_name}. "
            f"Industry terms are {df.loc[i, 'industry_terms']}. "
            f"Data granularity is {df.loc[i, 'data_granularity']}. "
            f"Main business purpose is {df.loc[i, 'main_business_purpose']}. "
            f"Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. "
            f"Unique insights are {df.loc[i, 'unique_insights']}. "
            f"Example questions: {questions_text}"
        )
        
        doc = Document(
            page_content=description,
            metadata={"table_name": table_name.split(".")[-1]}
        )
        documents.append(doc)
    
    return documents

def evaluate_retrieval_methods():
    """Compare different retrieval methods"""
    
    # Prepare documents
    print("Preparing documents...")
    documents = prepare_enhanced_documents()
    
    # Initialize enhanced retriever
    print("Initializing enhanced retriever...")
    enhanced_retriever = EnhancedTableRetriever(documents)
    
    # Test queries
    test_queries = [
        "Can you plot the wellhead pressure vs. time for the top 5 producing wells over the last 30 days?",
        "What is the total oil production by field for the current month?",
        "Show me wells with abnormal pressure readings",
        "Compare gas production rates between different reservoirs",
        "Which wells have the highest water cut?"
    ]
    
    print("\n" + "="*80)
    print("TESTING ENHANCED RETRIEVAL SYSTEM")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Test without enhancements
        print("\n1. Basic Hybrid Search (no query expansion, no reranking):")
        basic_results = enhanced_retriever.retrieve(
            query, 
            top_k=5, 
            use_query_expansion=False, 
            use_reranking=False
        )
        for i, doc in enumerate(basic_results):
            table_name = doc.metadata.get('table_name', 'Unknown')
            print(f"   {i+1}. {table_name}")
        
        # Test with query expansion only
        print("\n2. With Query Expansion:")
        expansion_results = enhanced_retriever.retrieve(
            query, 
            top_k=5, 
            use_query_expansion=True, 
            use_reranking=False
        )
        for i, doc in enumerate(expansion_results):
            table_name = doc.metadata.get('table_name', 'Unknown')
            print(f"   {i+1}. {table_name}")
        
        # Test with full enhancements
        print("\n3. With Query Expansion + Cross-Encoder Reranking:")
        full_results = enhanced_retriever.retrieve(
            query, 
            top_k=5, 
            use_query_expansion=True, 
            use_reranking=True
        )
        for i, doc in enumerate(full_results):
            table_name = doc.metadata.get('table_name', 'Unknown')
            print(f"   {i+1}. {table_name}")
        
        # Show expanded queries
        print("\n4. Query Expansions:")
        expansions = enhanced_retriever.expand_query(query)
        for i, exp in enumerate(expansions):
            print(f"   - {exp}")

def test_with_ground_truth():
    """Test retrieval against ground truth data"""
    
    # Prepare documents
    documents = prepare_enhanced_documents()
    enhanced_retriever = EnhancedTableRetriever(documents)
    
    print("\n" + "="*80)
    print("TESTING AGAINST GROUND TRUTH")
    print("="*80)
    
    # Test first 5 queries from ground truth
    for idx in range(min(5, len(ground_truth))):
        query = ground_truth.iloc[idx]['NL_QUERY']
        expected_tables = ground_truth.iloc[idx]['TABLES'].split(',') if pd.notna(ground_truth.iloc[idx]['TABLES']) else []
        
        print(f"\nQuery {idx+1}: {query}")
        print(f"Expected tables: {expected_tables}")
        
        # Retrieve with full enhancements
        results = enhanced_retriever.retrieve(query, top_k=5)
        retrieved_tables = [doc.metadata.get('table_name', '') for doc in results]
        
        print(f"Retrieved tables: {retrieved_tables}")
        
        # Check if expected tables are in results
        matches = [table for table in expected_tables if any(table in ret for ret in retrieved_tables)]
        print(f"Matches: {matches} ({len(matches)}/{len(expected_tables)})")

if __name__ == "__main__":
    # Run evaluation
    evaluate_retrieval_methods()
    
    # Test with ground truth
    test_with_ground_truth()
"""
Enhanced version of main.py using the improved retrieval system
"""
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from enhanced_retrieval import EnhancedTableRetriever
from config import CFG
from default_descriptions import table_description_list_default
import json

# Load environment variables
load_dotenv(CFG.env_file)

# Initialize LLM
llm = ChatOpenAI(model='gpt-4.1')

# Load data
df = pd.read_excel(CFG.table_description)
ground_truth = pd.read_csv(CFG.golden_dataset)[['NL_QUERY', "TABLES"]]

# Load generated questions
try:
    with open('src/result_full.json', 'r') as f:
        result_full = json.load(f)
except:
    print("Warning: result_full.json not found. Running without example questions.")
    result_full = {}

# Prepare enhanced documents
def create_enhanced_documents():
    """Create documents with comprehensive table descriptions"""
    documents = []
    
    # Use the default descriptions with full table names
    for table_info in table_description_list_default:
        full_table_name = table_info["table_name"]
        table_description = table_info["table_description"]
        
        # Extract just the table name (last part after dot)
        short_table_name = full_table_name.split(".")[-1]
        
        # Get example questions for this table if available
        questions = result_full.get(full_table_name, [])
        questions_text = ' '.join(questions[:5]) if questions else ""
        
        # Create comprehensive description
        description = (
            f"Table name is {short_table_name}. "
            f"Description: {table_description} "
            f"Example questions: {questions_text}"
        )
        
        doc = Document(
            page_content=description,
            metadata={"table_name": short_table_name}
        )
        documents.append(doc)
    
    print(f"Created documents for {len(documents)} tables: {[doc.metadata['table_name'] for doc in documents]}")
    
    return documents

# Initialize enhanced retriever
print("Initializing enhanced retrieval system...")
documents = create_enhanced_documents()
retriever = EnhancedTableRetriever(
    documents=documents,
    embed_model="text-embedding-3-large",
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# Function to retrieve tables for a query
def get_relevant_tables(query: str, top_k: int = 5, verbose: bool = False):
    """
    Retrieve relevant tables for a given query using enhanced retrieval
    
    Args:
        query: User's natural language query
        top_k: Number of tables to retrieve
        verbose: Whether to print detailed information
        
    Returns:
        List of table names
    """
    if verbose:
        print(f"\nProcessing query: {query}")
        print("-" * 50)
    
    # Retrieve with all enhancements
    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        use_query_expansion=True,
        use_reranking=True,
        alpha=0.6  # Slightly favor BM25 for keyword matching
    )
    
    # Extract table names
    table_names = []
    for i, doc in enumerate(results):
        table_name = doc.metadata.get('table_name', 'Unknown')
        table_names.append(table_name)
        
        if verbose:
            print(f"{i+1}. Table: {table_name}")
            # Show first 200 chars of description
            desc_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"   Description: {desc_preview}")
    
    return table_names

# Example usage
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Can you plot the wellhead pressure vs. time for the top 5 producing wells over the last 30 days?",
        "What is the total oil production by field for the current month?",
        "Show me wells with abnormal pressure readings",
        "Compare gas production rates between different reservoirs",
        "Which wells have the highest water cut?"
    ]
    
    print("\n" + "="*80)
    print("ENHANCED TABLE RETRIEVAL EXAMPLES")
    print("="*80)
    
    for query in test_queries:
        tables = get_relevant_tables(query, top_k=3, verbose=True)
        print(f"\nTop tables for this query: {tables}")
        print("\n" + "="*80)
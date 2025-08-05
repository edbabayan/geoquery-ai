"""
Knowledge Graph Retrieval Models for evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List
import pandas as pd

# Import knowledge graph components
from knowledge_graph_builder import OilGasKnowledgeGraphBuilder
from graph_based_retriever import GraphBasedTableRetriever
from retrieval_models import table_description_list_default, ground_truth_renamed


class KnowledgeGraphRetrieverWrapper:
    """Wrapper to make GraphBasedTableRetriever compatible with evaluate_rag_with_mrr"""
    
    def __init__(self, graph_retriever: GraphBasedTableRetriever, top_k: int = 5):
        self.graph_retriever = graph_retriever
        self.top_k = top_k
    
    def invoke(self, query: str) -> List[Document]:
        """
        Invoke method that matches the expected interface
        
        Args:
            query: The search query
            
        Returns:
            List of Document objects with table_name in metadata
        """
        return self.graph_retriever.retrieve(query, top_k=self.top_k)


# Build knowledge graph from existing metadata
print("Building knowledge graph...")
kg_builder = OilGasKnowledgeGraphBuilder()
knowledge_graph = kg_builder.build_from_metadata(
    table_descriptions=table_description_list_default,
    column_metadata=None,  # Add if available
    query_patterns=None    # Add if available
)

# Create graph-based retriever
graph_retriever_base = GraphBasedTableRetriever(knowledge_graph)

# Create wrapped versions with different configurations
retriever_kg_basic = KnowledgeGraphRetrieverWrapper(graph_retriever_base, top_k=5)

# Create a version with different LLM for entity extraction
graph_retriever_gpt4 = GraphBasedTableRetriever(knowledge_graph, llm_model="gpt-4")
retriever_kg_gpt4 = KnowledgeGraphRetrieverWrapper(graph_retriever_gpt4, top_k=5)

# Export for evaluation
__all__ = [
    'retriever_kg_basic',
    'retriever_kg_gpt4',
    'ground_truth_renamed'
]

# Optional: Create additional variants with different parameters
class KnowledgeGraphRetrieverEnhanced:
    """Enhanced version with query expansion and hybrid scoring"""
    
    def __init__(self, knowledge_graph, embed_model="text-embedding-3-large"):
        self.kg = knowledge_graph
        self.graph_retriever = GraphBasedTableRetriever(knowledge_graph)
        self.embeddings = OpenAIEmbeddings(model=embed_model)
        
    def invoke(self, query: str) -> List[Document]:
        """Enhanced retrieval with multiple strategies"""
        # Get graph-based results
        graph_docs = self.graph_retriever.retrieve(query, top_k=10)
        
        # You could add additional retrieval strategies here
        # For now, just return the graph results
        return graph_docs[:5]


# Create enhanced retriever
retriever_kg_enhanced = KnowledgeGraphRetrieverEnhanced(knowledge_graph)

# Add to exports
__all__.extend(['retriever_kg_enhanced'])
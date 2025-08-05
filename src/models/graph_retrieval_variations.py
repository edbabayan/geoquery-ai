"""
Knowledge Graph Retrieval Models with Different Description Variations
Parallel to retrieval_models.py but using graph-based approach
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Set
import pandas as pd
import networkx as nx
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import base components
from retrieval_models import table_description_list_default, ground_truth_renamed
from knowledge_graph_schema import TableKnowledgeGraph, TableNode, NodeType


class GraphRetrieverBase:
    """Base class for graph-based retrieval with embeddings"""
    
    def __init__(self, description_format: str = "default"):
        self.kg = TableKnowledgeGraph()
        self.description_format = description_format
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.table_embeddings = {}
        self._build_graph()
        
    def _build_graph(self):
        """Build knowledge graph from table descriptions"""
        # Read additional metadata
        try:
            df = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/table_description 1.xlsx")
        except:
            df = None
            
        for i, table_desc in enumerate(table_description_list_default):
            table_name = table_desc["table_name"].split(".")[-1]
            
            # Get description based on format
            if self.description_format == "default":
                description = table_desc["table_description"]
            elif self.description_format == "artem_v1" and df is not None and i < len(df):
                # Full description with all fields
                description = f"Table name is {df.loc[i, 'name']}. Industry terms are {df.loc[i, 'industry_terms']}. Data granularity is {df.loc[i, 'data_granularity']}. Main business purpose is {df.loc[i, 'main_business_purpose']}. Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}."
            elif self.description_format == "artem_v2" and df is not None and i < len(df):
                # Main purpose + unique insights only
                description = f"Table name is {df.loc[i, 'name']}. Main business purpose is {df.loc[i, 'main_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}."
            elif self.description_format == "artem_v3" and df is not None and i < len(df):
                # Full description (same as v1 but structured for consistency)
                description = f"Table name is {df.loc[i, 'name']}. Industry terms are {df.loc[i, 'industry_terms']}. Data granularity is {df.loc[i, 'data_granularity']}. Main business purpose is {df.loc[i, 'main_business_purpose']}. Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}."
            elif self.description_format == "artem_v4" and df is not None and i < len(df):
                # Minimal - main purpose + insights without "Table name is"
                description = f"Table name is {df.loc[i, 'name']}. Main business purpose is {df.loc[i, 'main_business_purpose']} Unique insights are {df.loc[i, 'unique_insights']}"
            else:
                description = table_desc["table_description"]
            
            # Create table node
            table_node = TableNode(
                name=table_name,
                full_name=table_desc["table_name"],
                description=description,
                main_business_purpose=table_desc.get("main_business_purpose", ""),
                alternative_business_purpose=table_desc.get("alternative_business_purpose"),
                industry_terms=table_desc.get("industry_terms", []),
                data_granularity=table_desc.get("data_granularity"),
                update_frequency=table_desc.get("update_frequency"),
                unique_insights=table_desc.get("unique_insights", [])
            )
            
            node_id = self.kg.add_table(table_node)
            
            # Generate and store embedding for the description
            embedding = self.sentence_model.encode(description)
            self.table_embeddings[table_name] = embedding
            
        # Add relationships between tables
        self._add_table_relationships()
        
    def _add_table_relationships(self):
        """Add known relationships between tables"""
        relationships = [
            ("daily_allocation", "well", {"daily_allocation.uwi": "well.uwi"}),
            ("daily_allocation", "well_reservoir", {"daily_allocation.uwi": "well_reservoir.uwi"}),
            ("string_event", "well", {"string_event.uwi": "well.uwi"}),
            ("well", "field", {"well.field_code": "field.field_code"}),
            ("well_reservoir", "well", {"well_reservoir.uwi": "well.uwi"}),
            ("flow_test", "well", {"flow_test.uwi": "well.uwi"}),
            ("unified_pressure_test", "well_reservoir", {
                "unified_pressure_test.uwi": "well_reservoir.uwi",
                "unified_pressure_test.reservoir": "well_reservoir.reservoir"
            }),
            ("real_time_corporate_pi", "well", {"real_time_corporate_pi.uwi": "well.uwi"}),
            ("inactive_string", "well", {"inactive_string.uwi": "well.uwi"}),
            ("well_completion", "wellbore", {"well_completion.wellbore_id": "wellbore.wellbore_id"}),
            ("wellbore", "well", {"wellbore.uwi": "well.uwi"})
        ]
        
        for source, target, join_condition in relationships:
            self.kg.add_table_relationship(source, target, join_condition, 0.9)
            
    def retrieve_with_embeddings(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve using embedding similarity and graph structure"""
        # Encode query
        query_embedding = self.sentence_model.encode(query)
        
        # Calculate similarities
        similarities = {}
        for table_name, table_embedding in self.table_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                table_embedding.reshape(1, -1)
            )[0][0]
            similarities[table_name] = similarity
            
        # Get top tables by similarity
        top_tables = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k*2]
        
        # Use graph to boost related tables
        final_scores = {}
        for table_name, sim_score in top_tables:
            final_scores[table_name] = sim_score
            
            # Boost scores for related tables
            related = self.kg.get_related_tables(table_name, max_hops=1)
            for related_table, distance in related:
                if related_table in similarities:
                    boost = 0.1 / distance  # Smaller boost for more distant tables
                    current_score = final_scores.get(related_table, similarities[related_table])
                    final_scores[related_table] = min(1.0, current_score + boost)
                    
        # Sort by final score and create documents
        sorted_tables = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        documents = []
        for table_name, score in sorted_tables:
            table_id = f"table:{table_name}"
            if table_id in self.kg.graph:
                table_data = self.kg.graph.nodes[table_id]
                doc = Document(
                    page_content=table_data.get('description', ''),
                    metadata={
                        'table_name': table_name,
                        'score': score,
                        'retrieval_method': f'graph_{self.description_format}'
                    }
                )
                documents.append(doc)
                
        return documents
    
    def invoke(self, query: str) -> List[Document]:
        """Invoke method for compatibility with evaluation framework"""
        return self.retrieve_with_embeddings(query, top_k=5)


class GraphRetrieverWithBM25:
    """Graph retriever that combines BM25 and embeddings like the original"""
    
    def __init__(self, description_format: str = "default", weights: List[float] = [0.5, 0.5]):
        self.base_retriever = GraphRetrieverBase(description_format)
        self.weights = weights  # [bm25_weight, embedding_weight]
        self.description_format = description_format
        
    def invoke(self, query: str) -> List[Document]:
        """Retrieve using both BM25 and embeddings"""
        # Get embedding-based results
        embedding_docs = self.base_retriever.retrieve_with_embeddings(query, top_k=10)
        embedding_scores = {doc.metadata['table_name']: doc.metadata['score'] 
                           for doc in embedding_docs}
        
        # Simple BM25 scoring based on word overlap
        query_words = set(query.lower().split())
        bm25_scores = {}
        
        for table_name, embedding in self.base_retriever.table_embeddings.items():
            table_id = f"table:{table_name}"
            if table_id in self.base_retriever.kg.graph:
                table_data = self.base_retriever.kg.graph.nodes[table_id]
                description = table_data.get('description', '').lower()
                desc_words = set(description.split())
                
                # Calculate BM25-like score (simplified)
                overlap = len(query_words.intersection(desc_words))
                doc_length = len(desc_words)
                
                # Simple BM25 approximation
                if doc_length > 0:
                    bm25_scores[table_name] = overlap / (1 + np.log(doc_length))
                else:
                    bm25_scores[table_name] = 0
                    
        # Normalize scores
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            if max_bm25 > 0:
                bm25_scores = {k: v/max_bm25 for k, v in bm25_scores.items()}
                
        # Combine scores
        final_scores = {}
        all_tables = set(embedding_scores.keys()).union(set(bm25_scores.keys()))
        
        for table in all_tables:
            embed_score = embedding_scores.get(table, 0)
            bm25_score = bm25_scores.get(table, 0)
            final_scores[table] = (self.weights[0] * bm25_score + 
                                  self.weights[1] * embed_score)
            
        # Create final documents
        sorted_tables = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        documents = []
        for table_name, score in sorted_tables:
            table_id = f"table:{table_name}"
            if table_id in self.base_retriever.kg.graph:
                table_data = self.base_retriever.kg.graph.nodes[table_id]
                doc = Document(
                    page_content=table_data.get('description', ''),
                    metadata={
                        'table_name': table_name,
                        'score': score,
                        'retrieval_method': f'graph_hybrid_{self.description_format}'
                    }
                )
                documents.append(doc)
                
        return documents


# Create retriever instances matching the original retrieval_models.py

# Default description (original table descriptions)
retriever_kg_default = GraphRetrieverWithBM25(description_format="default", weights=[0.5, 0.5])

# Artem V1 - Full descriptions with all metadata
retriever_kg_artem_v1 = GraphRetrieverWithBM25(description_format="artem_v1", weights=[0.5, 0.5])

# Artem V2 - Main purpose + unique insights only  
retriever_kg_artem_v2 = GraphRetrieverWithBM25(description_format="artem_v2", weights=[0.5, 0.5])

# Artem V3 - Full descriptions (same as V1 for consistency)
retriever_kg_artem_v3 = GraphRetrieverWithBM25(description_format="artem_v3", weights=[0.5, 0.5])

# Artem V4 - Minimal description
retriever_kg_artem_v4 = GraphRetrieverWithBM25(description_format="artem_v4", weights=[0.5, 0.5])

# Also create variants with different weight combinations for the best performer
retriever_kg_default_bm25 = GraphRetrieverWithBM25(description_format="default", weights=[0.7, 0.3])
retriever_kg_default_vector = GraphRetrieverWithBM25(description_format="default", weights=[0.3, 0.7])

# Export all retrievers
__all__ = [
    'retriever_kg_default',
    'retriever_kg_artem_v1', 
    'retriever_kg_artem_v2',
    'retriever_kg_artem_v3',
    'retriever_kg_artem_v4',
    'retriever_kg_default_bm25',
    'retriever_kg_default_vector',
    'ground_truth_renamed'
]
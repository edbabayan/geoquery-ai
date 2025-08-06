"""
Improved Knowledge Graph Retrieval Models - Step by Step Improvements
Each class represents a progressive improvement over the original graph approach
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Set
import pandas as pd
import networkx as nx
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import base components
from retrieval_models import table_description_list_default, ground_truth_renamed
from knowledge_graph_schema import TableKnowledgeGraph, TableNode, NodeType
from questions_generation import result_full


class ImprovedKGRetrieverStep1:
    """Step 1: Improved BM25 using proper BM25Retriever instead of word overlap"""
    
    def __init__(self, description_format: str = "default_with_questions", weights: List[float] = [0.3, 0.7]):
        self.kg = TableKnowledgeGraph()
        self.description_format = description_format
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.table_embeddings = {}
        self.weights = weights  # [bm25_weight, embedding_weight]
        self.documents = []
        self._build_graph_and_documents()
        self._setup_retrievers()
        
    def _build_graph_and_documents(self):
        """Build knowledge graph and create documents for retrievers"""
        # Read additional metadata
        try:
            df = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/table_description 1.xlsx")
        except:
            df = None
            
        for i, table_desc in enumerate(table_description_list_default):
            table_name = table_desc["table_name"].split(".")[-1]
            
            # Get description based on format (same as original)
            if self.description_format == "default":
                description = table_desc["table_description"]
            elif self.description_format == "default_with_questions":
                # Default description + generated questions
                base_description = table_desc["table_description"]
                table_questions = result_full.get(table_name, [])
                questions_text = " Example questions: " + " ".join(table_questions[:5]) if table_questions else ""
                description = f"{base_description}{questions_text}"
            else:
                description = table_desc["table_description"]
            
            # Create table node for graph
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
            
            # Create document for retrievers
            doc = Document(
                page_content=description,
                metadata={"table_name": table_name}
            )
            self.documents.append(doc)
            
            # Generate and store embedding
            embedding = self.embeddings.embed_query(description)
            self.table_embeddings[table_name] = np.array(embedding)
            
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
    
    def _setup_retrievers(self):
        """Setup proper BM25 and vector retrievers"""
        # BM25 retriever using proper implementation
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 10  # Get more candidates for ensemble
        
        # Vector retriever - still using manual approach for now (will improve in step 2)
        # This maintains consistency with current approach while improving BM25
        
    def invoke(self, query: str) -> List[Document]:
        """Retrieve using improved BM25 + manual vector + graph boost"""
        # Get BM25 results (proper implementation)
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_scores = {doc.metadata['table_name']: 1.0 / (i + 1) for i, doc in enumerate(bm25_docs)}
        
        # Get embedding-based results
        query_embedding = np.array(self.embeddings.embed_query(query))
        embedding_scores = {}
        
        for table_name, table_embedding in self.table_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                table_embedding.reshape(1, -1)
            )[0][0]
            embedding_scores[table_name] = similarity
            
        # Normalize embedding scores
        if embedding_scores:
            max_embed = max(embedding_scores.values())
            if max_embed > 0:
                embedding_scores = {k: v/max_embed for k, v in embedding_scores.items()}
        
        # Get all unique tables from both retrievers
        all_tables = set(bm25_scores.keys()).union(set(embedding_scores.keys()))
        
        # Combine scores with graph boost
        final_scores = {}
        for table_name in all_tables:
            bm25_score = bm25_scores.get(table_name, 0)
            embed_score = embedding_scores.get(table_name, 0)
            
            # Basic ensemble combination
            base_score = (self.weights[0] * bm25_score + self.weights[1] * embed_score)
            
            # Apply light graph boost
            graph_boost = self._get_graph_boost(table_name, query)
            final_scores[table_name] = base_score + graph_boost
            
        # Sort and create final documents
        sorted_tables = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
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
                        'retrieval_method': f'improved_kg_step1_{self.description_format}'
                    }
                )
                documents.append(doc)
                
        return documents
    
    def _get_graph_boost(self, table_name: str, query: str) -> float:
        """Apply light graph boost based on relationships"""
        boost = 0.0
        
        # Get related tables
        related = self.kg.get_related_tables(table_name, max_hops=1)
        if related:
            # Small boost for tables with many relationships
            boost = min(0.05, len(related) * 0.01)
            
        return boost


class ImprovedKGRetrieverStep2:
    """Step 2: Proper BM25 + Optimized Vector Search using Chroma"""
    
    def __init__(self, description_format: str = "default_with_questions", weights: List[float] = [0.3, 0.7]):
        self.kg = TableKnowledgeGraph()
        self.description_format = description_format
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.weights = weights
        self.documents = []
        self._build_graph_and_documents()
        self._setup_retrievers()
        
    def _build_graph_and_documents(self):
        """Build knowledge graph and create documents"""
        # Same as Step 1
        try:
            df = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/table_description 1.xlsx")
        except:
            df = None
            
        for i, table_desc in enumerate(table_description_list_default):
            table_name = table_desc["table_name"].split(".")[-1]
            
            if self.description_format == "default":
                description = table_desc["table_description"]
            elif self.description_format == "default_with_questions":
                base_description = table_desc["table_description"]
                table_questions = result_full.get(table_name, [])
                questions_text = " Example questions: " + " ".join(table_questions[:5]) if table_questions else ""
                description = f"{base_description}{questions_text}"
            else:
                description = table_desc["table_description"]
            
            # Create table node for graph
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
            
            # Create document
            doc = Document(
                page_content=description,
                metadata={"table_name": table_name}
            )
            self.documents.append(doc)
            
        # Add relationships
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
    
    def _setup_retrievers(self):
        """Setup proper BM25 and Chroma vector retrievers"""
        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 10
        
        # Chroma vector retriever
        self.vectorstore = Chroma.from_documents(
            self.documents, 
            self.embeddings, 
            collection_name="improved_kg_step2"
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
    def invoke(self, query: str) -> List[Document]:
        """Retrieve using proper BM25 + Chroma vector + graph boost"""
        # Get BM25 results
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_scores = {doc.metadata['table_name']: 1.0 / (i + 1) for i, doc in enumerate(bm25_docs)}
        
        # Get vector results from Chroma
        vector_docs = self.vector_retriever.invoke(query)
        vector_scores = {doc.metadata['table_name']: 1.0 / (i + 1) for i, doc in enumerate(vector_docs)}
        
        # Get all unique tables
        all_tables = set(bm25_scores.keys()).union(set(vector_scores.keys()))
        
        # Combine scores with graph boost
        final_scores = {}
        for table_name in all_tables:
            bm25_score = bm25_scores.get(table_name, 0)
            vector_score = vector_scores.get(table_name, 0)
            
            # Ensemble combination
            base_score = (self.weights[0] * bm25_score + self.weights[1] * vector_score)
            
            # Apply light graph boost
            graph_boost = self._get_graph_boost(table_name, query)
            final_scores[table_name] = base_score + graph_boost
            
        # Sort and create final documents
        sorted_tables = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        documents = []
        for table_name, score in sorted_tables:
            # Find the original document content
            doc_content = ""
            for doc in self.documents:
                if doc.metadata['table_name'] == table_name:
                    doc_content = doc.page_content
                    break
                    
            doc = Document(
                page_content=doc_content,
                metadata={
                    'table_name': table_name,
                    'score': score,
                    'retrieval_method': f'improved_kg_step2_{self.description_format}'
                }
            )
            documents.append(doc)
                
        return documents
    
    def _get_graph_boost(self, table_name: str, query: str) -> float:
        """Apply light graph boost based on relationships"""
        boost = 0.0
        
        # Get related tables
        related = self.kg.get_related_tables(table_name, max_hops=1)
        if related:
            # Small boost for tables with many relationships
            boost = min(0.05, len(related) * 0.01)
            
        return boost


# Create model instances for evaluation
retriever_improved_kg_step1 = ImprovedKGRetrieverStep1(
    description_format="default_with_questions", 
    weights=[0.3, 0.7]
)

retriever_improved_kg_step2 = ImprovedKGRetrieverStep2(
    description_format="default_with_questions", 
    weights=[0.3, 0.7]
)

# Export retrievers
__all__ = [
    'retriever_improved_kg_step1',
    'retriever_improved_kg_step2'
]
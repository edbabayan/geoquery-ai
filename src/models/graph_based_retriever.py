"""
Graph-based retrieval system using knowledge graph
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import networkx as nx
from collections import defaultdict
import re

from knowledge_graph_schema import TableKnowledgeGraph, NodeType, EdgeType


class GraphBasedTableRetriever:
    """Retriever that uses knowledge graph structure for table discovery"""
    
    def __init__(self, 
                 knowledge_graph: TableKnowledgeGraph,
                 llm_model: str = "gpt-4-turbo"):
        """
        Initialize graph-based retriever
        
        Args:
            knowledge_graph: The constructed knowledge graph
            llm_model: LLM model for entity extraction
        """
        self.kg = knowledge_graph
        self.entity_extractor_llm = ChatOpenAI(model=llm_model, temperature=0)
        self._setup_entity_extraction()
        
    def _setup_entity_extraction(self):
        """Set up entity extraction chain"""
        self.entity_extraction_prompt = PromptTemplate(
            template="""You are an oil and gas data expert. Extract entities from the user query that match our database schema.

Extract the following types of entities:
1. Field names (e.g., BAB, ASAB, SAHIL, ZAKUM)
2. Well identifiers (e.g., BA-234, SH118, ZK-8)
3. Reservoir names (e.g., KHARAIB-1, ARAB-A, THAMAMA)
4. Material types (oil, gas, water)
5. Time periods (last month, 2024, past 30 days)
6. Measurement types (pressure, temperature, production, flow rate)

Query: {query}

Output as JSON with keys: fields, wells, reservoirs, materials, time_periods, measurements
Each key should have a list of extracted entities (empty list if none found).""",
            input_variables=["query"]
        )
        
        self.entity_extraction_chain = (
            self.entity_extraction_prompt | 
            self.entity_extractor_llm | 
            JsonOutputParser()
        )
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract domain entities from query"""
        try:
            # Use LLM for sophisticated extraction
            entities = self.entity_extraction_chain.invoke({"query": query})
        except:
            # Fallback to pattern matching
            entities = self._extract_entities_regex(query)
        
        return entities
    
    def _extract_entities_regex(self, query: str) -> Dict[str, List[str]]:
        """Fallback entity extraction using regex patterns"""
        entities = {
            "fields": [],
            "wells": [],
            "reservoirs": [],
            "materials": [],
            "time_periods": [],
            "measurements": []
        }
        
        # Field patterns
        field_pattern = r'\b(BAB|ASAB|BU HASA|BUHASA|SAHIL|ZAKUM|UMM SHAIF|RUMAITHA)\b'
        entities["fields"] = re.findall(field_pattern, query.upper())
        
        # Well patterns (letter-number combinations)
        well_pattern = r'\b([A-Z]{2,3}[-\s]?\d{1,4})\b'
        entities["wells"] = re.findall(well_pattern, query.upper())
        
        # Material types
        material_pattern = r'\b(oil|gas|water|injection)\b'
        entities["materials"] = re.findall(material_pattern, query.lower())
        
        # Measurements
        measurement_pattern = r'\b(pressure|temperature|production|flow rate|GOR|water cut|BHP|choke)\b'
        entities["measurements"] = re.findall(measurement_pattern, query.lower())
        
        return entities
    
    def find_tables_by_entities(self, entities: Dict[str, List[str]]) -> Set[str]:
        """Find tables connected to extracted entities"""
        relevant_tables = set()
        
        # Map entity types to node prefixes
        entity_mappings = {
            "fields": "field:",
            "reservoirs": "reservoir:",
            "wells": "well:"  # Note: wells are usually referenced in tables, not as nodes
        }
        
        # Find tables connected to entity nodes
        for entity_type, prefix in entity_mappings.items():
            for entity in entities.get(entity_type, []):
                # Try to find the entity node
                entity_node_id = f"{prefix}{entity}"
                if entity_node_id in self.kg.graph:
                    # Find all table nodes connected to this entity
                    for neighbor in nx.descendants(self.kg.graph, entity_node_id):
                        if self.kg.graph.nodes[neighbor].get('node_type') == NodeType.TABLE.value:
                            table_name = self.kg.graph.nodes[neighbor].get('name')
                            relevant_tables.add(table_name)
        
        # Handle materials and measurements differently (they relate to columns/tables)
        if entities.get("materials"):
            # Tables that typically contain material-related data
            material_tables = ["daily_allocation", "flow_test", "well_allowable_limits"]
            relevant_tables.update(material_tables)
        
        if entities.get("measurements"):
            measurement_table_mapping = {
                "pressure": ["unified_pressure_test", "real_time_corporate_pi", "string_event"],
                "temperature": ["real_time_corporate_pi", "string_event", "flow_test"],
                "production": ["daily_allocation", "flow_test"],
                "flow rate": ["daily_allocation", "flow_test"],
                "gor": ["flow_test"],
                "water cut": ["flow_test"],
                "bhp": ["unified_pressure_test"],
                "choke": ["string_event", "flow_test", "real_time_corporate_pi"]
            }
            
            for measurement in entities.get("measurements", []):
                if measurement in measurement_table_mapping:
                    relevant_tables.update(measurement_table_mapping[measurement])
        
        return relevant_tables
    
    def find_tables_by_pattern(self, query: str) -> Set[str]:
        """Find tables based on query patterns"""
        relevant_tables = set()
        
        # Get all query pattern nodes
        pattern_nodes = self.kg.get_nodes_by_type(NodeType.QUERY_PATTERN)
        
        # Simple keyword matching for patterns
        query_lower = query.lower()
        pattern_keywords = {
            "production_analysis": ["production", "output", "yield", "volume", "produce"],
            "well_status": ["status", "inactive", "down", "operational", "flowing"],
            "pressure_monitoring": ["pressure", "psi", "bhp", "wellhead pressure"],
            "flow_test_analysis": ["flow test", "gor", "water cut", "gas oil ratio"]
        }
        
        for pattern_node_id in pattern_nodes:
            pattern_data = self.kg.graph.nodes[pattern_node_id]
            pattern_name = pattern_data.get('pattern')
            
            # Check if query matches pattern keywords
            if pattern_name in pattern_keywords:
                if any(keyword in query_lower for keyword in pattern_keywords[pattern_name]):
                    # Get tables associated with this pattern
                    for successor in self.kg.graph.successors(pattern_node_id):
                        edge_data = self.kg.graph.get_edge_data(pattern_node_id, successor)
                        if edge_data.get('relation') == EdgeType.USED_IN_QUERY.value:
                            table_name = self.kg.graph.nodes[successor].get('name')
                            relevant_tables.add(table_name)
        
        return relevant_tables
    
    def expand_tables_by_relationships(self, 
                                     seed_tables: Set[str], 
                                     max_hops: int = 1) -> List[Tuple[str, float]]:
        """Expand table set using graph relationships"""
        table_scores = defaultdict(float)
        
        # Initialize seed tables with high scores
        for table in seed_tables:
            table_scores[table] = 1.0
        
        # Expand using graph relationships
        for table in seed_tables:
            table_id = f"table:{table}"
            if table_id not in self.kg.graph:
                continue
            
            # Find related tables
            related = self.kg.get_related_tables(table, max_hops)
            for related_table, distance in related:
                # Score based on distance (closer = higher score)
                score = 1.0 / (distance + 1)
                table_scores[related_table] = max(table_scores[related_table], score * 0.5)
        
        # Sort by score
        return sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    
    def score_tables_by_relevance(self, 
                                 tables: List[Tuple[str, float]], 
                                 query: str,
                                 entities: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        """Score tables based on multiple relevance factors"""
        scored_tables = []
        
        for table_name, graph_score in tables:
            table_id = f"table:{table_name}"
            if table_id not in self.kg.graph:
                continue
            
            table_data = self.kg.graph.nodes[table_id]
            
            # Calculate relevance scores
            scores = {
                'graph': graph_score,
                'description': self._calculate_description_match(table_data, query),
                'entity': self._calculate_entity_match(table_name, entities),
                'centrality': self._calculate_centrality_score(table_id)
            }
            
            # Weighted combination
            final_score = (
                0.3 * scores['graph'] +
                0.3 * scores['description'] +
                0.2 * scores['entity'] +
                0.2 * scores['centrality']
            )
            
            scored_tables.append((table_name, final_score))
        
        return sorted(scored_tables, key=lambda x: x[1], reverse=True)
    
    def _calculate_description_match(self, table_data: Dict[str, Any], query: str) -> float:
        """Calculate how well table description matches query"""
        description = table_data.get('description', '').lower()
        purpose = table_data.get('main_business_purpose', '').lower()
        terms = ' '.join(table_data.get('industry_terms', [])).lower()
        
        combined_text = f"{description} {purpose} {terms}"
        query_words = set(query.lower().split())
        
        # Simple word overlap score
        text_words = set(combined_text.split())
        overlap = len(query_words.intersection(text_words))
        
        return min(overlap / max(len(query_words), 1), 1.0)
    
    def _calculate_entity_match(self, table_name: str, entities: Dict[str, List[str]]) -> float:
        """Calculate if table is likely to contain extracted entities"""
        score = 0.0
        
        # Check if table typically contains wells (most tables reference wells)
        if entities.get("wells") and table_name in [
            "daily_allocation", "well", "string_event", "flow_test", 
            "unified_pressure_test", "real_time_corporate_pi"
        ]:
            score += 0.5
        
        # Check for field/reservoir relationships
        if (entities.get("fields") or entities.get("reservoirs")) and table_name in [
            "well", "well_reservoir", "daily_allocation", "field"
        ]:
            score += 0.5
        
        return min(score, 1.0)
    
    def _calculate_centrality_score(self, table_id: str) -> float:
        """Calculate importance of table based on graph centrality"""
        try:
            # Use degree centrality as a measure of table importance
            centrality = nx.degree_centrality(self.kg.graph)
            return centrality.get(table_id, 0.0)
        except:
            return 0.0
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Main retrieval method using graph structure
        
        Args:
            query: User query
            top_k: Number of tables to retrieve
            
        Returns:
            List of Document objects with table information
        """
        # 1. Extract entities from query
        entities = self.extract_entities(query)
        
        # 2. Find tables by entities
        entity_tables = self.find_tables_by_entities(entities)
        
        # 3. Find tables by query patterns
        pattern_tables = self.find_tables_by_pattern(query)
        
        # 4. Combine and expand
        seed_tables = entity_tables.union(pattern_tables)
        
        # If no seed tables found, use most central tables
        if not seed_tables:
            all_table_ids = self.kg.get_nodes_by_type(NodeType.TABLE)
            centrality = nx.degree_centrality(self.kg.graph)
            top_central = sorted(all_table_ids, key=lambda x: centrality.get(x, 0), reverse=True)[:3]
            seed_tables = {self.kg.graph.nodes[tid]['name'] for tid in top_central}
        
        # 5. Expand using relationships
        expanded_tables = self.expand_tables_by_relationships(seed_tables, max_hops=1)
        
        # 6. Score and rank
        scored_tables = self.score_tables_by_relevance(expanded_tables, query, entities)
        
        # 7. Create documents for top-k tables
        documents = []
        for table_name, score in scored_tables[:top_k]:
            table_id = f"table:{table_name}"
            if table_id in self.kg.graph:
                table_data = self.kg.graph.nodes[table_id]
                
                # Get columns for additional context
                columns = self.kg.get_table_columns(table_name)
                column_names = [col['name'] for col in columns[:10]]  # Limit to 10 columns
                
                # Create enriched document
                content = (
                    f"Table: {table_name}\n"
                    f"Description: {table_data.get('description', '')}\n"
                    f"Purpose: {table_data.get('main_business_purpose', '')}\n"
                    f"Key columns: {', '.join(column_names)}\n"
                    f"Industry terms: {', '.join(table_data.get('industry_terms', []))}"
                )
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'table_name': table_name,
                        'full_name': table_data.get('full_name', ''),
                        'score': score,
                        'retrieval_method': 'graph',
                        'entities_matched': entities
                    }
                )
                documents.append(doc)
        
        return documents
    
    def explain_retrieval(self, query: str, table_name: str) -> str:
        """Explain why a particular table was retrieved for a query"""
        entities = self.extract_entities(query)
        table_id = f"table:{table_name}"
        
        explanation_parts = []
        
        # Check entity connections
        if entities:
            explanation_parts.append(f"Query entities found: {entities}")
        
        # Check pattern matches
        pattern_matches = self.find_tables_by_pattern(query)
        if table_name in pattern_matches:
            explanation_parts.append("Table matches query pattern")
        
        # Check relationships
        if table_id in self.kg.graph:
            # Find shortest paths to entity nodes
            for field in entities.get('fields', []):
                field_id = f"field:{field}"
                if field_id in self.kg.graph:
                    path = self.kg.find_join_path(field_id, table_id)
                    if path:
                        explanation_parts.append(f"Connected to field {field} via: {path}")
        
        return "\n".join(explanation_parts) if explanation_parts else "No specific connections found"
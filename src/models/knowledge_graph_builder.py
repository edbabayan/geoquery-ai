"""
Knowledge Graph Builder for Oil & Gas Table Metadata
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from knowledge_graph_schema import (
    TableKnowledgeGraph, 
    TableNode, 
    ColumnNode, 
    FieldNode, 
    ReservoirNode,
    QueryPatternNode
)


class OilGasKnowledgeGraphBuilder:
    """Builder for constructing knowledge graph from oil & gas metadata"""
    
    def __init__(self):
        self.kg = TableKnowledgeGraph()
        
    def build_from_metadata(self, 
                           table_descriptions: List[Dict[str, Any]],
                           column_metadata: Optional[pd.DataFrame] = None,
                           query_patterns: Optional[Dict[str, List[str]]] = None) -> TableKnowledgeGraph:
        """Build knowledge graph from metadata sources"""
        
        # 1. Add table nodes
        for table_desc in table_descriptions:
            self._add_table_from_description(table_desc)
        
        # 2. Add column nodes if available
        if column_metadata is not None:
            self._add_columns_from_metadata(column_metadata)
        
        # 3. Add domain entities (fields, reservoirs)
        self._add_domain_entities()
        
        # 4. Add query patterns if available
        if query_patterns:
            self._add_query_patterns(query_patterns)
        
        # 5. Infer and add relationships
        self._infer_relationships()
        
        return self.kg
    
    def _add_table_from_description(self, table_desc: Dict[str, Any]):
        """Add a table node from description dictionary"""
        table_name = table_desc["table_name"].split(".")[-1]  # Get table name without schema
        
        table_node = TableNode(
            name=table_name,
            full_name=table_desc["table_name"],
            description=table_desc.get("table_description", ""),
            main_business_purpose=table_desc.get("main_business_purpose", ""),
            alternative_business_purpose=table_desc.get("alternative_business_purpose"),
            industry_terms=table_desc.get("industry_terms", []),
            data_granularity=table_desc.get("data_granularity"),
            update_frequency=table_desc.get("update_frequency"),
            unique_insights=table_desc.get("unique_insights", [])
        )
        
        self.kg.add_table(table_node)
    
    def _add_columns_from_metadata(self, column_df: pd.DataFrame):
        """Add column nodes from DataFrame metadata"""
        for _, row in column_df.iterrows():
            column_node = ColumnNode(
                name=row['column_name'],
                table_name=row['table_name'],
                data_type=row.get('data_type', 'unknown'),
                description=row.get('description', ''),
                is_primary_key=row.get('is_primary_key', False),
                is_foreign_key=row.get('is_foreign_key', False),
                references_table=row.get('references_table'),
                references_column=row.get('references_column'),
                example_values=row.get('example_values', []),
                unit_of_measure=row.get('unit_of_measure')
            )
            
            self.kg.add_column(column_node)
    
    def _add_domain_entities(self):
        """Add oil & gas domain entities based on known patterns"""
        # Add known fields
        known_fields = [
            {"name": "BAB", "code": "BAB", "company": "ADNOC", "location_type": "onshore"},
            {"name": "ASAB", "code": "ASB", "company": "ADNOC", "location_type": "onshore"},
            {"name": "BU HASA", "code": "BH", "company": "ADNOC", "location_type": "onshore"},
            {"name": "SAHIL", "code": "SH", "company": "ADNOC", "location_type": "onshore"},
            {"name": "ZAKUM", "code": "ZK", "company": "ADNOC", "location_type": "offshore"},
            {"name": "UMM SHAIF", "code": "US", "company": "ADNOC", "location_type": "offshore"},
            {"name": "RUMAITHA", "code": "RA", "company": "ADNOC", "location_type": "onshore"}
        ]
        
        for field_data in known_fields:
            field_node = FieldNode(**field_data)
            self.kg.add_field(field_node)
        
        # Add known reservoirs
        known_reservoirs = [
            {"name": "KHARAIB-1", "field_name": "BAB", "formation": "Kharaib"},
            {"name": "KHARAIB-2", "field_name": "BAB", "formation": "Kharaib"},
            {"name": "ARAB-A", "field_name": "ASAB", "formation": "Arab"},
            {"name": "ARAB-C", "field_name": "ASAB", "formation": "Arab"},
            {"name": "THAMAMA", "field_name": "BU HASA", "formation": "Thamama"},
            {"name": "UPPER ZAKUM", "field_name": "ZAKUM", "formation": "Arab"}
        ]
        
        for reservoir_data in known_reservoirs:
            reservoir_node = ReservoirNode(**reservoir_data)
            self.kg.add_reservoir(reservoir_node)
    
    def _add_query_patterns(self, query_patterns: Dict[str, List[str]]):
        """Add query patterns from the provided examples"""
        pattern_categories = {
            "production_analysis": {
                "pattern": "production_analysis",
                "description": "Queries about oil/gas/water production volumes and rates",
                "complexity": "medium",
                "tables": ["daily_allocation", "well", "field"],
                "examples": [
                    "What is the total oil production for well X?",
                    "Show production trends for field Y"
                ]
            },
            "well_status": {
                "pattern": "well_status",
                "description": "Queries about well operational status and events",
                "complexity": "simple",
                "tables": ["string_event", "inactive_string", "well"],
                "examples": [
                    "What is the current status of well X?",
                    "Show inactive wells in field Y"
                ]
            },
            "pressure_monitoring": {
                "pattern": "pressure_monitoring",
                "description": "Queries about pressure tests and real-time pressure data",
                "complexity": "complex",
                "tables": ["unified_pressure_test", "real_time_corporate_pi"],
                "examples": [
                    "Show BHP surveys for well X",
                    "Display wellhead pressure trends"
                ]
            },
            "flow_test_analysis": {
                "pattern": "flow_test_analysis",
                "description": "Queries about flow test results and performance",
                "complexity": "complex",
                "tables": ["flow_test", "well_reservoir"],
                "examples": [
                    "What is the GOR for well X?",
                    "Show water cut progression"
                ]
            }
        }
        
        for pattern_key, pattern_data in pattern_categories.items():
            # Count frequency from actual query examples
            frequency = len(query_patterns.get(pattern_data['tables'][0], []))
            
            pattern_node = QueryPatternNode(
                pattern=pattern_data['pattern'],
                description=pattern_data['description'],
                complexity=pattern_data['complexity'],
                tables_involved=pattern_data['tables'],
                example_queries=pattern_data['examples'],
                frequency_score=frequency / 100.0  # Normalize
            )
            
            self.kg.add_query_pattern(pattern_node)
    
    def _infer_relationships(self):
        """Infer relationships between tables based on common patterns"""
        # Define common join relationships in oil & gas data
        table_relationships = [
            {
                "source": "daily_allocation",
                "target": "well",
                "join_condition": {"daily_allocation.uwi": "well.uwi"},
                "strength": 0.9
            },
            {
                "source": "daily_allocation",
                "target": "well_reservoir",
                "join_condition": {"daily_allocation.uwi": "well_reservoir.uwi"},
                "strength": 0.8
            },
            {
                "source": "string_event",
                "target": "well",
                "join_condition": {"string_event.uwi": "well.uwi"},
                "strength": 0.9
            },
            {
                "source": "well",
                "target": "field",
                "join_condition": {"well.field_code": "field.field_code"},
                "strength": 0.9
            },
            {
                "source": "well_reservoir",
                "target": "well",
                "join_condition": {"well_reservoir.uwi": "well.uwi"},
                "strength": 1.0
            },
            {
                "source": "flow_test",
                "target": "well",
                "join_condition": {"flow_test.uwi": "well.uwi"},
                "strength": 0.8
            },
            {
                "source": "unified_pressure_test",
                "target": "well_reservoir",
                "join_condition": {
                    "unified_pressure_test.uwi": "well_reservoir.uwi",
                    "unified_pressure_test.reservoir": "well_reservoir.reservoir"
                },
                "strength": 0.9
            },
            {
                "source": "real_time_corporate_pi",
                "target": "well",
                "join_condition": {"real_time_corporate_pi.uwi": "well.uwi"},
                "strength": 0.8
            },
            {
                "source": "inactive_string",
                "target": "well",
                "join_condition": {"inactive_string.uwi": "well.uwi"},
                "strength": 0.9
            },
            {
                "source": "well_completion",
                "target": "wellbore",
                "join_condition": {"well_completion.wellbore_id": "wellbore.wellbore_id"},
                "strength": 0.9
            },
            {
                "source": "wellbore",
                "target": "well",
                "join_condition": {"wellbore.uwi": "well.uwi"},
                "strength": 1.0
            }
        ]
        
        # Add relationships to graph
        for rel in table_relationships:
            self.kg.add_table_relationship(
                rel["source"],
                rel["target"],
                rel["join_condition"],
                rel["strength"]
            )
        
        # Add similarity relationships based on shared purpose
        self._add_similarity_relationships()
    
    def _add_similarity_relationships(self):
        """Add similarity edges between related tables"""
        similarity_groups = [
            # Production-related tables
            ["daily_allocation", "flow_test", "well_allowable_limits"],
            # Well status tables
            ["string_event", "inactive_string", "well"],
            # Pressure/sensor data
            ["unified_pressure_test", "real_time_corporate_pi"],
            # Well construction
            ["wellbore", "well_completion", "well_log_index"]
        ]
        
        for group in similarity_groups:
            # Add similarity edges between all pairs in group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    table1_id = f"table:{group[i]}"
                    table2_id = f"table:{group[j]}"
                    self.kg.add_similarity_edge(table1_id, table2_id, similarity_score=0.7)
    
    @staticmethod
    def from_existing_metadata(table_descriptions_list: List[Dict[str, Any]], 
                              excel_metadata_path: Optional[str] = None,
                              query_examples: Optional[Dict[str, List[str]]] = None) -> TableKnowledgeGraph:
        """
        Create knowledge graph from existing metadata sources
        
        Args:
            table_descriptions_list: List of table description dictionaries
            excel_metadata_path: Path to Excel file with additional metadata
            query_examples: Dictionary of table names to example queries
            
        Returns:
            Constructed TableKnowledgeGraph
        """
        builder = OilGasKnowledgeGraphBuilder()
        
        # Load Excel metadata if provided
        column_metadata = None
        if excel_metadata_path:
            try:
                column_metadata = pd.read_excel(excel_metadata_path, sheet_name='columns')
            except:
                pass
        
        return builder.build_from_metadata(
            table_descriptions=table_descriptions_list,
            column_metadata=column_metadata,
            query_patterns=query_examples
        )
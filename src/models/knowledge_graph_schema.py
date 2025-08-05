"""
Knowledge Graph Schema for Table Retrieval System
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import json


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    TABLE = "table"
    COLUMN = "column"
    FIELD = "field"
    RESERVOIR = "reservoir"
    WELL = "well"
    ASSET = "asset"
    QUERY_PATTERN = "query_pattern"
    MATERIAL_TYPE = "material_type"  # oil, gas, water
    TIME_GRANULARITY = "time_granularity"  # daily, monthly, real-time


class EdgeType(Enum):
    """Types of relationships in the knowledge graph"""
    HAS_COLUMN = "has_column"
    REFERENCES = "references"  # Foreign key relationships
    BELONGS_TO = "belongs_to"
    JOINS_WITH = "joins_with"
    USED_IN_QUERY = "used_in_query"
    SIMILAR_TO = "similar_to"
    PRODUCES = "produces"  # Well produces material
    LOCATED_IN = "located_in"  # Well located in field/reservoir
    MEASURES = "measures"  # Column measures specific attribute
    TEMPORAL_RELATION = "temporal_relation"  # Time-based relationships


@dataclass
class TableNode:
    """Node representing a database table"""
    name: str
    full_name: str  # Including schema
    description: str
    main_business_purpose: str
    alternative_business_purpose: Optional[str] = None
    industry_terms: List[str] = field(default_factory=list)
    data_granularity: Optional[str] = None
    update_frequency: Optional[str] = None
    unique_insights: List[str] = field(default_factory=list)
    row_count_estimate: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": NodeType.TABLE.value,
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "main_business_purpose": self.main_business_purpose,
            "alternative_business_purpose": self.alternative_business_purpose,
            "industry_terms": self.industry_terms,
            "data_granularity": self.data_granularity,
            "update_frequency": self.update_frequency,
            "unique_insights": self.unique_insights,
            "row_count_estimate": self.row_count_estimate
        }


@dataclass
class ColumnNode:
    """Node representing a table column"""
    name: str
    table_name: str
    data_type: str
    description: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    example_values: List[str] = field(default_factory=list)
    unit_of_measure: Optional[str] = None
    value_range: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": NodeType.COLUMN.value,
            "name": self.name,
            "table_name": self.table_name,
            "data_type": self.data_type,
            "description": self.description,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "references_table": self.references_table,
            "references_column": self.references_column,
            "example_values": self.example_values,
            "unit_of_measure": self.unit_of_measure,
            "value_range": self.value_range
        }


@dataclass
class FieldNode:
    """Node representing an oil/gas field"""
    name: str
    code: str
    company: str
    location_type: str  # onshore, offshore
    region: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": NodeType.FIELD.value,
            "name": self.name,
            "code": self.code,
            "company": self.company,
            "location_type": self.location_type,
            "region": self.region
        }


@dataclass
class ReservoirNode:
    """Node representing a reservoir"""
    name: str
    field_name: str
    formation: Optional[str] = None
    depth_range: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": NodeType.RESERVOIR.value,
            "name": self.name,
            "field_name": self.field_name,
            "formation": self.formation,
            "depth_range": self.depth_range
        }


@dataclass
class QueryPatternNode:
    """Node representing common query patterns"""
    pattern: str
    description: str
    complexity: str  # simple, medium, complex
    tables_involved: List[str] = field(default_factory=list)
    example_queries: List[str] = field(default_factory=list)
    frequency_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": NodeType.QUERY_PATTERN.value,
            "pattern": self.pattern,
            "description": self.description,
            "complexity": self.complexity,
            "tables_involved": self.tables_involved,
            "example_queries": self.example_queries,
            "frequency_score": self.frequency_score
        }


class TableKnowledgeGraph:
    """Knowledge graph for table relationships and metadata"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_index = {}  # Quick lookup by node type
        
    def add_table(self, table: TableNode) -> str:
        """Add a table node to the graph"""
        node_id = f"table:{table.name}"
        self.graph.add_node(node_id, **table.to_dict())
        self._update_index(node_id, NodeType.TABLE)
        return node_id
    
    def add_column(self, column: ColumnNode) -> str:
        """Add a column node and connect to its table"""
        node_id = f"column:{column.table_name}.{column.name}"
        table_id = f"table:{column.table_name}"
        
        self.graph.add_node(node_id, **column.to_dict())
        self._update_index(node_id, NodeType.COLUMN)
        
        # Connect to table
        if table_id in self.graph:
            self.graph.add_edge(
                table_id, 
                node_id, 
                relation=EdgeType.HAS_COLUMN.value,
                weight=1.0
            )
        
        # Add foreign key relationship if exists
        if column.is_foreign_key and column.references_table and column.references_column:
            ref_column_id = f"column:{column.references_table}.{column.references_column}"
            if ref_column_id in self.graph:
                self.graph.add_edge(
                    node_id,
                    ref_column_id,
                    relation=EdgeType.REFERENCES.value,
                    weight=1.0
                )
        
        return node_id
    
    def add_field(self, field_node: FieldNode) -> str:
        """Add a field node to the graph"""
        node_id = f"field:{field_node.code}"
        self.graph.add_node(node_id, **field_node.to_dict())
        self._update_index(node_id, NodeType.FIELD)
        return node_id
    
    def add_reservoir(self, reservoir: ReservoirNode) -> str:
        """Add a reservoir node and connect to its field"""
        node_id = f"reservoir:{reservoir.field_name}:{reservoir.name}"
        field_id = f"field:{reservoir.field_name}"
        
        self.graph.add_node(node_id, **reservoir.to_dict())
        self._update_index(node_id, NodeType.RESERVOIR)
        
        # Connect to field
        if field_id in self.graph:
            self.graph.add_edge(
                field_id,
                node_id,
                relation=EdgeType.BELONGS_TO.value,
                weight=1.0
            )
        
        return node_id
    
    def add_query_pattern(self, pattern: QueryPatternNode) -> str:
        """Add a query pattern node and connect to involved tables"""
        node_id = f"pattern:{hash(pattern.pattern) % 1000000}"
        self.graph.add_node(node_id, **pattern.to_dict())
        self._update_index(node_id, NodeType.QUERY_PATTERN)
        
        # Connect to involved tables
        for table_name in pattern.tables_involved:
            table_id = f"table:{table_name}"
            if table_id in self.graph:
                self.graph.add_edge(
                    node_id,
                    table_id,
                    relation=EdgeType.USED_IN_QUERY.value,
                    weight=pattern.frequency_score
                )
        
        return node_id
    
    def add_table_relationship(self, 
                              source_table: str, 
                              target_table: str,
                              join_condition: Dict[str, str],
                              relationship_strength: float = 1.0):
        """Add a relationship between tables"""
        source_id = f"table:{source_table}"
        target_id = f"table:{target_table}"
        
        if source_id in self.graph and target_id in self.graph:
            self.graph.add_edge(
                source_id,
                target_id,
                relation=EdgeType.JOINS_WITH.value,
                join_condition=join_condition,
                weight=relationship_strength
            )
    
    def add_similarity_edge(self, node1: str, node2: str, similarity_score: float):
        """Add similarity relationship between nodes"""
        if node1 in self.graph and node2 in self.graph:
            self.graph.add_edge(
                node1,
                node2,
                relation=EdgeType.SIMILAR_TO.value,
                weight=similarity_score
            )
    
    def _update_index(self, node_id: str, node_type: NodeType):
        """Update the node type index"""
        if node_type not in self._node_index:
            self._node_index[node_type] = set()
        self._node_index[node_type].add(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[str]:
        """Get all nodes of a specific type"""
        return list(self._node_index.get(node_type, set()))
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get all columns for a table"""
        table_id = f"table:{table_name}"
        columns = []
        
        if table_id in self.graph:
            for successor in self.graph.successors(table_id):
                edge_data = self.graph.get_edge_data(table_id, successor)
                if edge_data.get('relation') == EdgeType.HAS_COLUMN.value:
                    columns.append(self.graph.nodes[successor])
        
        return columns
    
    def get_related_tables(self, table_name: str, max_hops: int = 2) -> List[Tuple[str, int]]:
        """Get tables related to a given table within max_hops"""
        table_id = f"table:{table_name}"
        related_tables = []
        
        if table_id not in self.graph:
            return related_tables
        
        # BFS to find related tables
        visited = {table_id}
        current_level = {table_id}
        
        for hop in range(1, max_hops + 1):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        
                        # Check if it's a table node
                        if self.graph.nodes[neighbor].get('node_type') == NodeType.TABLE.value:
                            table_name = self.graph.nodes[neighbor].get('name')
                            related_tables.append((table_name, hop))
            
            current_level = next_level
        
        return related_tables
    
    def find_join_path(self, source_table: str, target_table: str) -> Optional[List[Dict[str, Any]]]:
        """Find the join path between two tables"""
        source_id = f"table:{source_table}"
        target_id = f"table:{target_table}"
        
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            join_path = []
            
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                if edge_data:
                    join_path.append({
                        'from': path[i],
                        'to': path[i + 1],
                        'relation': edge_data.get('relation'),
                        'join_condition': edge_data.get('join_condition', {})
                    })
            
            return join_path
        except nx.NetworkXNoPath:
            return None
    
    def export_to_json(self, filepath: str):
        """Export the graph to JSON format"""
        data = {
            'nodes': [
                {'id': node, **self.graph.nodes[node]} 
                for node in self.graph.nodes()
            ],
            'edges': [
                {'source': u, 'target': v, **data}
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_from_json(self, filepath: str):
        """Import graph from JSON format"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing graph
        self.graph.clear()
        self._node_index.clear()
        
        # Add nodes
        for node_data in data['nodes']:
            node_id = node_data['id']
            node_attrs = {k: v for k, v in node_data.items() if k != 'id'}
            self.graph.add_node(node_id, **node_attrs)
            
            # Update index
            node_type_str = node_attrs.get('node_type')
            if node_type_str:
                node_type = NodeType(node_type_str)
                self._update_index(node_id, node_type)
        
        # Add edges
        for edge_data in data['edges']:
            self.graph.add_edge(
                edge_data['source'],
                edge_data['target'],
                **{k: v for k, v in edge_data.items() if k not in ['source', 'target']}
            )
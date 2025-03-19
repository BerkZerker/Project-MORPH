"""
Knowledge Graph module for MORPH.

This module provides components for managing the relationships between experts,
concept specialization, and expert similarities.
"""

from src.core.knowledge_graph.graph_base import GraphBase
from src.core.knowledge_graph.graph_operations import GraphOperations
from src.core.knowledge_graph.graph_analysis import GraphAnalysis
from src.core.knowledge_graph.graph_serialization import GraphSerialization


class KnowledgeGraph(GraphBase, GraphOperations, GraphAnalysis, 
                    GraphSerialization):
    """
    Knowledge Graph for MORPH model.
    
    Manages the relationships between experts, concept specialization,
    and expert similarities. Provides advanced querying capabilities for
    expert routing and consolidation.
    
    This class provides methods for:
    1. Adding and removing experts
    2. Updating edge weights and expert activations
    3. Finding similar experts and concepts
    4. Analyzing expert importance and centrality
    5. Visualizing the expert network
    6. Saving and loading graph state
    """
    pass  # All functionality inherited from mixins


__all__ = [
    'GraphBase',
    'GraphOperations',
    'GraphAnalysis',
    'GraphSerialization',
    'KnowledgeGraph'
]

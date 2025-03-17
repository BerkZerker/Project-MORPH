"""
Knowledge Graph for MORPH model.

This module provides a thin interface to the knowledge graph components,
which manage the relationships between experts, concept specialization,
and expert similarities.
"""

from morph.core.knowledge_graph.graph_base import GraphBase
from morph.core.knowledge_graph.graph_operations import GraphOperations
from morph.core.knowledge_graph.graph_analysis import GraphAnalysis
from morph.core.knowledge_graph.graph_visualization import GraphVisualization
from morph.core.knowledge_graph.graph_serialization import GraphSerialization


class KnowledgeGraph(GraphBase, GraphOperations, GraphAnalysis, 
                    GraphVisualization, GraphSerialization):
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

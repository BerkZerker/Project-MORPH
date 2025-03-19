import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_knowledge_graph_initialization():
    """Test that knowledge graph initializes correctly."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Check graph is initialized
    assert isinstance(kg.graph, nx.Graph)
    assert len(kg.graph.nodes) == 0
    assert len(kg.graph.edges) == 0
    
    # Check concept structures
    assert isinstance(kg.concept_embeddings, dict)
    assert isinstance(kg.expert_concepts, dict)
    
    # Check concept hierarchy
    assert isinstance(kg.concept_hierarchy, nx.DiGraph)

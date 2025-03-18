import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_add_expert():
    """Test adding an expert to the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add an expert with visualization
    with capture_test_state(kg, "Add Expert"):
        kg.add_expert(
            expert_id=0,
            specialization_score=0.6,
            adaptation_rate=0.8
        )
    
    # Check expert was added
    assert 0 in kg.graph.nodes
    assert kg.graph.nodes[0]['type'] == 'expert'
    assert kg.graph.nodes[0]['specialization_score'] == 0.6
    assert kg.graph.nodes[0]['adaptation_rate'] == 0.8
    assert kg.graph.nodes[0]['activation_count'] == 0
    
    # Check expert-concept mapping was initialized
    assert 0 in kg.expert_concepts
    assert len(kg.expert_concepts[0]) == 0

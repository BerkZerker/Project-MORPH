import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph




def test_update_expert_specialization():
    """Test updating expert specialization and adaptation rate."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Add an expert
    kg.add_expert(0, specialization_score=0.5, adaptation_rate=1.0)
    
    # Update specialization
    kg.update_expert_specialization(0, specialization_score=0.8)
    
    # Check specialization was updated
    assert kg.graph.nodes[0]['specialization_score'] == 0.8
    
    # Check adaptation rate was updated based on specialization
    # Higher specialization should lead to lower adaptation rate
    assert kg.graph.nodes[0]['adaptation_rate'] < 1.0

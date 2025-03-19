import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test
def test_link_expert_to_concept():
    """Test linking experts to concepts."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add expert and concept
    kg.add_expert(0)
    embedding = torch.randn(10, device=device)
    kg.add_concept("concept1", embedding)
    
    # Link expert to concept with visualization
    with capture_test_state(kg, "Link Expert to Concept"):
        kg.link_expert_to_concept(0, "concept1", strength=0.9)
    
    # Check linking worked
    assert "concept1" in kg.expert_concepts[0]
    
    # Test with non-existent concept
    kg.link_expert_to_concept(0, "nonexistent", strength=0.5)  # Should not raise error

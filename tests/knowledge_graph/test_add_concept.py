import torch
import pytest
import networkx as nx
from src.config import MorphConfig
from src.core.knowledge_graph import KnowledgeGraph




def test_add_concept():
    """Test adding concepts to the knowledge graph."""
    config = MorphConfig()
    kg = KnowledgeGraph(config)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add a concept
    embedding = torch.randn(10, device=device)
    kg.add_concept("concept1", embedding)
    
    # Check concept was added
    assert "concept1" in kg.concept_embeddings
    assert torch.equal(kg.concept_embeddings["concept1"], embedding)
    assert "concept1" in kg.concept_hierarchy.nodes
    
    # Add a child concept
    child_embedding = torch.randn(10, device=device)
    kg.add_concept("concept1.1", child_embedding, parent_concept="concept1")
    
    # Check parent-child relationship
    assert kg.concept_hierarchy.has_edge("concept1", "concept1.1")

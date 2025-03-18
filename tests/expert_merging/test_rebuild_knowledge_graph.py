import torch
import pytest


def test_rebuild_knowledge_graph(model):
    """Test knowledge graph rebuilding after merging/pruning."""
    # Number of experts before
    num_experts_before = len(model.experts)
    
    # Add some test edges
    model.knowledge_graph.add_edge(0, 1, weight=0.5)
    model.knowledge_graph.add_edge(1, 2, weight=0.3)
    
    # Manually remove an expert (simulating pruning)
    del model.experts[2]
    
    # Update expert IDs (normally done in prune method)
    for i, expert in enumerate(model.experts):
        expert.expert_id = i
        
    # Rebuild knowledge graph
    model._rebuild_knowledge_graph()
    
    # Check that graph was properly rebuilt
    assert len(model.knowledge_graph.graph.nodes) == len(model.experts), "Graph should have same number of nodes as experts"
    assert model.knowledge_graph.graph.has_edge(0, 1), "Edge between remaining experts should be preserved"

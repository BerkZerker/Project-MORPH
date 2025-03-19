import torch
import pytest
from src.config import MorphConfig
from src.core.model import MorphModel

from src.utils.gpu_utils import get_optimal_worker_count



def test_expert_reorganization():
    """Test expert reorganization based on activation patterns."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=3,
        
        # Enable reorganization
        enable_expert_reorganization=True,
        specialization_threshold=0.7,
        overlap_threshold=0.3,
        
        # Use GPU if available, otherwise CPU
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Enable mixed precision if CUDA is available
        enable_mixed_precision=torch.cuda.is_available(),
        # Optimize data loading
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available()
    )
    
    model = MorphModel(config)
    
    # Create specialization metrics
    specialization_metrics = {
        0: {'specialization_score': 0.8, 'activation_count': 100, 'unique_inputs': 10},
        1: {'specialization_score': 0.9, 'activation_count': 50, 'unique_inputs': 5},
        2: {'specialization_score': 0.3, 'activation_count': 200, 'unique_inputs': 50}
    }
    
    # Add overlapping input distributions (experts 0 and 1 have overlap)
    common_features = {f"feature_{i}": 1 for i in range(10)}
    expert0_features = {f"feature_{i}": 1 for i in range(5, 15)}
    expert1_features = {f"feature_{i}": 1 for i in range(0, 10)}
    
    model.expert_input_distributions[0] = expert0_features
    model.expert_input_distributions[1] = expert1_features
    model.expert_input_distributions[2] = {f"feature_{i}": 1 for i in range(20, 70)}
    
    # Perform reorganization
    result = model._reorganize_experts(specialization_metrics)
    
    # Verify reorganization occurred
    assert result is True
    
    # Check that a specialization edge was created between experts 0 and 1
    assert model.knowledge_graph.graph.has_edge(0, 1)
    edge_data = model.knowledge_graph.graph.get_edge_data(0, 1)
    assert 'relation_type' in edge_data
    
    # Expert 2 should not have a specialization edge with others (no overlap)
    if model.knowledge_graph.graph.has_edge(0, 2):
        edge_data = model.knowledge_graph.graph.get_edge_data(0, 2)
        assert edge_data.get('relation_type', '') != 'specialization_split'
        
    if model.knowledge_graph.graph.has_edge(1, 2):
        edge_data = model.knowledge_graph.graph.get_edge_data(1, 2)
        assert edge_data.get('relation_type', '') != 'specialization_split'

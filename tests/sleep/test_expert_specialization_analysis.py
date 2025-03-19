import torch
import pytest
from src.config import MorphConfig
from src.core.model import MorphModel

from src.utils.gpu_utils import get_optimal_worker_count



def test_expert_specialization_analysis():
    """Test expert specialization analysis."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=3,
        
        # Use GPU if available, otherwise CPU
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Enable mixed precision if CUDA is available
        enable_mixed_precision=torch.cuda.is_available(),
        # Optimize data loading
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available()
    )
    
    model = MorphModel(config)
    
    # Initialize the expert input distributions with different patterns
    # Expert 0: Diverse inputs (low specialization)
    model.expert_input_distributions[0] = {f"feature_{i}": 1 for i in range(50)}
    
    # Expert 1: Focused inputs (high specialization)
    model.expert_input_distributions[1] = {f"feature_1": 50}
    
    # Expert 2: Moderately specialized
    model.expert_input_distributions[2] = {f"feature_{i}": 10 for i in range(5)}
    
    # Analyze specialization
    metrics = model._analyze_expert_specialization()
    
    # Check the results
    assert 0 in metrics and 1 in metrics and 2 in metrics
    
    # Expert 0 should have low specialization score (near 0)
    assert metrics[0]['specialization_score'] < 0.3
    
    # Expert 1 should have high specialization score (near 1)
    assert metrics[1]['specialization_score'] > 0.7
    
    # Expert 2 should have moderate specialization score
    assert 0.3 < metrics[2]['specialization_score'] < 0.7
    
    # Check that knowledge graph node attributes were updated
    assert 'specialization_score' in model.knowledge_graph.graph.nodes[0]
    assert 'adaptation_rate' in model.knowledge_graph.graph.nodes[0]
    
    # More specialized experts should have lower adaptation rates
    assert model.knowledge_graph.graph.nodes[0]['adaptation_rate'] > model.knowledge_graph.graph.nodes[1]['adaptation_rate']

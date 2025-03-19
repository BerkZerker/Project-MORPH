import torch
import pytest
from src.config import MorphConfig
from src.core.model import MorphModel

from src.utils.gpu_utils import get_optimal_worker_count



def test_adaptive_sleep_scheduling():
    """Test adaptive sleep scheduling."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=3,
        
        # Sleep settings
        enable_sleep=True,
        sleep_cycle_frequency=100,
        enable_adaptive_sleep=True,
        min_sleep_frequency=50,
        max_sleep_frequency=200,
        
        # Use GPU if available, otherwise CPU
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Enable mixed precision if CUDA is available
        enable_mixed_precision=torch.cuda.is_available(),
        # Optimize data loading
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available()
    )
    
    model = MorphModel(config)
    
    # Store initial values
    initial_frequency = model.adaptive_sleep_frequency
    initial_next_step = model.next_sleep_step
    
    # Update the sleep schedule
    model._update_sleep_schedule()
    
    # Check that the sleep cycle counter was incremented
    assert model.sleep_cycles_completed == 1
    
    # Check that next_sleep_step was updated
    assert model.next_sleep_step > model.step_count
    
    # Add some experts to trigger more frequent sleeping
    for _ in range(5):
        expert = model.experts[0].clone()
        expert.expert_id = len(model.experts)
        model.experts.append(expert)
        model.knowledge_graph.add_expert(
            expert.expert_id,
            specialization_score=0.5,
            adaptation_rate=1.0
        )
        # Update activation count and last_activated
        model.knowledge_graph.graph.nodes[expert.expert_id]['activation_count'] = 0
        model.knowledge_graph.graph.nodes[expert.expert_id]['last_activated'] = model.step_count
    
    # Update sleep schedule again
    model._update_sleep_schedule()
    
    # With many experts, frequency should decrease (sleep more often)
    if len(model.experts) > config.num_initial_experts * 2:
        assert model.adaptive_sleep_frequency < initial_frequency
        
    # Check bounds are respected
    assert model.adaptive_sleep_frequency >= config.min_sleep_frequency
    assert model.adaptive_sleep_frequency <= config.max_sleep_frequency

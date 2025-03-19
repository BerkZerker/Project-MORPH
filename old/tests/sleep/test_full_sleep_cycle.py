import torch
import pytest
from src.config import MorphConfig
from src.core.model import MorphModel
from src.utils.testing.decorators import visualize_test, capture_test_state
from src.utils.gpu_utils import get_optimal_worker_count


@visualize_test
def test_full_sleep_cycle():
    """Test a complete sleep cycle with all components."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=4,
        expert_k=2,
        
        # Sleep settings
        enable_sleep=True,
        sleep_cycle_frequency=100,
        enable_adaptive_sleep=True,
        
        # Expert creation
        enable_dynamic_experts=True,
        
        # Expert reorganization
        enable_expert_reorganization=True,
        
        # Use GPU if available, otherwise CPU
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Enable mixed precision if CUDA is available
        enable_mixed_precision=torch.cuda.is_available(),
        # Optimize data loading
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available()
    )
    
    model = MorphModel(config)
    
    # Create some fake activations
    for expert_idx in range(len(model.experts)):
        for _ in range(10):
            # Create inputs on the same device as the model
            inputs = torch.randn(2, config.input_size).to(model.device)
            outputs = model.experts[expert_idx](inputs)
            
            # Store CPU tensors in the buffer to avoid device issues
            model.activation_buffer.append({
                'expert_idx': expert_idx,
                'inputs': inputs.cpu(),
                'outputs': outputs.cpu(),
                'routing_weight': 1.0,
                'step': 0,
                'batch_size': inputs.size(0),
                'input_features': torch.mean(inputs, dim=0).cpu(),
                'uncertainty': 0.1
            })
    
    # Initialize experts with some specific specializations
    model.expert_input_distributions[0] = {f"feature_{i}": 1 for i in range(50)}
    model.expert_input_distributions[1] = {f"feature_1": 50}
    model.expert_input_distributions[2] = {f"feature_{i}": 10 for i in range(5)}
    model.expert_input_distributions[3] = {f"feature_{i}": 10 for i in range(4, 9)}
    
    # Initialize knowledge graph with some data
    for i in range(len(model.experts)):
        model.knowledge_graph.graph.nodes[i]['activation_count'] = 50 + i * 10
        model.knowledge_graph.graph.nodes[i]['last_activated'] = model.step_count - i * 10
    
    # Add similar weight parameters to first two experts to trigger merging
    with torch.no_grad():
        for p1, p2 in zip(model.experts[0].parameters(), model.experts[1].parameters()):
            p2.data = p1.data * 0.95 + torch.randn_like(p1.data) * 0.05
    
    # Perform sleep cycle with visualization
    with capture_test_state(model, "Full Sleep Cycle"):
        model.sleep()
    
    # Check that sleep cycle was performed
    assert model.sleep_cycles_completed == 1
    assert model.next_sleep_step > model.step_count
    
    # Verify that expert analysis was performed
    for node in model.knowledge_graph.graph.nodes:
        assert 'specialization_score' in model.knowledge_graph.graph.nodes[node]
        assert 'adaptation_rate' in model.knowledge_graph.graph.nodes[node]

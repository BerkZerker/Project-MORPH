import torch
import pytest
import numpy as np
from morph.config import MorphConfig
from morph.core.model import MorphModel


def test_sleep_cycle_initialization():
    """Test that sleep cycle metrics are properly initialized."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=3,
        
        # Sleep settings
        enable_sleep=True,
        sleep_cycle_frequency=100,
        enable_adaptive_sleep=True
    )
    
    model = MorphModel(config)
    
    # Check sleep cycle counters
    assert model.sleep_cycles_completed == 0
    assert model.next_sleep_step == config.sleep_cycle_frequency
    assert model.adaptive_sleep_frequency == config.sleep_cycle_frequency
    

def test_memory_replay():
    """Test that memory replay works as expected."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=2,
        expert_k=1,
        
        # Sleep settings
        enable_sleep=True,
        sleep_cycle_frequency=100
    )
    
    model = MorphModel(config)
    
    # Create some fake activations
    for expert_idx in range(len(model.experts)):
        for _ in range(10):
            inputs = torch.randn(2, config.input_size)
            outputs = model.experts[expert_idx](inputs)
            
            model.activation_buffer.append({
                'expert_idx': expert_idx,
                'inputs': inputs,
                'outputs': outputs,
                'routing_weight': 1.0,
                'step': 0,
                'batch_size': inputs.size(0),
                'input_features': torch.mean(inputs, dim=0),
                'uncertainty': 0.1
            })
    
    # Verify activation buffer has been populated
    assert len(model.activation_buffer) == 20
    
    # Perform memory replay
    result = model._perform_memory_replay()
    
    # Verify memory replay worked
    assert result is True
    assert len(model.activation_buffer) == 0  # Buffer should be cleared


def test_expert_specialization_analysis():
    """Test expert specialization analysis."""
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=3
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
        max_sleep_frequency=200
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
        overlap_threshold=0.3
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
        enable_expert_reorganization=True
    )
    
    model = MorphModel(config)
    
    # Create some fake activations
    for expert_idx in range(len(model.experts)):
        for _ in range(10):
            inputs = torch.randn(2, config.input_size)
            outputs = model.experts[expert_idx](inputs)
            
            model.activation_buffer.append({
                'expert_idx': expert_idx,
                'inputs': inputs,
                'outputs': outputs,
                'routing_weight': 1.0,
                'step': 0,
                'batch_size': inputs.size(0),
                'input_features': torch.mean(inputs, dim=0),
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
    
    # Perform sleep cycle
    model.sleep()
    
    # Check that sleep cycle was performed
    assert model.sleep_cycles_completed == 1
    assert model.next_sleep_step > model.step_count
    
    # Verify that expert analysis was performed
    for node in model.knowledge_graph.graph.nodes:
        assert 'specialization_score' in model.knowledge_graph.graph.nodes[node]
        assert 'adaptation_rate' in model.knowledge_graph.graph.nodes[node]

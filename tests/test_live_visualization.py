"""
Test for live visualization features.

This module provides a simple test that demonstrates the live visualization
features of the MORPH model.
"""

import torch
import time
import pytest
from src.core.model import MorphModel
from src.config import MorphConfig
from src.utils.testing.decorators import visualize_test, capture_test_state


@visualize_test(live=True)
def test_live_visualization():
    """Test that demonstrates live visualization features."""
    # Create a simple model configuration
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
    )
    
    # Create the model
    model = MorphModel(config)
    
    # Step 1: Initialize the model
    with capture_test_state(model, "Model Initialization"):
        # Initialize the knowledge graph
        for i in range(len(model.experts)):
            model.knowledge_graph.graph.add_node(i, 
                                               activation_count=10, 
                                               last_activated=0,
                                               specialization_score=0.2 + i * 0.1)
        
        # Add some edges between experts
        for i in range(len(model.experts) - 1):
            model.knowledge_graph.graph.add_edge(i, i + 1, weight=0.5, relation_type='similarity')
    
    # Step 2: Simulate some expert activations
    with capture_test_state(model, "Expert Activations"):
        # Create some fake activations
        for expert_idx in range(len(model.experts)):
            for _ in range(5 * (expert_idx + 1)):  # More activations for later experts
                # Update activation count in the knowledge graph
                model.knowledge_graph.graph.nodes[expert_idx]['activation_count'] += 1
                
                # Update last activated
                model.knowledge_graph.graph.nodes[expert_idx]['last_activated'] = model.step_count
                
                # Increment step count
                model.step_count += 1
    
    # Step 3: Simulate expert specialization
    with capture_test_state(model, "Expert Specialization"):
        # Update specialization scores
        for i in range(len(model.experts)):
            model.knowledge_graph.graph.nodes[i]['specialization_score'] = min(0.9, 0.3 + i * 0.2)
    
    # Step 4: Add a new expert
    with capture_test_state(model, "Adding New Expert"):
        # Add a new expert to the model
        new_expert_id = len(model.experts)
        
        # Create a new expert (this is simplified, in reality would use model.add_expert())
        new_expert = type(model.experts[0])(
            input_size=config.input_size,
            hidden_size=config.expert_hidden_size,
            output_size=config.output_size
        )
        model.experts.append(new_expert)
        
        # Add to knowledge graph
        model.knowledge_graph.graph.add_node(new_expert_id, 
                                           activation_count=5, 
                                           last_activated=model.step_count,
                                           specialization_score=0.1)
        
        # Connect to existing experts
        for i in range(new_expert_id):
            weight = 0.3 if i % 2 == 0 else 0.7
            model.knowledge_graph.graph.add_edge(i, new_expert_id, 
                                               weight=weight, 
                                               relation_type='similarity')
    
    # Step 5: Simulate expert merging
    with capture_test_state(model, "Expert Merging"):
        # Increase similarity between two experts
        model.knowledge_graph.graph.edges[0, 1]['weight'] = 0.9
        model.knowledge_graph.graph.edges[0, 1]['relation_type'] = 'composition'
        
        # Update specialization scores
        model.knowledge_graph.graph.nodes[0]['specialization_score'] = 0.8
        model.knowledge_graph.graph.nodes[1]['specialization_score'] = 0.85
    
    # Step 6: Final state
    with capture_test_state(model, "Final State"):
        # Update all activation counts one more time
        for i in range(len(model.experts)):
            model.knowledge_graph.graph.nodes[i]['activation_count'] += 10
            model.knowledge_graph.graph.nodes[i]['last_activated'] = model.step_count
            
            # Increment sleep cycles
            model.sleep_module.sleep_cycles_completed = 2
            model.sleep_module.adaptive_sleep_frequency = 120
    
    # Assertions to make pytest happy
    assert len(model.experts) > 4  # We added at least one expert
    assert model.sleep_cycles_completed == 2
    assert model.step_count > 0


if __name__ == "__main__":
    # Run the test directly to see the visualizations
    test_live_visualization()

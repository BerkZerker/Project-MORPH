"""
Test visualization framework for MORPH models.

This module provides the core functionality for capturing and tracking
the state of MORPH models during test execution for live visualization.
"""

import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from src.core.model import MorphModel


class TestVisualizer:
    """
    Captures and tracks the state of MORPH models during test execution.
    
    This class provides methods to track model state changes and capture key metrics
    for live visualization.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the test visualizer.
        
        Args:
            output_dir: Ignored parameter, kept for backward compatibility.
        """
        # State tracking
        self.test_name = None
        self.start_time = None
        self.end_time = None
        self.state_snapshots = []
        self.model_snapshots = []
        
    def start_test(self, test_name: str):
        """
        Start tracking a new test.
        
        Args:
            test_name: Name of the test being executed
        """
        self.test_name = test_name
        self.start_time = time.time()
        self.state_snapshots = []
        self.model_snapshots = []
        
        print(f"\n📊 Live test visualization enabled for: {test_name}")
    
    def end_test(self):
        """
        End tracking for the current test.
        """
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        print(f"\n✅ Test completed in {duration:.2f}s")
        print(f"📊 Captured {len(self.state_snapshots)} state snapshots")
    
    def capture_state(self, model: MorphModel, step_name: str, 
                     additional_data: Optional[Dict[str, Any]] = None):
        """
        Capture the current state of the model.
        
        Args:
            model: The MorphModel instance being tested
            step_name: Name of the current test step
            additional_data: Any additional data to capture
        """
        if not self.test_name:
            raise RuntimeError("Must call start_test before capturing state")
        
        # Create a snapshot of the current state
        snapshot = {
            'step_name': step_name,
            'timestamp': time.time(),
            'step_number': len(self.state_snapshots),
            'model_state': self._capture_model_state(model),
            'additional_data': additional_data or {}
        }
        
        self.state_snapshots.append(snapshot)
        
        return snapshot
    
    def _capture_model_state(self, model: MorphModel) -> Dict[str, Any]:
        """
        Capture key aspects of the model's current state.
        
        Args:
            model: The MorphModel instance
            
        Returns:
            Dictionary containing the captured state
        """
        state = {
            'num_experts': len(model.experts),
            'sleep_cycles_completed': getattr(model, 'sleep_cycles_completed', 0),
            'step_count': getattr(model, 'step_count', 0),
            'next_sleep_step': getattr(model, 'next_sleep_step', None),
            'adaptive_sleep_frequency': getattr(model, 'adaptive_sleep_frequency', None),
            'activation_buffer_size': len(getattr(model, 'activation_buffer', [])),
            'knowledge_graph_nodes': len(model.knowledge_graph.graph.nodes),
            'knowledge_graph_edges': len(model.knowledge_graph.graph.edges),
        }
        
        # Capture expert states
        expert_states = {}
        for i, expert in enumerate(model.experts):
            expert_states[i] = {
                'expert_id': expert.expert_id,
                'activation_count': expert.activation_count,
                'last_activated': expert.last_activated,
            }
        
        state['expert_states'] = expert_states
        
        # Capture knowledge graph node attributes
        kg_node_attrs = {}
        for node in model.knowledge_graph.graph.nodes:
            kg_node_attrs[node] = dict(model.knowledge_graph.graph.nodes[node])
        
        state['kg_node_attrs'] = kg_node_attrs
        
        return state
# Global instance for use by decorators
_default_visualizer = TestVisualizer()

def get_default_visualizer() -> TestVisualizer:
    """Get the default TestVisualizer instance."""
    return _default_visualizer

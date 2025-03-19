"""
Stub implementation of test visualization framework for MORPH models.

This module provides empty implementations of the visualization classes
to maintain backward compatibility with existing tests.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from src.core.model import MorphModel


class TestVisualizer:
    """
    Stub implementation of TestVisualizer that does nothing.
    
    This is a placeholder for the original TestVisualizer class
    to maintain backward compatibility.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the test visualizer."""
        self.test_name = None
        self.start_time = None
        self.end_time = None
        self.state_snapshots = []
        
    def start_test(self, test_name: str):
        """Start tracking a new test."""
        self.test_name = test_name
    
    def end_test(self):
        """End tracking for the current test."""
        pass
    
    def capture_state(self, model: MorphModel, step_name: str, 
                     additional_data: Optional[Dict[str, Any]] = None):
        """Capture the current state of the model."""
        return {}
    
    def _capture_model_state(self, model: MorphModel) -> Dict[str, Any]:
        """Capture key aspects of the model's current state."""
        return {}


# Global instance for use by decorators
_default_visualizer = TestVisualizer()

def get_default_visualizer() -> TestVisualizer:
    """Get the default TestVisualizer instance."""
    return _default_visualizer

"""
Decorators for live test visualization in MORPH models.

This module provides decorators and context managers that make it easy
to add live visualization capabilities to tests.
"""

import time
import inspect
import functools
import contextlib
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from src.core.model import MorphModel
from src.utils.testing.visualizer import get_default_visualizer, TestVisualizer
from src.utils.testing import live_server, progress_tracker, live_visualizer


def visualize_test(func=None, *, enabled=True, output_dir=None, live=True):
    """
    Decorator to enable live visualization for a test function.
    
    This decorator captures the state of the model before and after the test,
    and sends it to the live visualization server.
    
    Args:
        func: The test function to decorate
        enabled: Whether visualization is enabled (default: True)
        output_dir: Ignored parameter, kept for backward compatibility
        live: Whether to enable live visualization (default: True)
        
    Returns:
        Decorated test function
    
    Example:
        @visualize_test
        def test_sleep_cycle():
            # Test code here
            ...
    """
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            if not enabled or not live:
                return test_func(*args, **kwargs)
            
            # Get the test name
            test_name = test_func.__name__
            
            # Get visualizer
            visualizer = get_default_visualizer()
            
            # Start tracking the test
            visualizer.start_test(test_name)
            
            # Find MorphModel instances in args or kwargs
            model = _find_model_in_args(args, kwargs)
            
            # Capture initial state with live visualization if we found a model
            if model:
                # Estimate total steps based on test name
                estimated_steps = _estimate_test_steps(test_name)
                
                # Capture initial state
                _capture_live_state(model, "Initial State", 0, estimated_steps)
                
                # Also track in visualizer for state management
                visualizer.capture_state(model, "Initial State")
            
            # Run the test
            try:
                result = test_func(*args, **kwargs)
                
                # Capture final state if we found a model
                if model:
                    # Estimate total steps based on test name
                    estimated_steps = _estimate_test_steps(test_name)
                    
                    # Capture final state
                    _capture_live_state(model, "Final State", estimated_steps, estimated_steps)
                    
                    # Also track in visualizer for state management
                    visualizer.capture_state(model, "Final State")
                
                return result
            except Exception as e:
                # Capture error state if we found a model
                if model:
                    # Estimate total steps based on test name
                    estimated_steps = _estimate_test_steps(test_name)
                    
                    # Capture error state
                    _capture_live_state(
                        model, 
                        "Error State", 
                        0, 
                        estimated_steps,
                        {"error": str(e)}
                    )
                    
                    # Also track in visualizer for state management
                    visualizer.capture_state(
                        model, 
                        "Error State", 
                        additional_data={"error": str(e)}
                    )
                raise
            finally:
                # End tracking
                visualizer.end_test()
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


@contextlib.contextmanager
def capture_test_state(model: MorphModel, step_name: str, 
                      additional_data: Optional[Dict[str, Any]] = None,
                      visualizer: Optional[TestVisualizer] = None,
                      live: bool = True):
    """
    Context manager to capture the state of a model at a specific point in a test.
    
    Args:
        model: The MorphModel instance
        step_name: Name of the current test step
        additional_data: Any additional data to capture
        visualizer: TestVisualizer instance (default: None, uses default visualizer)
        live: Whether to enable live visualization (default: True)
        
    Example:
        with capture_test_state(model, "After Memory Replay"):
            model._perform_memory_replay()
    """
    if not live:
        # If live visualization is disabled, just yield and return
        yield
        return
        
    # Get visualizer
    vis = visualizer or get_default_visualizer()
    
    # Get current test name
    test_name = _get_current_test_name()
    
    # Estimate total steps
    estimated_steps = _estimate_test_steps(test_name)
    
    # Get current step number
    step_number = len(vis.state_snapshots) if hasattr(vis, 'state_snapshots') else 0
    
    # Capture state before with live visualization
    _capture_live_state(model, f"{step_name} (Before)", step_number, estimated_steps, additional_data)
    
    # Also track in visualizer for state management
    vis.capture_state(model, f"{step_name} (Before)", additional_data)
    
    try:
        # Execute the context
        yield
    finally:
        # Increment step number
        step_number = len(vis.state_snapshots) if hasattr(vis, 'state_snapshots') else 0
        
        # Capture state after with live visualization
        _capture_live_state(model, f"{step_name} (After)", step_number, estimated_steps, additional_data)
        
        # Also track in visualizer for state management
        vis.capture_state(model, f"{step_name} (After)", additional_data)


def _find_model_in_args(args, kwargs) -> Optional[MorphModel]:
    """
    Find a MorphModel instance in function arguments.
    
    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        MorphModel instance if found, None otherwise
    """
    # Check args
    for arg in args:
        if isinstance(arg, MorphModel):
            return arg
    
    # Check kwargs
    for _, value in kwargs.items():
        if isinstance(value, MorphModel):
            return value
    
    return None


def _get_current_test_name() -> str:
    """
    Get the name of the current test function.
    
    Returns:
        Name of the current test function
    """
    # Get the current stack frame
    stack = inspect.stack()
    
    # Look for a test function in the stack
    for frame_info in stack:
        if frame_info.function.startswith('test_'):
            return frame_info.function
    
    # If no test function found, return a default name
    return 'unknown_test'


def _estimate_test_steps(test_name: str) -> int:
    """
    Estimate the number of steps in a test based on its name.
    
    Args:
        test_name: Name of the test
        
    Returns:
        Estimated number of steps
    """
    # Define step estimates for known tests
    step_estimates = {
        'test_sleep_cycle': 10,
        'test_expert_initialization': 5,
        'test_knowledge_graph_initialization': 5,
        'test_add_expert': 5,
        'test_add_concept': 5,
        'test_add_edge': 5,
        'test_link_expert_to_concept': 5,
        'test_expert_forward': 5,
        'test_expert_clone': 5,
        'test_parameter_similarity': 5,
        'test_get_similar_experts': 5,
        'test_get_dormant_experts': 5,
        'test_update_expert_activation': 5,
        'test_update_expert_specialization': 5,
        'test_decay_edges': 5,
        'test_prune_isolated_experts': 5,
        'test_rebuild_graph': 5,
        'test_memory_replay': 8,
        'test_sleep_cycle_initialization': 5,
        'test_expert_reorganization': 8,
        'test_expert_specialization_analysis': 8,
        'test_adaptive_sleep_scheduling': 8,
        'test_full_sleep_cycle': 15,
        'test_merge_expert_connections': 8,
    }
    
    # Return estimate if known, otherwise return a default value
    return step_estimates.get(test_name, 10)


def _capture_live_state(model: MorphModel, step_name: str, step_number: int, total_steps: int,
                       additional_data: Optional[Dict[str, Any]] = None):
    """
    Capture the state of a model for live visualization.
    
    Args:
        model: The MorphModel instance
        step_name: Name of the current test step
        step_number: Current step number
        total_steps: Total number of steps
        additional_data: Any additional data to capture
    """
    # Check if live visualizations are disabled via command line
    import sys
    if '--no-live-viz' in sys.argv:
        return
    
    # Get current test name
    test_name = _get_current_test_name()
    
    # Update progress tracker
    progress_tracker.record_step(step_name, step_number, total_steps)
    
    # Get model snapshot
    snapshot = live_visualizer.get_model_snapshot(model)
    
    # Add additional data if provided
    if additional_data:
        snapshot['additional_data'] = additional_data
    
    # Update live server
    live_server.update_test_state(test_name, step_name, step_number, total_steps, snapshot)

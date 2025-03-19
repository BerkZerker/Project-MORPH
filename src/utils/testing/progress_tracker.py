"""
Stub implementation of progress tracking for MORPH tests.

This module provides empty implementations of the progress tracking functions
to maintain backward compatibility with existing tests.
"""

from typing import Dict, List, Any, Optional, Union, Callable


def start_test(test_name: str):
    """
    Stub implementation that does nothing.
    
    This is a placeholder for the original start_test function
    to maintain backward compatibility.
    """
    pass


def end_test():
    """
    Stub implementation that does nothing.
    
    This is a placeholder for the original end_test function
    to maintain backward compatibility.
    """
    pass


def record_step(step_name: str, step_number: int, total_steps: int):
    """
    Stub implementation that does nothing.
    
    This is a placeholder for the original record_step function
    to maintain backward compatibility.
    """
    pass


def get_progress() -> Dict[str, Any]:
    """
    Stub implementation that returns an empty dictionary.
    
    This is a placeholder for the original get_progress function
    to maintain backward compatibility.
    """
    return {
        'test_name': None,
        'progress': 0,
        'elapsed': 0,
        'remaining': 0,
        'step_number': 0,
        'total_steps': 0
    }


def estimate_remaining_time(step_number: int, total_steps: int) -> float:
    """
    Stub implementation that returns 0.
    
    This is a placeholder for the original estimate_remaining_time function
    to maintain backward compatibility.
    """
    return 0.0

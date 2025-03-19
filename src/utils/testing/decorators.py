"""
Stub implementation of decorators for MORPH tests.

This module provides empty implementations of the visualization decorators
to maintain backward compatibility with existing tests.
"""

import functools
import contextlib
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from src.core.model import MorphModel


def visualize_test(func=None, *, enabled=True, output_dir=None, live=True):
    """
    Stub decorator that does nothing.
    
    This is a placeholder for the original visualize_test decorator
    to maintain backward compatibility.
    """
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            return test_func(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


@contextlib.contextmanager
def capture_test_state(model: MorphModel, step_name: str, 
                      additional_data: Optional[Dict[str, Any]] = None,
                      visualizer: Optional[Any] = None,
                      live: bool = True):
    """
    Stub context manager that does nothing.
    
    This is a placeholder for the original capture_test_state context manager
    to maintain backward compatibility.
    """
    yield

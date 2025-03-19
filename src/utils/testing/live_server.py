"""
Stub implementation of live visualization server for MORPH tests.

This module provides empty implementations of the live visualization server
to maintain backward compatibility with existing tests.
"""

from typing import Dict, List, Any, Optional, Union, Callable


def start_server(host: str = '127.0.0.1', port: int = 8080, 
                debug: bool = False, auto_open: bool = True):
    """
    Stub implementation that does nothing.
    
    This is a placeholder for the original start_server function
    to maintain backward compatibility.
    """
    pass


def stop_server():
    """
    Stub implementation that does nothing.
    
    This is a placeholder for the original stop_server function
    to maintain backward compatibility.
    """
    pass


def update_test_state(test_name: str, step_name: str, 
                     step_number: int, total_steps: int, 
                     snapshot: Dict[str, Any]):
    """
    Stub implementation that does nothing.
    
    This is a placeholder for the original update_test_state function
    to maintain backward compatibility.
    """
    pass


def start_test(test_name: str):
    """
    Stub implementation that does nothing.
    
    This is a placeholder for the original start_test function
    to maintain backward compatibility.
    """
    pass


def end_test(test_name: str):
    """
    Stub implementation that does nothing.
    
    This is a placeholder for the original end_test function
    to maintain backward compatibility.
    """
    pass

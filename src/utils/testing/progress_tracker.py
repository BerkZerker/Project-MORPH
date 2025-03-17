"""
Progress tracking for MORPH tests.

This module provides utilities for tracking test progress, estimating
remaining time, and managing test state.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable

# Global variables to track progress
_current_test = None
_test_history = {}
_test_start_time = None
_test_steps = {}
_lock = threading.RLock()


class TestProgressTracker:
    """
    Tracks progress of MORPH tests.
    
    This class provides methods to track test progress, estimate
    remaining time, and manage test state.
    """
    
    def __init__(self):
        """Initialize the test progress tracker."""
        self.current_test = None
        self.start_time = None
        self.steps = {}
        self.history = {}
    
    def start_test(self, test_name: str):
        """
        Start tracking a new test.
        
        Args:
            test_name: Name of the test being executed
        """
        self.current_test = test_name
        self.start_time = time.time()
        self.steps = {}
        
        print(f"\n⏱️ Progress tracking enabled for: {test_name}")
    
    def end_test(self):
        """End tracking for the current test."""
        if not self.current_test or not self.start_time:
            return
        
        # Calculate test duration
        duration = time.time() - self.start_time
        
        # Update test history
        self.history[self.current_test] = {
            'duration': duration,
            'steps': len(self.steps),
            'last_run': time.time()
        }
        
        print(f"\n✅ Test completed in {duration:.2f}s")
        
        # Reset current test
        self.current_test = None
        self.start_time = None
        self.steps = {}
    
    def record_step(self, step_name: str, step_number: int, total_steps: int):
        """
        Record a test step.
        
        Args:
            step_name: Name of the step
            step_number: Current step number
            total_steps: Total number of steps
        """
        if not self.current_test:
            return
        
        # Record step
        self.steps[step_number] = {
            'name': step_name,
            'timestamp': time.time(),
            'number': step_number,
            'total': total_steps
        }
        
        # Calculate progress
        progress = step_number / total_steps if total_steps > 0 else 0
        elapsed = time.time() - self.start_time
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed if progress > 0 else 0
        
        # Print progress
        print(f"\r⏱️ Progress: {step_number}/{total_steps} ({progress:.1%}) - "
              f"Elapsed: {elapsed:.1f}s - "
              f"Remaining: {remaining:.1f}s", end="")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information.
        
        Returns:
            Dictionary containing progress information
        """
        if not self.current_test or not self.start_time:
            return {
                'test_name': None,
                'progress': 0,
                'elapsed': 0,
                'remaining': 0,
                'step_number': 0,
                'total_steps': 0
            }
        
        # Get latest step
        if not self.steps:
            return {
                'test_name': self.current_test,
                'progress': 0,
                'elapsed': time.time() - self.start_time,
                'remaining': 0,
                'step_number': 0,
                'total_steps': 0
            }
        
        latest_step = max(self.steps.values(), key=lambda s: s['number'])
        
        # Calculate progress
        progress = latest_step['number'] / latest_step['total'] if latest_step['total'] > 0 else 0
        elapsed = time.time() - self.start_time
        estimated_total = elapsed / progress if progress > 0 else 0
        remaining = estimated_total - elapsed if progress > 0 else 0
        
        return {
            'test_name': self.current_test,
            'progress': progress,
            'elapsed': elapsed,
            'remaining': remaining,
            'step_number': latest_step['number'],
            'total_steps': latest_step['total'],
            'step_name': latest_step['name']
        }
    
    def estimate_remaining_time(self, step_number: int, total_steps: int) -> float:
        """
        Estimate remaining time based on current progress.
        
        Args:
            step_number: Current step number
            total_steps: Total number of steps
            
        Returns:
            Estimated remaining time in seconds
        """
        if not self.current_test or not self.start_time:
            return 0
        
        if step_number >= total_steps:
            return 0
        
        # If we have historical data for this test, use it
        if self.current_test in self.history:
            history = self.history[self.current_test]
            avg_step_time = history['duration'] / history['steps']
            return avg_step_time * (total_steps - step_number)
        
        # Otherwise, use current test data
        if step_number == 0:
            return 0
        
        elapsed = time.time() - self.start_time
        avg_step_time = elapsed / step_number
        return avg_step_time * (total_steps - step_number)


# Global instance
_tracker = TestProgressTracker()


def start_test(test_name: str):
    """
    Start tracking a new test.
    
    Args:
        test_name: Name of the test being executed
    """
    global _tracker
    
    with _lock:
        _tracker.start_test(test_name)


def end_test():
    """End tracking for the current test."""
    global _tracker
    
    with _lock:
        _tracker.end_test()


def record_step(step_name: str, step_number: int, total_steps: int):
    """
    Record a test step.
    
    Args:
        step_name: Name of the step
        step_number: Current step number
        total_steps: Total number of steps
    """
    global _tracker
    
    with _lock:
        _tracker.record_step(step_name, step_number, total_steps)


def get_progress() -> Dict[str, Any]:
    """
    Get current progress information.
    
    Returns:
        Dictionary containing progress information
    """
    global _tracker
    
    with _lock:
        return _tracker.get_progress()


def estimate_remaining_time(step_number: int, total_steps: int) -> float:
    """
    Estimate remaining time based on current progress.
    
    Args:
        step_number: Current step number
        total_steps: Total number of steps
        
    Returns:
        Estimated remaining time in seconds
    """
    global _tracker
    
    with _lock:
        return _tracker.estimate_remaining_time(step_number, total_steps)

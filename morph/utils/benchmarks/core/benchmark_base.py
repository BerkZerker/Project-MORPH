"""
Core benchmark base class for MORPH models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import Dataset


class BenchmarkBase:
    """
    Base class for continual learning benchmarks.
    
    This class provides the core functionality and attributes for benchmarks:
    1. Task management
    2. Performance tracking
    3. Basic configuration
    """
    
    def __init__(self, 
                 tasks: Dict[int, Dataset],
                 input_size: int,
                 output_size: int,
                 device: torch.device = torch.device("cpu"),
                 drift_detection: bool = False,
                 task_similarities: Optional[Dict[Tuple[int, int], float]] = None):
        """
        Initialize the benchmark.
        
        Args:
            tasks: Dictionary mapping task ID to dataset
            input_size: Dimension of input features
            output_size: Dimension of output features
            device: Device to run benchmark on
            drift_detection: Whether to enable concept drift detection
            task_similarities: Optional dictionary mapping task pairs (i,j) to similarity scores
        """
        self.tasks = tasks
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.drift_detection = drift_detection
        self.task_similarities = task_similarities or {}
        
        # Initial performance tracking
        self.accuracy_history = {}
        self.forgetting_metrics = {}
        
        # Drift detection setup
        if drift_detection:
            self.reference_distributions = {}
            self.distribution_shifts = {}
            
        # Expert utilization tracking for MORPH
        self.expert_utilization = {}
        self.expert_specialization = {}

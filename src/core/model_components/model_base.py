"""
Base class for the MORPH model.

This module contains the core attributes and simple utility methods for the MORPH model.
"""

import torch
import torch.nn as nn
import logging


class ModelBase(nn.Module):
    """
    Base class for the MORPH model.
    
    Contains core attributes and simple utility methods.
    """
    
    def __init__(self):
        """
        Initialize the ModelBase.
        
        Note: This is meant to be used as a mixin, not instantiated directly.
        The actual initialization happens in ModelInitialization.
        """
        super().__init__()
        
        # These will be initialized in ModelInitialization
        self.config = None
        self.step_count = 0
        self.device = None
        self.devices = None
        self.experts = None
        self.gating = None
        self.knowledge_graph = None
        self.sleep_module = None
        self.expert_device_map = None
        self.expert_input_distributions = None
        self.expert_performance_history = None
        self._wrapped_model = None
        
    # Properties to access sleep module attributes directly
    @property
    def sleep_cycles_completed(self):
        """Get the number of completed sleep cycles."""
        return self.sleep_module.sleep_cycles_completed
    
    @property
    def adaptive_sleep_frequency(self):
        """Get the adaptive sleep frequency."""
        return self.sleep_module.adaptive_sleep_frequency
    
    @property
    def next_sleep_step(self):
        """Get the step at which the next sleep cycle will occur."""
        return self.sleep_module.next_sleep_step
    
    @property
    def activation_buffer(self):
        """Get the activation buffer from the sleep module."""
        return self.sleep_module.activation_buffer
    
    def get_knowledge_graph(self):
        """
        Get the knowledge graph.
        
        Returns:
            NetworkX graph of expert relationships
        """
        from src.core.model_utils import get_knowledge_graph
        return get_knowledge_graph(self)
    
    def get_expert_metrics(self):
        """
        Get metrics about the current experts.
        
        Returns:
            Dictionary of expert metrics
        """
        from src.core.model_utils import get_expert_metrics
        return get_expert_metrics(self)
    
    def get_sleep_metrics(self):
        """
        Get metrics about sleep cycles.
        
        Returns:
            List of sleep cycle metrics
        """
        from src.core.model_utils import get_sleep_metrics
        return get_sleep_metrics(self)

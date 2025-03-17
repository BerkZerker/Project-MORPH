import torch
import logging
from typing import List, Dict, Any


class SleepCore:
    """
    Core functionality for the Sleep Module.
    
    Provides initialization and basic attributes for the sleep module.
    """
    
    def __init__(self, config, knowledge_graph):
        """
        Initialize the sleep module.
        
        Args:
            config: Configuration object with sleep parameters
            knowledge_graph: KnowledgeGraph instance
        """
        self.config = config
        self.knowledge_graph = knowledge_graph
        self.device = torch.device("cuda")
        
        # Sleep cycle tracking
        self.sleep_cycles_completed = 0
        
        # Use test-specific sleep frequency if in test mode
        if config.test_mode:
            self.next_sleep_step = config.test_sleep_frequency
            self.adaptive_sleep_frequency = config.test_sleep_frequency
        else:
            self.next_sleep_step = config.sleep_cycle_frequency
            self.adaptive_sleep_frequency = config.sleep_cycle_frequency
        
        # Memory replay buffer - use smaller buffer for tests if in test mode
        self.activation_buffer = []
        self.buffer_size = config.test_memory_buffer_size if config.test_mode else config.memory_buffer_size
        
        # Meta-learning state
        self.meta_learning_state = {
            'performance_history': [],
            'next_meta_update': config.meta_learning_intervals
        }
        
        # Sleep metrics tracking
        self.sleep_metrics = []

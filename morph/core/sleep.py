import torch
import logging
from typing import List, Dict, Tuple, Any, Optional

from morph.core.sleep_management.sleep_core import SleepCore
from morph.core.sleep_management.memory_management import add_to_memory_buffer, perform_memory_replay
from morph.core.sleep_management.expert_analysis import analyze_expert_specialization
from morph.core.sleep_management.expert_reorganization import ExpertReorganization
from morph.core.sleep_management.perform_sleep import perform_sleep_cycle
from morph.core.sleep_management.sleep_scheduling import update_sleep_schedule, should_sleep


class SleepModule(SleepCore, ExpertReorganization):
    """
    Sleep Module for MORPH model.
    
    Implements the 'sleep' phase of the model, responsible for:
    1. Memory replay and consolidation
    2. Expert merging and pruning
    3. Knowledge graph reorganization
    4. Meta-learning optimization
    """
    
    def __init__(self, config, knowledge_graph):
        """
        Initialize the sleep module.
        
        Args:
            config: Configuration object with sleep parameters
            knowledge_graph: KnowledgeGraph instance
        """
        SleepCore.__init__(self, config, knowledge_graph)
    
    def should_sleep(self, step_count: int) -> bool:
        """
        Determine if a sleep cycle should be triggered.
        
        Args:
            step_count: Current training step
            
        Returns:
            Boolean indicating whether to trigger sleep
        """
        return should_sleep(self, step_count)
    
    def add_to_memory_buffer(self, activation_data: Dict[str, Any]) -> None:
        """
        Add activation data to the memory buffer.
        
        Args:
            activation_data: Dictionary containing activation information
        """
        add_to_memory_buffer(self, activation_data)
    
    def perform_sleep_cycle(self, model, step_count: int) -> Dict[str, Any]:
        """
        Perform a complete sleep cycle.
        
        Args:
            model: The MORPH model
            step_count: Current training step
            
        Returns:
            Dictionary of metrics from the sleep cycle
        """
        return perform_sleep_cycle(self, model, step_count)
    
    def _perform_memory_replay(self, model) -> Dict[str, Any]:
        """
        Perform memory replay by replaying stored activations to experts.
        Uses batched processing for better performance.
        """
        return perform_memory_replay(self, model)
        
    def _prioritize_experiences(self, model) -> List[Dict[str, Any]]:
        """
        Prioritize experiences in the replay buffer.
        """
        # This is now handled internally by perform_memory_replay
        from morph.core.sleep_management.memory_management import _prioritize_experiences
        return _prioritize_experiences(self, model)
    
    def _analyze_expert_specialization(self, model) -> Dict[int, Dict[str, Any]]:
        """
        Analyze expert specialization based on input distributions.
        
        Returns a dictionary with expert indices as keys and specialization metrics as values.
        """
        return analyze_expert_specialization(self, model)
    
    def _merge_similar_experts(self, model) -> Tuple[bool, Dict[str, Any]]:
        """
        Find and merge experts that are too similar.
        """
        # Delegate to model's implementation
        return model._merge_similar_experts()
    
    def _prune_dormant_experts(self, model, step_count) -> Tuple[bool, Dict[str, Any]]:
        """
        Remove experts that haven't been activated for a long time.
        """
        # Delegate to model's implementation
        return model._prune_dormant_experts()
    
    def _update_sleep_schedule(self, model, step_count, experts_before, experts_after) -> None:
        """
        Update the adaptive sleep scheduling based on model performance.
        """
        update_sleep_schedule(self, model, step_count, experts_before, experts_after)

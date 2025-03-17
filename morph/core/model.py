"""
MORPH Model - Mixture Of experts with Recursive Post-processing & Hierarchy.

This module provides a thin interface to the MORPH model components,
which are implemented in separate modules for better organization and maintainability.
"""

from morph.core.model_components.model_base import ModelBase
from morph.core.model_components.model_forward import ModelForward
from morph.core.model_components.model_initialization import ModelInitialization
from morph.core.model_components.model_device import ModelDevice
from morph.core.model_components.model_mixed_precision import ModelMixedPrecision


class MorphModel(ModelBase, ModelForward, ModelInitialization, 
                ModelDevice, ModelMixedPrecision):
    """
    MORPH: Mixture Of experts with Recursive Post-processing & Hierarchy.
    
    This model implements a dynamic mixture of experts architecture with
    adaptive expert creation, knowledge graph routing, and a sleep cycle
    for knowledge consolidation.
    
    The implementation is split across multiple modules for better organization:
    - ModelBase: Core attributes and simple utility methods
    - ModelForward: Forward pass logic
    - ModelInitialization: Initialization logic
    - ModelDevice: Device management
    - ModelMixedPrecision: Mixed precision handling
    """
    
    def __init__(self, config):
        """
        Initialize the MORPH model.
        
        Args:
            config: Configuration object with model parameters
        """
        # Initialize all parent classes
        ModelBase.__init__(self)
        ModelForward.__init__(self)
        ModelInitialization.__init__(self, config)
        ModelDevice.__init__(self)
        ModelMixedPrecision.__init__(self)
        
    def forward(self, x, training=True):
        """
        Forward pass through the MORPH model.
        
        Args:
            x: Input tensor [batch_size, input_size]
            training: Whether in training mode
            
        Returns:
            Model output tensor [batch_size, output_size]
        """
        # Delegate to the ModelForward implementation
        return ModelForward.forward(self, x, training)
    
    def sleep(self):
        """
        Perform a sleep cycle to consolidate knowledge.
        
        This includes:
        1. Replaying stored activations for memory consolidation
        2. Analyzing expert specialization
        3. Merging similar experts
        4. Pruning dormant experts
        5. Reorganizing experts based on activation patterns
        """
        from morph.core.sleep_management import perform_sleep_cycle
        # Delegate sleep cycle to the sleep module
        return perform_sleep_cycle(self.sleep_module, self, self.step_count)
    
    def _merge_expert_parameters(self, idx1, idx2):
        """
        Merge parameters of two experts by weighted averaging.
        The first expert (idx1) will contain the merged parameters.
        
        Args:
            idx1: Index of first expert (destination)
            idx2: Index of second expert (to be merged)
        """
        from morph.core.expert_management import merge_expert_parameters
        merge_expert_parameters(self, idx1, idx2)
    
    def _merge_similar_experts(self):
        """
        Find and merge experts that are too similar.
        
        Returns:
            Tuple of (boolean indicating if any experts were merged, merge metrics dict)
        """
        from morph.core.expert_management import merge_similar_experts
        return merge_similar_experts(self)
    
    def _rebuild_knowledge_graph(self):
        """
        Rebuild the knowledge graph after expert count changes.
        """
        from morph.core.expert_management import rebuild_knowledge_graph
        rebuild_knowledge_graph(self)
    
    def _prune_dormant_experts(self):
        """
        Remove experts that haven't been activated for a long time.
        
        Returns:
            Tuple of (boolean indicating if any experts were pruned, pruning metrics dict)
        """
        from morph.core.expert_management import prune_dormant_experts
        return prune_dormant_experts(self)
    
    def _analyze_expert_specialization(self, model=None):
        """
        Analyze expert specialization based on input distributions.
        Delegates to sleep module.
        """
        from morph.core.model_utils import analyze_expert_specialization
        if model is None:
            model = self
        return analyze_expert_specialization(self, model)
    
    def _update_sleep_schedule(self, model=None):
        """
        Update the adaptive sleep scheduling based on model performance.
        Delegates to sleep module.
        """
        if model is None:
            model = self
        # Increment sleep cycle counter
        self.sleep_module.sleep_cycles_completed += 1
        
        # Calculate next sleep step
        experts_before = len(self.experts)
        experts_after = len(self.experts)
        self.sleep_module._update_sleep_schedule(model, self.step_count, experts_before, experts_after)
        return True
    
    def _reorganize_experts(self, specialization_metrics=None):
        """
        Reorganize experts based on activation patterns and specialization.
        Delegates to sleep module.
        """
        from morph.core.model_utils import reorganize_experts
        result, metrics = reorganize_experts(self, specialization_metrics)
        return result, metrics
    
    def _perform_memory_replay(self):
        """
        Perform memory replay by replaying stored activations to experts.
        Delegates to sleep module.
        """
        from morph.core.model_utils import perform_memory_replay
        return perform_memory_replay(self)
    
    def train_step(self, batch, optimizer, criterion):
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            
        Returns:
            Dictionary with loss and metrics
        """
        from morph.core.training import train_step
        return train_step(self, batch, optimizer, criterion)
    
    def evaluate(self, data_loader, criterion, device=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            criterion: Loss function
            device: Device to use (optional, defaults to model's device)
            
        Returns:
            Dictionary with evaluation metrics
        """
        from morph.core.training import evaluate
        return evaluate(self, data_loader, criterion, device)

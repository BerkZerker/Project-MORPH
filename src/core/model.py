"""
MORPH Model - Mixture Of experts with Recursive Post-processing & Hierarchy.

This module provides a thin interface to the MORPH model components,
which are implemented in separate modules for better organization and maintainability.
"""

import torch
import logging
from typing import Optional, Literal, Dict, Any, Union, List

from src.core.model_components.model_base import ModelBase
from src.core.model_components.model_forward import ModelForward
from src.core.model_components.model_initialization import ModelInitialization
from src.core.model_components.model_device import ModelDevice
from src.core.model_components.model_mixed_precision import ModelMixedPrecision


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
        
        # Compilation status
        self.is_compiled = False
        self.compilation_mode = None
        
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
        from src.core.sleep_management import perform_sleep_cycle
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
        from src.core.expert_management import merge_expert_parameters
        merge_expert_parameters(self, idx1, idx2)
    
    def _merge_similar_experts(self):
        """
        Find and merge experts that are too similar.
        
        Returns:
            Tuple of (boolean indicating if any experts were merged, merge metrics dict)
        """
        from src.core.expert_management import merge_similar_experts
        return merge_similar_experts(self)
    
    def _rebuild_knowledge_graph(self):
        """
        Rebuild the knowledge graph after expert count changes.
        """
        from src.core.expert_management import rebuild_knowledge_graph
        rebuild_knowledge_graph(self)
    
    def _prune_dormant_experts(self):
        """
        Remove experts that haven't been activated for a long time.
        
        Returns:
            Tuple of (boolean indicating if any experts were pruned, pruning metrics dict)
        """
        from src.core.expert_management import prune_dormant_experts
        return prune_dormant_experts(self)
    
    def _analyze_expert_specialization(self, model=None):
        """
        Analyze expert specialization based on input distributions.
        Delegates to sleep module.
        """
        from src.core.model_utils import analyze_expert_specialization
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
        from src.core.model_utils import reorganize_experts
        result, metrics = reorganize_experts(self, specialization_metrics)
        return result, metrics
    
    def _perform_memory_replay(self):
        """
        Perform memory replay by replaying stored activations to experts.
        Delegates to sleep module.
        """
        from src.core.model_utils import perform_memory_replay
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
        from src.core.training import train_step
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
        from src.core.training import evaluate
        return evaluate(self, data_loader, criterion, device)
    
    def compile_model(self, mode: str = 'default', 
                     compile_experts: bool = True, 
                     compile_gating: bool = True,
                     fullgraph: bool = False,
                     dynamic: bool = True) -> bool:
        """
        Compile the model using torch.compile() for faster execution on GPU.
        
        This requires PyTorch 2.0 or newer. For older versions, this is a no-op.
        
        Args:
            mode: Compilation mode:
                - 'default': Balance between compilation time and runtime performance
                - 'reduce-overhead': Minimize runtime overhead (faster execution)
                - 'max-autotune': Maximize performance through autotuning (slower compilation)
            compile_experts: Whether to compile expert models
            compile_gating: Whether to compile the gating network
            fullgraph: Whether to use full graph mode (less dynamic but faster)
            dynamic: Whether to allow dynamic shapes
            
        Returns:
            True if compilation was successful, False otherwise
        """
        # Check if PyTorch version supports torch.compile
        if not hasattr(torch, 'compile'):
            logging.warning("PyTorch version does not support torch.compile(). Skipping compilation.")
            return False
        
        try:
            compilation_count = 0
            
            # Compile experts
            if compile_experts and hasattr(self, 'experts'):
                for i, expert in enumerate(self.experts):
                    if hasattr(expert, 'weight'):  # Only compile modules with parameters
                        try:
                            self.experts[i] = torch.compile(
                                expert, 
                                mode=mode,
                                fullgraph=fullgraph,
                                dynamic=dynamic
                            )
                            compilation_count += 1
                        except Exception as e:
                            logging.warning(f"Failed to compile expert {i}: {e}")
            
            # Compile gating network
            if compile_gating and hasattr(self, 'gating') and self.gating is not None:
                try:
                    self.gating = torch.compile(
                        self.gating, 
                        mode=mode,
                        fullgraph=fullgraph,
                        dynamic=dynamic
                    )
                    compilation_count += 1
                except Exception as e:
                    logging.warning(f"Failed to compile gating network: {e}")
            
            # Mark model as compiled if any components were compiled
            if compilation_count > 0:
                self.is_compiled = True
                self.compilation_mode = mode
                logging.info(f"Model compiled with mode '{mode}': {compilation_count} components compiled")
                return True
            else:
                logging.warning("No components were compiled")
                return False
                
        except Exception as e:
            logging.error(f"Failed to compile model: {e}")
            self.is_compiled = False
            return False
    
    def get_compilation_status(self) -> Dict[str, Any]:
        """
        Get the compilation status of the model.
        
        Returns:
            Dictionary with compilation status information
        """
        status = {
            'is_compiled': getattr(self, 'is_compiled', False),
            'compilation_mode': getattr(self, 'compilation_mode', None),
            'pytorch_version': torch.__version__,
            'has_compile': hasattr(torch, 'compile'),
        }
        
        # Add CUDA compilation info if available
        if torch.cuda.is_available():
            status['cuda_version'] = torch.version.cuda
            
            # Check if we're using CUDA graphs
            if hasattr(torch.cuda, 'is_current_stream_capturing'):
                status['using_cuda_graphs'] = torch.cuda.is_current_stream_capturing()
            else:
                status['using_cuda_graphs'] = False
        
        return status

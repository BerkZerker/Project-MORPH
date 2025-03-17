"""
Forward pass logic for the MORPH model.

This module contains the forward pass logic for the MORPH model,
including routing and expert activation.
"""

import torch
import numpy as np
from torch.amp import autocast


class ModelForward:
    """
    Forward pass logic for the MORPH model.
    
    This class is responsible for the forward pass through the MORPH model,
    including routing inputs to experts and combining their outputs.
    """
    
    def __init__(self):
        """
        Initialize the ModelForward.
        
        Note: This is meant to be used as a mixin, not instantiated directly.
        """
        # No initialization needed for this mixin
        pass
    
    def forward(self, x, training=True):
        """
        Forward pass through the MORPH model.
        
        Args:
            x: Input tensor [batch_size, input_size]
            training: Whether in training mode
            
        Returns:
            Model output tensor [batch_size, output_size]
        """
        # If using a parallel wrapper, delegate to it
        if self._wrapped_model is not None and not isinstance(self._wrapped_model, type(self)):
            return self._wrapped_model(x, training=training)
            
        # Move input to the correct device
        x = x.to(self.device)
        batch_size = x.shape[0]
        
        # Use autocast for mixed precision if enabled
        with autocast('cuda', enabled=self.enable_mixed_precision):
            # Get routing weights from gating network
            routing_weights, expert_indices, uncertainty = self.gating(x, training)
            
            # Maybe create new expert if uncertainty is high
            if (training and 
                self.config.enable_dynamic_experts and 
                self.gating.should_create_new_expert(uncertainty)):
                self._create_new_expert()
            
            # Initialize output tensor
            outputs = torch.zeros(batch_size, self.config.output_size, device=self.device)
            
            # Route inputs to selected experts and combine outputs
            outputs = self._route_and_combine(
                x, routing_weights, expert_indices, uncertainty, 
                outputs, batch_size, training
            )
        
        # Increment step counter and maybe trigger sleep cycle
        if training:
            self.step_count += 1
            
            # Maybe trigger sleep cycle using adaptive scheduling
            if (self.config.enable_sleep and 
                self.sleep_module.should_sleep(self.step_count)):
                self.sleep()
        
        return outputs
    
    def _route_and_combine(self, x, routing_weights, expert_indices, uncertainty, 
                          outputs, batch_size, training):
        """
        Route inputs to selected experts and combine their outputs.
        
        Args:
            x: Input tensor [batch_size, input_size]
            routing_weights: Routing weights from gating network [batch_size, k]
            expert_indices: Expert indices from gating network [batch_size, k]
            uncertainty: Uncertainty from gating network
            outputs: Output tensor to fill [batch_size, output_size]
            batch_size: Batch size
            training: Whether in training mode
            
        Returns:
            Combined output tensor [batch_size, output_size]
        """
        # Route inputs to selected experts and combine outputs
        for i in range(self.gating.k):
            # Get the expert indices and routing weights for this slot
            indices = expert_indices[:, i]  # [batch_size]
            weights = routing_weights[:, i].unsqueeze(1)  # [batch_size, 1]
            
            # Process each expert
            for expert_idx in indices.unique():
                # Find which batch items use this expert
                mask = (indices == expert_idx)
                if not mask.any():
                    continue
                    
                # Get the expert and its device
                expert_idx_int = expert_idx.item()
                expert = self.experts[expert_idx_int]
                expert_device = self.expert_device_map.get(expert_idx_int, self.device)
                
                # Update expert activation counters
                expert.last_activated = self.step_count
                
                # If training, update knowledge graph
                if training:
                    self.knowledge_graph.update_expert_activation(expert_idx_int, self.step_count)
                
                # Process inputs with this expert (move to expert's device if needed)
                expert_inputs = x[mask]
                if expert_inputs.device != expert_device:
                    expert_inputs = expert_inputs.to(expert_device)
                    
                expert_outputs = expert(expert_inputs)
                
                # Weight outputs by routing weights (on expert's device)
                if weights[mask].device != expert_device:
                    expert_weights = weights[mask].to(expert_device)
                else:
                    expert_weights = weights[mask]
                    
                weighted_outputs = expert_outputs * expert_weights
                
                # Move back to primary device if needed and add to final outputs
                if weighted_outputs.device != self.device:
                    weighted_outputs = weighted_outputs.to(self.device)
                    
                outputs[mask] += weighted_outputs
                
                # Store activation for sleep cycle if training
                if training:
                    self._store_activation_data(
                        expert_idx, expert_inputs, expert_outputs, 
                        weights, mask, uncertainty
                    )
        
        return outputs
    
    def _store_activation_data(self, expert_idx, expert_inputs, expert_outputs, 
                              weights, mask, uncertainty):
        """
        Store activation data for the sleep module.
        
        Args:
            expert_idx: Expert index
            expert_inputs: Inputs to the expert
            expert_outputs: Outputs from the expert
            weights: Routing weights
            mask: Mask indicating which batch items use this expert
            uncertainty: Uncertainty from gating network
        """
        # Compute and store expert performance
        with torch.no_grad():
            # Store activation metadata for sleep module
            # Keep tensors on GPU if possible, only move to CPU for storage if needed
            if self.device.type == 'cuda':
                # Store features on GPU to reduce CPU-GPU transfers
                input_features = torch.mean(expert_inputs, dim=0)
                # Only detach and move to CPU for storage in memory buffer
                activation_data = {
                    'expert_idx': expert_idx.item(),
                    'inputs': expert_inputs.detach().cpu() if self.config.memory_buffer_size > 0 else None,
                    'outputs': expert_outputs.detach().cpu() if self.config.memory_buffer_size > 0 else None,
                    'routing_weight': weights[mask].mean().item(),
                    'step': self.step_count,
                    'batch_size': expert_inputs.size(0),
                    'input_features': input_features.detach().cpu(),
                    'uncertainty': uncertainty.item() if uncertainty is not None else 0.0
                }
            else:
                # On CPU, no need to move
                input_features = torch.mean(expert_inputs, dim=0).detach()
                activation_data = {
                    'expert_idx': expert_idx.item(),
                    'inputs': expert_inputs.detach() if self.config.memory_buffer_size > 0 else None,
                    'outputs': expert_outputs.detach() if self.config.memory_buffer_size > 0 else None,
                    'routing_weight': weights[mask].mean().item(),
                    'step': self.step_count,
                    'batch_size': expert_inputs.size(0),
                    'input_features': input_features,
                    'uncertainty': uncertainty.item() if uncertainty is not None else 0.0
                }
            
            self.sleep_module.add_to_memory_buffer(activation_data)
            
        # Track input distribution for specialization analysis
        if expert_idx.item() in self.expert_input_distributions:
            # Use vectorized operations for better performance
            input_features_np = input_features.cpu().numpy()
            feature_hash = hash(str(np.round(input_features_np, 2)))
            
            if feature_hash in self.expert_input_distributions[expert_idx.item()]:
                self.expert_input_distributions[expert_idx.item()][feature_hash] += 1
            else:
                self.expert_input_distributions[expert_idx.item()][feature_hash] = 1
    
    def _create_new_expert(self):
        """
        Create a new expert and update the gating network.
        """
        from morph.core.expert_management import create_new_expert
        return create_new_expert(self)

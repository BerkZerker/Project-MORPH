"""
Device management for the MORPH model.

This module contains the device management logic for the MORPH model,
including expert distribution across devices and device mapping.
"""

import torch
import logging
from morph.utils.gpu_utils import distribute_experts_across_gpus


class ModelDevice:
    """
    Device management for the MORPH model.
    
    This class is responsible for managing device assignments for the MORPH model,
    including distributing experts across devices and handling device transfers.
    """
    
    def __init__(self):
        """
        Initialize the ModelDevice.
        
        Note: This is meant to be used as a mixin, not instantiated directly.
        """
        # No initialization needed for this mixin
        pass
    
    def to(self, device):
        """
        Move the model to the specified device.
        
        This overrides the default PyTorch to() method to handle the expert device map.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self for chaining
        """
        # Only move components that should be on the primary device
        # Experts are managed separately through the expert_device_map
        if hasattr(self, 'gating') and self.gating is not None:
            self.gating = self.gating.to(device)
        
        # Update the primary device
        if hasattr(self, 'device'):
            self.device = device
        
        return self
    
    def redistribute_experts(self, devices=None):
        """
        Redistribute experts across available devices.
        
        This is useful after adding or removing experts.
        
        Args:
            devices: List of devices to distribute experts across.
                    If None, uses self.devices.
        
        Returns:
            Updated expert_device_map
        """
        if devices is None:
            devices = self.devices
        
        # Create a new expert device map
        num_experts = len(self.experts)
        new_expert_device_map = distribute_experts_across_gpus(num_experts, devices)
        
        # Move experts to their new devices
        for i, expert in enumerate(self.experts):
            if i in new_expert_device_map:
                target_device = new_expert_device_map[i]
                current_device = expert.weight.device if hasattr(expert, 'weight') else None
                
                # Only move if needed
                if current_device != target_device:
                    self.experts[i] = expert.to(target_device)
                    logging.info(f"Moved expert {i} from {current_device} to {target_device}")
        
        # Update the expert device map
        self.expert_device_map = new_expert_device_map
        if hasattr(self, 'config'):
            self.config.expert_device_map = new_expert_device_map
        
        return new_expert_device_map
    
    def get_expert_device(self, expert_idx):
        """
        Get the device for a specific expert.
        
        Args:
            expert_idx: Expert index
            
        Returns:
            Device for the expert
        """
        if expert_idx in self.expert_device_map:
            return self.expert_device_map[expert_idx]
        return self.device
    
    def move_expert_to_device(self, expert_idx, device):
        """
        Move a specific expert to a different device.
        
        Args:
            expert_idx: Expert index
            device: Target device
            
        Returns:
            True if the expert was moved, False otherwise
        """
        if expert_idx >= len(self.experts):
            logging.warning(f"Expert {expert_idx} does not exist")
            return False
        
        # Get the expert
        expert = self.experts[expert_idx]
        current_device = expert.weight.device if hasattr(expert, 'weight') else None
        
        # Only move if needed
        if current_device != device:
            self.experts[expert_idx] = expert.to(device)
            self.expert_device_map[expert_idx] = device
            logging.info(f"Moved expert {expert_idx} from {current_device} to {device}")
            return True
        
        return False

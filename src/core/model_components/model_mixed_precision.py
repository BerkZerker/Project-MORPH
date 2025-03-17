"""
Mixed precision handling for the MORPH model.

This module contains the mixed precision handling logic for the MORPH model,
including autocast and scaling.
"""

import torch
from torch.amp import autocast, GradScaler
import logging


class ModelMixedPrecision:
    """
    Mixed precision handling for the MORPH model.
    
    This class is responsible for managing mixed precision training,
    including autocast and gradient scaling.
    """
    
    def __init__(self):
        """
        Initialize the ModelMixedPrecision.
        
        Note: This is meant to be used as a mixin, not instantiated directly.
        """
        # No initialization needed for this mixin
        pass
    
    def enable_mixed_precision(self, enable=True):
        """
        Enable or disable mixed precision training.
        
        Args:
            enable: Whether to enable mixed precision training
            
        Returns:
            True if mixed precision is enabled, False otherwise
        """
        # Only enable if using CUDA
        if enable and any(d.type == 'cuda' for d in self.devices):
            self.enable_mixed_precision = True
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.scaler = GradScaler('cuda')
            logging.info("Mixed precision training enabled")
            return True
        else:
            self.enable_mixed_precision = False
            self.scaler = None
            if enable:
                logging.warning("Mixed precision training requires CUDA. Disabled.")
            return False
    
    def get_autocast_context(self, device_type='cuda'):
        """
        Get an autocast context for mixed precision.
        
        Args:
            device_type: Device type for autocast
            
        Returns:
            Autocast context manager
        """
        return autocast(device_type, enabled=self.enable_mixed_precision)
    
    def scale_loss(self, loss, optimizer):
        """
        Scale the loss for mixed precision training.
        
        Args:
            loss: Loss to scale
            optimizer: Optimizer to use
            
        Returns:
            Scaled loss
        """
        if self.enable_mixed_precision and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer):
        """
        Step the optimizer with gradient scaling if mixed precision is enabled.
        
        Args:
            optimizer: Optimizer to step
            
        Returns:
            True if the optimizer was stepped, False otherwise
        """
        if self.enable_mixed_precision and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        else:
            optimizer.step()
            return False

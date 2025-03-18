"""
Mixed precision handling for the MORPH model.

This module contains the mixed precision handling logic for the MORPH model,
including autocast and scaling with support for both FP16 and BF16 precision.
"""

import torch
from torch.amp import autocast, GradScaler
import logging
from typing import Optional, Literal


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
    
    def enable_mixed_precision(self, enable=True, precision_type='auto'):
        """
        Enable or disable mixed precision training with specified precision type.
        
        Args:
            enable: Whether to enable mixed precision training
            precision_type: Precision type to use ('auto', 'fp16', or 'bf16')
                - 'auto': Automatically select based on hardware capability
                - 'fp16': Use FP16 precision (works on all CUDA GPUs)
                - 'bf16': Use BF16 precision (Ampere+ GPUs, better numerical stability)
            
        Returns:
            True if mixed precision is enabled, False otherwise
        """
        # Only enable if using CUDA
        if enable and any(d.type == 'cuda' for d in self.devices):
            # Get the first CUDA device for capability check
            cuda_device = next((d for d in self.devices if d.type == 'cuda'), None)
            device_type = cuda_device.type if cuda_device else 'cuda'
            device_index = cuda_device.index if cuda_device else 0
            
            # Determine optimal precision type based on hardware
            if precision_type == 'auto':
                # Check if we have Ampere or newer GPU (compute capability >= 8.0)
                if torch.cuda.is_available() and hasattr(torch.cuda, 'get_device_capability'):
                    capability = torch.cuda.get_device_capability(device_index)
                    if capability[0] >= 8:  # Ampere or newer
                        precision_type = 'bf16'
                        logging.info(f"Auto-selected BF16 precision (GPU compute capability {capability[0]}.{capability[1]})")
                    else:
                        precision_type = 'fp16'
                        logging.info(f"Auto-selected FP16 precision (GPU compute capability {capability[0]}.{capability[1]})")
                else:
                    precision_type = 'fp16'
                    logging.info("Auto-selected FP16 precision (could not detect GPU capability)")
            
            # Store the precision type
            self.precision_type = precision_type
            self.mixed_precision_enabled = True
            
            # Create appropriate scaler based on precision type
            if precision_type == 'bf16':
                # BF16 doesn't need a scaler for numerical stability
                self.scaler = None
                logging.info("Mixed precision training enabled with BF16 precision (no scaling needed)")
            else:
                # For FP16, use GradScaler with appropriate device type
                if not hasattr(self, 'scaler') or self.scaler is None:
                    self.scaler = GradScaler(device_type=device_type)
                logging.info(f"Mixed precision training enabled with FP16 precision and gradient scaling")
            
            return True
        else:
            self.mixed_precision_enabled = False
            self.precision_type = None
            self.scaler = None
            if enable:
                logging.warning("Mixed precision training requires CUDA. Disabled.")
            return False
    
    def get_autocast_context(self, device_type=None):
        """
        Get an autocast context for mixed precision with appropriate dtype.
        
        Args:
            device_type: Device type for autocast (if None, uses the first CUDA device type)
            
        Returns:
            Autocast context manager
        """
        if not hasattr(self, 'mixed_precision_enabled') or not self.mixed_precision_enabled:
            return autocast(device_type or 'cuda', enabled=False)
            
        # Get device type from model if not specified
        if device_type is None:
            cuda_device = next((d for d in self.devices if d.type == 'cuda'), None)
            device_type = cuda_device.type if cuda_device else 'cuda'
        
        # Set dtype based on precision type
        dtype = None
        if hasattr(self, 'precision_type'):
            if self.precision_type == 'bf16':
                dtype = torch.bfloat16
            elif self.precision_type == 'fp16':
                dtype = torch.float16
        
        # Create autocast context with appropriate dtype
        return autocast(device_type, enabled=self.mixed_precision_enabled, dtype=dtype)
    
    def scale_loss(self, loss, optimizer):
        """
        Scale the loss for mixed precision training.
        
        Args:
            loss: Loss to scale
            optimizer: Optimizer to use
            
        Returns:
            Scaled loss
        """
        if hasattr(self, 'mixed_precision_enabled') and self.mixed_precision_enabled:
            # Only scale if using FP16 (BF16 doesn't need scaling)
            if self.scaler is not None:
                return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer):
        """
        Step the optimizer with gradient scaling if mixed precision is enabled.
        
        Args:
            optimizer: Optimizer to step
            
        Returns:
            True if the optimizer was stepped with scaling, False otherwise
        """
        if hasattr(self, 'mixed_precision_enabled') and self.mixed_precision_enabled:
            if self.scaler is not None:
                # FP16 path with scaling
                self.scaler.step(optimizer)
                self.scaler.update()
                return True
            else:
                # BF16 path without scaling
                optimizer.step()
                return True
        else:
            # Standard precision path
            optimizer.step()
            return False

import torch
import torch.nn as nn
from typing import List, Any, Dict, Optional


class DataParallelWrapper(nn.Module):
    """
    A wrapper for data parallel training of the MORPH model.
    This distributes batches across multiple GPUs.
    """
    
    def __init__(self, model, devices):
        """
        Initialize the data parallel wrapper.
        
        Args:
            model: The MORPH model to wrap
            devices: List of devices to use
        """
        super().__init__()
        self.model = model
        self.devices = devices
        self.data_parallel = nn.DataParallel(model, device_ids=[d.index for d in devices if d.type == 'cuda'])
        
    def forward(self, x, training=True):
        """
        Forward pass through the data parallel model.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Model output
        """
        return self.data_parallel(x, training=training)
    
    def train_step(self, batch, optimizer, criterion):
        """
        Perform a single training step with data parallelism.
        
        Args:
            batch: Tuple of (inputs, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            
        Returns:
            Dictionary with loss and metrics
        """
        return self.model.train_step(batch, optimizer, criterion)
    
    def evaluate(self, data_loader, criterion, device=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            criterion: Loss function
            device: Device to use (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        return self.model.evaluate(data_loader, criterion, device)
    
    # Forward all other methods to the wrapped model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
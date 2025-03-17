"""
Standard neural network model for comparison with MORPH.
"""

import torch
import torch.nn as nn


class StandardModel(nn.Module):
    """
    Standard neural network for comparison with MORPH.
    This model has a similar capacity but lacks specialized expert structure.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        """
        Initialize a standard neural network.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Size of hidden layers
            output_size: Dimension of output features
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        # Build layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, training=True):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            training: Whether in training mode (ignored, included for API compatibility with MORPH)
        
        Returns:
            Output tensor
        """
        return self.network(x)

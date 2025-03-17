"""
Elastic Weight Consolidation (EWC) model for continual learning.
"""

import torch
import torch.nn as nn


class EWCModel(nn.Module):
    """
    Implementation of Elastic Weight Consolidation (EWC) for continual learning.
    
    EWC is a regularization method that preserves important parameters for previous tasks
    by adding a penalty term to the loss function.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, ewc_lambda: float = 5000):
        """
        Initialize an EWC model.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Size of hidden layers
            output_size: Dimension of output features
            ewc_lambda: Importance of the EWC penalty term
        """
        super().__init__()
        
        # Standard network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # EWC parameters
        self.ewc_lambda = ewc_lambda
        self.fisher_information = {}
        self.optimal_parameters = {}
        self.task_count = 0
        
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
    
    def calculate_fisher_information(self, data_loader, device):
        """
        Calculate the Fisher information matrix for the current task.
        
        Args:
            data_loader: DataLoader for the current task
            device: Device to run calculation on
        """
        # Store current parameters
        self.optimal_parameters[self.task_count] = {}
        for name, param in self.named_parameters():
            self.optimal_parameters[self.task_count][name] = param.data.clone()
        
        # Initialize Fisher information
        self.fisher_information[self.task_count] = {}
        for name, param in self.named_parameters():
            self.fisher_information[self.task_count][name] = torch.zeros_like(param.data)
        
        # Calculate Fisher information
        self.eval()
        criterion = nn.CrossEntropyLoss()
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            
            self.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.fisher_information[self.task_count][name] += param.grad.data ** 2 / len(data_loader)
        
        self.task_count += 1
    
    def ewc_loss(self, current_loss):
        """
        Calculate the EWC loss (current loss + EWC penalty).
        
        Args:
            current_loss: Current task loss
            
        Returns:
            Total loss including EWC penalty
        """
        ewc_loss = current_loss
        
        # Add EWC penalty for each previous task
        for task_id in range(self.task_count):
            for name, param in self.named_parameters():
                if name in self.fisher_information[task_id] and name in self.optimal_parameters[task_id]:
                    fisher = self.fisher_information[task_id][name]
                    optimal_param = self.optimal_parameters[task_id][name]
                    ewc_loss += (self.ewc_lambda / 2) * torch.sum(fisher * (param - optimal_param) ** 2)
        
        return ewc_loss

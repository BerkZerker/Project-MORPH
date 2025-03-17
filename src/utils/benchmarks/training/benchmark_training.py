"""
Training functionality for continual learning benchmarks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import DataLoader

from src.core.model import MorphModel


class BenchmarkTraining:
    """
    Training functionality for continual learning benchmarks.
    
    This class provides methods for training models on tasks and tracking
    training metrics, expert utilization, and concept drift.
    """
    
    def train_model(self, 
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    task_id: int,
                    epochs: int = 5,
                    batch_size: int = 32,
                    is_morph: bool = False,
                    is_ewc: bool = False,
                    track_drift: bool = False):
        """
        Train a model on a specific task.
        
        Args:
            model: Model to train
            optimizer: Optimizer to use
            task_id: ID of task to train on
            epochs: Number of epochs to train
            batch_size: Batch size for training
            is_morph: Whether the model is a MORPH model
            is_ewc: Whether the model uses EWC
            track_drift: Whether to track distribution drift during training
            
        Returns:
            Training metrics
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Create data loader
        dataloader = DataLoader(
            self.tasks[task_id], 
            batch_size=batch_size, 
            shuffle=True
        )
        
        metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'drift_detected': False,
            'drift_magnitude': 0.0,
            'adaptation_rate': 0.0
        }
        
        # Setup for drift tracking
        if track_drift and self.drift_detection:
            # Sample feature distributions at the start
            initial_distribution = self._compute_feature_distribution(model, dataloader, is_morph)
            
            # Store as reference if this is the first time seeing this task
            if task_id not in self.reference_distributions:
                self.reference_distributions[task_id] = initial_distribution
        
        # Expert tracking for MORPH
        if is_morph:
            initial_expert_count = len(model.experts)
            metrics['initial_expert_count'] = initial_expert_count
            
            # Track initial expert specialization
            if hasattr(model, 'get_expert_metrics'):
                initial_expert_metrics = model.get_expert_metrics()
                metrics['initial_expert_metrics'] = initial_expert_metrics
        
        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            epoch_predictions = []
            epoch_targets = []
            
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Flatten inputs for MLP-based models
                inputs = inputs.view(inputs.size(0), -1)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                if is_morph:
                    outputs = model(inputs, training=True)
                else:
                    outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Add EWC penalty if applicable
                if is_ewc:
                    loss = model.ewc_loss(loss)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions for drift analysis
                if track_drift:
                    epoch_predictions.append(predicted.cpu().numpy())
                    epoch_targets.append(targets.cpu().numpy())
            
            # Epoch metrics
            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = 100. * correct / total
            
            metrics['train_loss'].append(epoch_loss)
            metrics['train_accuracy'].append(epoch_accuracy)
            
            # Drift detection after each epoch
            if track_drift and self.drift_detection and epoch > 0:
                # Check for concept drift by comparing to reference distribution
                current_distribution = self._compute_feature_distribution(model, dataloader, is_morph)
                
                drift_magnitude = self._compute_distribution_shift(
                    self.reference_distributions[task_id], 
                    current_distribution
                )
                
                if drift_magnitude > 0.3:  # Arbitrary threshold
                    metrics['drift_detected'] = True
                    metrics['drift_magnitude'] = max(metrics['drift_magnitude'], drift_magnitude)
                    
                    # Measure adaptation
                    all_preds = np.concatenate(epoch_predictions)
                    all_targets = np.concatenate(epoch_targets)
                    adaptation_rate = self._compute_adaptation_rate(all_preds, all_targets)
                    metrics['adaptation_rate'] = adaptation_rate
                    
                    # Store distribution shift for analysis
                    if task_id not in self.distribution_shifts:
                        self.distribution_shifts[task_id] = []
                    
                    self.distribution_shifts[task_id].append({
                        'epoch': epoch,
                        'magnitude': drift_magnitude,
                        'adaptation_rate': adaptation_rate
                    })
        
        # Track MORPH expert evolution
        if is_morph:
            # Track expert growth
            final_expert_count = len(model.experts)
            metrics['final_expert_count'] = final_expert_count
            metrics['expert_growth'] = final_expert_count - initial_expert_count
            
            # Track expert utilization
            if hasattr(model, 'get_expert_metrics'):
                final_expert_metrics = model.get_expert_metrics()
                metrics['final_expert_metrics'] = final_expert_metrics
                
                # Update expert tracking
                self.expert_utilization[task_id] = {
                    i: data.get('activation_count', 0) 
                    for i, data in final_expert_metrics.items()
                }
                
                self.expert_specialization[task_id] = {
                    i: data.get('specialization_score', 0.0) 
                    for i, data in final_expert_metrics.items()
                }
            
            # Track sleep metrics if available
            if hasattr(model, 'get_sleep_metrics'):
                sleep_metrics = model.get_sleep_metrics()
                if sleep_metrics:
                    metrics['sleep_cycles'] = len(sleep_metrics)
                    metrics['sleep_metrics'] = sleep_metrics
        
        # Store Fisher information for EWC
        if is_ewc:
            model.calculate_fisher_information(dataloader, self.device)
            
        return metrics

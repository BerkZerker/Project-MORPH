"""
Evaluation functionality for continual learning benchmarks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import DataLoader

from morph.core.model import MorphModel
from morph.utils.benchmarks.evaluation.metrics import (
    calculate_forgetting,
    calculate_forward_transfer,
    calculate_knowledge_retention
)
from morph.utils.benchmarks.evaluation.visualization import visualize_results


class BenchmarkEvaluation:
    """
    Evaluation functionality for continual learning benchmarks.
    
    This class provides methods for evaluating models on tasks and running
    full benchmarks with multiple models.
    """
    
    def evaluate_model(self, 
                       model: nn.Module,
                       task_ids: Optional[List[int]] = None,
                       batch_size: int = 32,
                       is_morph: bool = False,
                       detailed_metrics: bool = False):
        """
        Evaluate a model on one or more tasks.
        
        Args:
            model: Model to evaluate
            task_ids: List of task IDs to evaluate on (default: all tasks)
            batch_size: Batch size for evaluation
            is_morph: Whether the model is a MORPH model
            detailed_metrics: Whether to return detailed metrics beyond accuracy
            
        Returns:
            Dictionary mapping task IDs to metrics
        """
        model.eval()
        
        if task_ids is None:
            task_ids = list(self.tasks.keys())
            
        results = {}
        
        for task_id in task_ids:
            dataloader = DataLoader(
                self.tasks[task_id],
                batch_size=batch_size,
                shuffle=False
            )
            
            correct = 0
            total = 0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Flatten inputs for MLP-based models
                    inputs = inputs.view(inputs.size(0), -1)
                    
                    # Forward pass
                    if is_morph:
                        outputs = model(inputs, training=False)
                    else:
                        outputs = model(inputs)
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Store for detailed metrics
                    if detailed_metrics:
                        all_predictions.append(predicted.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())
            
            # Basic accuracy metric
            accuracy = 100. * correct / total
            
            if not detailed_metrics:
                results[task_id] = accuracy
            else:
                # Calculate detailed metrics
                all_preds = np.concatenate(all_predictions)
                all_targets = np.concatenate(all_targets)
                
                # Calculate per-class accuracy
                classes = set(all_targets)
                per_class_acc = {}
                
                for cls in classes:
                    mask = (all_targets == cls)
                    if np.any(mask):
                        class_correct = np.sum(all_preds[mask] == all_targets[mask])
                        class_total = np.sum(mask)
                        per_class_acc[int(cls)] = 100. * class_correct / class_total
                
                # Track expert activation for MORPH
                expert_activations = {}
                if is_morph and hasattr(model, 'gating'):
                    # This is a simplified version - in a real implementation, 
                    # we would track which experts activate for each input
                    expert_activations = {
                        i: 0 for i in range(len(model.experts))
                    }
                
                results[task_id] = {
                    'accuracy': accuracy,
                    'per_class_accuracy': per_class_acc,
                    'expert_activations': expert_activations
                }
        
        return results
    
    def run_benchmark(self, 
                      models: Dict[str, nn.Module],
                      optimizers: Dict[str, torch.optim.Optimizer],
                      epochs_per_task: int = 5,
                      batch_size: int = 32,
                      detailed_eval: bool = False):
        """
        Run a full continual learning benchmark with multiple models.
        
        Args:
            models: Dictionary mapping model names to models
            optimizers: Dictionary mapping model names to optimizers
            epochs_per_task: Number of epochs to train on each task
            batch_size: Batch size for training and evaluation
            detailed_eval: Whether to collect detailed evaluation metrics
            
        Returns:
            Dictionary of benchmark results
        """
        # Track accuracy history for each model and task
        model_accuracy_history = {name: {} for name in models.keys()}
        model_detailed_metrics = {name: {} for name in models.keys()}
        model_training_metrics = {name: {} for name in models.keys()}
        
        # For each task sequentially
        for task_id in sorted(self.tasks.keys()):
            print(f"Training on Task {task_id}")
            
            # For each model
            for model_name, model in models.items():
                print(f"  Model: {model_name}")
                
                # Train on current task
                is_morph = isinstance(model, MorphModel)
                is_ewc = model_name == "EWC"
                
                optimizer = optimizers[model_name]
                
                # Training with drift detection
                training_metrics = self.train_model(
                    model=model,
                    optimizer=optimizer,
                    task_id=task_id,
                    epochs=epochs_per_task,
                    batch_size=batch_size,
                    is_morph=is_morph,
                    is_ewc=is_ewc,
                    track_drift=self.drift_detection
                )
                
                # Store training metrics
                model_training_metrics[model_name][task_id] = training_metrics
                
                # Print drift detection results if applicable
                if self.drift_detection and training_metrics['drift_detected']:
                    print(f"    Drift detected! Magnitude: {training_metrics['drift_magnitude']:.4f}, "
                          f"Adaptation: {training_metrics['adaptation_rate']:.4f}")
                
                # Evaluate on all tasks
                evaluation_results = self.evaluate_model(
                    model=model,
                    task_ids=list(range(task_id + 1)),  # Evaluate on all tasks seen so far
                    batch_size=batch_size,
                    is_morph=is_morph,
                    detailed_metrics=detailed_eval
                )
                
                # Update accuracy history
                for eval_task_id, result in evaluation_results.items():
                    # Handle both simple and detailed metrics
                    accuracy = result if not detailed_eval else result['accuracy']
                    
                    if eval_task_id not in model_accuracy_history[model_name]:
                        model_accuracy_history[model_name][eval_task_id] = {}
                        if detailed_eval:
                            model_detailed_metrics[model_name][eval_task_id] = {}
                    
                    model_accuracy_history[model_name][eval_task_id][task_id] = accuracy
                    print(f"    Task {eval_task_id} accuracy: {accuracy:.2f}%")
                    
                    if detailed_eval:
                        model_detailed_metrics[model_name][eval_task_id][task_id] = result
        
        # Calculate forgetting metrics
        forgetting_metrics = {}
        for model_name, accuracy_history in model_accuracy_history.items():
            forgetting_metrics[model_name] = calculate_forgetting(accuracy_history)
        
        # Calculate retention metrics
        retention_metrics = {}
        for model_name, accuracy_history in model_accuracy_history.items():
            retention_metrics[model_name] = calculate_knowledge_retention(accuracy_history)
        
        # Calculate average forgetting for each model
        avg_forgetting = {}
        for model_name, forgetting in forgetting_metrics.items():
            if forgetting:
                avg_forgetting[model_name] = sum(forgetting.values()) / len(forgetting)
            else:
                avg_forgetting[model_name] = 0.0
        
        # Calculate final accuracy on all tasks
        final_accuracies = {}
        for model_name, model in models.items():
            is_morph = isinstance(model, MorphModel)
            final_accuracies[model_name] = self.evaluate_model(
                model=model,
                batch_size=batch_size,
                is_morph=is_morph,
                detailed_metrics=detailed_eval
            )
        
        # Calculate concept drift metrics if applicable
        drift_metrics = {}
        if self.drift_detection:
            drift_metrics = self.calculate_concept_drift_metrics()
        
        # Calculate expert utilization metrics for MORPH models
        expert_metrics = {}
        for model_name in models.keys():
            if isinstance(models[model_name], MorphModel):
                expert_metrics[model_name] = self.calculate_expert_utilization_metrics(model_name)
        
        # Compile results
        results = {
            'accuracy_history': model_accuracy_history,
            'forgetting_metrics': forgetting_metrics,
            'retention_metrics': retention_metrics,
            'avg_forgetting': avg_forgetting,
            'final_accuracies': final_accuracies,
            'training_metrics': model_training_metrics,
            'drift_metrics': drift_metrics,
            'expert_metrics': expert_metrics
        }
        
        if detailed_eval:
            results['detailed_metrics'] = model_detailed_metrics
        
        return results

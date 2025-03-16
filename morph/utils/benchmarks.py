import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score

from morph.config import MorphConfig
from morph.core.model import MorphModel


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


class ContinualLearningBenchmark:
    """
    Benchmark for comparing continual learning performance of different models.
    
    This class provides tools to:
    1. Set up sequential tasks with distribution shifts
    2. Train and evaluate models on these tasks
    3. Measure catastrophic forgetting and other continual learning metrics
    4. Compare MORPH with standard models and other continual learning approaches
    5. Detect concept drift and measure adaptation
    6. Evaluate knowledge transfer between related tasks
    """
    
    def __init__(self, 
                 tasks: Dict[int, Dataset],
                 input_size: int,
                 output_size: int,
                 device: torch.device = torch.device("cpu"),
                 drift_detection: bool = False,
                 task_similarities: Optional[Dict[Tuple[int, int], float]] = None):
        """
        Initialize the benchmark.
        
        Args:
            tasks: Dictionary mapping task ID to dataset
            input_size: Dimension of input features
            output_size: Dimension of output features
            device: Device to run benchmark on
            drift_detection: Whether to enable concept drift detection
            task_similarities: Optional dictionary mapping task pairs (i,j) to similarity scores
        """
        self.tasks = tasks
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.drift_detection = drift_detection
        self.task_similarities = task_similarities or {}
        
        # Initial performance tracking
        self.accuracy_history = {}
        self.forgetting_metrics = {}
        
        # Drift detection setup
        if drift_detection:
            self.reference_distributions = {}
            self.distribution_shifts = {}
            
        # Expert utilization tracking for MORPH
        self.expert_utilization = {}
        self.expert_specialization = {}
        
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
    
    def _compute_feature_distribution(self, model, dataloader, is_morph):
        """
        Compute the distribution of features in a dataset.
        
        Args:
            model: Model to use for feature extraction
            dataloader: DataLoader for the dataset
            is_morph: Whether the model is a MORPH model
            
        Returns:
            Feature distribution representation
        """
        # This is a simplified implementation
        # In practice, would use more sophisticated distribution modeling
        
        # Extract predictions on the dataset
        all_outputs = []
        
        model.eval()
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                inputs = inputs.view(inputs.size(0), -1)
                
                # Forward pass
                if is_morph:
                    outputs = model(inputs, training=False)
                else:
                    outputs = model(inputs)
                
                all_outputs.append(outputs.cpu().numpy())
        
        # Concatenate all outputs
        if all_outputs:
            all_outputs = np.concatenate(all_outputs)
            
            # Return simple distribution representation
            # (mean and variance of model outputs)
            return {
                'mean': np.mean(all_outputs, axis=0),
                'var': np.var(all_outputs, axis=0)
            }
        else:
            return {'mean': None, 'var': None}
    
    def _compute_distribution_shift(self, dist1, dist2):
        """
        Compute the shift between two distributions.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Magnitude of distribution shift
        """
        if dist1['mean'] is None or dist2['mean'] is None:
            return 0.0
            
        # Compute Wasserstein distance between distributions
        # This is a simplified version using just mean/variance
        # In practice, would use more sophisticated distance metrics
        
        # Mean shift
        mean_shift = np.mean(np.abs(dist1['mean'] - dist2['mean']))
        
        # Variance shift
        var_shift = np.mean(np.abs(np.sqrt(dist1['var']) - np.sqrt(dist2['var'])))
        
        # Combined shift
        return mean_shift + var_shift
    
    def _compute_adaptation_rate(self, predictions, targets):
        """
        Compute how well the model has adapted to distribution shift.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Adaptation rate metric
        """
        # Simple adaptation rate based on accuracy
        accuracy = np.mean(predictions == targets)
        
        # Scale to [0, 1] range (arbitrary scaling)
        return min(1.0, accuracy * 2)
    
    def calculate_forgetting(self, accuracy_history: Dict[int, Dict[int, float]]):
        """
        Calculate forgetting metrics for each task.
        
        Forgetting measures how much performance on previous tasks decreases
        after learning new tasks.
        
        Args:
            accuracy_history: Dictionary mapping task IDs to accuracy history
            
        Returns:
            Dictionary of forgetting metrics
        """
        forgetting = {}
        
        for task_id in self.tasks.keys():
            if task_id in accuracy_history:
                # Get maximum performance on this task before learning subsequent tasks
                history = accuracy_history[task_id]
                
                if len(history) > 1:
                    # Calculate forgetting as the difference between max performance and final performance
                    max_performance = max(list(history.values())[:-1])
                    final_performance = list(history.values())[-1]
                    forgetting[task_id] = max_performance - final_performance
                else:
                    forgetting[task_id] = 0.0
        
        return forgetting
        
    def calculate_forward_transfer(self, accuracy_history: Dict[int, Dict[int, float]]):
        """
        Calculate forward transfer metrics.
        
        Forward transfer measures how learning a task improves performance on future tasks.
        
        Args:
            accuracy_history: Dictionary mapping task IDs to accuracy history
            
        Returns:
            Dictionary of forward transfer metrics
        """
        forward_transfer = {}
        
        # Use task similarities if available
        if self.task_similarities:
            for (task_i, task_j), similarity in self.task_similarities.items():
                if task_i < task_j and task_i in accuracy_history and task_j in accuracy_history:
                    # Get accuracy on task_j before and after seeing it
                    pre_accuracy = accuracy_history[task_j].get(task_i, 0.0)
                    post_accuracy = list(accuracy_history[task_j].values())[-1]
                    
                    # Weight transfer by task similarity
                    forward_transfer[(task_i, task_j)] = (post_accuracy - pre_accuracy) * similarity
        
        return forward_transfer
    
    def calculate_knowledge_retention(self, accuracy_history: Dict[int, Dict[int, float]]):
        """
        Calculate knowledge retention metrics.
        
        Knowledge retention measures how well the model retains knowledge of previous tasks.
        
        Args:
            accuracy_history: Dictionary mapping task IDs to accuracy history
            
        Returns:
            Dictionary of knowledge retention metrics
        """
        retention = {}
        
        for task_id in sorted(self.tasks.keys()):
            if task_id in accuracy_history:
                history = accuracy_history[task_id]
                
                if len(history) > 1:
                    # Calculate retention as ratio of final to maximum performance
                    max_performance = max(list(history.values())[:-1])
                    final_performance = list(history.values())[-1]
                    
                    if max_performance > 0:
                        retention[task_id] = final_performance / max_performance
                    else:
                        retention[task_id] = 1.0  # No change
                else:
                    retention[task_id] = 1.0  # Perfect retention
        
        return retention
    
    def calculate_concept_drift_metrics(self):
        """
        Calculate metrics related to concept drift detection and adaptation.
        
        Returns:
            Dictionary of concept drift metrics
        """
        if not self.drift_detection or not self.distribution_shifts:
            return {}
            
        metrics = {
            'drift_detected_tasks': 0,
            'avg_drift_magnitude': 0.0,
            'avg_adaptation_rate': 0.0
        }
        
        total_magnitude = 0.0
        total_adaptation = 0.0
        
        for task_id, shifts in self.distribution_shifts.items():
            if shifts:
                metrics['drift_detected_tasks'] += 1
                
                # Max drift magnitude for this task
                task_max_magnitude = max(shift['magnitude'] for shift in shifts)
                total_magnitude += task_max_magnitude
                
                # Average adaptation rate for this task
                task_adaptation = np.mean([shift['adaptation_rate'] for shift in shifts])
                total_adaptation += task_adaptation
        
        # Calculate averages
        if metrics['drift_detected_tasks'] > 0:
            metrics['avg_drift_magnitude'] = total_magnitude / metrics['drift_detected_tasks']
            metrics['avg_adaptation_rate'] = total_adaptation / metrics['drift_detected_tasks']
        
        return metrics
    
    def calculate_expert_utilization_metrics(self, model_name: str):
        """
        Calculate metrics related to expert utilization for MORPH models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of expert utilization metrics
        """
        if model_name not in self.expert_utilization:
            return {}
            
        metrics = {
            'expert_utilization': {},
            'expert_specialization': {},
            'task_expert_overlap': {}
        }
        
        # Calculate expert utilization across tasks
        for task_id, utilization in self.expert_utilization.items():
            metrics['expert_utilization'][task_id] = utilization
            
        # Calculate expert specialization across tasks
        for task_id, specialization in self.expert_specialization.items():
            metrics['expert_specialization'][task_id] = specialization
            
        # Calculate task-expert overlap (which experts are shared between tasks)
        task_ids = sorted(self.expert_utilization.keys())
        for i, task_i in enumerate(task_ids):
            for j, task_j in enumerate(task_ids[i+1:], i+1):
                # Experts used in both tasks
                experts_i = set(self.expert_utilization[task_i].keys())
                experts_j = set(self.expert_utilization[task_j].keys())
                
                common_experts = experts_i & experts_j
                
                # Calculate Jaccard similarity
                if experts_i or experts_j:
                    overlap = len(common_experts) / len(experts_i | experts_j)
                else:
                    overlap = 0.0
                    
                metrics['task_expert_overlap'][(task_i, task_j)] = overlap
        
        return metrics
    
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
            forgetting_metrics[model_name] = self.calculate_forgetting(accuracy_history)
        
        # Calculate retention metrics
        retention_metrics = {}
        for model_name, accuracy_history in model_accuracy_history.items():
            retention_metrics[model_name] = self.calculate_knowledge_retention(accuracy_history)
        
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
    
    def visualize_results(self, results, title="Continual Learning Benchmark", output_path=None):
        """
        Visualize benchmark results.
        
        Args:
            results: Results from run_benchmark
            title: Plot title
            output_path: Path to save visualization (if None, show instead)
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Average accuracy across all tasks
        ax1 = axes[0, 0]
        
        model_names = list(results['final_accuracies'].keys())
        x = np.arange(len(model_names))
        avg_accuracies = [
            sum(accuracies.values()) / len(accuracies) 
            for accuracies in results['final_accuracies'].values()
        ]
        
        bars = ax1.bar(x, avg_accuracies, width=0.6)
        
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Average Accuracy Across All Tasks')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.set_ylim(0, 100)
        
        # Add values on top of bars
        for bar, acc in zip(bars, avg_accuracies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{acc:.1f}%',
                ha='center',
                va='bottom'
            )
        
        # Plot 2: Average forgetting
        ax2 = axes[0, 1]
        
        avg_forgetting = list(results['avg_forgetting'].values())
        
        bars = ax2.bar(x, avg_forgetting, width=0.6, color='orange')
        
        ax2.set_ylabel('Forgetting (%)')
        ax2.set_title('Average Forgetting')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.set_ylim(0, max(avg_forgetting) * 1.2 + 5 if avg_forgetting else 10)
        
        # Add values on top of bars
        for bar, forget in zip(bars, avg_forgetting):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{forget:.1f}%',
                ha='center',
                va='bottom'
            )
        
        # Plot 3: Task accuracy evolution for each model
        ax3 = axes[1, 0]
        
        # Get all tasks
        all_tasks = sorted(list(results['final_accuracies'][model_names[0]].keys()))
        
        # Plot accuracy for each model across tasks
        for model_name in model_names:
            task_accuracies = [results['final_accuracies'][model_name][task_id] for task_id in all_tasks]
            ax3.plot(all_tasks, task_accuracies, 'o-', label=model_name)
        
        ax3.set_xlabel('Task ID')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Final Accuracy by Task')
        ax3.set_xticks(all_tasks)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        
        # Plot 4: Forgetting by task for each model
        ax4 = axes[1, 1]
        
        # Get tasks that have forgetting metrics (all except the last one)
        forgetting_tasks = all_tasks[:-1] if len(all_tasks) > 1 else []
        
        if forgetting_tasks:
            for model_name in model_names:
                if model_name in results['forgetting_metrics']:
                    task_forgetting = [
                        results['forgetting_metrics'][model_name].get(task_id, 0) 
                        for task_id in forgetting_tasks
                    ]
                    ax4.plot(forgetting_tasks, task_forgetting, 'o-', label=model_name)
            
            ax4.set_xlabel('Task ID')
            ax4.set_ylabel('Forgetting (%)')
            ax4.set_title('Forgetting by Task')
            ax4.set_xticks(forgetting_tasks)
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()
        else:
            ax4.set_title('Forgetting by Task (Not enough tasks)')
            ax4.set_xlabel('Task ID')
            ax4.set_ylabel('Forgetting (%)')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_rotating_mnist_tasks(num_tasks=5, samples_per_task=1000, feature_dim=784):
    """
    Create a sequence of tasks with rotating MNIST digits.
    
    Args:
        num_tasks: Number of tasks to create
        samples_per_task: Number of samples per task
        feature_dim: Dimension of input features
        
    Returns:
        Dictionary mapping task IDs to datasets
    """
    # Load MNIST dataset
    from torchvision import datasets, transforms
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Create tasks
    tasks = {}
    
    for task_id in range(num_tasks):
        # Select a subset of MNIST
        indices = torch.randperm(len(train_dataset))[:samples_per_task]
        
        # Get data and targets
        data = torch.stack([train_dataset[i][0] for i in indices])
        targets = torch.tensor([train_dataset[i][1] for i in indices])
        
        # Apply rotation based on task ID
        rotation_angle = task_id * 15  # 15 degree increments
        
        # Apply rotation
        if rotation_angle > 0:
            rotated_data = transforms.functional.rotate(data, rotation_angle)
        else:
            rotated_data = data
        
        # Flatten images
        flat_data = rotated_data.view(-1, feature_dim)
        
        # Create dataset
        tasks[task_id] = TensorDataset(flat_data, targets)
    
    return tasks


def create_split_mnist_tasks(num_tasks=5):
    """
    Create a sequence of tasks by splitting MNIST into different digit groups.
    
    Args:
        num_tasks: Number of tasks to create
        
    Returns:
        Dictionary mapping task IDs to datasets
    """
    # Load MNIST dataset
    from torchvision import datasets, transforms
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Split digits into groups
    digits_per_task = 10 // num_tasks
    
    # Create tasks
    tasks = {}
    
    for task_id in range(num_tasks):
        # Determine digits for this task
        start_digit = task_id * digits_per_task
        end_digit = start_digit + digits_per_task
        
        # Select examples with these digits
        indices = [i for i, (_, target) in enumerate(train_dataset) 
                   if start_digit <= target < end_digit]
        
        # Get data and targets
        data = torch.stack([train_dataset[i][0] for i in indices])
        targets = torch.tensor([train_dataset[i][1] - start_digit for i in indices])  # Remap to 0-based
        
        # Flatten images
        flat_data = data.view(-1, 784)
        
        # Create dataset
        tasks[task_id] = TensorDataset(flat_data, targets)
    
    return tasks


def create_permuted_mnist_tasks(num_tasks=5, samples_per_task=5000):
    """
    Create a sequence of tasks with permuted MNIST digits.
    Each task applies a different fixed permutation to the pixels.
    
    Args:
        num_tasks: Number of tasks to create
        samples_per_task: Number of samples per task
        
    Returns:
        Dictionary mapping task IDs to datasets
    """
    # Load MNIST dataset
    from torchvision import datasets, transforms
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Create tasks
    tasks = {}
    
    for task_id in range(num_tasks):
        # Select a subset of MNIST
        indices = torch.randperm(len(train_dataset))[:samples_per_task]
        
        # Get data and targets
        data = torch.stack([train_dataset[i][0] for i in indices])
        targets = torch.tensor([train_dataset[i][1] for i in indices])
        
        # Flatten images
        flat_data = data.view(-1, 784)
        
        # Apply permutation (except for first task)
        if task_id > 0:
            # Generate a fixed permutation for this task
            permutation = torch.randperm(784)
            permuted_data = flat_data[:, permutation]
        else:
            permuted_data = flat_data
        
        # Create dataset
        tasks[task_id] = TensorDataset(permuted_data, targets)
    
    return tasks
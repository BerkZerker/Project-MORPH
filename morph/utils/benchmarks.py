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
    """
    
    def __init__(self, 
                 tasks: Dict[int, Dataset],
                 input_size: int,
                 output_size: int,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize the benchmark.
        
        Args:
            tasks: Dictionary mapping task ID to dataset
            input_size: Dimension of input features
            output_size: Dimension of output features
            device: Device to run benchmark on
        """
        self.tasks = tasks
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        
        # Initial performance tracking
        self.accuracy_history = {}
        self.forgetting_metrics = {}
        
    def train_model(self, 
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    task_id: int,
                    epochs: int = 5,
                    batch_size: int = 32,
                    is_morph: bool = False,
                    is_ewc: bool = False):
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
            'train_accuracy': []
        }
        
        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
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
            
            # Epoch metrics
            epoch_loss = running_loss / len(dataloader)
            epoch_accuracy = 100. * correct / total
            
            metrics['train_loss'].append(epoch_loss)
            metrics['train_accuracy'].append(epoch_accuracy)
        
        # Store Fisher information for EWC
        if is_ewc:
            model.calculate_fisher_information(dataloader, self.device)
            
        return metrics
    
    def evaluate_model(self, 
                       model: nn.Module,
                       task_ids: Optional[List[int]] = None,
                       batch_size: int = 32,
                       is_morph: bool = False):
        """
        Evaluate a model on one or more tasks.
        
        Args:
            model: Model to evaluate
            task_ids: List of task IDs to evaluate on (default: all tasks)
            batch_size: Batch size for evaluation
            is_morph: Whether the model is a MORPH model
            
        Returns:
            Dictionary mapping task IDs to accuracy
        """
        model.eval()
        
        if task_ids is None:
            task_ids = list(self.tasks.keys())
            
        accuracies = {}
        
        for task_id in task_ids:
            dataloader = DataLoader(
                self.tasks[task_id],
                batch_size=batch_size,
                shuffle=False
            )
            
            correct = 0
            total = 0
            
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
            
            accuracies[task_id] = 100. * correct / total
        
        return accuracies
    
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
    
    def run_benchmark(self, 
                      models: Dict[str, nn.Module],
                      optimizers: Dict[str, torch.optim.Optimizer],
                      epochs_per_task: int = 5,
                      batch_size: int = 32):
        """
        Run a full continual learning benchmark with multiple models.
        
        Args:
            models: Dictionary mapping model names to models
            optimizers: Dictionary mapping model names to optimizers
            epochs_per_task: Number of epochs to train on each task
            batch_size: Batch size for training and evaluation
            
        Returns:
            Dictionary of benchmark results
        """
        # Track accuracy history for each model and task
        model_accuracy_history = {name: {} for name in models.keys()}
        
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
                
                self.train_model(
                    model=model,
                    optimizer=optimizer,
                    task_id=task_id,
                    epochs=epochs_per_task,
                    batch_size=batch_size,
                    is_morph=is_morph,
                    is_ewc=is_ewc
                )
                
                # Evaluate on all tasks
                accuracies = self.evaluate_model(
                    model=model,
                    task_ids=list(range(task_id + 1)),  # Evaluate on all tasks seen so far
                    batch_size=batch_size,
                    is_morph=is_morph
                )
                
                # Update accuracy history
                for eval_task_id, accuracy in accuracies.items():
                    if eval_task_id not in model_accuracy_history[model_name]:
                        model_accuracy_history[model_name][eval_task_id] = {}
                    
                    model_accuracy_history[model_name][eval_task_id][task_id] = accuracy
                    print(f"    Task {eval_task_id} accuracy: {accuracy:.2f}%")
        
        # Calculate forgetting metrics
        forgetting_metrics = {}
        
        for model_name, accuracy_history in model_accuracy_history.items():
            forgetting_metrics[model_name] = self.calculate_forgetting(accuracy_history)
        
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
                is_morph=is_morph
            )
        
        # Compile results
        results = {
            'accuracy_history': model_accuracy_history,
            'forgetting_metrics': forgetting_metrics,
            'avg_forgetting': avg_forgetting,
            'final_accuracies': final_accuracies
        }
        
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
"""
MNIST dataset variants for continual learning benchmarks.
"""

import torch
from torch.utils.data import TensorDataset


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

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from typing import Tuple, List, Dict


def get_mnist_dataloaders(batch_size=64, num_workers=2):
    """
    Get MNIST dataset loaders for basic testing.
    
    Args:
        batch_size: Batch size for training/testing
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


class ContinualTaskDataset(Dataset):
    """
    Dataset for continual learning with distribution shifts.
    
    This dataset sequentially introduces new tasks, allowing for
    testing of catastrophic forgetting and adaptation.
    """
    
    def __init__(self, base_datasets, task_schedule, transform=None):
        """
        Initialize the continual learning dataset.
        
        Args:
            base_datasets: List of datasets to draw from
            task_schedule: Dict mapping step numbers to active tasks
            transform: Optional transform to be applied to the data
        """
        self.base_datasets = base_datasets
        self.task_schedule = task_schedule
        self.transform = transform
        self.current_step = 0
        self.active_tasks = self._get_active_tasks(0)
        
    def _get_active_tasks(self, step):
        """
        Determine which tasks are active at the current step.
        """
        active_tasks = []
        for task_id, schedule in self.task_schedule.items():
            if schedule[0] <= step and (schedule[1] is None or step < schedule[1]):
                active_tasks.append(task_id)
        return active_tasks
    
    def set_step(self, step):
        """
        Update the current training step.
        """
        self.current_step = step
        self.active_tasks = self._get_active_tasks(step)
        
    def __len__(self):
        """
        Get the total length of the dataset.
        """
        return sum(len(self.base_datasets[task_id]) for task_id in self.active_tasks)
    
    def __getitem__(self, idx):
        """
        Get an item from the active datasets.
        """
        # Find which dataset this index belongs to
        for task_id in self.active_tasks:
            task_len = len(self.base_datasets[task_id])
            if idx < task_len:
                # Get the item
                item = self.base_datasets[task_id][idx]
                
                # Add task_id as metadata
                if isinstance(item, tuple):
                    return item + (task_id,)
                else:
                    return (item, task_id)
            idx -= task_len
            
        raise IndexError("Index out of bounds")

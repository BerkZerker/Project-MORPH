import torch
from src.utils.data import ContinualTaskDataset
from tests.continual_learning.test_dataset import TestDataset


def test_continual_dataset_getitem():
    """Test retrieving items from the dataset."""
    # Create base datasets
    datasets = {
        0: TestDataset(size=100, feature_dim=10),
        1: TestDataset(size=200, feature_dim=10)
    }
    
    # Create task schedule
    task_schedule = {
        0: (0, 1000),
        1: (1000, 2000)
    }
    
    # Create continual dataset
    continual_dataset = ContinualTaskDataset(datasets, task_schedule)
    
    # Check retrieving items from task 0
    continual_dataset.set_step(500)
    item = continual_dataset[0]
    
    # Should return (data, target, task_id)
    assert len(item) == 3
    assert item[0].shape == torch.Size([10])  # Feature dimension
    assert item[2] == 0  # Task ID
    
    # Test with overlapping tasks
    task_schedule = {
        0: (0, 1500),
        1: (1000, 2000)
    }
    
    continual_dataset = ContinualTaskDataset(datasets, task_schedule)
    continual_dataset.set_step(1200)  # Both tasks active
    
    # First 100 items should be from task 0
    item = continual_dataset[50]
    assert item[2] == 0
    
    # Items after 100 should be from task 1
    item = continual_dataset[150]
    assert item[2] == 1

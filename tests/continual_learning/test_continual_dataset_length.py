import torch
from src.utils.data import ContinualTaskDataset
from tests.continual_learning.test_dataset import TestDataset


def test_continual_dataset_length():
    """Test that dataset length is the sum of active datasets."""
    # Create base datasets
    datasets = {
        0: TestDataset(size=100),
        1: TestDataset(size=200),
        2: TestDataset(size=150)
    }
    
    # Create task schedule
    task_schedule = {
        0: (0, 1000),
        1: (1000, 2000),
        2: (2000, None)
    }
    
    # Create continual dataset
    continual_dataset = ContinualTaskDataset(datasets, task_schedule)
    
    # Check length with different active tasks
    continual_dataset.set_step(500)  # Only task 0 active
    assert len(continual_dataset) == 100
    
    # Test overlapping tasks
    task_schedule = {
        0: (0, 1500),
        1: (1000, 2500),
        2: (2000, None)
    }
    
    continual_dataset = ContinualTaskDataset(datasets, task_schedule)
    
    continual_dataset.set_step(1200)  # Tasks 0 and 1 active
    assert len(continual_dataset) == 100 + 200

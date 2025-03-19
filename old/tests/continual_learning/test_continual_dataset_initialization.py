import torch
from src.utils.data import ContinualTaskDataset
from tests.continual_learning.test_dataset import TestDataset


def test_continual_dataset_initialization():
    """Test that ContinualTaskDataset initializes correctly."""
    # Create base datasets
    datasets = {
        0: TestDataset(size=100),
        1: TestDataset(size=200),
        2: TestDataset(size=150)
    }
    
    # Create task schedule
    task_schedule = {
        0: (0, 1000),     # Task 0 active from step 0 to 1000
        1: (1000, 2000),  # Task 1 active from step 1000 to 2000
        2: (2000, None)   # Task 2 active from step 2000 onwards
    }
    
    # Create continual dataset
    continual_dataset = ContinualTaskDataset(datasets, task_schedule)
    
    # Check initial state
    assert continual_dataset.current_step == 0
    assert continual_dataset.active_tasks == [0]  # Only task 0 should be active

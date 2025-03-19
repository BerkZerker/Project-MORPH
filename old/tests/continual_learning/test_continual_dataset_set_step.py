import torch
from src.utils.data import ContinualTaskDataset
from tests.continual_learning.test_dataset import TestDataset


def test_continual_dataset_set_step():
    """Test that setting the step updates active tasks."""
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
    
    # Set step to various points and check active tasks
    continual_dataset.set_step(500)
    assert continual_dataset.active_tasks == [0]
    
    continual_dataset.set_step(1500)
    assert continual_dataset.active_tasks == [1]
    
    continual_dataset.set_step(2500)
    assert continual_dataset.active_tasks == [2]
    
    # Test overlapping tasks
    task_schedule = {
        0: (0, 1500),      # Task 0 active from step 0 to 1500
        1: (1000, 2500),   # Task 1 active from step 1000 to 2500
        2: (2000, None)    # Task 2 active from step 2000 onwards
    }
    
    continual_dataset = ContinualTaskDataset(datasets, task_schedule)
    
    continual_dataset.set_step(1200)
    assert set(continual_dataset.active_tasks) == {0, 1}
    
    continual_dataset.set_step(2200)
    assert set(continual_dataset.active_tasks) == {1, 2}

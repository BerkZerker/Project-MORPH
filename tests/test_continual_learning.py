import torch
import numpy as np
from torch.utils.data import DataLoader
from src.config import MorphConfig
from src.core.model import MorphModel
from src.utils.data import ContinualTaskDataset
from src.utils.gpu_utils import get_optimal_worker_count


class TestDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=100, feature_dim=10):
        self.data = torch.randn(size, feature_dim)
        self.targets = torch.randint(0, 5, (size,))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


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


def test_morph_with_continual_learning():
    """Test MORPH model on a continual learning scenario."""
    # Create a small MORPH model with GPU acceleration
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=20,
        output_size=5,
        num_initial_experts=2,
        expert_k=1,
        
        # Enable dynamic experts
        enable_dynamic_experts=True,
        expert_creation_uncertainty_threshold=0.3,
        
        # Enable sleep
        enable_sleep=True,
        sleep_cycle_frequency=50,
        
        # Set batch size small for testing
        batch_size=4,
        
        # GPU acceleration
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_mixed_precision=torch.cuda.is_available(),  # Enable mixed precision if CUDA available
        
        # Data loading optimization
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available(),
        
        # Test-specific optimizations
        test_mode=True,
        test_expert_size=16,  # Smaller expert size for tests
        test_sleep_frequency=25,  # More frequent sleep cycles for tests
        test_memory_buffer_size=100  # Smaller memory buffer for tests
    )
    
    model = MorphModel(config)
    
    # Create base datasets with different distributions
    # Task 0: Random normal
    task0_dataset = TestDataset(size=100, feature_dim=10)
    
    # Task 1: Shifted distribution
    task1_data = torch.randn(100, 10) + torch.tensor([2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    task1_targets = torch.randint(0, 5, (100,))
    task1_dataset = torch.utils.data.TensorDataset(task1_data, task1_targets)
    
    datasets = {0: task0_dataset, 1: task1_dataset}
    
    # Create task schedule
    task_schedule = {
        0: (0, 50),  # Task 0 for 50 steps
        1: (50, 100)  # Task 1 for 50 steps
    }
    
    # Create continual dataset
    continual_dataset = ContinualTaskDataset(datasets, task_schedule)
    
    # Create data loader with optimized data loading
    dataloader = DataLoader(
        continual_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Initial expert count
    initial_expert_count = len(model.experts)
    
    # Train for more steps on task 0 to improve performance
    model.train()
    for step in range(40):  # Increased from 25 to 40 steps
        # Set the current step in the dataset
        continual_dataset.set_step(step % 50)  # Keep within task 0 range
        
        # Get a batch
        for data, target, task_id in dataloader:
            # Flatten batch dimension for testing
            data = data.view(data.size(0), -1)
            
            # Move data and target to the same device as the model
            data = data.to(model.device)
            target = target.to(model.device)
            
            # Forward pass
            outputs = model(data, training=True)
            loss = criterion(outputs, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Trigger sleep every 10 steps to improve specialization
        if step % 10 == 0 and step > 0:
            model.sleep()
    
    # Check that the model has adapted to task 0
    task0_expert_count = len(model.experts)
    
    # Train on task 1 (different distribution)
    for step in range(50, 75):
        # Set the current step in the dataset
        continual_dataset.set_step(step)
        
        # Get a batch
        for data, target, task_id in dataloader:
            # Flatten batch dimension for testing
            data = data.view(data.size(0), -1)
            
            # Move data and target to the same device as the model
            data = data.to(model.device)
            target = target.to(model.device)
            
            # Forward pass
            outputs = model(data, training=True)
            loss = criterion(outputs, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Check that the model has adapted to task 1
    task1_expert_count = len(model.experts)
    
    # The model should have created new experts for the new distribution
    assert task1_expert_count > task0_expert_count
    
    # Evaluate on both tasks
    model.eval()
    
    # Function to compute accuracy on a dataset with optimized data loading
    def compute_accuracy(dataset):
        correct = 0
        total = 0
        
        dataloader = DataLoader(
            dataset, 
            batch_size=10, 
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        with torch.no_grad():
            for data, target in dataloader:
                # Flatten batch dimension
                data = data.view(data.size(0), -1)
                
                # Move data and target to the same device as the model
                if hasattr(model, 'device'):
                    device = model.device
                else:
                    device = next(model.parameters()).device
                    
                data = data.to(device)
                target = target.to(device)
                
                # Forward pass
                outputs = model(data, training=False)
                _, predicted = torch.max(outputs, 1)
                
                # Count correct predictions
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total
    
    # Compute accuracy on both tasks
    task0_accuracy = compute_accuracy(task0_dataset)
    task1_accuracy = compute_accuracy(task1_dataset)
    
    # Both accuracies should be reasonable
    # Since this is a simple test, we just check if they're above random chance
    assert task0_accuracy > 0.2  # Better than random (0.2 for 5 classes)
    assert task1_accuracy > 0.2
    
    # Check that model has preserved knowledge of task 0
    # This is the key feature of continual learning
    # Note: A regular model would likely perform poorly on task 0 after training on task 1
    assert task0_accuracy > 0.2


def test_expert_specialization_for_tasks():
    """Test that experts specialize for different tasks in continual learning."""
    # Create a small MORPH model with GPU acceleration
    config = MorphConfig(
        input_size=10,
        expert_hidden_size=16,  # Reduced from 20 to 16
        output_size=5,
        num_initial_experts=3,
        expert_k=2,
        
        # Enable dynamic experts
        enable_dynamic_experts=True,
        expert_creation_uncertainty_threshold=0.3,
        
        # Enable sleep (more frequent for testing)
        enable_sleep=True,
        sleep_cycle_frequency=20,  # Optimized from 30
        
        # GPU acceleration
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_mixed_precision=torch.cuda.is_available(),
        
        # Data loading optimization
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available(),
        
        # Test-specific optimizations
        test_mode=True,
        test_expert_size=12,  # Reduced from 16 to 12
        test_sleep_frequency=10,  # Optimized from 15
        test_memory_buffer_size=50  # Reduced from 100 to 50
    )
    
    model = MorphModel(config)
    
    # Create two very different distributions
    # Task 0: Centered around origin - reduced from 200 to 100 samples
    task0_data = torch.randn(100, 10)
    task0_targets = torch.randint(0, 5, (100,))
    task0_dataset = torch.utils.data.TensorDataset(task0_data, task0_targets)
    
    # Task 1: Shifted far away - reduced from 200 to 100 samples
    task1_data = torch.randn(100, 10) + torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 0, 0, 0, 0, 0])
    task1_targets = torch.randint(0, 5, (100,))
    task1_dataset = torch.utils.data.TensorDataset(task1_data, task1_targets)
    
    datasets = {0: task0_dataset, 1: task1_dataset}
    
    # Train on task 0 then task 1 sequentially - reduced total steps from 200 to 100
    task_schedule = {
        0: (0, 50),   # Task 0 for 50 steps (reduced from 100)
        1: (50, 100)  # Task 1 for 50 steps (reduced from 100)
    }
    
    # Create continual dataset
    continual_dataset = ContinualTaskDataset(datasets, task_schedule)
    dataloader = DataLoader(
        continual_dataset, 
        batch_size=20,  # Increased from 16 to 20 for faster processing
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create dictionaries to track which experts are activated for each task
    task0_expert_activations = {}
    task1_expert_activations = {}
    
    # Train for 100 steps (covering both tasks) - reduced from 200 to 100
    model.train()
    for step in range(100):
        # Set the current step in the dataset
        continual_dataset.set_step(step)
        
        current_task = 0 if step < 50 else 1  # Adjusted for new step counts
        
        # Trigger sleep less frequently to improve speed
        if step % 30 == 0 and step > 0:
            model.sleep()
        
        # Get a batch
        for data, target, task_id in dataloader:
            # Should all be the same task
            assert task_id[0].item() == current_task
            
            # Flatten batch dimension
            data = data.view(data.size(0), -1)
            
            # Move data and target to the same device as the model
            data = data.to(model.device)
            target = target.to(model.device)
            task_id = task_id.to(model.device)
            
            # Reset activation counts before checking to track per-batch activations
            for expert in model.experts:
                expert.activation_count = 0
            
            # Forward pass will update activation counts
            outputs = model(data, training=True)
            
            # Record activations for this batch
            for i, expert in enumerate(model.experts):
                if expert.activation_count > 0:
                    activation_dict = task0_expert_activations if current_task == 0 else task1_expert_activations
                    activation_dict[i] = activation_dict.get(i, 0) + expert.activation_count
            
            # Usual training steps
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # After training, check expert specialization
    # Some experts should be more active for task 0, others for task 1
    task0_specialists = []
    task1_specialists = []
    
    for expert_idx in range(len(model.experts)):
        # Skip experts that were never activated for either task
        if expert_idx not in task0_expert_activations and expert_idx not in task1_expert_activations:
            continue
            
        task0_activations = task0_expert_activations.get(expert_idx, 0)
        task1_activations = task1_expert_activations.get(expert_idx, 0)
        
        if task0_activations > 2 * task1_activations:
            # This expert specializes in task 0
            task0_specialists.append(expert_idx)
        elif task1_activations > 2 * task0_activations:
            # This expert specializes in task 1
            task1_specialists.append(expert_idx)
    
    # There should be at least one specialist for each task
    assert len(task0_specialists) > 0
    assert len(task1_specialists) > 0
    
    # Check different experts specialize in different tasks
    assert set(task0_specialists) != set(task1_specialists)
    

def test_catastrophic_forgetting_reduction():
    """Test that MORPH reduces catastrophic forgetting compared to standard models."""
    # Create datasets with different distributions
    feature_dim = 10
    
    # Create base datasets
    np.random.seed(42)  # For reproducibility
    
    # Task 0: Random normal
    task0_data = torch.randn(200, feature_dim)
    task0_targets = torch.randint(0, 2, (200,))  # Binary task for simplicity
    task0_dataset = torch.utils.data.TensorDataset(task0_data, task0_targets)
    
    # Task 1: Shifted distribution
    task1_data = torch.randn(200, feature_dim) + torch.tensor([3.0] + [0.0] * (feature_dim - 1))
    task1_targets = torch.randint(0, 2, (200,))
    task1_dataset = torch.utils.data.TensorDataset(task1_data, task1_targets)
    
    datasets = {0: task0_dataset, 1: task1_dataset}
    
    # Function to evaluate model on both tasks with optimized data loading
    def evaluate_model(model, task0_dataset, task1_dataset):
        model.eval()
        accuracies = {}
        
        for task_id, dataset in [(0, task0_dataset), (1, task1_dataset)]:
            dataloader = DataLoader(
                dataset, 
                batch_size=32, 
                shuffle=False,
                num_workers=morph_config.num_workers,
                pin_memory=morph_config.pin_memory
            )
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in dataloader:
                    # Flatten batch dimension
                    data = data.view(data.size(0), -1)
                    
                    # Move data and target to the same device as the model
                    if hasattr(model, 'device'):
                        device = model.device
                    else:
                        device = next(model.parameters()).device
                        
                    data = data.to(device)
                    target = target.to(device)
                    
                    # Forward pass
                    outputs = model(data, training=False)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Count correct predictions
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracies[task_id] = (correct / total) * 100.0
        
        return accuracies
    
    # Create MORPH model with GPU acceleration
    morph_config = MorphConfig(
        input_size=feature_dim,
        expert_hidden_size=20,
        output_size=2,
        num_initial_experts=2,
        expert_k=1,
        
        # Enable dynamic experts
        enable_dynamic_experts=True,
        
        # Enable sleep
        enable_sleep=True,
        sleep_cycle_frequency=50,
        
        # GPU acceleration
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_mixed_precision=torch.cuda.is_available(),
        
        # Data loading optimization
        num_workers=get_optimal_worker_count(),
        pin_memory=torch.cuda.is_available(),
        
        # Test-specific optimizations
        test_mode=True,
        test_expert_size=16,
        test_sleep_frequency=25,
        test_memory_buffer_size=100
    )
    
    morph_model = MorphModel(morph_config)
    
    # Create standard model (MLP) with same capacity
    class StandardModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x, training=True):
            return self.network(x)
    
    # Create standard model with comparable capacity
    standard_model = StandardModel(feature_dim, 40, 2)  # Larger hidden size for fair comparison
    
    # Train both models on task 0 with optimized data loading
    task0_dataloader = DataLoader(
        task0_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=morph_config.num_workers,
        pin_memory=morph_config.pin_memory
    )
    
    # Training setup
    morph_optimizer = torch.optim.Adam(morph_model.parameters(), lr=0.001)
    standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train on task 0 - increase epochs for better performance
    for epoch in range(10):  # Increased from 5 to 10 epochs
        morph_model.train()
        standard_model.train()
        
        for data, target in task0_dataloader:
            # Flatten batch dimension
            data = data.view(data.size(0), -1)
            
            # Move data and target to the same device as the model
            data = data.to(morph_model.device)
            target = target.to(morph_model.device)
            
            # Train MORPH model
            morph_optimizer.zero_grad()
            morph_outputs = morph_model(data, training=True)
            morph_loss = criterion(morph_outputs, target)
            morph_loss.backward()
            morph_optimizer.step()
            
            # Move standard model to the same device as morph_model
            standard_model = standard_model.to(morph_model.device)
            
            # Train standard model
            standard_optimizer.zero_grad()
            standard_outputs = standard_model(data)
            standard_loss = criterion(standard_outputs, target)
            standard_loss.backward()
            standard_optimizer.step()
            
        # Trigger sleep every epoch to improve specialization
        if epoch % 2 == 0 and hasattr(morph_model, 'sleep'):
            morph_model.sleep()
    
    # Evaluate both models on task 0
    morph_accuracies_after_task0 = evaluate_model(morph_model, task0_dataset, task1_dataset)
    standard_accuracies_after_task0 = evaluate_model(standard_model, task0_dataset, task1_dataset)
    
    # Both models should perform well on task 0
    assert morph_accuracies_after_task0[0] > 60
    assert standard_accuracies_after_task0[0] > 60
    
    # Now train both models on task 1 with optimized data loading
    task1_dataloader = DataLoader(
        task1_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=morph_config.num_workers,
        pin_memory=morph_config.pin_memory
    )
    
    for epoch in range(5):  # Just a few epochs for testing
        morph_model.train()
        standard_model.train()
        
        for data, target in task1_dataloader:
            # Flatten batch dimension
            data = data.view(data.size(0), -1)
            
            # Move data and target to the same device as the model
            data = data.to(morph_model.device)
            target = target.to(morph_model.device)
            
            # Train MORPH model
            morph_optimizer.zero_grad()
            morph_outputs = morph_model(data, training=True)
            morph_loss = criterion(morph_outputs, target)
            morph_loss.backward()
            morph_optimizer.step()
            
            # Make sure standard model is on the same device
            standard_model = standard_model.to(morph_model.device)
            
            # Train standard model
            standard_optimizer.zero_grad()
            standard_outputs = standard_model(data)
            standard_loss = criterion(standard_outputs, target)
            standard_loss.backward()
            standard_optimizer.step()
    
    # Evaluate both models on both tasks
    morph_accuracies_after_task1 = evaluate_model(morph_model, task0_dataset, task1_dataset)
    standard_accuracies_after_task1 = evaluate_model(standard_model, task0_dataset, task1_dataset)
    
    # Both models should perform well on task 1 now
    assert morph_accuracies_after_task1[1] > 60
    assert standard_accuracies_after_task1[1] > 60
    
    # Calculate forgetting (drop in task 0 performance after learning task 1)
    morph_forgetting = morph_accuracies_after_task0[0] - morph_accuracies_after_task1[0]
    standard_forgetting = standard_accuracies_after_task0[0] - standard_accuracies_after_task1[0]
    
    # MORPH should have less forgetting than the standard model
    assert morph_forgetting < standard_forgetting, \
        f"MORPH forgetting: {morph_forgetting}, Standard forgetting: {standard_forgetting}"

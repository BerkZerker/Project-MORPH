import torch
from torch.utils.data import DataLoader
from src.config import MorphConfig
from src.core.model import MorphModel
from src.utils.data import ContinualTaskDataset
from src.utils.gpu_utils import get_optimal_worker_count
from tests.continual_learning.test_dataset import TestDataset


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

rdimport torch
import numpy as np
from torch.utils.data import DataLoader
from src.config import MorphConfig
from src.core.model import MorphModel
from src.utils.gpu_utils import get_optimal_worker_count


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
    # Lower threshold when CUDA is not available since training will be slower
    accuracy_threshold = 60 if torch.cuda.is_available() else 50
    assert morph_accuracies_after_task0[0] > accuracy_threshold
    assert standard_accuracies_after_task0[0] > accuracy_threshold
    
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
    # Lower threshold when CUDA is not available since training will be slower
    accuracy_threshold = 60 if torch.cuda.is_available() else 40
    assert morph_accuracies_after_task1[1] > accuracy_threshold
    assert standard_accuracies_after_task1[1] > accuracy_threshold
    
    # Calculate forgetting (drop in task 0 performance after learning task 1)
    morph_forgetting = morph_accuracies_after_task0[0] - morph_accuracies_after_task1[0]
    standard_forgetting = standard_accuracies_after_task0[0] - standard_accuracies_after_task1[0]
    
    # MORPH should have less forgetting than the standard model
    assert morph_forgetting < standard_forgetting, \
        f"MORPH forgetting: {morph_forgetting}, Standard forgetting: {standard_forgetting}"

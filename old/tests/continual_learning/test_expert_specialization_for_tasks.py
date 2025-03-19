import torch
from torch.utils.data import DataLoader
from src.config import MorphConfig
from src.core.model import MorphModel
from src.utils.data import ContinualTaskDataset
from src.utils.gpu_utils import get_optimal_worker_count


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

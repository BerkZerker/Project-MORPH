import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add parent directory to path to import morph
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from morph.config import MorphConfig
from morph.core.model import MorphModel
from morph.utils.data import get_mnist_dataloaders, ContinualTaskDataset
from morph.utils.visualization import (
    visualize_knowledge_graph, 
    plot_expert_activations,
    visualize_expert_lifecycle,
    visualize_sleep_metrics
)


class RotatedMNISTTask:
    """Helper class to generate a series of related but different tasks using MNIST."""
    
    def __init__(self, base_rotation=0, task_count=5, samples_per_task=10000, rotation_increment=15):
        """
        Initialize the RotatedMNISTTask generator.
        
        Args:
            base_rotation: Initial rotation in degrees
            task_count: Number of tasks to generate
            samples_per_task: Number of samples per task
            rotation_increment: Degrees to rotate for each new task
        """
        self.base_rotation = base_rotation
        self.task_count = task_count
        self.samples_per_task = samples_per_task
        self.rotation_increment = rotation_increment
        
        # Load MNIST dataset
        from torchvision import datasets, transforms
        
        self.base_dataset = datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
    
    def get_task_datasets(self):
        """
        Generate task datasets with increasingly rotated MNIST digits.
        
        Returns:
            Dictionary mapping task_id to task datasets
        """
        from torchvision import transforms
        
        task_datasets = {}
        
        for task_id in range(self.task_count):
            # Calculate rotation for this task
            rotation = self.base_rotation + (task_id * self.rotation_increment)
            
            # Create transform for this task
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation((rotation, rotation)),  # Fixed rotation
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Create a custom dataset with rotated images
            dataset = datasets.MNIST(
                root='./data', 
                train=True, 
                download=True, 
                transform=transform
            )
            
            # Subsample to desired size
            indices = torch.randperm(len(dataset))[:self.samples_per_task]
            task_datasets[task_id] = torch.utils.data.Subset(dataset, indices)
        
        return task_datasets


def evaluate_forgetting(model, task_datasets, device, n_samples=1000):
    """
    Evaluate catastrophic forgetting on previous tasks.
    
    Args:
        model: The MORPH model
        task_datasets: Dictionary of task datasets
        device: Device to run evaluation on
        n_samples: Number of samples to evaluate per task
        
    Returns:
        Dictionary mapping task_id to accuracy
    """
    model.eval()
    task_accuracies = {}
    
    for task_id, dataset in task_datasets.items():
        # Create a limited evaluation set
        indices = torch.randperm(len(dataset))[:n_samples]
        eval_set = torch.utils.data.Subset(dataset, indices)
        eval_loader = torch.utils.data.DataLoader(
            eval_set, batch_size=64, shuffle=False
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(device), target.to(device)
                
                # Flatten images
                data = data.view(data.size(0), -1)
                
                # Forward pass
                output = model(data, training=False)
                
                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        task_accuracies[task_id] = accuracy
        
    return task_accuracies


def visualize_task_performance(task_accuracies_history, output_path=None):
    """
    Visualize performance on each task over time.
    
    Args:
        task_accuracies_history: List of (step, {task_id: accuracy}) pairs
        output_path: Path to save visualization
    """
    plt.figure(figsize=(12, 6))
    
    # Extract steps and organize accuracies by task
    steps = [step for step, _ in task_accuracies_history]
    
    # Get all task_ids
    all_task_ids = set()
    for _, accuracies in task_accuracies_history:
        all_task_ids.update(accuracies.keys())
    
    # Colors for tasks
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_task_ids)))
    
    # Plot accuracy for each task over time
    for i, task_id in enumerate(sorted(all_task_ids)):
        accuracies = [acc.get(task_id, float('nan')) for _, acc in task_accuracies_history]
        
        # Label first occurrence of task
        task_steps = []
        for j, (step, acc_dict) in enumerate(task_accuracies_history):
            if task_id in acc_dict:
                task_steps.append(step)
                
        # Find first non-NaN value
        first_idx = next((i for i, val in enumerate(accuracies) if not np.isnan(val)), None)
        
        # Plot with appropriate label
        plt.plot(steps, accuracies, '-o', color=colors[i], linewidth=2, 
                label=f'Task {task_id}', alpha=0.8)
        
        # Mark first occurrence
        if first_idx is not None:
            plt.scatter([steps[first_idx]], [accuracies[first_idx]], color=colors[i], 
                       s=100, zorder=10, edgecolor='black')
    
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy (%)')
    plt.title('Task Performance Over Time (Catastrophic Forgetting Analysis)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Highlight regions when each task was the active task
    task_regions = defaultdict(list)
    active_task = None
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def main():
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/continual', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create rotated MNIST tasks
    task_generator = RotatedMNISTTask(
        base_rotation=0, 
        task_count=5,  # 5 tasks with increasing rotation
        samples_per_task=10000,
        rotation_increment=15  # Each task rotates digits 15 degrees more
    )
    
    task_datasets = task_generator.get_task_datasets()
    
    # Define task schedule
    task_schedule = {}
    steps_per_task = 2000  # Show each task for this many steps
    current_step = 0
    
    for task_id in range(len(task_datasets)):
        # Task starts at current_step and ends after steps_per_task
        task_schedule[task_id] = (current_step, current_step + steps_per_task)
        current_step += steps_per_task
    
    # Create continual learning dataset
    continual_dataset = ContinualTaskDataset(task_datasets, task_schedule)
    
    # Configure the MORPH model with settings optimized for continual learning
    config = MorphConfig(
        input_size=784,  # 28x28 MNIST images
        expert_hidden_size=256,
        output_size=10,  # 10 MNIST classes
        num_initial_experts=3,  # Start with fewer experts
        expert_k=2,
        
        # Dynamic expert creation settings - more aggressive creation
        enable_dynamic_experts=True,
        expert_creation_uncertainty_threshold=0.2,  # Lower threshold to encourage expert creation
        min_experts=3,
        max_experts=15,
        
        # Expert merging and pruning settings - less aggressive merging
        enable_sleep=True,
        sleep_cycle_frequency=250,  # More frequent sleep cycles
        enable_adaptive_sleep=True,
        min_sleep_frequency=100,
        max_sleep_frequency=500,
        expert_similarity_threshold=0.85,  # Higher threshold to prevent premature merging
        dormant_steps_threshold=1500,  # Give experts more time before pruning
        min_lifetime_activations=50,
        
        # Memory replay settings - more memory
        memory_replay_batch_size=64,
        memory_buffer_size=5000,  # Larger buffer to remember more examples
        replay_learning_rate=0.0001,
        
        # Expert reorganization - improved specialization
        enable_expert_reorganization=True,
        specialization_threshold=0.65,
        overlap_threshold=0.25,
        
        # Knowledge graph settings
        knowledge_edge_decay=0.98,  # Slower decay
        knowledge_edge_min=0.05,
        
        # Meta-learning - more frequent updates
        enable_meta_learning=True,
        meta_learning_intervals=1,  # Every sleep cycle
        
        # Training settings
        batch_size=64,
        num_epochs=1,  # Will control training by steps instead of epochs
        learning_rate=0.001
    )
    
    # Create the MORPH model
    model = MorphModel(config).to(device)
    logging.info(f"Created MORPH model with {len(model.experts)} initial experts")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create a data loader that will advance through the continual learning tasks
    dataloader = torch.utils.data.DataLoader(
        continual_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    # Tracking metrics
    expert_counts = []
    merge_events = []
    create_events = []
    sleep_events = []
    task_accuracies_history = []
    active_task_history = []
    
    # Variables for tracking
    step_counter = 0
    current_active_task = 0
    
    # Override the sleep method to track events
    original_sleep = model.sleep
    
    def instrumented_sleep():
        nonlocal step_counter, sleep_events
        
        logging.info(f"[Step {step_counter}] Starting sleep cycle")
        
        # Store metrics before sleep
        num_experts_before = len(model.experts)
        
        # Call original sleep method
        original_sleep()
        
        # Store sleep event
        expert_delta = len(model.experts) - num_experts_before
        sleep_metrics = {
            'expert_change': expert_delta,
            'expert_count_after': len(model.experts),
            'cycle_number': model.sleep_cycles_completed,
            'adaptive_frequency': getattr(model, 'adaptive_sleep_frequency',
                                         model.config.sleep_cycle_frequency)
        }
        sleep_events.append((step_counter, sleep_metrics))
        
        # Update expert lifecycle log
        expert_counts.append((step_counter, len(model.experts)))
        
        # Track sleep event
        logging.info(f"[Step {step_counter}] Sleep cycle completed - "
                   f"Next cycle in {sleep_metrics['adaptive_frequency']} steps")
    
    # Replace sleep method
    model.sleep = instrumented_sleep
    
    # Override expert creation method
    original_create = model._create_new_expert
    
    def instrumented_create():
        nonlocal create_events, step_counter
        
        # Call original method
        original_create()
        
        # Log creation
        if create_events and create_events[-1][0] == step_counter:
            create_events[-1] = (step_counter, create_events[-1][1] + 1)
        else:
            create_events.append((step_counter, 1))
        
        logging.info(f"[Step {step_counter}] Created new expert (total: {len(model.experts)})")
        
        # Update expert lifecycle log
        expert_counts.append((step_counter, len(model.experts)))
    
    # Replace create method
    model._create_new_expert = instrumented_create
    
    # Initial expert count
    expert_counts.append((0, len(model.experts)))
    
    # Maximum training steps
    total_steps = sum(end - start for start, end in task_schedule.values())
    logging.info(f"Training for {total_steps} steps across {len(task_schedule)} tasks")
    
    # Training loop
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create infinite data iterator
    data_iterator = iter(dataloader)
    
    # Evaluate all tasks at the start
    initial_task_accuracies = evaluate_forgetting(model, task_datasets, device)
    task_accuracies_history.append((0, initial_task_accuracies))
    
    # Main training loop
    for step in range(1, total_steps + 1):
        # Get next batch, reinitialize iterator if needed
        try:
            data, target, task_id = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            data, target, task_id = next(data_iterator)
        
        # Update active task if changed
        active_task = task_id[0].item()  # All items in batch have same task_id
        if active_task != current_active_task:
            logging.info(f"[Step {step}] Switching to task {active_task}")
            current_active_task = active_task
            active_task_history.append((step, active_task))
            
            # Update dataset's current step to ensure correct task is active
            continual_dataset.set_step(step)
        
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Flatten images for MLP-based model
        data = data.view(data.size(0), -1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with training=True to enable expert creation
        output = model(data, training=True)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update step counter (used by instrumented sleep and create methods)
        step_counter = step
        
        # Print progress every 100 steps
        if step % 100 == 0:
            accuracy = 100. * correct / total
            logging.info(f'Step: {step}/{total_steps} '
                  f'Task: {active_task} '
                  f'Loss: {running_loss/100:.6f} '
                  f'Acc: {accuracy:.2f}% '
                  f'Experts: {len(model.experts)}')
            
            running_loss = 0.0
            correct = 0
            total = 0
        
        # Periodically evaluate all tasks for catastrophic forgetting assessment
        if step % 500 == 0 or step == total_steps:
            # Evaluate all tasks
            task_accuracies = evaluate_forgetting(model, task_datasets, device)
            task_accuracies_history.append((step, task_accuracies))
            
            # Log results
            logging.info(f"[Step {step}] Task accuracies:")
            for task_id, acc in task_accuracies.items():
                logging.info(f"  Task {task_id}: {acc:.2f}%")
        
        # Update expert count log
        if step % 100 == 0:
            expert_counts.append((step, len(model.experts)))
            
        # Visualize knowledge graph periodically
        if step % 1000 == 0 or step == total_steps:
            visualize_knowledge_graph(
                model,
                output_path=f"results/continual/knowledge_graph_step_{step}.png"
            )
    
    # Final evaluation of all tasks
    final_task_accuracies = evaluate_forgetting(model, task_datasets, device)
    logging.info("Final task accuracies:")
    for task_id, acc in final_task_accuracies.items():
        logging.info(f"Task {task_id}: {acc:.2f}%")
    
    # Plot expert lifecycle
    visualize_expert_lifecycle(
        expert_counts=expert_counts,
        creation_events=create_events,
        merge_events=merge_events,
        sleep_events=sleep_events,
        output_path="results/continual/expert_lifecycle.png"
    )
    
    # Visualize sleep metrics
    visualize_sleep_metrics(
        model=model,
        sleep_events=sleep_events,
        output_path="results/continual/sleep_metrics.png"
    )
    
    # Plot task performance over time
    visualize_task_performance(
        task_accuracies_history=task_accuracies_history,
        output_path="results/continual/task_performance.png"
    )
    
    # Plot expert activations
    plot_expert_activations(
        model=model,
        n_steps=total_steps,
        output_path="results/continual/expert_activations.png"
    )
    
    # Visualize final knowledge graph
    visualize_knowledge_graph(
        model=model,
        output_path="results/continual/final_knowledge_graph.png"
    )
    
    # Create a custom visualization that shows task transitions and expert creation
    plt.figure(figsize=(14, 6))
    
    # Plot expert count
    steps, counts = zip(*expert_counts)
    plt.plot(steps, counts, 'b-', linewidth=2, label='Number of Experts')
    
    # Shade regions for different tasks
    colors = plt.cm.rainbow(np.linspace(0, 1, len(task_schedule)))
    
    for i, (task_id, (start, end)) in enumerate(sorted(task_schedule.items())):
        plt.axvspan(start, end, alpha=0.2, color=colors[i], label=f'Task {task_id} active')
    
    # Mark expert creations
    if create_events:
        create_steps, create_counts = zip(*create_events)
        plt.scatter(create_steps, [counts[steps.index(s)] if s in steps else counts[-1] for s in create_steps],
                  color='green', marker='^', s=100, label='Expert Creation')
    
    # Mark sleep events
    if sleep_events:
        sleep_steps = [event[0] for event in sleep_events]
        plt.scatter(sleep_steps, [counts[steps.index(s)] if s in steps else counts[-1] for s in sleep_steps],
                  color='purple', marker='*', s=120, label='Sleep Cycle')
    
    plt.xlabel('Training Step')
    plt.ylabel('Number of Experts')
    plt.title('Expert Dynamics During Continual Learning Tasks')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("results/continual/task_expert_dynamics.png", dpi=300, bbox_inches='tight')
    
    # Save the model
    torch.save(model.state_dict(), "results/continual/morph_continual_model.pt")
    logging.info("Model saved to results/continual/morph_continual_model.pt")
    
    # Create summary report
    with open("results/continual/continual_learning_summary.txt", "w") as f:
        f.write("MORPH Continual Learning Summary\n")
        f.write("===============================\n\n")
        f.write(f"Total tasks: {len(task_schedule)}\n")
        f.write(f"Initial experts: {config.num_initial_experts}\n")
        f.write(f"Final experts: {len(model.experts)}\n\n")
        
        f.write("Task accuracy summary:\n")
        for task_id, acc in final_task_accuracies.items():
            f.write(f"  Task {task_id}: {acc:.2f}%\n")
        
        f.write("\nExpert statistics:\n")
        f.write(f"  Experts created: {sum(count for _, count in create_events)}\n")
        f.write(f"  Sleep cycles: {model.sleep_cycles_completed}\n")
        
        # Calculate catastrophic forgetting metrics
        if len(task_accuracies_history) >= 2:
            first_accs = task_accuracies_history[0][1]
            forgetting = {}
            
            for task_id in first_accs:
                if task_id in final_task_accuracies:
                    forgetting[task_id] = first_accs[task_id] - final_task_accuracies[task_id]
            
            f.write("\nCatastrophic forgetting analysis:\n")
            for task_id, forget_amount in forgetting.items():
                f.write(f"  Task {task_id}: {forget_amount:.2f}% accuracy drop\n")
            
            # Average forgetting
            if forgetting:
                avg_forgetting = sum(forgetting.values()) / len(forgetting)
                f.write(f"\nAverage accuracy drop: {avg_forgetting:.2f}%\n")
    
    logging.info("Continual learning experiment completed. Results saved to results/continual/")


if __name__ == "__main__":
    main()
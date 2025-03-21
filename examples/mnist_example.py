import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add parent directory to path to import from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.core.model import Model
from src.data.data import get_mnist_dataloaders
from src.visualization.visualization import visualize_knowledge_graph, plot_expert_activations


def train(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Flatten images for MLP-based model
        data = data.view(data.size(0), -1)
        
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
        
        # Print progress
        if batch_idx % 100 == 0:
            logging.info(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f} Acc: {100. * correct / total:.2f}%')
    
    return running_loss / len(train_loader), 100. * correct / total


def test(model, test_loader, criterion, device):
    """
    Test the model.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Flatten images for MLP-based model
            data = data.view(data.size(0), -1)
            
            # Forward pass with training=False to disable expert creation
            output = model(data, training=False)
            test_loss += criterion(output, target).item()
            
            # Track statistics
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    logging.info(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return test_loss, accuracy


def main():
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/experts', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configure the model with enhanced sleep settings
    config = Config(
        input_size=784,  # 28x28 MNIST images
        expert_hidden_size=256,
        output_size=10,  # 10 MNIST classes
        num_initial_experts=4,
        expert_k=2,
        
        # Dynamic expert creation settings
        enable_dynamic_experts=True,
        expert_creation_uncertainty_threshold=0.25,  # Lower threshold to encourage more expert creation
        min_experts=3,
        max_experts=20,
        
        # Expert merging and pruning settings
        enable_sleep=True,
        sleep_cycle_frequency=300,  # Base frequency for sleep cycles
        enable_adaptive_sleep=True,  # Enable adaptive sleep scheduling
        min_sleep_frequency=150,     # Minimum steps between sleep cycles
        max_sleep_frequency=600,     # Maximum steps between sleep cycles
        expert_similarity_threshold=0.75,  # Similarity threshold for merging
        dormant_steps_threshold=1000,  # Steps before considering an expert dormant
        min_lifetime_activations=50,  # Min activations to avoid pruning
        
        # Memory replay settings
        memory_replay_batch_size=32,  # Batch size for memory replay
        memory_buffer_size=2000,      # Maximum size of activation buffer
        replay_learning_rate=0.0001,  # Learning rate for replay fine-tuning
        
        # Expert reorganization
        enable_expert_reorganization=True,  # Whether to reorganize experts
        specialization_threshold=0.7,  # Threshold for considering an expert specialized
        overlap_threshold=0.3,  # Threshold for considering expert overlap significant
        
        # Knowledge graph settings
        knowledge_edge_decay=0.95,
        knowledge_edge_min=0.1,
        
        # Meta-learning
        enable_meta_learning=True,  # Whether to enable meta-learning optimizations
        meta_learning_intervals=2,  # Sleep cycles between meta-learning updates
        
        # Training settings
        batch_size=64,
        num_epochs=5
    )
    
    # Create model
    model = Model(config).to(device)
    logging.info(f"Created model with {len(model.experts)} initial experts")
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Get data loaders
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=config.batch_size
    )
    
    # Tracking metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    expert_counts = []
    merge_events = []
    prune_events = []
    create_events = []
    
    # Track number of experts over time
    step_counter = 0
    
    # Function to track expert lifecycle events
    def log_expert_lifecycle():
        nonlocal step_counter
        expert_counts.append((step_counter, len(model.experts)))
        
        # Save expert count for visualization
        with open("results/expert_lifecycle.csv", "a") as f:
            f.write(f"{step_counter},{len(model.experts)}\n")
    
    # Create expert lifecycle log file
    with open("results/expert_lifecycle.csv", "w") as f:
        f.write("step,num_experts\n")
    
    # Log initial state
    log_expert_lifecycle()
    
    # Track sleep events for visualization
    sleep_events = []
    
    # Override sleep method to track merge/prune events and sleep metrics
    original_sleep = model.sleep
    
    def instrumented_sleep():
        nonlocal merge_events, prune_events, step_counter, sleep_events
        
        logging.info(f"[Step {step_counter}] Starting sleep cycle")
        
        # Check expert count before
        num_experts_before = len(model.experts)
        
        # Store pre-sleep metrics
        pre_sleep_metrics = {
            'expert_count': num_experts_before,
            'step': step_counter
        }
        
        # Call original sleep
        original_sleep()
        
        # Check expert count after
        num_experts_after = len(model.experts)
        
        # Calculate expert delta
        expert_delta = num_experts_after - num_experts_before
        
        # Track merging/pruning events
        if num_experts_after < num_experts_before:
            delta = num_experts_before - num_experts_after
            if len(merge_events) > 0 and merge_events[-1][0] == step_counter:
                # Update the last event if it's at the same step
                merge_events[-1] = (step_counter, merge_events[-1][1] + delta)
            else:
                merge_events.append((step_counter, delta))
            logging.info(f"[Step {step_counter}] Merged/Pruned {delta} experts")
        
        # Store sleep event with metrics
        sleep_metrics = {
            'expert_change': expert_delta,
            'expert_count_after': num_experts_after,
            'cycle_number': model.sleep_cycles_completed,
            'adaptive_frequency': getattr(model, 'adaptive_sleep_frequency', 
                                        model.config.sleep_cycle_frequency)
        }
        sleep_events.append((step_counter, sleep_metrics))
        logging.info(f"[Step {step_counter}] Sleep cycle completed - " +
                   f"Next cycle in {sleep_metrics['adaptive_frequency']} steps")
        
        # Update expert lifecycle log
        log_expert_lifecycle()
    
    # Replace sleep method
    model.sleep = instrumented_sleep
    
    # Override create_new_expert method to track creation events
    original_create = model._create_new_expert
    
    def instrumented_create():
        nonlocal create_events, step_counter
        
        # Call original method
        original_create()
        
        # Log creation
        if len(create_events) > 0 and create_events[-1][0] == step_counter:
            # Update the last event if it's at the same step
            create_events[-1] = (step_counter, create_events[-1][1] + 1)
        else:
            create_events.append((step_counter, 1))
        
        logging.info(f"[Step {step_counter}] Created new expert (total: {len(model.experts)})")
        
        # Update expert lifecycle log
        log_expert_lifecycle()
    
    # Replace create method
    model._create_new_expert = instrumented_create
    
    for epoch in range(1, config.num_epochs + 1):
        logging.info(f"Starting epoch {epoch}")
        
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Update step counter
        step_counter += len(train_loader)
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Log expert status
        logging.info(f"Epoch {epoch} completed. Current experts: {len(model.experts)}")
        
        # Visualize knowledge graph after each epoch
        visualize_knowledge_graph(
            model, 
            output_path=f"results/knowledge_graph_epoch_{epoch}.png"
        )
        
        # Manually trigger sleep at the end of each epoch for demonstration
        if epoch < config.num_epochs:  # Skip on last epoch as we'll do final evaluation
            logging.info(f"Manually triggering sleep cycle after epoch {epoch}")
            model.sleep()
    
    # Final evaluation
    logging.info("Final evaluation:")
    test_loss, test_acc = test(model, test_loader, criterion, device)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig("results/training_curves.png")
    
    # Plot expert activations
    plot_expert_activations(
        model, 
        n_steps=config.num_epochs * len(train_loader),
        output_path="results/expert_activations.png"
    )
    
    # Use the visualization utility to plot expert lifecycle
    from src.visualization.visualization import visualize_expert_lifecycle, visualize_sleep_metrics
    
    # Plot expert lifecycle with all events
    visualize_expert_lifecycle(
        expert_counts=expert_counts,
        creation_events=create_events,
        merge_events=merge_events,
        sleep_events=sleep_events,
        output_path="results/expert_lifecycle.png"
    )
    
    # Add epoch boundaries to the saved image
    img = plt.imread("results/expert_lifecycle.png")
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    
    # Get steps and counts for annotation
    steps, counts = zip(*expert_counts)
    max_count = max(counts)
    
    # Add epoch boundaries
    for epoch in range(1, config.num_epochs + 1):
        step = epoch * len(train_loader)
        # Calculate pixel position (approximate)
        relative_x = step / steps[-1]
        x_pos = relative_x * img.shape[1]
        plt.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.5)
        plt.text(x_pos, img.shape[0] * 0.05, f'Epoch {epoch}', 
                ha='center', va='bottom', fontsize=10, color='black',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.savefig("results/expert_lifecycle_with_epochs.png", bbox_inches='tight')
    plt.close()
    
    # Visualize sleep metrics
    visualize_sleep_metrics(
        model=model,
        sleep_events=sleep_events,
        output_path="results/sleep_metrics.png"
    )
    
    # Visualization for knowledge graph evolution
    logging.info("Generating knowledge graph evolution visualization...")
    
    # Create animation to show knowledge graph evolution (simple version)
    plt.figure(figsize=(10, 8))
    plt.scatter([0], [0], color='red', s=100, label='Merged Experts')
    plt.scatter([0], [0], color='green', s=100, label='Created Experts')
    plt.title('Expert Knowledge Graph Evolution')
    plt.text(0.5, 0.5, "Knowledge graph evolved from\n"
             f"{config.num_initial_experts} initial experts to {len(model.experts)} final experts\n"
             f"Created: {sum([count for _, count in create_events])}\n"
             f"Merged/Pruned: {sum([count for _, count in merge_events])}", 
             ha='center', va='center', fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    plt.legend()
    plt.grid(False)
    plt.axis('off')
    plt.savefig("results/knowledge_graph_evolution.png")
    
    # Save summary statistics
    with open("results/expert_summary.txt", "w") as f:
        f.write(f"Expert Lifecycle Summary\n")
        f.write(f"==============================\n\n")
        f.write(f"Initial experts: {config.num_initial_experts}\n")
        f.write(f"Final experts: {len(model.experts)}\n\n")
        f.write(f"Expert creation events: {len(create_events)}\n")
        f.write(f"Total experts created: {sum([count for _, count in create_events])}\n\n")
        f.write(f"Expert merging/pruning events: {len(merge_events)}\n")
        f.write(f"Total experts merged/pruned: {sum([count for _, count in merge_events])}\n\n")
        f.write(f"Final model accuracy: {test_acc:.2f}%\n")
    
    # Save model
    torch.save(model.state_dict(), "results/model.pt")
    logging.info("Model saved to results/model.pt")
    logging.info(f"Expert summary saved to results/expert_summary.txt")


if __name__ == "__main__":
    main()

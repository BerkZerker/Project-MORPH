import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
from tqdm import tqdm

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
from morph.utils.data import get_mnist_dataloaders
from morph.utils.visualization import (
    visualize_knowledge_graph, 
    plot_expert_activations,
    visualize_expert_lifecycle,
    visualize_sleep_metrics
)


class ContinualMNISTLoader:
    """
    Wrapper around MNIST dataset that introduces concept drift by
    gradually introducing new digit classes.
    """
    
    def __init__(self, train_loader, phases=5, batch_size=64):
        """
        Initialize the continual MNIST loader.
        
        Args:
            train_loader: Original MNIST train loader
            phases: Number of training phases (for concept drift)
            batch_size: Batch size for each phase
        """
        self.train_loader = train_loader
        self.phases = phases
        self.batch_size = batch_size
        
        # Store all data
        self.all_data = []
        self.all_targets = []
        
        # Extract all data from the train loader
        for data, targets in train_loader:
            self.all_data.append(data)
            self.all_targets.append(targets)
            
        self.all_data = torch.cat(self.all_data)
        self.all_targets = torch.cat(self.all_targets)
        
        # Split data by digit
        self.digit_data = {}
        self.digit_indices = {}
        
        for digit in range(10):
            indices = (self.all_targets == digit).nonzero(as_tuple=True)[0]
            self.digit_indices[digit] = indices
            
        # Prepare phases
        self.num_digits_per_phase = [2, 4, 6, 8, 10]
        
    def get_phase_data(self, phase_idx):
        """
        Get data for a specific phase.
        
        Args:
            phase_idx: Phase index (0-based)
            
        Returns:
            DataLoader for this phase
        """
        if phase_idx >= len(self.num_digits_per_phase):
            phase_idx = len(self.num_digits_per_phase) - 1
            
        # Get active digits for this phase
        num_digits = self.num_digits_per_phase[phase_idx]
        active_digits = list(range(num_digits))
        
        # Collect data for active digits
        phase_indices = []
        for digit in active_digits:
            phase_indices.append(self.digit_indices[digit])
            
        # Combine indices
        if phase_indices:
            phase_indices = torch.cat(phase_indices)
            
            # Shuffle indices
            phase_indices = phase_indices[torch.randperm(len(phase_indices))]
            
            # Get data for this phase
            phase_data = self.all_data[phase_indices]
            phase_targets = self.all_targets[phase_indices]
            
            # Create TensorDataset
            phase_dataset = torch.utils.data.TensorDataset(phase_data, phase_targets)
            
            # Create DataLoader
            phase_loader = torch.utils.data.DataLoader(
                phase_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            
            return phase_loader
        else:
            # Default to empty loader
            return None


def visualize_sleep_benefits(model, test_loader, device, output_path="results/sleep_benefits.png"):
    """
    Visualize the benefits of sleep by comparing model performance before and after sleep.
    
    Args:
        model: MorphModel instance
        test_loader: Test data loader
        device: Device for computation
        output_path: Path to save visualization
    """
    # Store original state
    original_expert_count = len(model.experts)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate pre-sleep
    pre_sleep_metrics = model.evaluate(test_loader, criterion, device)
    
    # Extract specific subsets of test data
    digit_accs_pre = evaluate_per_digit(model, test_loader, device)
    
    # Trigger sleep cycle
    logging.info("Triggering sleep cycle for visualization...")
    model.sleep()
    
    # Evaluate post-sleep
    post_sleep_metrics = model.evaluate(test_loader, criterion, device)
    
    # Extract specific subsets of test data again
    digit_accs_post = evaluate_per_digit(model, test_loader, device)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot overall accuracy
    plt.subplot(2, 2, 1)
    plt.bar(['Pre-Sleep', 'Post-Sleep'], 
           [pre_sleep_metrics['accuracy'], post_sleep_metrics['accuracy']], 
           color=['blue', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.title('Overall Accuracy Before and After Sleep')
    plt.grid(axis='y', alpha=0.3)
    
    # Annotate with values
    for i, v in enumerate([pre_sleep_metrics['accuracy'], post_sleep_metrics['accuracy']]):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    
    # Plot expert count
    plt.subplot(2, 2, 2)
    plt.bar(['Pre-Sleep', 'Post-Sleep'], 
           [original_expert_count, len(model.experts)], 
           color=['blue', 'green'])
    plt.ylabel('Number of Experts')
    plt.title('Expert Count Before and After Sleep')
    plt.grid(axis='y', alpha=0.3)
    
    # Annotate with values
    for i, v in enumerate([original_expert_count, len(model.experts)]):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    # Plot per-digit accuracy
    plt.subplot(2, 1, 2)
    x = np.arange(10)
    width = 0.35
    
    pre_values = [digit_accs_pre.get(i, 0) for i in range(10)]
    post_values = [digit_accs_post.get(i, 0) for i in range(10)]
    
    plt.bar(x - width/2, pre_values, width, label='Pre-Sleep', color='blue', alpha=0.7)
    plt.bar(x + width/2, post_values, width, label='Post-Sleep', color='green', alpha=0.7)
    
    plt.xlabel('Digit')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Digit Accuracy Before and After Sleep')
    plt.xticks(x, [str(i) for i in range(10)])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logging.info(f"Sleep benefit visualization saved to {output_path}")


def evaluate_per_digit(model, test_loader, device):
    """
    Evaluate model accuracy per digit.
    
    Args:
        model: MorphModel instance
        test_loader: Test data loader
        device: Device for computation
        
    Returns:
        Dictionary mapping digit -> accuracy
    """
    model.eval()
    
    # Initialize counters for each digit
    correct_by_digit = {i: 0 for i in range(10)}
    total_by_digit = {i: 0 for i in range(10)}
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Flatten images for MLP-based model
            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), -1)
                
            # Forward pass
            outputs = model(inputs, training=False)
            _, predicted = outputs.max(1)
            
            # Update counters for each digit
            for digit in range(10):
                digit_mask = (targets == digit)
                if digit_mask.any():
                    digit_correct = predicted[digit_mask].eq(targets[digit_mask]).sum().item()
                    digit_total = digit_mask.sum().item()
                    
                    correct_by_digit[digit] += digit_correct
                    total_by_digit[digit] += digit_total
    
    # Calculate accuracy for each digit
    accuracy_by_digit = {}
    for digit in range(10):
        if total_by_digit[digit] > 0:
            accuracy_by_digit[digit] = 100.0 * correct_by_digit[digit] / total_by_digit[digit]
        else:
            accuracy_by_digit[digit] = 0.0
            
    return accuracy_by_digit


def main():
    """
    Run the MORPH sleep module example with continual learning on MNIST.
    """
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/sleep_module', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configure the MORPH model with enhanced sleep settings
    config = MorphConfig(
        input_size=784,  # 28x28 MNIST images
        expert_hidden_size=256,
        output_size=10,  # 10 MNIST classes
        num_initial_experts=3,  # Start with fewer experts
        expert_k=2,
        
        # Dynamic expert creation settings
        enable_dynamic_experts=True,
        expert_creation_uncertainty_threshold=0.25,  # Lower threshold to encourage more expert creation
        min_experts=2,
        max_experts=15,
        
        # Expert merging and pruning settings
        enable_sleep=True,
        sleep_cycle_frequency=300,  # Base frequency for sleep cycles
        enable_adaptive_sleep=True,  # Enable adaptive sleep scheduling
        min_sleep_frequency=150,     # Minimum steps between sleep cycles
        max_sleep_frequency=600,     # Maximum steps between sleep cycles
        expert_similarity_threshold=0.75,  # Similarity threshold for merging
        dormant_steps_threshold=500,  # Steps before considering an expert dormant
        min_lifetime_activations=50,  # Min activations to avoid pruning
        
        # Memory replay settings
        memory_replay_batch_size=32,  # Batch size for memory replay
        memory_buffer_size=1000,      # Maximum size of activation buffer
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
        num_epochs=3
    )
    
    # Create model
    model = MorphModel(config).to(device)
    logging.info(f"Created MORPH model with {len(model.experts)} initial experts")
    
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
    
    # Create concept drift wrapper
    continual_loader = ContinualMNISTLoader(train_loader, phases=5, batch_size=config.batch_size)
    
    # Tracking metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    expert_counts = []
    
    # Create expert lifecycle log file
    with open("results/sleep_module/expert_lifecycle.csv", "w") as f:
        f.write("step,phase,num_experts,sleep_cycles\n")
    
    # Training loop with concept drift phases
    total_steps = 0
    for phase in range(len(continual_loader.num_digits_per_phase)):
        logging.info(f"Starting phase {phase+1} with {continual_loader.num_digits_per_phase[phase]} digits")
        
        # Get phase-specific data
        phase_loader = continual_loader.get_phase_data(phase)
        
        # Train for the current phase
        for epoch in range(config.num_epochs):
            logging.info(f"Phase {phase+1}, Epoch {epoch+1}")
            
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(phase_loader, desc="Training")):
                data, target = data.to(device), target.to(device)
                
                # Flatten images for MLP-based model
                data = data.view(data.size(0), -1)
                
                # Get metrics from train step
                metrics = model.train_step((data, target), optimizer, criterion)
                
                # Track statistics
                running_loss += metrics['loss']
                correct += metrics['accuracy'] * target.size(0) / 100.0
                total += target.size(0)
                
                # Increment step counter
                total_steps += 1
                
                # Log expert count periodically
                if batch_idx % 20 == 0:
                    expert_counts.append((total_steps, len(model.experts)))
                    with open("results/sleep_module/expert_lifecycle.csv", "a") as f:
                        f.write(f"{total_steps},{phase},{len(model.experts)},{model.sleep_cycles_completed}\n")
                
                # Log progress
                if batch_idx % 50 == 0:
                    logging.info(f"Phase {phase+1} Epoch {epoch+1} Batch {batch_idx}: "
                              f"Loss {metrics['loss']:.4f} Acc {metrics['accuracy']:.2f}% "
                              f"Experts {metrics['num_experts']}")
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(phase_loader)
            epoch_acc = 100.0 * correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            
            # Test
            test_metrics = model.evaluate(test_loader, criterion, device)
            test_losses.append(test_metrics['loss'])
            test_accs.append(test_metrics['accuracy'])
            
            logging.info(f"Phase {phase+1} Epoch {epoch+1} complete: "
                      f"Train Loss {epoch_loss:.4f} Train Acc {epoch_acc:.2f}% "
                      f"Test Loss {test_metrics['loss']:.4f} Test Acc {test_metrics['accuracy']:.2f}% "
                      f"Experts {test_metrics['num_experts']}")
        
        # Visualize after each phase
        visualize_knowledge_graph(
            model, 
            output_path=f"results/sleep_module/knowledge_graph_phase_{phase+1}.png"
        )
        
        # Force a sleep cycle after each phase
        if phase < len(continual_loader.num_digits_per_phase) - 1:
            logging.info(f"Forcing sleep cycle after phase {phase+1}")
            model.sleep()
            
            # Visualize sleep benefits
            visualize_sleep_benefits(
                model, 
                test_loader, 
                device, 
                output_path=f"results/sleep_module/sleep_benefits_phase_{phase+1}.png"
            )
    
    # Final evaluation
    logging.info("Final evaluation:")
    final_metrics = model.evaluate(test_loader, criterion, device)
    
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
    plt.savefig("results/sleep_module/training_curves.png")
    
    # Plot expert activations
    plot_expert_activations(
        model, 
        n_steps=total_steps,
        output_path="results/sleep_module/expert_activations.png"
    )
    
    # Parse expert lifecycle data for visualization
    expert_counts = []
    sleep_events = []
    
    with open("results/sleep_module/expert_lifecycle.csv", "r") as f:
        # Skip header
        next(f)
        for line in f:
            step, phase, num_experts, sleep_cycles = line.strip().split(",")
            step = int(step)
            num_experts = int(num_experts)
            sleep_cycles = int(sleep_cycles)
            expert_counts.append((step, num_experts))
            
    # Extract sleep events from model
    sleep_metrics = model.get_sleep_metrics()
    for metrics in sleep_metrics:
        sleep_events.append((
            metrics['step'], 
            {
                'expert_change': metrics.get('experts_after', 0) - metrics.get('experts_before', 0),
                'expert_count_after': metrics.get('experts_after', 0),
                'cycle_number': metrics.get('cycle_number', 0),
                'adaptive_frequency': metrics.get('next_sleep', 0) - metrics.get('step', 0)
            }
        ))
    
    # Visualize expert lifecycle
    visualize_expert_lifecycle(
        expert_counts=expert_counts,
        creation_events=[],  # Would need to track these separately
        merge_events=[],     # Would need to track these separately
        sleep_events=sleep_events,
        output_path="results/sleep_module/expert_lifecycle.png"
    )
    
    # Visualize sleep metrics
    visualize_sleep_metrics(
        model=model,
        sleep_events=sleep_events,
        output_path="results/sleep_module/sleep_metrics.png"
    )
    
    # Generate final knowledge graph visualization
    visualize_knowledge_graph(
        model, 
        output_path="results/sleep_module/final_knowledge_graph.png",
        highlight_dormant=True,
        highlight_similar=True,
        highlight_specialization=True
    )
    
    # Save model
    torch.save(model.state_dict(), "results/sleep_module/morph_model.pt")
    logging.info(f"Model saved to results/sleep_module/morph_model.pt")
    
    # Print final summary
    logging.info(f"Final accuracy: {final_metrics['accuracy']:.2f}%")
    logging.info(f"Final number of experts: {len(model.experts)}")
    logging.info(f"Sleep cycles completed: {model.sleep_cycles_completed}")
    logging.info(f"View results in the results/sleep_module directory")


if __name__ == "__main__":
    main()
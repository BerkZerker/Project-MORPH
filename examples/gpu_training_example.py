import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import logging
import argparse
import time

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
from morph.utils.visualization import visualize_knowledge_graph, plot_expert_activations
from morph.utils.gpu_utils import detect_available_gpus, get_gpu_memory_info, setup_gpu_environment


def train_epoch(model, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Flatten images for MLP-based model
        data = data.view(data.size(0), -1)
        
        # Forward pass with training=True to enable expert creation
        metrics = model.train_step((data, target), optimizer, criterion)
        
        # Track statistics
        running_loss += metrics['loss']
        correct += metrics['accuracy'] * target.size(0) / 100
        total += target.size(0)
        
        # Print progress
        if batch_idx % 100 == 0:
            logging.info(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {metrics["loss"]:.6f} Acc: {metrics["accuracy"]:.2f}% '
                  f'Experts: {metrics["num_experts"]}')
    
    epoch_time = time.time() - start_time
    logging.info(f'Epoch {epoch} completed in {epoch_time:.2f} seconds')
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion):
    """
    Evaluate the model on the test set.
    """
    # Use model's evaluate method which handles device placement
    metrics = model.evaluate(test_loader, criterion)
    
    logging.info(f'Test set: Average loss: {metrics["loss"]:.4f}, '
          f'Accuracy: {metrics["accuracy"]:.2f}%, '
          f'Experts: {metrics["num_experts"]}')
    
    return metrics["loss"], metrics["accuracy"]


def main():
    parser = argparse.ArgumentParser(description='MORPH GPU Training Example')
    parser.add_argument('--gpu-mode', type=str, default='auto', 
                        choices=['auto', 'cpu', 'single_gpu', 'multi_gpu'],
                        help='GPU mode to use')
    parser.add_argument('--parallel-strategy', type=str, default='data_parallel',
                        choices=['data_parallel', 'expert_parallel'],
                        help='Parallelization strategy for multi-GPU')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--auto-batch-size', action='store_true',
                        help='Automatically determine optimal batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--experts', type=int, default=8,
                        help='Number of initial experts')
    args = parser.parse_args()
    
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/gpu_training', exist_ok=True)
    
    # Set up GPU environment
    setup_gpu_environment()
    
    # Detect available GPUs
    available_gpus = detect_available_gpus()
    logging.info(f"Available GPUs: {available_gpus}")
    
    # Get GPU memory info
    if available_gpus:
        memory_info = get_gpu_memory_info()
        for gpu_idx, info in memory_info.items():
            logging.info(f"GPU {gpu_idx}: {info['total']:.2f} GB total, {info['free']:.2f} GB free")
    
    # Configure the MORPH model with GPU settings
    config = MorphConfig(
        # Model architecture
        input_size=784,  # 28x28 MNIST images
        expert_hidden_size=256,
        output_size=10,  # 10 MNIST classes
        num_initial_experts=args.experts,
        expert_k=2,
        
        # Dynamic expert creation settings
        enable_dynamic_experts=True,
        expert_creation_uncertainty_threshold=0.25,
        min_experts=3,
        max_experts=32,
        
        # Sleep cycle settings
        enable_sleep=True,
        sleep_cycle_frequency=300,
        enable_adaptive_sleep=True,
        
        # GPU settings
        gpu_mode=args.gpu_mode,
        parallel_strategy=args.parallel_strategy,
        enable_mixed_precision=args.mixed_precision,
        auto_batch_size=args.auto_batch_size,
        batch_size=args.batch_size,
        
        # Training settings
        num_epochs=args.epochs
    )
    
    # Create model
    model = MorphModel(config)
    
    # Log GPU configuration
    logging.info(f"GPU Mode: {config.gpu_mode}")
    logging.info(f"Devices: {[str(d) for d in config.devices]}")
    logging.info(f"Primary Device: {config.device}")
    logging.info(f"Parallel Strategy: {config.parallel_strategy}")
    logging.info(f"Mixed Precision: {config.enable_mixed_precision}")
    logging.info(f"Auto Batch Size: {config.auto_batch_size}")
    logging.info(f"Batch Size: {config.batch_size}")
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Get data loaders
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Tracking metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    epoch_times = []
    
    # Training loop
    total_start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Log expert status
        logging.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds. "
                    f"Current experts: {len(model.experts)}")
        
        # Visualize knowledge graph after each epoch
        visualize_knowledge_graph(
            model, 
            output_path=f"results/gpu_training/knowledge_graph_epoch_{epoch}.png"
        )
    
    total_time = time.time() - total_start_time
    logging.info(f"Training completed in {total_time:.2f} seconds")
    
    # Final evaluation
    logging.info("Final evaluation:")
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.subplot(1, 3, 3)
    plt.plot(epoch_times)
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Training Time')
    
    plt.tight_layout()
    plt.savefig("results/gpu_training/training_metrics.png")
    
    # Plot expert activations
    plot_expert_activations(
        model, 
        n_steps=config.num_epochs * len(train_loader),
        output_path="results/gpu_training/expert_activations.png"
    )
    
    # Save expert device distribution
    if config.gpu_mode == 'multi_gpu':
        device_counts = {}
        for device in config.devices:
            device_counts[str(device)] = 0
            
        for expert_idx, device in model.expert_device_map.items():
            device_str = str(device)
            if device_str in device_counts:
                device_counts[device_str] += 1
            else:
                device_counts[device_str] = 1
        
        plt.figure(figsize=(10, 6))
        devices = list(device_counts.keys())
        counts = list(device_counts.values())
        plt.bar(devices, counts)
        plt.xlabel('Device')
        plt.ylabel('Number of Experts')
        plt.title('Expert Distribution Across Devices')
        plt.savefig("results/gpu_training/expert_device_distribution.png")
        
        # Log expert device distribution
        logging.info("Expert distribution across devices:")
        for device, count in device_counts.items():
            logging.info(f"  {device}: {count} experts")
    
    # Save performance summary
    with open("results/gpu_training/performance_summary.txt", "w") as f:
        f.write(f"MORPH GPU Training Performance Summary\n")
        f.write(f"=====================================\n\n")
        f.write(f"GPU Mode: {config.gpu_mode}\n")
        f.write(f"Devices: {[str(d) for d in config.devices]}\n")
        f.write(f"Parallel Strategy: {config.parallel_strategy}\n")
        f.write(f"Mixed Precision: {config.enable_mixed_precision}\n")
        f.write(f"Batch Size: {config.batch_size}\n\n")
        
        f.write(f"Initial experts: {config.num_initial_experts}\n")
        f.write(f"Final experts: {len(model.experts)}\n\n")
        
        f.write(f"Training time: {total_time:.2f} seconds\n")
        f.write(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds\n\n")
        
        f.write(f"Final model accuracy: {test_acc:.2f}%\n")
    
    # Save model
    torch.save(model.state_dict(), "results/gpu_training/morph_model.pt")
    logging.info("Model saved to results/gpu_training/morph_model.pt")
    logging.info(f"Performance summary saved to results/gpu_training/performance_summary.txt")


if __name__ == "__main__":
    main()

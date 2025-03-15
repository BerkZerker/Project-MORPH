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

# Add parent directory to path to import morph
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from morph.config import MorphConfig
from morph.core.model import MorphModel
from morph.utils.data import get_mnist_dataloaders
from morph.utils.visualization import visualize_knowledge_graph, plot_expert_activations


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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configure the MORPH model
    config = MorphConfig(
        input_size=784,  # 28x28 MNIST images
        expert_hidden_size=256,
        output_size=10,  # 10 MNIST classes
        num_initial_experts=4,
        expert_k=2,
        enable_dynamic_experts=True,
        enable_sleep=True,
        sleep_cycle_frequency=500,  # Sleep every 500 batches
        expert_similarity_threshold=0.8,
        dormant_steps_threshold=2000,
        min_experts=2,
        max_experts=16,
        batch_size=64,
        num_epochs=5
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
    
    # Training loop
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Log expert status
        logging.info(f"Number of experts: {len(model.experts)}")
        
        # Visualize knowledge graph
        visualize_knowledge_graph(
            model, 
            output_path=f"results/knowledge_graph_epoch_{epoch}.png"
        )
    
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
    
    # Save model
    torch.save(model.state_dict(), "results/morph_model.pt")
    logging.info("Model saved to results/morph_model.pt")


if __name__ == "__main__":
    main()

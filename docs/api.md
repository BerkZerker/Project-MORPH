# MORPH API Reference

## Core Classes

### MorphModel

Main model class that implements the Dynamic Mixture of Experts architecture.

```python
class MorphModel(nn.Module):
    """
    MORPH: Mixture Of experts with Recursive Post-processing & Hierarchy.
    
    This model implements a dynamic mixture of experts architecture with
    adaptive expert creation, knowledge graph routing, and a sleep cycle
    for knowledge consolidation.
    """
    
    def __init__(self, config: MorphConfig):
        """
        Initialize the MORPH model.
        
        Args:
            config: Configuration object with model parameters
        """
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Forward pass through the MORPH model.
        
        Args:
            x: Input tensor [batch_size, input_size]
            training: Whether in training mode
            
        Returns:
            Model output tensor [batch_size, output_size]
        """
        
    def sleep(self):
        """
        Perform a sleep cycle to consolidate knowledge.
        
        This includes:
        1. Replaying stored activations
        2. Merging similar experts
        3. Pruning dormant experts
        """
```

### Expert

Individual expert network that specializes in a particular subset of the data.

```python
class Expert(nn.Module):
    """
    Base expert network that specializes in a particular subset of the data.
    
    Each expert is a simple feed-forward network that can be customized.
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 output_size: int, num_layers: int = 2):
        """
        Initialize an expert network.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Size of hidden layers
            output_size: Dimension of output features
            num_layers: Number of hidden layers
        """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert.
        
        Args:
            x: Input tensor
            
        Returns:
            Expert output
        """
        
    def clone(self) -> 'Expert':
        """
        Create a clone of this expert with the same architecture but 
        re-initialized weights.
        
        Returns:
            A new Expert instance
        """
        
    def get_parameter_similarity(self, other_expert: 'Expert') -> float:
        """
        Compute cosine similarity between this expert's parameters and another expert.
        
        Args:
            other_expert: Another Expert instance to compare with
            
        Returns:
            Similarity score between 0 and 1
        """
```

### GatingNetwork

Network that determines which experts to use for a given input.

```python
class GatingNetwork(nn.Module):
    """
    Gating network that determines which experts to use for a given input.
    
    The gating network routes inputs to the most appropriate experts based on
    the input features.
    """
    
    def __init__(self, input_size: int, num_experts: int, 
                 k: int = 2, routing_type: str = "top_k"):
        """
        Initialize the gating network.
        
        Args:
            input_size: Dimension of input features
            num_experts: Number of experts to route between
            k: Number of experts to activate per input (for top-k routing)
            routing_type: Type of routing mechanism ("top_k" or "noisy_top_k")
        """
        
    def forward(self, x: torch.Tensor, training: bool = True) -> tuple:
        """
        Compute routing probabilities for each expert.
        
        Args:
            x: Input tensor [batch_size, input_size]
            training: Whether in training mode (affects routing)
            
        Returns:
            Tuple of (routing_weights, expert_indices, uncertainty)
            - routing_weights: Tensor of shape [batch_size, k]
            - expert_indices: Tensor of shape [batch_size, k]
            - uncertainty: Scalar representing routing uncertainty
        """
        
    def should_create_new_expert(self, uncertainty: float) -> bool:
        """
        Determine if a new expert should be created based on uncertainty.
        
        Args:
            uncertainty: Routing uncertainty score
            
        Returns:
            Boolean indicating whether to create a new expert
        """
        
    def update_num_experts(self, num_experts: int):
        """
        Update the gating network when the number of experts changes.
        
        Args:
            num_experts: New number of experts
        """
```

## Configuration

### MorphConfig

Configuration dataclass for MORPH model parameters.

```python
@dataclass
class MorphConfig:
    """
    Configuration class for MORPH model.
    """
    # Model architecture
    input_size: int = 784  # Default for MNIST
    expert_hidden_size: int = 256
    output_size: int = 10  # Default for MNIST (10 classes)
    num_initial_experts: int = 4
    expert_k: int = 2  # Number of experts to route to for each input
    
    # Dynamic expert creation
    enable_dynamic_experts: bool = True
    expert_creation_uncertainty_threshold: float = 0.3
    min_experts: int = 2  # Minimum number of experts to maintain
    max_experts: int = 32  # Maximum number of experts
    
    # Sleep cycle
    enable_sleep: bool = True
    sleep_cycle_frequency: int = 1000  # Steps between sleep cycles
    expert_similarity_threshold: float = 0.8  # Threshold for merging experts
    dormant_steps_threshold: int = 5000  # Steps of inactivity before pruning
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 10
```

## Utility Functions

### Visualization

```python
def visualize_knowledge_graph(model: MorphModel, output_path: str = None):
    """
    Visualize the knowledge graph of experts.
    
    Args:
        model: MorphModel instance
        output_path: Path to save the visualization (if None, display instead)
    """
    
def plot_expert_activations(model: MorphModel, n_steps: int, 
                           output_path: str = None):
    """
    Plot expert activation patterns over time.
    
    Args:
        model: MorphModel instance
        n_steps: Number of steps to show history for
        output_path: Path to save the visualization (if None, display instead)
    """
```

### Data Utilities

```python
def get_mnist_dataloaders(batch_size: int = 64, num_workers: int = 2) -> tuple:
    """
    Get MNIST dataset loaders for basic testing.
    
    Args:
        batch_size: Batch size for training/testing
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    
class ContinualTaskDataset(Dataset):
    """
    Dataset for continual learning with distribution shifts.
    
    This dataset sequentially introduces new tasks, allowing for
    testing of catastrophic forgetting and adaptation.
    """
    
    def __init__(self, base_datasets: List[Dataset], 
                 task_schedule: Dict[int, tuple], 
                 transform=None):
        """
        Initialize the continual learning dataset.
        
        Args:
            base_datasets: List of datasets to draw from
            task_schedule: Dict mapping step numbers to active tasks
            transform: Optional transform to be applied to the data
        """
```
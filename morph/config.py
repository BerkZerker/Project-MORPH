from dataclasses import dataclass


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

from dataclasses import dataclass
import torch


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
    enable_adaptive_sleep: bool = True  # Whether to adjust sleep frequency dynamically
    min_sleep_frequency: int = 500  # Minimum steps between sleep cycles
    max_sleep_frequency: int = 2000  # Maximum steps between sleep cycles
    memory_replay_batch_size: int = 32  # Batch size for memory replay
    memory_buffer_size: int = 2000  # Maximum size of activation buffer
    replay_learning_rate: float = 0.0001  # Learning rate for replay fine-tuning
    
    # Expert reorganization
    enable_expert_reorganization: bool = True  # Whether to reorganize experts
    specialization_threshold: float = 0.7  # Threshold for considering an expert specialized
    overlap_threshold: float = 0.3  # Threshold for considering expert overlap significant
    
    # Expert merging
    expert_similarity_threshold: float = 0.8  # Threshold for merging experts
    
    # Expert pruning
    dormant_steps_threshold: int = 5000  # Steps of inactivity before pruning
    min_lifetime_activations: int = 100  # Minimum activations to avoid pruning
    
    # Knowledge graph
    knowledge_edge_decay: float = 0.99  # Decay factor for edge weights
    knowledge_edge_min: float = 0.1  # Minimum edge weight before pruning
    knowledge_relation_types: list = None  # Types of relations to track in graph
    
    # Meta-learning
    enable_meta_learning: bool = True  # Whether to enable meta-learning optimizations
    meta_learning_intervals: int = 10  # Sleep cycles between meta-learning updates
    
    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 10
    
    # GPU Acceleration
    device: str = "auto"  # "auto", "cuda", or "cpu" - Auto-detected in __post_init__
    enable_mixed_precision: bool = False  # Whether to use mixed precision training (fp16)
    
    # Data Loading Optimization
    num_workers: int = 4  # Number of workers for DataLoader
    pin_memory: bool = True  # Whether to use pinned memory for faster CPU->GPU transfer
    
    # Test-specific optimizations
    test_mode: bool = False  # Whether to use test-specific optimizations
    test_expert_size: int = 64  # Smaller expert size for tests
    test_sleep_frequency: int = 100  # Reduced sleep frequency for tests
    test_memory_buffer_size: int = 200  # Smaller memory buffer for tests
    
    def __post_init__(self):
        """Initialize default values for complex types and auto-detect device"""
        if self.knowledge_relation_types is None:
            self.knowledge_relation_types = [
                "similarity",        # For similar experts
                "specialization",    # For experts specialized in a domain
                "dependency",        # For experts with input/output dependencies
                "composition",       # For experts that are compositional
                "specialization_split", # For experts that split specialization
            ]
        
        # Auto-detect device if set to "auto" or None
        if self.device is None or self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Enable mixed precision only if using CUDA
        if self.enable_mixed_precision and self.device != "cuda":
            self.enable_mixed_precision = False
            
        # Enable pin_memory only if using CUDA
        if self.pin_memory and self.device != "cuda":
            self.pin_memory = False

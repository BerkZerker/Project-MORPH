"""
Initialization logic for the MORPH model.

This module contains the initialization logic for the MORPH model,
including expert initialization, gating network setup, and device management.
"""

import torch
import torch.nn as nn
import logging
from torch.amp import GradScaler

from morph.core.expert import Expert
from morph.core.gating import GatingNetwork
from morph.core.knowledge_graph import KnowledgeGraph
from morph.core.sleep import SleepModule
from morph.utils.gpu_utils import setup_gpu_environment, distribute_experts_across_gpus, estimate_max_batch_size
from morph.utils.distributed import create_parallel_wrapper


class ModelInitialization:
    """
    Initialization logic for the MORPH model.
    
    This class is responsible for initializing the MORPH model components,
    including experts, gating network, knowledge graph, and sleep module.
    """
    
    def __init__(self, config):
        """
        Initialize the MORPH model.
        
        Args:
            config: Configuration object with the following attributes:
                - input_size: Dimension of input features
                - hidden_size: Size of expert hidden layers
                - output_size: Dimension of output features
                - num_initial_experts: Number of experts to start with
                - expert_k: Number of experts to route to for each input
                - enable_dynamic_experts: Whether to enable dynamic expert creation
                - enable_sleep: Whether to enable sleep cycles
                - sleep_cycle_frequency: How often to trigger sleep cycles
                - expert_similarity_threshold: Threshold for merging similar experts
                - device: Device to use (cuda or cpu)
                - enable_mixed_precision: Whether to use mixed precision training
        """
        self.config = config
        self.step_count = 0
        
        # Set up GPU environment
        setup_gpu_environment()
        
        # Get primary device from config
        self.device = torch.device(config.device)
        self.devices = config.devices if config.devices else [self.device]
            
        logging.info(f"Primary device: {self.device}")
        logging.info(f"All devices: {self.devices}")
        
        # Set up mixed precision training if enabled and using CUDA
        self.enable_mixed_precision = config.enable_mixed_precision and any(d.type == 'cuda' for d in self.devices)
        if self.enable_mixed_precision:
            logging.info("Mixed precision training enabled")
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        # Initialize experts
        self._initialize_experts(config)
        
        # Initialize gating network (always on primary device)
        self._initialize_gating_network(config)
        
        # Initialize knowledge graph
        self._initialize_knowledge_graph(config)
        
        # Initialize sleep module
        self._initialize_sleep_module(config)
        
        # Expert specialization metrics
        self.expert_input_distributions = {i: {} for i in range(config.num_initial_experts)}
        self.expert_performance_history = {i: [] for i in range(config.num_initial_experts)}
        
        # Store the expert device map in the config
        config.expert_device_map = self.expert_device_map
        
        # Determine optimal batch size if auto_batch_size is enabled
        self._configure_batch_size(config)
        
        # Create parallel wrapper if using multi-GPU
        self._initialize_parallel_wrapper(config)
            
        # Move model to the specified device
        self.to(self.device)
    
    def _initialize_experts(self, config):
        """
        Initialize the experts for the MORPH model.
        
        Args:
            config: Configuration object
        """
        # Use smaller expert size for tests if in test mode
        expert_hidden_size = config.test_expert_size if config.test_mode else config.expert_hidden_size
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert(
                config.input_size, 
                expert_hidden_size, 
                config.output_size
            ) for _ in range(config.num_initial_experts)
        ])
        
        # Set expert IDs
        for i, expert in enumerate(self.experts):
            expert.expert_id = i
            
        # Distribute experts across devices if using multi-GPU
        if config.gpu_mode == "multi_gpu" and len(self.devices) > 1:
            self.expert_device_map = distribute_experts_across_gpus(
                config.num_initial_experts, self.devices
            )
            
            # Move experts to their assigned devices
            for i, expert in enumerate(self.experts):
                if i in self.expert_device_map:
                    expert_device = self.expert_device_map[i]
                    self.experts[i] = expert.to(expert_device)
                    logging.info(f"Expert {i} assigned to {expert_device}")
        else:
            # Single device mode - all experts on the same device
            self.expert_device_map = {i: self.device for i in range(config.num_initial_experts)}
            self.experts = self.experts.to(self.device)
    
    def _initialize_gating_network(self, config):
        """
        Initialize the gating network for the MORPH model.
        
        Args:
            config: Configuration object
        """
        # Initialize gating network (always on primary device)
        self.gating = GatingNetwork(
            config.input_size,
            config.num_initial_experts,
            k=config.expert_k
        ).to(self.device)
    
    def _initialize_knowledge_graph(self, config):
        """
        Initialize the knowledge graph for the MORPH model.
        
        Args:
            config: Configuration object
        """
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph(config)
        
        # Add initial experts to knowledge graph
        for i in range(config.num_initial_experts):
            self.knowledge_graph.add_expert(i, specialization_score=0.5, adaptation_rate=1.0)
    
    def _initialize_sleep_module(self, config):
        """
        Initialize the sleep module for the MORPH model.
        
        Args:
            config: Configuration object
        """
        # Initialize sleep module
        self.sleep_module = SleepModule(config, self.knowledge_graph)
    
    def _configure_batch_size(self, config):
        """
        Configure the batch size based on available GPU memory.
        
        Args:
            config: Configuration object
        """
        # Determine optimal batch size if auto_batch_size is enabled
        if config.auto_batch_size and any(d.type == 'cuda' for d in self.devices):
            try:
                # Create a dummy input to estimate batch size
                dummy_input_shape = (config.input_size,)
                optimal_batch_size = estimate_max_batch_size(
                    self, dummy_input_shape, self.device, max_memory_fraction=0.8
                )
                
                # Update batch size in config if estimated batch size is smaller
                if optimal_batch_size < config.batch_size:
                    logging.info(f"Adjusting batch size from {config.batch_size} to {optimal_batch_size} based on GPU memory")
                    config.batch_size = optimal_batch_size
            except Exception as e:
                logging.warning(f"Failed to estimate optimal batch size: {e}")
    
    def _initialize_parallel_wrapper(self, config):
        """
        Initialize the parallel wrapper for multi-GPU training.
        
        Args:
            config: Configuration object
        """
        # Create parallel wrapper if using multi-GPU
        if config.gpu_mode == "multi_gpu" and len(self.devices) > 1:
            self._wrapped_model = create_parallel_wrapper(self, config)
            logging.info(f"Created parallel wrapper with strategy: {config.parallel_strategy}")
        else:
            self._wrapped_model = None

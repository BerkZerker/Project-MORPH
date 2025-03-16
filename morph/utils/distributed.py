import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from morph.utils.gpu_utils import distribute_experts_across_gpus


class DataParallelWrapper(nn.Module):
    """
    A wrapper for data parallel training of the MORPH model.
    This distributes batches across multiple GPUs.
    """
    
    def __init__(self, model, devices):
        """
        Initialize the data parallel wrapper.
        
        Args:
            model: The MORPH model to wrap
            devices: List of devices to use
        """
        super().__init__()
        self.model = model
        self.devices = devices
        self.data_parallel = nn.DataParallel(model, device_ids=[d.index for d in devices if d.type == 'cuda'])
        
    def forward(self, x, training=True):
        """
        Forward pass through the data parallel model.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Model output
        """
        return self.data_parallel(x, training=training)
    
    def train_step(self, batch, optimizer, criterion):
        """
        Perform a single training step with data parallelism.
        
        Args:
            batch: Tuple of (inputs, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            
        Returns:
            Dictionary with loss and metrics
        """
        return self.model.train_step(batch, optimizer, criterion)
    
    def evaluate(self, data_loader, criterion, device=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            criterion: Loss function
            device: Device to use (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        return self.model.evaluate(data_loader, criterion, device)
    
    # Forward all other methods to the wrapped model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class ExpertParallelWrapper(nn.Module):
    """
    A wrapper for expert parallel training of the MORPH model.
    This distributes experts across multiple GPUs.
    """
    
    def __init__(self, model, devices):
        """
        Initialize the expert parallel wrapper.
        
        Args:
            model: The MORPH model to wrap
            devices: List of devices to use
        """
        super().__init__()
        self.model = model
        self.devices = devices
        
        # Distribute experts across devices
        self.expert_device_map = distribute_experts_across_gpus(
            len(model.experts), devices
        )
        
        # Move experts to their assigned devices
        for expert_idx, device in self.expert_device_map.items():
            if expert_idx < len(model.experts):
                model.experts[expert_idx] = model.experts[expert_idx].to(device)
        
        # Store the expert device map in the model config
        model.config.expert_device_map = self.expert_device_map
        
        # Move gating network to the first device
        model.gating = model.gating.to(devices[0])
        
        # Log expert distribution
        logging.info(f"Distributed {len(model.experts)} experts across {len(devices)} devices")
        for device in devices:
            expert_count = sum(1 for d in self.expert_device_map.values() if d == device)
            logging.info(f"  Device {device}: {expert_count} experts")
    
    def forward(self, x, training=True):
        """
        Forward pass with expert parallelism.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Model output
        """
        # Move input to the first device
        x = x.to(self.devices[0])
        
        # Get routing weights from gating network (on first device)
        routing_weights, expert_indices, uncertainty = self.model.gating(x, training)
        
        # Initialize output tensor on the first device
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, self.model.config.output_size, device=self.devices[0])
        
        # Process each expert on its assigned device
        for i in range(self.model.gating.k):
            # Get the expert indices and routing weights for this slot
            indices = expert_indices[:, i]  # [batch_size]
            weights = routing_weights[:, i].unsqueeze(1)  # [batch_size, 1]
            
            # Process each expert
            for expert_idx in indices.unique():
                # Find which batch items use this expert
                mask = (indices == expert_idx)
                if not mask.any():
                    continue
                
                # Get the expert and its device
                expert_idx_int = expert_idx.item()
                if expert_idx_int >= len(self.model.experts):
                    continue
                    
                expert = self.model.experts[expert_idx_int]
                expert_device = self.expert_device_map.get(expert_idx_int, self.devices[0])
                
                # Move relevant inputs to expert's device
                expert_inputs = x[mask].to(expert_device)
                expert_weights = weights[mask].to(expert_device)
                
                # Process inputs with this expert
                expert_outputs = expert(expert_inputs)
                
                # Weight outputs by routing weights
                weighted_outputs = expert_outputs * expert_weights
                
                # Move outputs back to the first device and add to final outputs
                outputs[mask] += weighted_outputs.to(self.devices[0])
                
                # Update expert activation counters
                expert.last_activated = self.model.step_count
                
                # If training, update knowledge graph
                if training:
                    self.model.knowledge_graph.update_expert_activation(expert_idx_int, self.model.step_count)
                    
                    # Store activation for sleep cycle if training
                    with torch.no_grad():
                        # Compute and store expert performance
                        # Store activation metadata for sleep module
                        input_features = torch.mean(expert_inputs, dim=0)
                        activation_data = {
                            'expert_idx': expert_idx_int,
                            'inputs': expert_inputs.detach().cpu() if self.model.config.memory_buffer_size > 0 else None,
                            'outputs': expert_outputs.detach().cpu() if self.model.config.memory_buffer_size > 0 else None,
                            'routing_weight': expert_weights.mean().item(),
                            'step': self.model.step_count,
                            'batch_size': expert_inputs.size(0),
                            'input_features': input_features.detach().cpu(),
                            'uncertainty': uncertainty.item() if uncertainty is not None else 0.0
                        }
                        
                        self.model.sleep_module.add_to_memory_buffer(activation_data)
                        
                        # Track input distribution for specialization analysis
                        if expert_idx_int in self.model.expert_input_distributions:
                            input_features_np = input_features.cpu().numpy()
                            feature_hash = hash(str(input_features_np.round(2)))
                            
                            if feature_hash in self.model.expert_input_distributions[expert_idx_int]:
                                self.model.expert_input_distributions[expert_idx_int][feature_hash] += 1
                            else:
                                self.model.expert_input_distributions[expert_idx_int][feature_hash] = 1
        
        # Increment step counter
        if training:
            self.model.step_count += 1
            
            # Maybe trigger sleep cycle
            if (self.model.config.enable_sleep and 
                self.model.sleep_module.should_sleep(self.model.step_count)):
                self.model.sleep()
        
        return outputs
    
    def train_step(self, batch, optimizer, criterion):
        """
        Perform a single training step with expert parallelism.
        
        Args:
            batch: Tuple of (inputs, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            
        Returns:
            Dictionary with loss and metrics
        """
        inputs, targets = batch
        
        # Move data to the first device
        inputs = inputs.to(self.devices[0])
        targets = targets.to(self.devices[0])
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Mixed precision training if enabled
        if self.model.enable_mixed_precision and any(d.type == 'cuda' for d in self.devices):
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                outputs = self(inputs, training=True)
                loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            self.model.scaler.scale(loss).backward()
            self.model.scaler.step(optimizer)
            self.model.scaler.update()
        else:
            # Standard training
            outputs = self(inputs, training=True)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100. * correct / targets.size(0)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'num_experts': len(self.model.experts)
        }
    
    def evaluate(self, data_loader, criterion, device=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            criterion: Loss function
            device: Device to use (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # Use first device if none provided
        device = device or self.devices[0]
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass (with training=False to disable expert creation)
                outputs = self(inputs, training=False)
                test_loss += criterion(outputs, targets).item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss /= len(data_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'num_experts': len(self.model.experts)
        }
    
    # Forward all other methods to the wrapped model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def create_parallel_wrapper(model, config):
    """
    Create a parallel wrapper for the model based on the configuration.
    
    Args:
        model: The MORPH model to wrap
        config: The configuration object
        
    Returns:
        Wrapped model
    """
    # If no CUDA devices or only one device, return the model as is
    if config.gpu_mode == "cpu" or (config.gpu_mode == "single_gpu" and len(config.devices) <= 1):
        logging.info("Using single device mode")
        return model
    
    # Create wrapper based on parallel strategy
    if config.parallel_strategy == "data_parallel":
        logging.info("Using data parallel strategy")
        return DataParallelWrapper(model, config.devices)
    elif config.parallel_strategy == "expert_parallel":
        logging.info("Using expert parallel strategy")
        return ExpertParallelWrapper(model, config.devices)
    else:
        logging.warning(f"Unknown parallel strategy: {config.parallel_strategy}, using model as is")
        return model


def setup_distributed_environment(rank=0, world_size=1):
    """
    Set up the distributed environment for multi-node training.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
    """
    if world_size <= 1:
        return
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU training
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    logging.info(f"Initialized process group: rank={rank}, world_size={world_size}")

import torch
import torch.nn as nn
import logging
import gc
from typing import Dict, List, Optional, Tuple, Union, Any

from src.utils.gpu_utils import (
    distribute_experts_across_gpus, 
    clear_gpu_cache, 
    optimize_tensor_memory,
    gradient_accumulation_step
)


class ExpertParallelWrapper(nn.Module):
    """
    A wrapper for expert parallel training of the MORPH model.
    This distributes experts across multiple GPUs.
    """
    
    def __init__(self, model, devices, expert_complexities=None):
        """
        Initialize the expert parallel wrapper.
        
        Args:
            model: The MORPH model to wrap
            devices: List of devices to use
            expert_complexities: Optional list of expert complexity scores (higher = more complex)
                                If provided, more complex experts will be assigned to more powerful GPUs
        """
        super().__init__()
        self.model = model
        self.devices = devices
        
        # Clear GPU cache before distribution
        for device in devices:
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    clear_gpu_cache()
        
        # Distribute experts across devices based on complexity if provided
        self.expert_device_map = distribute_experts_across_gpus(
            len(model.experts), 
            devices,
            expert_complexities=expert_complexities
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
    
    def forward(self, x, training=True, optimize_memory=True):
        """
        Forward pass with expert parallelism.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            optimize_memory: Whether to optimize memory usage
        Returns:
            Model output
        """
        # Optimize tensor memory if requested
        if optimize_memory and x.device.type == 'cuda':
            x = optimize_tensor_memory(x)
        
        # Move input to the first device (non-blocking for async transfer)
        x = x.to(self.devices[0], non_blocking=True)
        
        # Get routing weights from gating network (on first device)
        routing_weights, expert_indices, uncertainty = self.model.gating(x, training)
        
        # Initialize output tensor on the first device
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, self.model.config.output_size, device=self.devices[0])
        
        # Use mixed precision if enabled
        use_mixed_precision = getattr(self.model, 'mixed_precision_enabled', False)
        
        # Process each expert slot
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
                expert_inputs = x[mask].to(expert_device, non_blocking=True)
                expert_weights = weights[mask].to(expert_device, non_blocking=True)
                
                # Process inputs with this expert using mixed precision if enabled
                if use_mixed_precision and hasattr(self.model, 'get_autocast_context'):
                    with self.model.get_autocast_context(device_type=expert_device.type):
                        expert_outputs = expert(expert_inputs)
                else:
                    expert_outputs = expert(expert_inputs)
                
                # Weight outputs by routing weights
                weighted_outputs = expert_outputs * expert_weights
                
                # Move outputs back to the first device and add to final outputs
                outputs[mask] += weighted_outputs.to(self.devices[0], non_blocking=True)
                
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
        
        # Clear cache after forward pass if using CUDA
        if any(d.type == 'cuda' for d in self.devices):
            # Only clear every few steps to avoid overhead
            if self.model.step_count % 10 == 0:
                clear_gpu_cache()
        
        return outputs
    
    def train_step(self, batch, optimizer, criterion, accumulation_steps=1, current_step=None):
        """
        Perform a single training step with expert parallelism.
        
        Args:
            batch: Tuple of (inputs, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            accumulation_steps: Number of steps to accumulate gradients over
            current_step: Current step index (if None, uses model.step_count)
        Returns:
            Dictionary with loss and metrics
        """
        # Get current step if not provided
        if current_step is None:
            current_step = getattr(self.model, 'step_count', 0)
            
        inputs, targets = batch
        
        # Optimize tensor memory
        if inputs.device.type == 'cuda':
            inputs = optimize_tensor_memory(inputs)
        
        # Move data to the first device (non-blocking for async transfer)
        inputs = inputs.to(self.devices[0], non_blocking=True)
        targets = targets.to(self.devices[0], non_blocking=True)
        
        # Only zero gradients at the start of accumulation
        if current_step % accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Mixed precision training if enabled
        use_mixed_precision = getattr(self.model, 'mixed_precision_enabled', False)
        
        if use_mixed_precision and hasattr(self.model, 'get_autocast_context'):
            # Forward pass with autocast
            with self.model.get_autocast_context(device_type=self.devices[0].type):
                outputs = self(inputs, training=True)
                loss = criterion(outputs, targets)
            
            # Use gradient accumulation with mixed precision
            gradient_accumulation_step(
                loss=loss,
                optimizer=optimizer,
                scaler=getattr(self.model, 'scaler', None),
                accumulation_steps=accumulation_steps,
                current_step=current_step
            )
        else:
            # Standard training with gradient accumulation
            outputs = self(inputs, training=True)
            loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
                
            # Backward pass
            loss.backward()
            
            # Only step optimizer after accumulation
            if (current_step + 1) % accumulation_steps == 0:
                optimizer.step()
                
                # Clear GPU cache after optimizer step to free memory
                if any(d.type == 'cuda' for d in self.devices):
                    clear_gpu_cache()
        
        # Calculate accuracy
        with torch.no_grad():
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100. * correct / targets.size(0)
        
        metrics = {
            'loss': loss.item() * (accumulation_steps if accumulation_steps > 1 else 1),  # Scale loss back for reporting
            'accuracy': accuracy,
            'num_experts': len(self.model.experts)
        }
        
        # Add GPU memory info if available
        if any(d.type == 'cuda' for d in self.devices):
            try:
                metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated(self.devices[0].index) / (1024**3)  # GB
                metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved(self.devices[0].index) / (1024**3)    # GB
            except:
                pass
        
        return metrics
    
    def evaluate(self, data_loader, criterion, device=None, use_mixed_precision=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            criterion: Loss function
            device: Device to use (optional)
            use_mixed_precision: Whether to use mixed precision (if None, uses model's setting)
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # Use first device if none provided
        device = device or self.devices[0]
        
        # Determine whether to use mixed precision
        if use_mixed_precision is None:
            use_mixed_precision = getattr(self.model, 'mixed_precision_enabled', False)
        
        # Clear GPU cache before evaluation
        if device.type == 'cuda':
            clear_gpu_cache()
        
        batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                # Memory optimization for inputs
                if inputs.device.type == 'cuda':
                    inputs = optimize_tensor_memory(inputs)
                
                # Move data to the correct device with non-blocking transfers
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                try:
                    # Use mixed precision if enabled
                    if use_mixed_precision and hasattr(self.model, 'get_autocast_context'):
                        with self.model.get_autocast_context(device_type=device.type):
                            # Forward pass (with training=False to disable expert creation)
                            outputs = self(inputs, training=False)
                            loss = criterion(outputs, targets)
                    else:
                        # Standard evaluation
                        outputs = self(inputs, training=False)
                        loss = criterion(outputs, targets)
                    
                    # Accumulate loss
                    test_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    batch_count += 1
                    
                    # Periodically clear cache for long evaluations
                    if batch_count % 50 == 0 and device.type == 'cuda':
                        clear_gpu_cache()
                        
                except Exception as e:
                    logging.error(f"Error during evaluation: {e}")
                    # Continue with next batch
                    continue
        
        # Avoid division by zero
        test_loss /= max(1, batch_count)
        accuracy = 100. * correct / total
        
        metrics = {
            'loss': test_loss,
            'accuracy': accuracy,
            'num_experts': len(self.model.experts),
            'total_samples': total,
            'correct_samples': correct
        }
        
        # Add GPU memory info if available
        if device.type == 'cuda':
            try:
                metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated(device.index) / (1024**3)  # GB
                metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved(device.index) / (1024**3)    # GB
            except:
                pass
        
        # Final cache clear
        if device.type == 'cuda':
            clear_gpu_cache()
        
        return metrics
    
    # Forward all other methods to the wrapped model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

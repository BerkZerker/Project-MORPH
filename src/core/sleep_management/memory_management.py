import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np


def add_to_memory_buffer(sleep_module, activation_data):
    """
    Add activation data to the memory buffer.
    
    Args:
        sleep_module: The SleepModule instance
        activation_data: Dictionary containing activation information
    """
    # If buffer is full, remove oldest items
    while len(sleep_module.activation_buffer) >= sleep_module.buffer_size:
        sleep_module.activation_buffer.pop(0)
        
    # Add new activation
    sleep_module.activation_buffer.append(activation_data)


def perform_memory_replay(sleep_module, model):
    """
    Perform memory replay by replaying stored activations to experts.
    Uses batched processing for better performance.
    
    Args:
        sleep_module: The SleepModule instance
        model: The MORPH model
        
    Returns:
        Dictionary of replay statistics
    """
    if not sleep_module.activation_buffer:
        return {'samples_replayed': 0}
        
    # Prioritize replay experiences
    prioritized_buffer = _prioritize_experiences(sleep_module, model)
    
    # Group activations by expert for batched processing
    expert_activations = {}
    for activation in prioritized_buffer:
        expert_idx = activation['expert_idx']
        if expert_idx not in expert_activations:
            expert_activations[expert_idx] = []
        expert_activations[expert_idx].append(activation)
    
    # Process each expert's activations in batches
    replay_stats = {
        'samples_replayed': len(prioritized_buffer),
        'expert_updates': 0,
        'avg_loss': 0.0
    }
    
    # Use smaller batch size for tests if in test mode
    batch_size = sleep_module.config.memory_replay_batch_size
    if sleep_module.config.test_mode:
        batch_size = min(batch_size, 8)  # Smaller batch size for tests
    
    # Process each expert's activations in batches
    total_loss = 0.0
    update_count = 0
    
    # Use mixed precision if enabled and CUDA is available
    cuda_available = torch.cuda.is_available()
    device_is_cuda = sleep_module.device.type == 'cuda' if hasattr(sleep_module.device, 'type') else False
    use_amp = getattr(model, 'enable_mixed_precision', False) and device_is_cuda and cuda_available
    scaler = getattr(model, 'scaler', None) if use_amp else None
    
    for expert_idx, activations in expert_activations.items():
        # Skip if expert no longer exists (might have been pruned)
        if expert_idx >= len(model.experts):
            continue
            
        expert = model.experts[expert_idx]
        
        # Create a small optimizer for this expert
        expert_optimizer = torch.optim.Adam(expert.parameters(), lr=sleep_module.config.replay_learning_rate)
        
        # Process in batches
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size]
            
            # Skip empty batches
            if not batch:
                continue
            
            # Collect inputs and expected outputs
            valid_batch = [a for a in batch if a['inputs'] is not None and a['outputs'] is not None]
            if not valid_batch:
                continue
            
            # Move data to the appropriate device
            inputs = torch.cat([a['inputs'] for a in valid_batch])
            expected_outputs = torch.cat([a['outputs'] for a in valid_batch])
            
            # Only move to device if the tensors aren't already on the correct device
            if hasattr(sleep_module, 'device'):
                current_device = inputs.device
                target_device = sleep_module.device
                
                # Check if we're trying to move to a CUDA device when CUDA is not available
                if target_device.type == 'cuda' and not torch.cuda.is_available():
                    # If CUDA is not available, use CPU instead
                    target_device = torch.device('cpu')
                
                if current_device != target_device:
                    inputs = inputs.to(target_device)
                    expected_outputs = expected_outputs.to(target_device)
            
            # Skip empty batches
            if inputs.size(0) == 0:
                continue
            
            # Zero gradients
            expert_optimizer.zero_grad()
            
            # Forward pass with autocast if mixed precision is enabled
            # Only use autocast with CUDA devices
            device_type = sleep_module.device.type if hasattr(sleep_module.device, 'type') else 'cpu'
            with torch.autocast(device_type=device_type, enabled=use_amp):
                # Process inputs with expert
                outputs = expert(inputs)
                
                # Calculate loss (mean squared error)
                loss = F.mse_loss(outputs, expected_outputs)
                total_loss += loss.item()
            
            # Backward pass with gradient scaling if mixed precision is enabled
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(expert_optimizer)
                scaler.update()
            else:
                loss.backward()
                expert_optimizer.step()
            
            # Update stats
            update_count += 1
            
        # Update expert's specialization score based on processed activations
        if hasattr(expert, 'update_confidence') and update_count > 0:
            expert.update_confidence(total_loss / update_count if update_count > 0 else 0.0)
    
    # Update replay stats
    replay_stats['expert_updates'] = update_count
    replay_stats['avg_loss'] = total_loss / update_count if update_count > 0 else 0.0
    
    # Clear activation buffer after replay
    sleep_module.activation_buffer = []
    
    return replay_stats


def _prioritize_experiences(sleep_module, model):
    """
    Prioritize experiences in the replay buffer.
    
    Args:
        sleep_module: The SleepModule instance
        model: The MORPH model
        
    Returns:
        List of prioritized activation data
    """
    if not sleep_module.activation_buffer:
        return []
        
    # Sort by priority (highest first)
    prioritized_buffer = sorted(
        [a.copy() for a in sleep_module.activation_buffer],
        key=lambda x: x.get('uncertainty', 0.0),
        reverse=True
    )
    
    return prioritized_buffer

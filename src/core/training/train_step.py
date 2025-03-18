import torch
from torch.amp import autocast
import logging
from src.utils.gpu_utils import clear_gpu_cache, gradient_accumulation_step, optimize_tensor_memory


def train_step(model, batch, optimizer, criterion, accumulation_steps=1, current_step=None):
    """
    Perform a single training step.
    
    Args:
        model: The MORPH model
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
        current_step = getattr(model, 'step_count', 0)
    
    # If using a parallel wrapper, delegate to it
    if hasattr(model, '_wrapped_model') and model._wrapped_model is not None and not isinstance(model._wrapped_model, type(model)):
        return model._wrapped_model.train_step(batch, optimizer, criterion, 
                                              accumulation_steps, current_step)
        
    inputs, targets = batch
    
    # Memory optimization for inputs
    if inputs.device.type == 'cuda':
        inputs = optimize_tensor_memory(inputs)
    
    # Move data to the correct device
    inputs = inputs.to(model.device, non_blocking=True)  # Use non-blocking for async transfer
    targets = targets.to(model.device, non_blocking=True)
    
    # Only zero gradients at the start of accumulation
    if current_step % accumulation_steps == 0:
        optimizer.zero_grad()
    
    # Mixed precision training if enabled
    if hasattr(model, 'mixed_precision_enabled') and model.mixed_precision_enabled:
        # Get the appropriate autocast context
        with model.get_autocast_context():
            outputs = model(inputs, training=True)
            loss = criterion(outputs, targets)
        
        # Use gradient accumulation with mixed precision
        gradient_accumulation_step(
            loss=loss,
            optimizer=optimizer,
            scaler=getattr(model, 'scaler', None),
            accumulation_steps=accumulation_steps,
            current_step=current_step
        )
    else:
        # Standard training with gradient accumulation
        outputs = model(inputs, training=True)
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
            if any(p.device.type == 'cuda' for p in model.parameters()):
                clear_gpu_cache()
    
    # Calculate accuracy
    with torch.no_grad():
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = 100. * correct / targets.size(0)
    
    metrics = {
        'loss': loss.item() * (accumulation_steps if accumulation_steps > 1 else 1),  # Scale loss back for reporting
        'accuracy': accuracy,
        'num_experts': len(model.experts) if hasattr(model, 'experts') else 0
    }
    
    # Add GPU memory info if available
    if torch.cuda.is_available():
        try:
            metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)  # GB
            metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)    # GB
        except:
            pass
    
    return metrics

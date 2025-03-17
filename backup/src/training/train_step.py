import torch
from torch.amp import autocast


def train_step(model, batch, optimizer, criterion):
    """
    Perform a single training step.
    
    Args:
        model: The MORPH model
        batch: Tuple of (inputs, targets)
        optimizer: Optimizer to use
        criterion: Loss function
        
    Returns:
        Dictionary with loss and metrics
    """
    # If using a parallel wrapper, delegate to it
    if model._wrapped_model is not None and not isinstance(model._wrapped_model, type(model)):
        return model._wrapped_model.train_step(batch, optimizer, criterion)
        
    inputs, targets = batch
    
    # Move data to the correct device
    inputs = inputs.to(model.device)
    targets = targets.to(model.device)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Mixed precision training if enabled
    if model.enable_mixed_precision and any(d.type == 'cuda' for d in model.devices):
        # Forward pass with autocast
        with autocast('cuda'):
            outputs = model(inputs, training=True)
            loss = criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        model.scaler.scale(loss).backward()
        model.scaler.step(optimizer)
        model.scaler.update()
    else:
        # Standard training
        outputs = model(inputs, training=True)
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
        'num_experts': len(model.experts)
    }

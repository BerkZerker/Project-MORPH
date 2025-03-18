import torch
import logging
from src.utils.gpu_utils import clear_gpu_cache, optimize_tensor_memory


def evaluate(model, data_loader, criterion, device=None, use_mixed_precision=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The MORPH model
        data_loader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to use (optional, defaults to model's device)
        use_mixed_precision: Whether to use mixed precision (if None, uses model's setting)
    Returns:
        Dictionary with evaluation metrics
    """
    # If using a parallel wrapper, delegate to it
    if hasattr(model, '_wrapped_model') and model._wrapped_model is not None and not isinstance(model._wrapped_model, type(model)):
        return model._wrapped_model.evaluate(data_loader, criterion, device)
        
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Use model's device if none provided
    device = device or model.device
    
    # Determine whether to use mixed precision
    if use_mixed_precision is None:
        use_mixed_precision = getattr(model, 'mixed_precision_enabled', False)
    
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
                if use_mixed_precision and hasattr(model, 'get_autocast_context'):
                    with model.get_autocast_context():
                        # Forward pass (with training=False to disable expert creation)
                        outputs = model(inputs, training=False)
                        loss = criterion(outputs, targets)
                else:
                    # Standard evaluation
                    outputs = model(inputs, training=False)
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
        'num_experts': len(model.experts) if hasattr(model, 'experts') else 0,
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

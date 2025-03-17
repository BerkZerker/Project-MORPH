import torch


def evaluate(model, data_loader, criterion, device=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The MORPH model
        data_loader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to use (optional, defaults to model's device)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # If using a parallel wrapper, delegate to it
    if model._wrapped_model is not None and not isinstance(model._wrapped_model, type(model)):
        return model._wrapped_model.evaluate(data_loader, criterion, device)
        
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # Use model's device if none provided
    device = device or model.device
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass (with training=False to disable expert creation)
            outputs = model(inputs, training=False)
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
        'num_experts': len(model.experts)
    }

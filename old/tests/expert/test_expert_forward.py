import torch
from src.core.expert import Expert
from src.utils.testing.decorators import visualize_test, capture_test_state


def test_expert_forward():
    """Test that an expert forward pass works correctly."""
    expert = Expert(input_size=10, hidden_size=20, output_size=5)
    
    # Move expert to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert = expert.to(device)
    
    # Create a batch of inputs and move to the same device
    inputs = torch.randn(32, 10, device=device)
    
    # Forward pass
    outputs = expert(inputs)
    
    # Check output shape
    assert outputs.shape == (32, 5)
    assert outputs.device == device  # Ensure output is on the correct device

    # Check activation count
    assert expert.activation_count == 32

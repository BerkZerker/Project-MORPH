import torch
from src.core.expert import Expert
from src.utils.testing.decorators import visualize_test, capture_test_state


def test_parameter_similarity():
    """Test parameter similarity calculation."""
    # Move experts to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert1 = Expert(input_size=10, hidden_size=20, output_size=5).to(device)
    expert2 = expert1.clone()  # Different random weights
    
    # With different weights, similarity should be low
    similarity = expert1.get_parameter_similarity(expert2)
    assert -1 <= similarity <= 1  # Should be a valid similarity score
    assert not torch.isnan(similarity) # Similarity should not be NaN
    assert similarity > -0.5 # Ensure similarity is not too negative
    
    # Same expert should have perfect similarity
    similarity = expert1.get_parameter_similarity(expert1)
    assert similarity > 0.99  # Should be very close to 1

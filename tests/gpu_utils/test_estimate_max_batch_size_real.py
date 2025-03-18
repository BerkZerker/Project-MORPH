import pytest
import torch

from src.utils.gpu_utils import estimate_max_batch_size
from tests.gpu_utils.simple_model import SimpleModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_estimate_max_batch_size_real():
    """Test batch size estimation with real GPU if available."""
    # Only run this test if CUDA is actually available
    model = SimpleModel()
    input_shape = (10,)  # Match model input size
    device = torch.device("cuda:0")
    
    # Move model to GPU
    model = model.to(device)
    
    # Estimate batch size
    batch_size = estimate_max_batch_size(model, input_shape, device)
    
    # Just check that we get a reasonable positive number
    assert batch_size > 0
    assert isinstance(batch_size, int)

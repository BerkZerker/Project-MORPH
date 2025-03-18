import pytest
import torch
from src.utils.distributed import DataParallelWrapper
from tests.distributed.simple_model import SimpleModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_data_parallel_wrapper_real():
    """Test DataParallelWrapper with real GPU if available."""
    # Only run this test if CUDA is actually available
    model = SimpleModel()
    devices = [torch.device("cuda:0")]
    
    # Create wrapper
    wrapper = DataParallelWrapper(model, devices)
    
    # Test forward pass
    x = torch.randn(4, 10).to(devices[0])
    output = wrapper(x)
    
    # Check output shape
    assert output.shape == (4, 5)

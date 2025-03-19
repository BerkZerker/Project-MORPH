import torch
import pytest
from src.core.model import MorphModel
from src.config import MorphConfig


@pytest.fixture
def model_config(optimized_test_config):
    """
    Create a model configuration for expert merging tests.
    
    This uses the optimized test configuration as a base and adds
    expert merging specific parameters.
    """
    config = optimized_test_config
    
    # Expert merging specific parameters
    config.num_initial_experts = 3  # Need at least 3 for merging tests
    config.expert_k = 2
    config.enable_dynamic_experts = True
    config.enable_sleep = True
    config.expert_similarity_threshold = 0.8
    config.min_experts = 2
    
    return config


@pytest.fixture
def model(model_config):
    model = MorphModel(model_config)
    # Process a batch to initialize
    device = torch.device(model_config.device)
    batch = torch.randn(10, model_config.input_size, device=device)
    model(batch, training=True)
    return model

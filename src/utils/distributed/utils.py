import logging
from typing import Optional

from src.utils.distributed.data_parallel import DataParallelWrapper
from src.utils.distributed.expert_parallel import ExpertParallelWrapper


def create_parallel_wrapper(model, config):
    """
    Create a parallel wrapper for the model based on the configuration.
    
    Args:
        model: The MORPH model to wrap
        config: The configuration object
        
    Returns:
        Wrapped model
    """
    # If no CUDA devices or only one device, return the model as is
    if config.gpu_mode == "cpu" or (config.gpu_mode == "single_gpu" and len(config.devices) <= 1):
        logging.info("Using single device mode")
        return model
    
    # Create wrapper based on parallel strategy
    if config.parallel_strategy == "data_parallel":
        logging.info("Using data parallel strategy")
        return DataParallelWrapper(model, config.devices)
    elif config.parallel_strategy == "expert_parallel":
        logging.info("Using expert parallel strategy")
        return ExpertParallelWrapper(model, config.devices)
    else:
        logging.warning(f"Unknown parallel strategy: {config.parallel_strategy}, using model as is")
        return model

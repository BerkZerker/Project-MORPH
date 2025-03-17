import torch
import torch.distributed as dist
import logging
from typing import Optional


def setup_distributed_environment(rank: int = 0, world_size: int = 1) -> None:
    """
    Set up the distributed environment for multi-node training.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
    """
    if world_size <= 1:
        return
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU training
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    logging.info(f"Initialized process group: rank={rank}, world_size={world_size}")
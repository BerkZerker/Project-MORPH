"""
Distributed computing utilities for MORPH.

This module provides tools for distributed training across multiple GPUs,
including data parallelism and expert parallelism.
"""

# Re-export from distributed modules
from src.utils.distributed.base import setup_distributed_environment
from src.utils.distributed.data_parallel import DataParallelWrapper
from src.utils.distributed.expert_parallel import ExpertParallelWrapper
from src.utils.distributed.utils import create_parallel_wrapper

# Define exports
__all__ = [
    'setup_distributed_environment',
    'DataParallelWrapper',
    'ExpertParallelWrapper',
    'create_parallel_wrapper'
]

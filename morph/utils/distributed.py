"""
Distributed computing utilities for MORPH.

This module provides tools for distributed training across multiple GPUs,
including data parallelism and expert parallelism.
"""

# Re-export from distributed modules
from morph.utils.distributed.base import setup_distributed_environment
from morph.utils.distributed.data_parallel import DataParallelWrapper
from morph.utils.distributed.expert_parallel import ExpertParallelWrapper
from morph.utils.distributed.utils import create_parallel_wrapper

# Define exports
__all__ = [
    'setup_distributed_environment',
    'DataParallelWrapper',
    'ExpertParallelWrapper',
    'create_parallel_wrapper'
]

"""
Dataset creation utilities for benchmarking.
"""

from morph.utils.benchmarks.datasets.mnist_variants import (
    create_rotating_mnist_tasks,
    create_split_mnist_tasks,
    create_permuted_mnist_tasks
)

__all__ = [
    'create_rotating_mnist_tasks',
    'create_split_mnist_tasks',
    'create_permuted_mnist_tasks'
]

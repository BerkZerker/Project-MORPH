"""
Benchmarking utilities for MORPH models.

This package provides tools for benchmarking MORPH models against standard
neural networks and other continual learning approaches.
"""

from morph.utils.benchmarks.models.standard_model import StandardModel
from morph.utils.benchmarks.models.ewc_model import EWCModel
from morph.utils.benchmarks.benchmark import ContinualLearningBenchmark
from morph.utils.benchmarks.datasets.mnist_variants import (
    create_rotating_mnist_tasks,
    create_split_mnist_tasks,
    create_permuted_mnist_tasks
)

__all__ = [
    'StandardModel',
    'EWCModel',
    'ContinualLearningBenchmark',
    'create_rotating_mnist_tasks',
    'create_split_mnist_tasks',
    'create_permuted_mnist_tasks'
]

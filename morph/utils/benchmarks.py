"""
Benchmarking utilities for MORPH models.

This module is maintained for backward compatibility.
All functionality has been moved to the benchmarks/ package.
"""

from morph.utils.benchmarks.models.standard_model import StandardModel
from morph.utils.benchmarks.models.ewc_model import EWCModel
from morph.utils.benchmarks.benchmark import ContinualLearningBenchmark
from morph.utils.benchmarks.datasets.mnist_variants import (
    create_rotating_mnist_tasks,
    create_split_mnist_tasks,
    create_permuted_mnist_tasks
)
from morph.utils.benchmarks.evaluation.visualization import visualize_results

# Re-export all components for backward compatibility
__all__ = [
    'StandardModel',
    'EWCModel',
    'ContinualLearningBenchmark',
    'create_rotating_mnist_tasks',
    'create_split_mnist_tasks',
    'create_permuted_mnist_tasks',
    'visualize_results'
]

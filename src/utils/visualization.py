"""
Visualization utilities for MORPH models.

This module is maintained for backward compatibility.
All functionality has been moved to the visualization/ package.
"""

from src.utils.visualization.knowledge_graph import visualize_knowledge_graph
from src.utils.visualization.expert_activation import (
    plot_expert_activations,
    visualize_expert_lifecycle,
    visualize_expert_specialization_over_time
)
from src.utils.visualization.performance_plots import visualize_concept_drift_adaptation
from src.utils.visualization.training_progress import visualize_sleep_metrics

# Re-export all components for backward compatibility
__all__ = [
    'visualize_knowledge_graph',
    'plot_expert_activations',
    'visualize_expert_lifecycle',
    'visualize_expert_specialization_over_time',
    'visualize_concept_drift_adaptation',
    'visualize_sleep_metrics'
]

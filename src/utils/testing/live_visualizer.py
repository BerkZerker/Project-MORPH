"""
Live visualization generator for MORPH tests.

This module provides utilities for generating real-time visualizations
of MORPH models during test execution.
"""

import os
import io
import base64
import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import networkx as nx

from src.core.model import MorphModel
from src.utils.visualization.knowledge_graph import visualize_knowledge_graph
from src.utils.visualization.expert_activation import (
    plot_expert_activations,
    visualize_expert_lifecycle,
    visualize_expert_specialization_over_time
)


class LiveModelVisualizer:
    """
    Generates real-time visualizations of MORPH models.
    
    This class provides methods to generate visualizations of model state
    that can be displayed in a web interface.
    """
    
    def __init__(self):
        """Initialize the live model visualizer."""
        self.last_update = 0
        self.update_interval = 0.5  # Minimum time between updates (seconds)
    
    def visualize_model(self, model: MorphModel) -> Dict[str, Any]:
        """
        Generate visualizations for a model.
        
        Args:
            model: The MorphModel instance
            
        Returns:
            Dictionary containing visualization data
        """
        # Throttle updates to avoid excessive CPU usage
        now = time.time()
        if now - self.last_update < self.update_interval:
            return {}
        
        self.last_update = now
        
        # Generate visualizations
        visualizations = {
            'knowledge_graph': self._visualize_knowledge_graph(model),
            'expert_activations': self._visualize_expert_activations(model),
            'model_metrics': self._visualize_model_metrics(model),
            'expert_specialization': self._visualize_expert_specialization(model)
        }
        
        return visualizations
    
    def _visualize_knowledge_graph(self, model: MorphModel) -> Dict[str, Any]:
        """
        Generate knowledge graph visualization.
        
        Args:
            model: The MorphModel instance
            
        Returns:
            Dictionary containing visualization data
        """
        # Create a buffer for the image
        buf = io.BytesIO()
        
        # Generate knowledge graph visualization
        plt.figure(figsize=(10, 8))
        
        # Get the knowledge graph
        G = model.knowledge_graph.graph
        
        if len(G.nodes) > 0:  # Only draw if there are nodes
            # Create positions for nodes
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos,
                node_size=500,
                node_color='lightblue',
                alpha=0.8
            )
            
            # Draw edges
            if len(G.edges) > 0:  # Only draw if there are edges
                nx.draw_networkx_edges(
                    G, pos,
                    width=1.0,
                    alpha=0.5
                )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=10,
                font_family='sans-serif'
            )
        
        plt.title('Knowledge Graph')
        plt.axis('off')
        plt.tight_layout()
        
        # Save to buffer
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            'image': f'data:image/png;base64,{img_str}',
            'nodes': len(G.nodes),
            'edges': len(G.edges)
        }
    
    def _visualize_expert_activations(self, model: MorphModel) -> Dict[str, Any]:
        """
        Generate expert activations visualization.
        
        Args:
            model: The MorphModel instance
            
        Returns:
            Dictionary containing visualization data
        """
        # Create a buffer for the image
        buf = io.BytesIO()
        
        # Generate expert activations visualization
        plt.figure(figsize=(10, 6))
        
        # Get expert activation counts
        expert_ids = [e.expert_id for e in model.experts]
        activation_counts = [e.activation_count for e in model.experts]
        
        # Create bar chart
        plt.bar(range(len(expert_ids)), activation_counts)
        plt.xlabel('Expert Index')
        plt.ylabel('Activation Count')
        plt.title('Expert Activation Counts')
        plt.xticks(range(len(expert_ids)), expert_ids)
        plt.tight_layout()
        
        # Save to buffer
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            'image': f'data:image/png;base64,{img_str}',
            'experts': len(expert_ids),
            'max_activation': max(activation_counts) if activation_counts else 0
        }
    
    def _visualize_model_metrics(self, model: MorphModel) -> Dict[str, Any]:
        """
        Generate model metrics visualization.
        
        Args:
            model: The MorphModel instance
            
        Returns:
            Dictionary containing visualization data
        """
        # Create a buffer for the image
        buf = io.BytesIO()
        
        # Generate model metrics visualization
        plt.figure(figsize=(10, 6))
        
        # Collect metrics
        metrics = [
            ('Number of Experts', len(model.experts)),
            ('Sleep Cycles', getattr(model, 'sleep_cycles_completed', 0)),
            ('Step Count', getattr(model, 'step_count', 0)),
            ('KG Nodes', len(model.knowledge_graph.graph.nodes)),
            ('KG Edges', len(model.knowledge_graph.graph.edges)),
            ('Buffer Size', len(getattr(model, 'activation_buffer', [])))
        ]
        
        labels, values = zip(*metrics)
        
        # Create bar chart
        plt.bar(labels, values)
        plt.ylabel('Value')
        plt.title('Model Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save to buffer
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            'image': f'data:image/png;base64,{img_str}',
            'metrics': dict(metrics)
        }
    
    def _visualize_expert_specialization(self, model: MorphModel) -> Dict[str, Any]:
        """
        Generate expert specialization visualization.
        
        Args:
            model: The MorphModel instance
            
        Returns:
            Dictionary containing visualization data
        """
        # Create a buffer for the image
        buf = io.BytesIO()
        
        # Check if we have specialization scores
        has_specialization = all(
            'specialization_score' in model.knowledge_graph.graph.nodes[e.expert_id] 
            for e in model.experts if e.expert_id is not None
        )
        
        if has_specialization:
            # Generate expert specialization visualization
            plt.figure(figsize=(10, 6))
            
            # Get expert specialization scores
            expert_ids = [e.expert_id for e in model.experts if e.expert_id is not None]
            spec_scores = [
                model.knowledge_graph.graph.nodes[e.expert_id].get('specialization_score', 0)
                for e in model.experts if e.expert_id is not None
            ]
            
            # Create bar chart
            plt.bar(range(len(expert_ids)), spec_scores)
            plt.xlabel('Expert Index')
            plt.ylabel('Specialization Score')
            plt.title('Expert Specialization Scores')
            plt.xticks(range(len(expert_ids)), expert_ids)
            plt.tight_layout()
            
            # Save to buffer
            plt.savefig(buf, format='png')
            plt.close()
            
            # Convert to base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return {
                'image': f'data:image/png;base64,{img_str}',
                'experts': len(expert_ids),
                'max_score': max(spec_scores) if spec_scores else 0
            }
        else:
            # No specialization scores available
            return {
                'image': None,
                'experts': 0,
                'max_score': 0
            }


# Global instance
_visualizer = LiveModelVisualizer()


def visualize_model(model: MorphModel) -> Dict[str, Any]:
    """
    Generate visualizations for a model.
    
    Args:
        model: The MorphModel instance
        
    Returns:
        Dictionary containing visualization data
    """
    global _visualizer
    return _visualizer.visualize_model(model)


def get_model_snapshot(model: MorphModel) -> Dict[str, Any]:
    """
    Get a snapshot of the model state.
    
    Args:
        model: The MorphModel instance
        
    Returns:
        Dictionary containing model state
    """
    # Capture model state
    state = {
        'num_experts': len(model.experts) if hasattr(model, 'experts') else 0,
        'sleep_cycles_completed': getattr(model, 'sleep_cycles_completed', 0),
        'step_count': getattr(model, 'step_count', 0),
        'next_sleep_step': getattr(model, 'next_sleep_step', None),
        'adaptive_sleep_frequency': getattr(model, 'adaptive_sleep_frequency', None),
        'activation_buffer_size': len(getattr(model, 'activation_buffer', [])),
        'knowledge_graph_nodes': len(model.knowledge_graph.graph.nodes),
        'knowledge_graph_edges': len(model.knowledge_graph.graph.edges),
    }
    
    # Capture expert states
    expert_states = {}
    if hasattr(model, 'experts'):
        for i, expert in enumerate(model.experts):
            expert_states[i] = {
                'expert_id': expert.expert_id,
                'activation_count': expert.activation_count,
                'last_activated': expert.last_activated,
            }
    
    state['expert_states'] = expert_states
    
    # Generate visualizations
    state['visualizations'] = visualize_model(model)
    
    return state

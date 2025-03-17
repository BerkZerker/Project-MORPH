"""
Visualization methods for the Knowledge Graph.

This module provides methods for visualizing the knowledge graph,
such as generating graph layouts and node colors.
"""

import networkx as nx
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class GraphVisualization:
    """
    Visualization methods for the Knowledge Graph.
    
    Provides methods for visualizing the knowledge graph structure.
    """
    
    def visualize_graph(self, layout_type: str = 'spring') -> Dict[str, Any]:
        """
        Generate visualization data for the knowledge graph.
        
        Args:
            layout_type: Type of layout algorithm to use ('spring', 'circular', etc.)
            
        Returns:
            Dictionary with visualization data (positions, colors, etc.)
        """
        if len(self.graph.nodes) == 0:
            return {'positions': {}, 'node_colors': {}, 'edge_weights': {}}
        
        # Get positions
        positions = self.get_graph_layout(layout_type)
        
        # Get node colors based on specialization
        node_colors = self.get_node_colors()
        
        # Get edge weights
        edge_weights = {
            (u, v): data.get('weight', 0.5)
            for u, v, data in self.graph.edges(data=True)
        }
        
        return {
            'positions': positions,
            'node_colors': node_colors,
            'edge_weights': edge_weights
        }
    
    def get_graph_layout(self, layout_type: str = 'spring') -> Dict[int, Tuple[float, float]]:
        """
        Get positions for graph nodes using a specified layout algorithm.
        
        Args:
            layout_type: Type of layout algorithm to use
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if len(self.graph.nodes) == 0:
            return {}
        
        # Choose layout algorithm
        if layout_type == 'spring':
            layout_func = nx.spring_layout
        elif layout_type == 'circular':
            layout_func = nx.circular_layout
        elif layout_type == 'kamada_kawai':
            layout_func = nx.kamada_kawai_layout
        elif layout_type == 'spectral':
            layout_func = nx.spectral_layout
        else:
            layout_func = nx.spring_layout
        
        # Generate layout
        positions = layout_func(self.graph)
        
        # Convert to dictionary of (x, y) tuples
        return {
            node: (pos[0], pos[1])
            for node, pos in positions.items()
        }
    
    def get_node_colors(self) -> Dict[int, Tuple[float, float, float]]:
        """
        Generate colors for nodes based on their specialization scores.
        
        Returns:
            Dictionary mapping node IDs to RGB color tuples
        """
        node_colors = {}
        
        for node in self.graph.nodes:
            # Get specialization score (default to 0.5)
            specialization = self.graph.nodes[node].get('specialization_score', 0.5)
            
            # Generate color based on specialization
            # More specialized experts are more red, less specialized are more blue
            r = min(1.0, 0.5 + specialization * 0.5)
            g = 0.4
            b = min(1.0, 0.5 + (1.0 - specialization) * 0.5)
            
            node_colors[node] = (r, g, b)
        
        return node_colors
    
    def get_subgraph_for_experts(self, expert_ids: List[int]) -> nx.Graph:
        """
        Extract a subgraph containing only the specified experts and their connections.
        
        Args:
            expert_ids: List of expert IDs to include
            
        Returns:
            NetworkX graph object for the subgraph
        """
        # Filter to only include valid expert IDs
        valid_ids = [eid for eid in expert_ids if eid in self.graph.nodes]
        
        # Extract subgraph
        return self.graph.subgraph(valid_ids)

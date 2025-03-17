"""
Serialization methods for the Knowledge Graph.

This module provides methods for saving and loading the knowledge graph,
as well as exporting graph data for external use.
"""

import networkx as nx
import torch
import json
import os
import logging
from typing import Dict, Any, Optional


class GraphSerialization:
    """
    Serialization methods for the Knowledge Graph.
    
    Provides methods for saving and loading the knowledge graph.
    """
    
    def save_graph(self, filepath: str) -> bool:
        """
        Save the knowledge graph to a file.
        
        Args:
            filepath: Path to save the graph to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Prepare data for serialization
            graph_data = {
                'nodes': [],
                'edges': [],
                'concepts': {},
                'expert_concepts': {}
            }
            
            # Serialize nodes
            for node, data in self.graph.nodes(data=True):
                node_data = {'id': node}
                node_data.update(data)
                graph_data['nodes'].append(node_data)
            
            # Serialize edges
            for u, v, data in self.graph.edges(data=True):
                edge_data = {'source': u, 'target': v}
                edge_data.update(data)
                graph_data['edges'].append(edge_data)
            
            # Serialize concept embeddings
            for concept_id, embedding in self.concept_embeddings.items():
                graph_data['concepts'][concept_id] = embedding.tolist()
            
            # Serialize expert-to-concept mappings
            for expert_id, concepts in self.expert_concepts.items():
                graph_data['expert_concepts'][str(expert_id)] = list(concepts)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            return True
        except Exception as e:
            logging.error(f"Error saving knowledge graph: {e}")
            return False
    
    def load_graph(self, filepath: str) -> bool:
        """
        Load the knowledge graph from a file.
        
        Args:
            filepath: Path to load the graph from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logging.error(f"Graph file not found: {filepath}")
                return False
            
            # Load from file
            with open(filepath, 'r') as f:
                graph_data = json.load(f)
            
            # Create new graph
            self.graph = nx.Graph()
            
            # Load nodes
            for node_data in graph_data['nodes']:
                node_id = node_data.pop('id')
                self.graph.add_node(node_id, **node_data)
            
            # Load edges
            for edge_data in graph_data['edges']:
                source = edge_data.pop('source')
                target = edge_data.pop('target')
                self.graph.add_edge(source, target, **edge_data)
            
            # Load concept embeddings
            self.concept_embeddings = {}
            for concept_id, embedding_list in graph_data['concepts'].items():
                self.concept_embeddings[concept_id] = torch.tensor(embedding_list)
            
            # Load expert-to-concept mappings
            self.expert_concepts = {}
            for expert_id_str, concepts in graph_data['expert_concepts'].items():
                self.expert_concepts[int(expert_id_str)] = set(concepts)
            
            return True
        except Exception as e:
            logging.error(f"Error loading knowledge graph: {e}")
            return False
    
    def export_graph_data(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """
        Export graph data in a format suitable for external use.
        
        Args:
            include_embeddings: Whether to include concept embeddings
            
        Returns:
            Dictionary with graph data
        """
        # Prepare data for export
        export_data = {
            'nodes': [],
            'edges': [],
            'concepts': []
        }
        
        # Export nodes
        for node, data in self.graph.nodes(data=True):
            node_data = {'id': node}
            # Include selected node attributes
            for attr in ['type', 'specialization_score', 'activation_count']:
                if attr in data:
                    node_data[attr] = data[attr]
            export_data['nodes'].append(node_data)
        
        # Export edges
        for u, v, data in self.graph.edges(data=True):
            edge_data = {
                'source': u,
                'target': v,
                'weight': data.get('weight', 0.5),
                'type': data.get('relation_type', 'similarity')
            }
            export_data['edges'].append(edge_data)
        
        # Export concepts
        for concept_id in self.concept_embeddings:
            concept_data = {'id': concept_id}
            
            # Include embeddings if requested
            if include_embeddings:
                concept_data['embedding'] = self.concept_embeddings[concept_id].tolist()
            
            # Find experts associated with this concept
            concept_data['experts'] = [
                expert_id for expert_id, concepts in self.expert_concepts.items()
                if concept_id in concepts
            ]
            
            export_data['concepts'].append(concept_data)
        
        return export_data

"""
Operations for the Knowledge Graph.

This module provides methods for manipulating the knowledge graph,
such as adding experts, adding edges, and updating expert information.
"""

import logging
import torch
from typing import Dict, List, Tuple, Optional, Any, Set


class GraphOperations:
    """
    Operations for the Knowledge Graph.
    
    Provides methods for manipulating the knowledge graph structure.
    """
    
    def add_expert(self, expert_id: int, specialization_score: float = 0.5,
                  adaptation_rate: float = 1.0) -> None:
        """
        Add a new expert to the knowledge graph.
        
        Args:
            expert_id: Unique identifier for the expert
            specialization_score: Initial specialization score (0.0-1.0)
            adaptation_rate: Initial adaptation rate (0.0-1.0)
        """
        self.graph.add_node(
            expert_id,
            type='expert',
            activation_count=0,
            last_activated=0,
            specialization_score=specialization_score,
            adaptation_rate=adaptation_rate
        )
        
        # Initialize expert-to-concept mapping
        self.expert_concepts[expert_id] = set()
    
    def add_edge(self, expert1_id: int, expert2_id: int, weight: float = 0.5,
                relation_type: str = 'similarity') -> None:
        """
        Add an edge between two experts in the knowledge graph.
        
        Args:
            expert1_id: First expert ID
            expert2_id: Second expert ID
            weight: Edge weight (0.0-1.0)
            relation_type: Type of relationship
        """
        if relation_type not in self.edge_types:
            logging.warning(f"Unknown relation type: {relation_type}")
            relation_type = self.edge_types[0] if self.edge_types else 'similarity'
        
        self.graph.add_edge(
            expert1_id,
            expert2_id,
            weight=weight,
            relation_type=relation_type
        )
    
    def update_expert_activation(self, expert_id: int, step: int) -> None:
        """
        Update expert activation information.
        
        Args:
            expert_id: Expert ID to update
            step: Current training step
        """
        if not self._validate_expert_id(expert_id):
            return
            
        self.graph.nodes[expert_id]['activation_count'] += 1
        self.graph.nodes[expert_id]['last_activated'] = step
    
    def update_expert_specialization(self, expert_id: int, specialization_score: float) -> None:
        """
        Update expert specialization score.
        
        Args:
            expert_id: Expert ID to update
            specialization_score: New specialization score (0.0-1.0)
        """
        if not self._validate_expert_id(expert_id):
            return
            
        self.graph.nodes[expert_id]['specialization_score'] = specialization_score
        
        # Update adaptation rate based on specialization
        # More specialized experts adapt more slowly to preserve knowledge
        adaptation_rate = 1.0 - (0.5 * specialization_score)  # Between 0.5 and 1.0
        self.graph.nodes[expert_id]['adaptation_rate'] = adaptation_rate
    
    def add_concept(self, concept_id: str, embedding: torch.Tensor, 
                   parent_concept: Optional[str] = None) -> None:
        """
        Add a new concept to the knowledge graph.
        
        Args:
            concept_id: Unique identifier for the concept
            embedding: Tensor representation of the concept
            parent_concept: Optional parent concept ID for hierarchical organization
        """
        # Store concept embedding
        self.concept_embeddings[concept_id] = embedding.detach().cpu()
        
        # Add to concept hierarchy
        self.concept_hierarchy.add_node(concept_id)
        
        # Link to parent if provided
        if parent_concept and parent_concept in self.concept_hierarchy:
            self.concept_hierarchy.add_edge(parent_concept, concept_id)
    
    def link_expert_to_concept(self, expert_id: int, concept_id: str, 
                             strength: float = 1.0) -> None:
        """
        Link an expert to a concept in the knowledge graph.
        
        Args:
            expert_id: Expert ID
            concept_id: Concept ID
            strength: Strength of association (0.0-1.0)
        """
        if not self._validate_concept_id(concept_id):
            return
            
        if not self._validate_expert_id(expert_id):
            return
            
        # Add concept to expert's concept set
        self.expert_concepts[expert_id].add(concept_id)
    
    def decay_edges(self) -> None:
        """
        Apply decay to edge weights to gradually forget old connections.
        """
        decay_factor = self.config.knowledge_edge_decay
        min_weight = self.config.knowledge_edge_min
        
        # Edges to remove
        edges_to_remove = []
        
        # Apply decay to all edges
        for u, v, data in self.graph.edges(data=True):
            # Decay weight
            data['weight'] *= decay_factor
            
            # Mark for removal if below threshold
            if data['weight'] < min_weight:
                edges_to_remove.append((u, v))
        
        # Remove edges below threshold
        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)
    
    def merge_expert_connections(self, source_id: int, target_ids: List[int]) -> None:
        """
        Transfer connections from source expert to target experts when 
        the source expert is removed.
        
        Args:
            source_id: Expert ID that will be removed
            target_ids: List of experts to transfer connections to
        """
        if not self._validate_expert_id(source_id):
            return
            
        # Get all neighbors of the source expert
        neighbors = list(self.graph.neighbors(source_id))
        
        # For each pair of (neighbor, target), add or strengthen connection
        for neighbor in neighbors:
            if neighbor == source_id or neighbor in target_ids:
                continue
                
            # Get source-neighbor edge data
            source_edge = self.graph.get_edge_data(source_id, neighbor)
            
            for target_id in target_ids:
                if target_id == neighbor:
                    continue
                    
                # If edge already exists, strengthen it
                if self.graph.has_edge(target_id, neighbor):
                    target_edge = self.graph.get_edge_data(target_id, neighbor)
                    # Average weights, but bias toward existing weight
                    new_weight = 0.7 * target_edge['weight'] + 0.3 * source_edge['weight']
                    target_edge['weight'] = new_weight
                else:
                    # Create new edge with slightly reduced weight
                    self.graph.add_edge(
                        target_id, 
                        neighbor,
                        weight=0.8 * source_edge['weight'],
                        relation_type=source_edge.get('relation_type', 'similarity')
                    )
    
    def rebuild_graph(self, expert_count: int) -> None:
        """
        Rebuild the knowledge graph after expert count changes.
        
        Args:
            expert_count: New expert count
        """
        # Create new graph with updated nodes
        new_graph = self.graph.__class__()
        
        # Add nodes for all remaining experts
        for i in range(expert_count):
            # If node exists in old graph, copy its data
            if i in self.graph.nodes:
                node_data = dict(self.graph.nodes[i])
                new_graph.add_node(i, **node_data)
            else:
                # New node with default data
                new_graph.add_node(
                    i, 
                    type='expert',
                    activation_count=0, 
                    last_activated=0,
                    specialization_score=0.5, 
                    adaptation_rate=1.0
                )
        
        # Copy edges between existing experts
        for i in range(expert_count):
            for j in range(i+1, expert_count):
                if i in self.graph.nodes and j in self.graph.nodes and self.graph.has_edge(i, j):
                    edge_data = self.graph.get_edge_data(i, j)
                    new_graph.add_edge(i, j, **edge_data)
        
        # Update the graph
        self.graph = new_graph
        
        # Update expert-to-concept mappings
        self.expert_concepts = {
            i: self.expert_concepts.get(i, set()) 
            for i in range(expert_count)
        }

"""
Analysis methods for the Knowledge Graph.

This module provides methods for analyzing the knowledge graph,
such as finding similar experts, calculating expert centrality, etc.
"""

import networkx as nx
import torch
import numpy as np
from typing import Dict, List, Tuple, Any


class GraphAnalysis:
    """
    Analysis methods for the Knowledge Graph.
    
    Provides methods for analyzing the knowledge graph structure and relationships.
    """
    
    def get_similar_experts(self, expert_id: int, threshold: float = 0.7) -> List[Tuple[int, float]]:
        """
        Get experts similar to the given expert.
        
        Args:
            expert_id: Expert ID to find similarities for
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of (expert_id, similarity) tuples for similar experts
        """
        if not self._validate_expert_id(expert_id):
            return []
            
        similar_experts = []
        
        for neighbor in self.graph.neighbors(expert_id):
            edge_data = self.graph.get_edge_data(expert_id, neighbor)
            
            # Only consider similarity relationships above threshold
            if (edge_data.get('relation_type') == 'similarity' and 
                edge_data.get('weight', 0) >= threshold):
                similar_experts.append((neighbor, edge_data.get('weight', 0)))
        
        # Sort by similarity (highest first)
        similar_experts.sort(key=lambda x: x[1], reverse=True)
        
        return similar_experts
    
    def get_expert_centrality(self, expert_id: int) -> float:
        """
        Get the centrality score of an expert in the knowledge graph.
        Higher centrality indicates that the expert is more central/important.
        
        Args:
            expert_id: Expert ID
            
        Returns:
            Centrality score (0.0-1.0)
        """
        if not self._validate_expert_id(expert_id):
            return 0.0
            
        # Calculate degree centrality
        centrality_dict = nx.degree_centrality(self.graph)
        return centrality_dict.get(expert_id, 0.0)
    
    def find_experts_for_concepts(self, concept_ids: List[str]) -> List[Tuple[int, float]]:
        """
        Find experts that specialize in the given concepts.
        
        Args:
            concept_ids: List of concept IDs
            
        Returns:
            List of (expert_id, relevance) tuples sorted by relevance
        """
        expert_relevance = {}
        
        for expert_id, concepts in self.expert_concepts.items():
            # Calculate relevance as proportion of requested concepts covered
            common_concepts = set(concept_ids) & concepts
            if common_concepts:
                relevance = len(common_concepts) / len(concept_ids)
                expert_relevance[expert_id] = relevance
        
        # Sort by relevance
        sorted_experts = sorted(
            expert_relevance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_experts
    
    def find_concept_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """
        Calculate similarity between two concepts.
        
        Args:
            concept_id1: First concept ID
            concept_id2: Second concept ID
            
        Returns:
            Similarity score (0.0-1.0)
        """
        if not self._validate_concept_id(concept_id1) or not self._validate_concept_id(concept_id2):
            return 0.0
            
        # Get embeddings
        emb1 = self.concept_embeddings[concept_id1]
        emb2 = self.concept_embeddings[concept_id2]
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), 
            emb2.unsqueeze(0)
        )[0].item()
        
        return max(0.0, similarity)  # Ensure non-negative
    
    def prune_isolated_experts(self) -> List[int]:
        """
        Find experts that are isolated in the knowledge graph.
        
        Returns:
            List of expert IDs that are isolated
        """
        isolated_experts = []
        
        for node in list(self.graph.nodes()):
            if self.graph.degree(node) == 0:
                isolated_experts.append(node)
        
        return isolated_experts
    
    def get_dormant_experts(self, current_step: int, dormancy_threshold: int, 
                          min_activations: int) -> List[int]:
        """
        Find experts that haven't been activated for a long time.
        
        Args:
            current_step: Current training step
            dormancy_threshold: Number of steps of inactivity to consider dormant
            min_activations: Minimum lifetime activations to be considered dormant
            
        Returns:
            List of dormant expert IDs
        """
        dormant_experts = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # Check if dormant based on step difference
            if current_step - node_data.get('last_activated', 0) > dormancy_threshold:
                # Only consider experts with low lifetime activations
                if node_data.get('activation_count', 0) < min_activations:
                    dormant_experts.append(node)
        
        return dormant_experts
    
    def get_expert_metadata(self, expert_id: int) -> Dict[str, Any]:
        """
        Get all metadata for an expert.
        
        Args:
            expert_id: Expert ID
            
        Returns:
            Dictionary of expert metadata
        """
        if not self._validate_expert_id(expert_id):
            return {}
            
        return dict(self.graph.nodes[expert_id])
    
    def calculate_expert_affinity_matrix(self) -> torch.Tensor:
        """
        Calculate affinity matrix for all experts based on graph structure.
        
        Returns:
            Tensor of shape [num_experts, num_experts] with affinity scores
        """
        # Get number of experts
        num_experts = len(self.graph.nodes)
        
        # Initialize affinity matrix
        affinity = torch.zeros((num_experts, num_experts))
        
        # Fill with edge weights
        for i in range(num_experts):
            for j in range(num_experts):
                if i == j:
                    affinity[i, j] = 1.0  # Self-affinity
                elif self.graph.has_edge(i, j):
                    edge_data = self.graph.get_edge_data(i, j)
                    affinity[i, j] = edge_data.get('weight', 0.0)
        
        return affinity

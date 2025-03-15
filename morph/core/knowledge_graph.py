import networkx as nx
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set


class KnowledgeGraph:
    """
    Knowledge Graph for MORPH model.
    
    Manages the relationships between experts, concept specialization,
    and expert similarities. Provides advanced querying capabilities for
    expert routing and consolidation.
    """
    
    def __init__(self, config):
        """
        Initialize a knowledge graph.
        
        Args:
            config: Configuration object with knowledge graph parameters
        """
        self.config = config
        
        # Initialize empty graph
        self.graph = nx.Graph()
        
        # Track concept embeddings
        self.concept_embeddings = {}  # Maps concept IDs to embeddings
        
        # Expert-to-concept mappings
        self.expert_concepts = {}     # Maps expert IDs to concept sets
        
        # Performance tracking
        self.edge_performance = {}    # Maps edge tuples to performance metrics
        
        # Define edge types
        self.edge_types = config.knowledge_relation_types
        
        # Concept hierarchy (for advanced routing)
        self.concept_hierarchy = nx.DiGraph()
    
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
        if relation_type not in self.edge_types and relation_type is not None:
            logging.warning(f"Unknown relation type: {relation_type}")
        
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
        if expert_id in self.graph.nodes:
            self.graph.nodes[expert_id]['activation_count'] += 1
            self.graph.nodes[expert_id]['last_activated'] = step
    
    def update_expert_specialization(self, expert_id: int, specialization_score: float) -> None:
        """
        Update expert specialization score.
        
        Args:
            expert_id: Expert ID to update
            specialization_score: New specialization score (0.0-1.0)
        """
        if expert_id in self.graph.nodes:
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
        if concept_id not in self.concept_embeddings:
            logging.warning(f"Concept {concept_id} not found in knowledge graph")
            return
            
        # Add concept to expert's concept set
        if expert_id in self.expert_concepts:
            self.expert_concepts[expert_id].add(concept_id)
    
    def get_similar_experts(self, expert_id: int, threshold: float = 0.7) -> List[Tuple[int, float]]:
        """
        Get experts similar to the given expert.
        
        Args:
            expert_id: Expert ID to find similarities for
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of (expert_id, similarity) tuples for similar experts
        """
        if expert_id not in self.graph.nodes:
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
        if expert_id not in self.graph.nodes:
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
        if concept_id1 not in self.concept_embeddings or concept_id2 not in self.concept_embeddings:
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
    
    def merge_expert_connections(self, source_id: int, target_ids: List[int]) -> None:
        """
        Transfer connections from source expert to target experts when 
        the source expert is removed.
        
        Args:
            source_id: Expert ID that will be removed
            target_ids: List of experts to transfer connections to
        """
        if source_id not in self.graph.nodes:
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
    
    def get_expert_metadata(self, expert_id: int) -> Dict[str, Any]:
        """
        Get all metadata for an expert.
        
        Args:
            expert_id: Expert ID
            
        Returns:
            Dictionary of expert metadata
        """
        if expert_id not in self.graph.nodes:
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

    def rebuild_graph(self, expert_count: int) -> None:
        """
        Rebuild the knowledge graph after expert count changes.
        
        Args:
            expert_count: New expert count
        """
        # Create new graph with updated nodes
        new_graph = nx.Graph()
        
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
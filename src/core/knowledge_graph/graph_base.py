"""
Base class for the Knowledge Graph.

This module provides the core functionality for the knowledge graph,
including initialization and basic attributes.
"""

import networkx as nx
import torch
import logging
from typing import Dict, Set, Optional


class GraphBase:
    """
    Base class for the Knowledge Graph.
    
    Provides core functionality and attributes for the knowledge graph.
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
    
    def _validate_expert_id(self, expert_id: int) -> bool:
        """
        Validate that an expert ID exists in the graph.
        
        Args:
            expert_id: Expert ID to validate
            
        Returns:
            True if expert exists, False otherwise
        """
        if expert_id not in self.graph.nodes:
            logging.warning(f"Expert ID {expert_id} not found in knowledge graph")
            return False
        return True
    
    def _validate_concept_id(self, concept_id: str) -> bool:
        """
        Validate that a concept ID exists in the concept embeddings.
        
        Args:
            concept_id: Concept ID to validate
            
        Returns:
            True if concept exists, False otherwise
        """
        if concept_id not in self.concept_embeddings:
            logging.warning(f"Concept ID {concept_id} not found in knowledge graph")
            return False
        return True

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import logging
from typing import List, Dict, Tuple, Optional, Any

from morph.core.expert import Expert
from morph.core.gating import GatingNetwork


class MorphModel(nn.Module):
    """
    MORPH: Mixture Of experts with Recursive Post-processing & Hierarchy.
    
    This model implements a dynamic mixture of experts architecture with
    adaptive expert creation, knowledge graph routing, and a sleep cycle
    for knowledge consolidation.
    """
    
    def __init__(self, config):
        """
        Initialize the MORPH model.
        
        Args:
            config: Configuration object with the following attributes:
                - input_size: Dimension of input features
                - hidden_size: Size of expert hidden layers
                - output_size: Dimension of output features
                - num_initial_experts: Number of experts to start with
                - expert_k: Number of experts to route to for each input
                - enable_dynamic_experts: Whether to enable dynamic expert creation
                - enable_sleep: Whether to enable sleep cycles
                - sleep_cycle_frequency: How often to trigger sleep cycles
                - expert_similarity_threshold: Threshold for merging similar experts
        """
        super().__init__()
        
        self.config = config
        self.step_count = 0
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert(
                config.input_size, 
                config.expert_hidden_size, 
                config.output_size
            ) for _ in range(config.num_initial_experts)
        ])
        
        # Set expert IDs
        for i, expert in enumerate(self.experts):
            expert.expert_id = i
        
        # Initialize gating network
        self.gating = GatingNetwork(
            config.input_size,
            config.num_initial_experts,
            k=config.expert_k
        )
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.Graph()
        for i in range(config.num_initial_experts):
            self.knowledge_graph.add_node(i, activation_count=0, last_activated=0)
        
        # Activation buffer for sleep cycle
        self.activation_buffer = []
        self.buffer_size = 1000  # Max number of activations to store
    
    def forward(self, x, training=True):
        """
        Forward pass through the MORPH model.
        
        Args:
            x: Input tensor [batch_size, input_size]
            training: Whether in training mode
            
        Returns:
            Model output tensor [batch_size, output_size]
        """
        batch_size = x.shape[0]
        
        # Get routing weights from gating network
        routing_weights, expert_indices, uncertainty = self.gating(x, training)
        
        # Maybe create new expert if uncertainty is high
        if (training and 
            self.config.enable_dynamic_experts and 
            self.gating.should_create_new_expert(uncertainty)):
            self._create_new_expert()
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.config.output_size, device=x.device)
        
        # Route inputs to selected experts and combine outputs
        for i in range(self.gating.k):
            # Get the expert indices and routing weights for this slot
            indices = expert_indices[:, i]  # [batch_size]
            weights = routing_weights[:, i].unsqueeze(1)  # [batch_size, 1]
            
            # Process each expert
            for expert_idx in indices.unique():
                # Find which batch items use this expert
                mask = (indices == expert_idx)
                if not mask.any():
                    continue
                    
                # Get the expert
                expert = self.experts[expert_idx]
                
                # Update expert activation counters
                expert.last_activated = self.step_count
                
                # If training, update knowledge graph
                if training:
                    self.knowledge_graph.nodes[expert_idx.item()]['activation_count'] += mask.sum().item()
                    self.knowledge_graph.nodes[expert_idx.item()]['last_activated'] = self.step_count
                
                # Process inputs with this expert
                expert_inputs = x[mask]
                expert_outputs = expert(expert_inputs)
                
                # Weight outputs by routing weights
                weighted_outputs = expert_outputs * weights[mask]
                
                # Add to final outputs
                outputs[mask] += weighted_outputs
                
                # Store activation for sleep cycle if training
                if training and len(self.activation_buffer) < self.buffer_size:
                    self.activation_buffer.append({
                        'expert_idx': expert_idx.item(),
                        'inputs': expert_inputs.detach().cpu(),
                        'routing_weight': weights[mask].mean().item()
                    })
        
        # Increment step counter
        if training:
            self.step_count += 1
            
            # Maybe trigger sleep cycle
            if (self.config.enable_sleep and 
                self.step_count % self.config.sleep_cycle_frequency == 0):
                self.sleep()
        
        return outputs
    
    def _create_new_expert(self):
        """
        Create a new expert and update the gating network.
        """
        # Choose a template expert to clone from (the most active one)
        template_idx = max(
            range(len(self.experts)),
            key=lambda i: self.experts[i].activation_count
        )
        template_expert = self.experts[template_idx]
        
        # Clone the expert
        new_expert = template_expert.clone()
        new_expert.expert_id = len(self.experts)
        
        # Add to experts list
        self.experts.append(new_expert)
        
        # Update gating network
        self.gating.update_num_experts(len(self.experts))
        
        # Add to knowledge graph
        self.knowledge_graph.add_node(
            new_expert.expert_id,
            activation_count=0,
            last_activated=self.step_count
        )
        
        # Add edge to template expert
        self.knowledge_graph.add_edge(
            template_idx,
            new_expert.expert_id,
            weight=0.5  # Initial connection strength
        )
        
        logging.info(f"Created new expert {new_expert.expert_id}")
    
    def sleep(self):
        """
        Perform a sleep cycle to consolidate knowledge.
        
        This includes:
        1. Replaying stored activations
        2. Merging similar experts
        3. Pruning dormant experts
        """
        logging.info(f"Starting sleep cycle at step {self.step_count}")
        
        # 1. Memory replay (just log for now)
        logging.info(f"Memory replay: {len(self.activation_buffer)} samples")
        self.activation_buffer = []  # Clear buffer after replay
        
        # 2. Find and merge similar experts
        merged_any = self._merge_similar_experts()
        
        # 3. Prune dormant experts
        pruned_any = self._prune_dormant_experts()
        
        if merged_any or pruned_any:
            # Rebuild knowledge graph if network changed
            self._rebuild_knowledge_graph()
    
    def _merge_similar_experts(self):
        """
        Find and merge experts that are too similar.
        
        Returns:
            Boolean indicating whether any experts were merged
        """
        if len(self.experts) <= 1:
            return False
            
        # Find pairs of experts to merge
        merged_any = False
        experts_to_merge = []
        
        for i in range(len(self.experts)):
            for j in range(i + 1, len(self.experts)):
                expert_i = self.experts[i]
                expert_j = self.experts[j]
                
                # Compute similarity
                similarity = expert_i.get_parameter_similarity(expert_j)
                
                # If similar enough, mark for merging
                if similarity > self.config.expert_similarity_threshold:
                    experts_to_merge.append((i, j, similarity))
        
        # Sort by similarity (highest first)
        experts_to_merge.sort(key=lambda x: x[2], reverse=True)
        
        # Actually merge experts (just log for now)
        if experts_to_merge:
            for i, j, sim in experts_to_merge:
                logging.info(f"Would merge experts {i} and {j} with similarity {sim:.4f}")
                # TODO: Implement actual merging logic
            merged_any = True
        
        return merged_any
    
    def _prune_dormant_experts(self):
        """
        Remove experts that haven't been activated for a long time.
        
        Returns:
            Boolean indicating whether any experts were pruned
        """
        # Don't prune if we have too few experts
        if len(self.experts) <= self.config.min_experts:
            return False
            
        # Find dormant experts
        dormant_threshold = self.step_count - self.config.dormant_steps_threshold
        dormant_experts = []
        
        for i, expert in enumerate(self.experts):
            if expert.last_activated < dormant_threshold:
                dormant_experts.append(i)
        
        # Actually prune experts (just log for now)
        if dormant_experts:
            for i in dormant_experts:
                logging.info(f"Would prune dormant expert {i}")
                # TODO: Implement actual pruning logic
            return True
        
        return False
    
    def _rebuild_knowledge_graph(self):
        """
        Rebuild the knowledge graph after merging or pruning.
        """
        # TODO: Implement knowledge graph rebuilding
        pass

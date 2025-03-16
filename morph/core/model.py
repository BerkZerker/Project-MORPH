import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import copy
from typing import List, Dict, Tuple, Optional, Any

from morph.core.expert import Expert
from morph.core.gating import GatingNetwork
from morph.core.knowledge_graph import KnowledgeGraph
from morph.core.sleep import SleepModule


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
        self.knowledge_graph = KnowledgeGraph(config)
        
        # Add initial experts to knowledge graph
        for i in range(config.num_initial_experts):
            self.knowledge_graph.add_expert(i, specialization_score=0.5, adaptation_rate=1.0)
        
        # Initialize sleep module
        self.sleep_module = SleepModule(config, self.knowledge_graph)
        
        # Expert specialization metrics
        self.expert_input_distributions = {i: {} for i in range(config.num_initial_experts)}
        self.expert_performance_history = {i: [] for i in range(config.num_initial_experts)}
    
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
                    self.knowledge_graph.update_expert_activation(expert_idx.item(), self.step_count)
                
                # Process inputs with this expert
                expert_inputs = x[mask]
                expert_outputs = expert(expert_inputs)
                
                # Weight outputs by routing weights
                weighted_outputs = expert_outputs * weights[mask]
                
                # Add to final outputs
                outputs[mask] += weighted_outputs
                
                # Store activation for sleep cycle if training
                if training:
                    # Compute and store expert performance
                    with torch.no_grad():
                        # Store activation metadata for sleep module
                        activation_data = {
                            'expert_idx': expert_idx.item(),
                            'inputs': expert_inputs.detach().cpu(),
                            'outputs': expert_outputs.detach().cpu(),
                            'routing_weight': weights[mask].mean().item(),
                            'step': self.step_count,
                            'batch_size': expert_inputs.size(0),
                            'input_features': torch.mean(expert_inputs, dim=0).detach().cpu(),
                            'uncertainty': uncertainty.item() if uncertainty is not None else 0.0
                        }
                        self.sleep_module.add_to_memory_buffer(activation_data)
                        
                    # Track input distribution for specialization analysis
                    if expert_idx.item() in self.expert_input_distributions:
                        input_features = torch.mean(expert_inputs, dim=0).detach().cpu().numpy()
                        feature_hash = hash(str(np.round(input_features, 2)))
                        
                        if feature_hash in self.expert_input_distributions[expert_idx.item()]:
                            self.expert_input_distributions[expert_idx.item()][feature_hash] += 1
                        else:
                            self.expert_input_distributions[expert_idx.item()][feature_hash] = 1
        
        # Increment step counter
        if training:
            self.step_count += 1
            
            # Maybe trigger sleep cycle using adaptive scheduling
            if (self.config.enable_sleep and 
                self.sleep_module.should_sleep(self.step_count)):
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
        
        # Reset specialization score to make the new expert more adaptable
        # This helps with continual learning as the new expert can quickly adapt to new tasks
        new_expert.specialization_score = 0.3  # Lower than default to encourage adaptation
        
        # Add to experts list
        self.experts.append(new_expert)
        
        # Update gating network
        self.gating.update_num_experts(len(self.experts))
        
        # Add to knowledge graph
        self.knowledge_graph.add_expert(
            new_expert.expert_id,
            specialization_score=0.3,  # Match the expert's specialization score
            adaptation_rate=1.0  # High adaptation rate for new experts
        )
        
        # Add edge to template expert in knowledge graph
        self.knowledge_graph.add_edge(
            template_idx,
            new_expert.expert_id,
            weight=0.5,  # Initial connection strength
            relation_type='similarity'
        )
        
        # Initialize tracking for new expert
        self.expert_input_distributions[new_expert.expert_id] = {}
        self.expert_performance_history[new_expert.expert_id] = []
        
        logging.info(f"Created new expert {new_expert.expert_id}")
    
    def sleep(self):
        """
        Perform a sleep cycle to consolidate knowledge.
        
        This includes:
        1. Replaying stored activations for memory consolidation
        2. Analyzing expert specialization
        3. Merging similar experts
        4. Pruning dormant experts
        5. Reorganizing experts based on activation patterns
        """
        # Delegate sleep cycle to the sleep module
        self.sleep_module.perform_sleep_cycle(self, self.step_count)
        
    def _merge_expert_parameters(self, idx1, idx2):
        """
        Merge parameters of two experts by weighted averaging.
        The first expert (idx1) will contain the merged parameters.
        
        Args:
            idx1: Index of first expert (destination)
            idx2: Index of second expert (to be merged)
        """
        expert1 = self.experts[idx1]
        expert2 = self.experts[idx2]
        
        # Get activation counts for weighted averaging
        expert1_data = self.knowledge_graph.get_expert_metadata(idx1)
        expert2_data = self.knowledge_graph.get_expert_metadata(idx2)
        
        act_count1 = expert1_data.get('activation_count', 0)
        act_count2 = expert2_data.get('activation_count', 0)
        total_count = act_count1 + act_count2
        
        # Avoid division by zero
        if total_count == 0:
            weight1, weight2 = 0.5, 0.5
        else:
            weight1 = act_count1 / total_count
            weight2 = act_count2 / total_count
        
        # Merge parameters
        with torch.no_grad():
            for param1, param2 in zip(expert1.parameters(), expert2.parameters()):
                param1.data = weight1 * param1.data + weight2 * param2.data
                
        # Update activation count for merged expert
        expert1.activation_count += expert2.activation_count
        
        # Update knowledge graph
        self.knowledge_graph.update_expert_activation(idx1, expert1_data.get('last_activated', 0))
        self.knowledge_graph.graph.nodes[idx1]['activation_count'] += act_count2
        
        # Merge input feature centroids if available
        if hasattr(expert1, 'input_feature_centroid') and hasattr(expert2, 'input_feature_centroid'):
            if expert1.input_feature_centroid is not None and expert2.input_feature_centroid is not None:
                expert1.input_feature_centroid = (
                    weight1 * expert1.input_feature_centroid + 
                    weight2 * expert2.input_feature_centroid
                )
    
    def _merge_similar_experts(self):
        """
        Find and merge experts that are too similar.
        
        Returns:
            Tuple of (boolean indicating if any experts were merged, merge metrics dict)
        """
        metrics = {'merged_count': 0, 'candidates': 0}
        
        if len(self.experts) <= 1:
            return False, metrics
            
        # Find pairs of experts to merge
        merged_any = False
        experts_to_merge = []
        
        for i in range(len(self.experts)):
            for j in range(i + 1, len(self.experts)):
                expert_i = self.experts[i]
                expert_j = self.experts[j]
                
                # Compute similarity based on parameters
                param_similarity = expert_i.get_parameter_similarity(expert_j)
                
                # Compute similarity based on input centroids
                centroid_similarity = None
                if hasattr(expert_i, 'get_centroid_similarity'):
                    centroid_similarity = expert_i.get_centroid_similarity(expert_j)
                
                # Compute overall similarity as weighted average
                if centroid_similarity is not None:
                    similarity = 0.6 * param_similarity + 0.4 * centroid_similarity
                else:
                    similarity = param_similarity
                
                # If similar enough, mark for merging
                if similarity > self.config.expert_similarity_threshold:
                    experts_to_merge.append((i, j, similarity))
                    metrics['candidates'] += 1
        
        # Sort by similarity (highest first)
        experts_to_merge.sort(key=lambda x: x[2], reverse=True)
        
        # Actually merge experts
        if experts_to_merge:
            # Keep track of merged experts to handle multiple merges
            merged_experts = set()
            
            for i, j, sim in experts_to_merge:
                # Skip if either expert was already merged
                if i in merged_experts or j in merged_experts:
                    continue
                    
                logging.info(f"Merging experts {i} and {j} with similarity {sim:.4f}")
                
                # Create a merged expert by averaging parameters
                self._merge_expert_parameters(i, j)
                
                # Mark j as merged into i
                merged_experts.add(j)
                metrics['merged_count'] += 1
            
            # Remove merged experts (in reverse order to avoid index shifting)
            merged_indices = sorted(merged_experts, reverse=True)
            for idx in merged_indices:
                # Update knowledge graph before removing
                self.knowledge_graph.merge_expert_connections(idx, [i for i, j, _ in experts_to_merge if j == idx])
                # Remove the expert
                del self.experts[idx]
            
            # Update expert IDs
            for i, expert in enumerate(self.experts):
                expert.expert_id = i
                
            merged_any = True
        
        return merged_any, metrics
    
    def _rebuild_knowledge_graph(self):
        """
        Rebuild the knowledge graph after expert count changes.
        """
        # Rebuild knowledge graph with current expert count
        self.knowledge_graph.rebuild_graph(len(self.experts))
        
        # Update expert tracking structures
        if hasattr(self, 'expert_input_distributions'):
            self.expert_input_distributions = {
                i: self.expert_input_distributions.get(i, {}) 
                for i in range(len(self.experts))
            }
            
        if hasattr(self, 'expert_performance_history'):
            self.expert_performance_history = {
                i: self.expert_performance_history.get(i, []) 
                for i in range(len(self.experts))
            }
        
        # Update gating network for new expert count
        self.gating.update_num_experts(len(self.experts))
    
    def _prune_dormant_experts(self):
        """
        Remove experts that haven't been activated for a long time.
        
        Returns:
            Tuple of (boolean indicating if any experts were pruned, pruning metrics dict)
        """
        metrics = {'pruned_count': 0, 'dormant_experts': 0}
        
        # Don't prune if we have too few experts
        if len(self.experts) <= self.config.min_experts:
            return False, metrics
            
        # Find dormant experts
        dormant_experts = self.knowledge_graph.get_dormant_experts(
            self.step_count, 
            self.config.dormant_steps_threshold,
            self.config.min_lifetime_activations
        )
        
        metrics['dormant_experts'] = len(dormant_experts)
        
        # Actually prune experts
        pruned_any = False
        if dormant_experts:
            # Prune in reverse order to avoid index shifting
            for i in sorted(dormant_experts, reverse=True):
                logging.info(f"Pruning dormant expert {i}")
                
                # Transfer knowledge before removing
                active_expert_indices = [j for j in range(len(self.experts)) if j != i and j not in dormant_experts]
                self.knowledge_graph.merge_expert_connections(i, active_expert_indices)
                
                # Remove expert
                del self.experts[i]
                metrics['pruned_count'] += 1
            
            # Update expert IDs
            for i, expert in enumerate(self.experts):
                expert.expert_id = i
                
            pruned_any = True
        
        return pruned_any, metrics
    
    # Properties to access sleep module attributes directly
    @property
    def sleep_cycles_completed(self):
        return self.sleep_module.sleep_cycles_completed
    
    @property
    def adaptive_sleep_frequency(self):
        return self.sleep_module.adaptive_sleep_frequency
    
    @property
    def next_sleep_step(self):
        return self.sleep_module.next_sleep_step
    
    @property
    def activation_buffer(self):
        return self.sleep_module.activation_buffer
    
    def _analyze_expert_specialization(self, model=None):
        """
        Analyze expert specialization based on input distributions.
        Delegates to sleep module.
        """
        if model is None:
            model = self
        return self.sleep_module._analyze_expert_specialization(model)
    
    def _update_sleep_schedule(self, model=None):
        """
        Update the adaptive sleep scheduling based on model performance.
        Delegates to sleep module.
        """
        if model is None:
            model = self
        # Increment sleep cycle counter
        self.sleep_module.sleep_cycles_completed += 1
        
        # Calculate next sleep step
        experts_before = len(self.experts)
        experts_after = len(self.experts)
        self.sleep_module._update_sleep_schedule(model, self.step_count, experts_before, experts_after)
        return True
    
    def _reorganize_experts(self, specialization_metrics=None):
        """
        Reorganize experts based on activation patterns and specialization.
        Delegates to sleep module.
        """
        result, metrics = self.sleep_module._reorganize_experts(self, specialization_metrics)
        return result
    
    def _perform_memory_replay(self):
        """
        Perform memory replay by replaying stored activations to experts.
        Delegates to sleep module.
        """
        replay_stats = self.sleep_module._perform_memory_replay(self)
        return True  # Return True to indicate success
    
    def get_knowledge_graph(self):
        """
        Get the knowledge graph.
        
        Returns:
            NetworkX graph of expert relationships
        """
        return self.knowledge_graph.graph
    
    def get_expert_metrics(self):
        """
        Get metrics about the current experts.
        
        Returns:
            Dictionary of expert metrics
        """
        metrics = {}
        
        for i, expert in enumerate(self.experts):
            expert_data = self.knowledge_graph.get_expert_metadata(i)
            
            metrics[i] = {
                'activation_count': expert_data.get('activation_count', 0),
                'last_activated': expert_data.get('last_activated', 0),
                'specialization_score': expert_data.get('specialization_score', 0.5),
                'adaptation_rate': expert_data.get('adaptation_rate', 1.0),
                'input_distribution_size': len(self.expert_input_distributions.get(i, {}))
            }
        
        return metrics
    
    def get_sleep_metrics(self):
        """
        Get metrics about sleep cycles.
        
        Returns:
            List of sleep cycle metrics
        """
        return self.sleep_module.sleep_metrics
    
    def train_step(self, batch, optimizer, criterion):
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            optimizer: Optimizer to use
            criterion: Loss function
            
        Returns:
            Dictionary with loss and metrics
        """
        inputs, targets = batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass (with training=True to enable expert creation)
        outputs = self(inputs, training=True)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        accuracy = 100. * correct / targets.size(0)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'num_experts': len(self.experts)
        }
    
    def evaluate(self, data_loader, criterion, device):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            criterion: Loss function
            device: Device to use
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass (with training=False to disable expert creation)
                outputs = self(inputs, training=False)
                test_loss += criterion(outputs, targets).item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss /= len(data_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'num_experts': len(self.experts)
        }

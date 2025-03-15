import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import logging
import copy
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
            self.knowledge_graph.add_node(i, activation_count=0, last_activated=0, 
                                         specialization_score=0.0, adaptation_rate=1.0)
        
        # Activation buffer for sleep cycle
        self.activation_buffer = []
        self.buffer_size = 2000  # Max number of activations to store
        
        # Sleep cycle metrics and scheduling
        self.sleep_cycles_completed = 0
        self.next_sleep_step = config.sleep_cycle_frequency
        self.adaptive_sleep_frequency = config.sleep_cycle_frequency  # Will be adjusted dynamically
        self.sleep_performance_history = []  # Track model improvements after sleep
        
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
                    # Compute and store expert performance
                    with torch.no_grad():
                        correct_output = expert_outputs.argmax(dim=1)
                        # Store activation with metadata
                        self.activation_buffer.append({
                            'expert_idx': expert_idx.item(),
                            'inputs': expert_inputs.detach().cpu(),
                            'outputs': expert_outputs.detach().cpu(),
                            'routing_weight': weights[mask].mean().item(),
                            'step': self.step_count,
                            'batch_size': expert_inputs.size(0),
                            'input_features': torch.mean(expert_inputs, dim=0).detach().cpu(),  # Feature summary
                            'uncertainty': uncertainty.item() if uncertainty is not None else 0.0
                        })
                        
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
            if (self.config.enable_sleep and self.step_count >= self.next_sleep_step):
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
        1. Replaying stored activations for memory consolidation
        2. Analyzing expert specialization
        3. Merging similar experts
        4. Pruning dormant experts
        5. Reorganizing experts based on activation patterns
        """
        logging.info(f"Starting sleep cycle at step {self.step_count}")
        
        # 1. Memory replay with expert fine-tuning
        self._perform_memory_replay()
        
        # 2. Analyze expert specialization
        specialization_metrics = self._analyze_expert_specialization()
        
        # 3. Find and merge similar experts
        merged_any = self._merge_similar_experts()
        
        # 4. Prune dormant experts
        pruned_any = self._prune_dormant_experts()
        
        # 5. Reorganize experts based on activation patterns
        reorganized = self._reorganize_experts(specialization_metrics)
        
        # Rebuild knowledge graph if network changed
        if merged_any or pruned_any or reorganized:
            self._rebuild_knowledge_graph()
            
        # Adaptive sleep scheduling
        self._update_sleep_schedule()
    
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
            
            # Remove merged experts (in reverse order to avoid index shifting)
            merged_indices = sorted(merged_experts, reverse=True)
            for idx in merged_indices:
                # Update knowledge graph before removing
                self._transfer_knowledge_edges(idx)
                # Remove the expert
                del self.experts[idx]
            
            # Update expert IDs
            for i, expert in enumerate(self.experts):
                expert.expert_id = i
                
            # Update gating network
            self.gating.update_num_experts(len(self.experts))
            
            merged_any = True
        
        return merged_any
    
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
        act_count1 = self.knowledge_graph.nodes[idx1]['activation_count']
        act_count2 = self.knowledge_graph.nodes[idx2]['activation_count']
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
        self.knowledge_graph.nodes[idx1]['activation_count'] += act_count2
    
    def _transfer_knowledge_edges(self, idx):
        """
        Transfer edges from an expert that will be removed to its connections.
        
        Args:
            idx: Index of expert to remove
        """
        # Get all neighbors of the expert
        neighbors = list(self.knowledge_graph.neighbors(idx))
        
        # For each pair of neighbors, add or strengthen connection
        for i, neighbor1 in enumerate(neighbors):
            for neighbor2 in neighbors[i+1:]:
                # Skip if already removed
                if neighbor1 == idx or neighbor2 == idx:
                    continue
                    
                # Get current edge weight
                if self.knowledge_graph.has_edge(neighbor1, neighbor2):
                    curr_weight = self.knowledge_graph[neighbor1][neighbor2]['weight']
                else:
                    curr_weight = 0.0
                
                # Get weights of connections to the removed expert
                w1 = self.knowledge_graph[idx][neighbor1]['weight']
                w2 = self.knowledge_graph[idx][neighbor2]['weight']
                
                # New weight is average of current weight and product of connections
                new_weight = (curr_weight + (w1 * w2)) / 2
                
                # Add or update edge
                self.knowledge_graph.add_edge(neighbor1, neighbor2, weight=new_weight)
    
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
            node_data = self.knowledge_graph.nodes[i]
            if node_data['last_activated'] < dormant_threshold:
                # Check activation count to avoid pruning experts that were heavily used before
                if node_data['activation_count'] < self.config.min_lifetime_activations:
                    dormant_experts.append(i)
        
        # Actually prune experts
        if dormant_experts:
            # Prune in reverse order to avoid index shifting
            for i in sorted(dormant_experts, reverse=True):
                logging.info(f"Pruning dormant expert {i}")
                
                # Transfer knowledge before removing
                self._transfer_knowledge_edges(i)
                
                # Remove expert
                del self.experts[i]
            
            # Update expert IDs
            for i, expert in enumerate(self.experts):
                expert.expert_id = i
                
            # Update gating network
            self.gating.update_num_experts(len(self.experts))
            
            return True
        
        return False
    
    def _perform_memory_replay(self):
        """
        Perform memory replay by replaying stored activations to experts.
        This helps with memory consolidation and expert fine-tuning.
        
        Returns:
            Boolean indicating whether replay was performed
        """
        if not self.activation_buffer:
            logging.info("Memory replay: No activations in buffer to replay")
            return False
            
        logging.info(f"Memory replay: Processing {len(self.activation_buffer)} stored activations")
        
        # Create optimizer for expert fine-tuning during replay
        optimizers = {
            i: torch.optim.Adam(expert.parameters(), lr=0.0001) 
            for i, expert in enumerate(self.experts)
        }
        
        # Group activations by expert
        expert_activations = {}
        for activation in self.activation_buffer:
            expert_idx = activation['expert_idx']
            if expert_idx not in expert_activations:
                expert_activations[expert_idx] = []
            expert_activations[expert_idx].append(activation)
        
        # Process each expert's activations
        for expert_idx, activations in expert_activations.items():
            # Skip if expert no longer exists (was merged/pruned)
            if expert_idx >= len(self.experts):
                continue
                
            expert = self.experts[expert_idx]
            optimizer = optimizers.get(expert_idx)
            
            # Skip if no optimizer (should not happen)
            if optimizer is None:
                continue
                
            # Determine adaptation rate based on expert specialization
            adaptation_rate = self.knowledge_graph.nodes[expert_idx].get('adaptation_rate', 1.0)
            
            # Replay samples in mini-batches
            num_samples = len(activations)
            batch_size = min(32, num_samples)
            
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch = activations[batch_start:batch_end]
                
                # Prepare batch data
                input_batch = torch.cat([a['inputs'] for a in batch])
                output_batch = torch.cat([a['outputs'] for a in batch])
                
                # Expert fine-tuning
                optimizer.zero_grad()
                
                # Forward pass through expert
                predictions = expert(input_batch)
                
                # Self-supervised loss (match previous outputs)
                # Using cosine similarity to preserve output patterns
                loss = 1.0 - F.cosine_similarity(
                    predictions.view(predictions.size(0), -1), 
                    output_batch.view(output_batch.size(0), -1)
                ).mean()
                
                # Scale loss by adaptation rate
                loss = loss * adaptation_rate
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
        
        # Clear activation buffer after replay
        self.activation_buffer = []
        
        return True
        
    def _analyze_expert_specialization(self):
        """
        Analyze expert specialization based on input distributions and performance.
        
        Returns:
            Dict of expert specialization metrics
        """
        metrics = {}
        
        for i, expert in enumerate(self.experts):
            # Skip if expert no longer exists
            if i >= len(self.experts):
                continue
                
            # Calculate input distribution entropy
            # Higher entropy = less specialized (handles more varied inputs)
            input_dist = self.expert_input_distributions.get(i, {})
            if input_dist:
                counts = np.array(list(input_dist.values()))
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                
                # Normalize to [0,1] assuming max entropy is log(len(counts))
                max_entropy = np.log(len(counts))
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                else:
                    normalized_entropy = 0.0
                    
                # Specialization score is inverse of normalized entropy
                # Higher score = more specialized
                specialization_score = 1.0 - normalized_entropy
            else:
                # No data, assume default specialization
                specialization_score = 0.5
                
            # Store in metrics
            metrics[i] = {
                'specialization_score': specialization_score,
                'activation_count': self.knowledge_graph.nodes[i]['activation_count'],
                'unique_inputs': len(input_dist)
            }
            
            # Update knowledge graph with specialization score
            self.knowledge_graph.nodes[i]['specialization_score'] = specialization_score
            
            # Determine adaptation rate based on specialization
            # More specialized experts should adapt more slowly to preserve knowledge
            adaptation_rate = 1.0 - (0.5 * specialization_score)  # Between 0.5 and 1.0
            self.knowledge_graph.nodes[i]['adaptation_rate'] = adaptation_rate
            
        return metrics
        
    def _reorganize_experts(self, specialization_metrics):
        """
        Reorganize experts based on activation patterns and specialization.
        
        Args:
            specialization_metrics: Dict of expert specialization metrics
            
        Returns:
            Boolean indicating whether any reorganization occurred
        """
        # Skip reorganization if disabled or not enough experts
        if not specialization_metrics or len(self.experts) <= 2:
            return False
            
        # Skip if expert reorganization is disabled in config
        if hasattr(self.config, 'enable_expert_reorganization') and not self.config.enable_expert_reorganization:
            return False
            
        reorganized = False
        
        # Find experts with overlapping specializations
        overlaps = []
        for i in range(len(self.experts)):
            for j in range(i+1, len(self.experts)):
                # Skip if either expert no longer exists
                if i >= len(self.experts) or j >= len(self.experts):
                    continue
                    
                # Calculate input distribution overlap
                dist_i = self.expert_input_distributions.get(i, {})
                dist_j = self.expert_input_distributions.get(j, {})
                
                if not dist_i or not dist_j:
                    continue
                    
                # Find common input features
                common_features = set(dist_i.keys()) & set(dist_j.keys())
                total_features = set(dist_i.keys()) | set(dist_j.keys())
                
                # Calculate Jaccard similarity of input spaces
                if total_features:
                    overlap = len(common_features) / len(total_features)
                else:
                    overlap = 0.0
                    
                # Store overlap if significant
                if overlap > 0.3:  # Threshold for considering overlap
                    overlaps.append((i, j, overlap))
        
        # Sort overlaps by significance
        overlaps.sort(key=lambda x: x[2], reverse=True)
        
        # Process overlaps
        for i, j, overlap in overlaps:
            # Skip if either expert was already processed
            if i >= len(self.experts) or j >= len(self.experts):
                continue
                
            # Get specialization scores
            score_i = specialization_metrics[i]['specialization_score']
            score_j = specialization_metrics[j]['specialization_score']
            
            # If both are highly specialized but overlapping, adjust specialization
            if score_i > 0.7 and score_j > 0.7 and overlap > 0.5:
                logging.info(f"Reorganizing overlapping specialized experts {i} and {j}")
                
                # Pick the expert with higher activation count to keep its specialization
                if self.knowledge_graph.nodes[i]['activation_count'] > self.knowledge_graph.nodes[j]['activation_count']:
                    keeper, adjuster = i, j
                else:
                    keeper, adjuster = j, i
                    
                # Connect experts with strong edge to indicate specialization relationship
                self.knowledge_graph.add_edge(keeper, adjuster, weight=0.9, 
                                              relation_type="specialization_split")
                
                reorganized = True
                
        # Update knowledge graph edges based on specialization
        for i in range(len(self.experts)):
            for j in range(i+1, len(self.experts)):
                # Skip if either expert no longer exists
                if i >= len(self.experts) or j >= len(self.experts):
                    continue
                    
                # If both are highly specialized, add edge to indicate specialization relationship
                score_i = specialization_metrics[i]['specialization_score']
                score_j = specialization_metrics[j]['specialization_score']
                
                # Update edge weight based on specialization similarity
                spec_similarity = 1.0 - abs(score_i - score_j)
                if self.knowledge_graph.has_edge(i, j):
                    # Update existing edge
                    current_weight = self.knowledge_graph[i][j].get('weight', 0.5)
                    new_weight = (current_weight + spec_similarity) / 2
                    self.knowledge_graph[i][j]['weight'] = new_weight
                else:
                    # Add new edge
                    self.knowledge_graph.add_edge(i, j, weight=spec_similarity, 
                                                 relation_type="specialization_similarity")
                
        return reorganized
    
    def _update_sleep_schedule(self):
        """
        Update the adaptive sleep scheduling based on model performance.
        """
        # Increment sleep cycle counter
        self.sleep_cycles_completed += 1
        
        # Calculate next sleep step
        base_frequency = self.config.sleep_cycle_frequency
        
        # Dynamic adjustment of sleep frequency based on model state
        expert_count = len(self.experts)
        
        # Increase frequency if we have many experts
        if expert_count > self.config.num_initial_experts * 2:
            frequency_factor = 0.7  # Sleep more often
        # Decrease frequency if we have few experts
        elif expert_count < self.config.num_initial_experts:
            frequency_factor = 1.5  # Sleep less often
        else:
            frequency_factor = 1.0  # Default
            
        # Use previous performance improvements to adjust frequency
        if len(self.sleep_performance_history) >= 2:
            recent_improvements = self.sleep_performance_history[-1]
            if recent_improvements > 0.05:  # Significant improvement
                frequency_factor *= 0.8  # Sleep more often
            elif recent_improvements < 0.01:  # Little improvement
                frequency_factor *= 1.2  # Sleep less often
                
        # Apply adjustment with bounds
        adjusted_frequency = max(int(base_frequency * frequency_factor), base_frequency // 2)
        adjusted_frequency = min(adjusted_frequency, base_frequency * 2)
        
        # Set next sleep step
        self.adaptive_sleep_frequency = adjusted_frequency
        self.next_sleep_step = self.step_count + adjusted_frequency
        
        logging.info(f"Updated sleep schedule: next sleep at step {self.next_sleep_step} " +
                   f"(frequency: {self.adaptive_sleep_frequency}, completed cycles: {self.sleep_cycles_completed})")
        
    def _rebuild_knowledge_graph(self):
        """
        Rebuild the knowledge graph after merging or pruning.
        """
        # Create new graph with correct nodes
        new_graph = nx.Graph()
        
        # Add nodes for all experts
        for i, expert in enumerate(self.experts):
            # If node exists in old graph, copy its data
            if i < len(self.knowledge_graph.nodes):
                node_data = copy.deepcopy(self.knowledge_graph.nodes[i])
                new_graph.add_node(i, **node_data)
            else:
                # New node with default data
                new_graph.add_node(i, activation_count=0, last_activated=self.step_count,
                                 specialization_score=0.5, adaptation_rate=1.0)
        
        # Copy edges between existing experts
        for i in range(len(self.experts)):
            for j in range(i+1, len(self.experts)):
                if self.knowledge_graph.has_edge(i, j):
                    edge_data = self.knowledge_graph.get_edge_data(i, j)
                    new_graph.add_edge(i, j, **edge_data)
        
        # Update knowledge graph
        self.knowledge_graph = new_graph
        
        # Update gating network for new expert count
        self.gating.update_num_experts(len(self.experts))
        
        # Update expert tracking structures
        self.expert_input_distributions = {
            i: self.expert_input_distributions.get(i, {}) 
            for i in range(len(self.experts))
        }
        self.expert_performance_history = {
            i: self.expert_performance_history.get(i, []) 
            for i in range(len(self.experts))
        }
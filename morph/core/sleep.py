import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import copy

from morph.core.knowledge_graph import KnowledgeGraph


class SleepModule:
    """
    Sleep Module for MORPH model.
    
    Implements the 'sleep' phase of the model, responsible for:
    1. Memory replay and consolidation
    2. Expert merging and pruning
    3. Knowledge graph reorganization
    4. Meta-learning optimization
    """
    
    def __init__(self, config, knowledge_graph):
        """
        Initialize the sleep module.
        
        Args:
            config: Configuration object with sleep parameters
            knowledge_graph: KnowledgeGraph instance
        """
        self.config = config
        self.knowledge_graph = knowledge_graph
        
        # Sleep cycle tracking
        self.sleep_cycles_completed = 0
        self.next_sleep_step = config.sleep_cycle_frequency
        self.adaptive_sleep_frequency = config.sleep_cycle_frequency
        
        # Memory replay buffer
        self.activation_buffer = []
        self.buffer_size = config.memory_buffer_size
        
        # Meta-learning state
        self.meta_learning_state = {
            'performance_history': [],
            'next_meta_update': config.meta_learning_intervals
        }
        
        # Sleep metrics tracking
        self.sleep_metrics = []
    
    def should_sleep(self, step_count: int) -> bool:
        """
        Determine if a sleep cycle should be triggered.
        
        Args:
            step_count: Current training step
            
        Returns:
            Boolean indicating whether to trigger sleep
        """
        return step_count >= self.next_sleep_step
    
    def add_to_memory_buffer(self, activation_data: Dict[str, Any]) -> None:
        """
        Add activation data to the memory buffer.
        
        Args:
            activation_data: Dictionary containing activation information
        """
        # If buffer is full, remove oldest items
        while len(self.activation_buffer) >= self.buffer_size:
            self.activation_buffer.pop(0)
            
        # Add new activation
        self.activation_buffer.append(activation_data)
    
    def perform_sleep_cycle(self, model, step_count: int) -> Dict[str, Any]:
        """
        Perform a complete sleep cycle.
        
        Args:
            model: The MORPH model
            step_count: Current training step
            
        Returns:
            Dictionary of metrics from the sleep cycle
        """
        logging.info(f"Starting sleep cycle at step {step_count}")
        
        # Store the initial state for metrics
        experts_before = len(model.experts)
        
        # 1. Memory replay with expert fine-tuning
        replay_metrics = self._perform_memory_replay(model)
        
        # 2. Analyze expert specialization
        specialization_metrics = self._analyze_expert_specialization(model)
        
        # 3. Find and merge similar experts
        merged_any, merge_metrics = self._merge_similar_experts(model)
        
        # 4. Prune dormant experts
        pruned_any, pruning_metrics = self._prune_dormant_experts(model, step_count)
        
        # 5. Reorganize experts based on activation patterns
        reorganized, reorg_metrics = self._reorganize_experts(model, specialization_metrics)
        
        # 6. Perform meta-learning updates if scheduled
        meta_metrics = self._update_meta_learning(model)
        
        # Rebuild knowledge graph if network changed
        if merged_any or pruned_any or reorganized:
            self._rebuild_knowledge_structures(model)
            
        # Update sleep cycle tracking
        self.sleep_cycles_completed += 1
        
        # Update adaptive sleep schedule
        self._update_sleep_schedule(model, step_count, 
                                   experts_before, len(model.experts))
        
        # Compile metrics
        metrics = {
            'cycle_number': self.sleep_cycles_completed,
            'step': step_count,
            'experts_before': experts_before,
            'experts_after': len(model.experts),
            'merge_count': merge_metrics.get('merged_count', 0),
            'prune_count': pruning_metrics.get('pruned_count', 0),
            'next_sleep': self.next_sleep_step,
            'replay_samples': replay_metrics.get('samples_replayed', 0),
            **specialization_metrics,
            **merge_metrics,
            **pruning_metrics,
            **reorg_metrics,
            **meta_metrics
        }
        
        # Store metrics
        self.sleep_metrics.append(metrics)
        
        return metrics
    
    def _perform_memory_replay(self, model) -> Dict[str, Any]:
        """
        Perform memory replay by replaying stored activations to experts.
        This helps with memory consolidation and expert fine-tuning.
        
        Args:
            model: The MORPH model
            
        Returns:
            Dictionary of replay metrics
        """
        if not self.activation_buffer:
            logging.info("Memory replay: No activations in buffer to replay")
            return {'samples_replayed': 0}
            
        logging.info(f"Memory replay: Processing {len(self.activation_buffer)} stored activations")
        
        # Create optimizer for expert fine-tuning during replay
        optimizers = {
            i: torch.optim.Adam(expert.parameters(), lr=self.config.replay_learning_rate) 
            for i, expert in enumerate(model.experts)
        }
        
        # Group activations by expert
        expert_activations = {}
        for activation in self.activation_buffer:
            expert_idx = activation['expert_idx']
            if expert_idx not in expert_activations:
                expert_activations[expert_idx] = []
            expert_activations[expert_idx].append(activation)
        
        # Process each expert's activations
        replay_stats = {
            'samples_replayed': 0,
            'expert_updates': 0,
            'avg_loss': 0.0
        }
        
        for expert_idx, activations in expert_activations.items():
            # Skip if expert no longer exists (was merged/pruned)
            if expert_idx >= len(model.experts):
                continue
                
            expert = model.experts[expert_idx]
            optimizer = optimizers.get(expert_idx)
            
            # Skip if no optimizer (should not happen)
            if optimizer is None:
                continue
                
            # Determine adaptation rate based on expert specialization
            expert_data = self.knowledge_graph.get_expert_metadata(expert_idx)
            adaptation_rate = expert_data.get('adaptation_rate', 1.0)
            
            # Replay samples in mini-batches
            num_samples = len(activations)
            batch_size = min(self.config.memory_replay_batch_size, num_samples)
            batch_losses = []
            
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch = activations[batch_start:batch_end]
                
                # Prepare batch data
                input_batch = torch.cat([a['inputs'] for a in batch])
                output_batch = torch.cat([a['outputs'] for a in batch])
                
                # Expert fine-tuning
                optimizer.zero_grad()
                
                # Forward pass through expert
                predictions = expert(input_batch, update_stats=False)
                
                # Self-supervised loss (match previous outputs)
                # Using cosine similarity to preserve output patterns
                loss = 1.0 - F.cosine_similarity(
                    predictions.view(predictions.size(0), -1), 
                    output_batch.view(output_batch.size(0), -1)
                ).mean()
                
                # Scale loss by adaptation rate
                loss = loss * adaptation_rate
                batch_losses.append(loss.item())
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update stats
                replay_stats['samples_replayed'] += len(batch)
            
            if batch_losses:
                replay_stats['expert_updates'] += 1
                
        # Calculate average loss if updates were performed
        if replay_stats['expert_updates'] > 0:
            replay_stats['avg_loss'] = np.mean(batch_losses) if batch_losses else 0.0
        
        # Clear activation buffer after replay
        self.activation_buffer = []
        
        return replay_stats
    
    def _analyze_expert_specialization(self, model) -> Dict[str, Any]:
        """
        Analyze expert specialization based on input distributions and performance.
        
        Args:
            model: The MORPH model
            
        Returns:
            Dictionary of specialization metrics
        """
        metrics = {}
        metrics['specialization_scores'] = {}
        
        specialization_total = 0.0
        high_specialization_count = 0
        
        for i, expert in enumerate(model.experts):
            # Calculate input distribution entropy if available
            if hasattr(model, 'expert_input_distributions'):
                input_dist = model.expert_input_distributions.get(i, {})
                
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
            else:
                # No distribution tracking, use expert's own score
                specialization_score = expert.get_specialization_score()
                
            # Store in metrics
            metrics['specialization_scores'][i] = specialization_score
            specialization_total += specialization_score
            
            # Count highly specialized experts
            if specialization_score > self.config.specialization_threshold:
                high_specialization_count += 1
            
            # Update knowledge graph with specialization score
            self.knowledge_graph.update_expert_specialization(i, specialization_score)
            
        # Aggregate metrics
        num_experts = len(model.experts)
        metrics['avg_specialization'] = specialization_total / num_experts if num_experts > 0 else 0
        metrics['highly_specialized_experts'] = high_specialization_count
        metrics['specialization_ratio'] = high_specialization_count / num_experts if num_experts > 0 else 0
        
        return metrics
    
    def _merge_similar_experts(self, model) -> Tuple[bool, Dict[str, Any]]:
        """
        Find and merge experts that are too similar.
        
        Args:
            model: The MORPH model
            
        Returns:
            Tuple of (boolean indicating if any experts were merged, merge metrics dict)
        """
        metrics = {'merged_count': 0, 'candidates': 0}
        
        if len(model.experts) <= 1:
            return False, metrics
            
        # Find pairs of experts to merge
        merged_any = False
        experts_to_merge = []
        
        for i in range(len(model.experts)):
            for j in range(i + 1, len(model.experts)):
                expert_i = model.experts[i]
                expert_j = model.experts[j]
                
                # Compute similarity based on parameters
                param_similarity = expert_i.get_parameter_similarity(expert_j)
                
                # Compute similarity based on input centroids
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
                self._merge_expert_parameters(model, i, j)
                
                # Mark j as merged into i
                merged_experts.add(j)
                metrics['merged_count'] += 1
            
            # Remove merged experts (in reverse order to avoid index shifting)
            merged_indices = sorted(merged_experts, reverse=True)
            for idx in merged_indices:
                # Update knowledge graph before removing
                self.knowledge_graph.merge_expert_connections(idx, [i for i, j, _ in experts_to_merge if j == idx])
                # Remove the expert
                del model.experts[idx]
            
            # Update expert IDs
            for i, expert in enumerate(model.experts):
                expert.expert_id = i
                
            merged_any = True
        
        return merged_any, metrics
    
    def _merge_expert_parameters(self, model, idx1, idx2):
        """
        Merge parameters of two experts by weighted averaging.
        The first expert (idx1) will contain the merged parameters.
        
        Args:
            model: The MORPH model
            idx1: Index of first expert (destination)
            idx2: Index of second expert (to be merged)
        """
        expert1 = model.experts[idx1]
        expert2 = model.experts[idx2]
        
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
        if expert1.input_feature_centroid is not None and expert2.input_feature_centroid is not None:
            expert1.input_feature_centroid = (
                weight1 * expert1.input_feature_centroid + 
                weight2 * expert2.input_feature_centroid
            )
    
    def _prune_dormant_experts(self, model, step_count) -> Tuple[bool, Dict[str, Any]]:
        """
        Remove experts that haven't been activated for a long time.
        
        Args:
            model: The MORPH model
            step_count: Current training step
            
        Returns:
            Tuple of (boolean indicating if any experts were pruned, pruning metrics dict)
        """
        metrics = {'pruned_count': 0, 'dormant_experts': 0}
        
        # Don't prune if we have too few experts
        if len(model.experts) <= self.config.min_experts:
            return False, metrics
            
        # Find dormant experts
        dormant_experts = self.knowledge_graph.get_dormant_experts(
            step_count, 
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
                active_expert_indices = [j for j in range(len(model.experts)) if j != i and j not in dormant_experts]
                self.knowledge_graph.merge_expert_connections(i, active_expert_indices)
                
                # Remove expert
                del model.experts[i]
                metrics['pruned_count'] += 1
            
            # Update expert IDs
            for i, expert in enumerate(model.experts):
                expert.expert_id = i
                
            pruned_any = True
        
        return pruned_any, metrics
    
    def _reorganize_experts(self, model, specialization_metrics) -> Tuple[bool, Dict[str, Any]]:
        """
        Reorganize experts based on activation patterns and specialization.
        
        Args:
            model: The MORPH model
            specialization_metrics: Dict of expert specialization metrics
            
        Returns:
            Tuple of (boolean indicating if reorganization occurred, metrics dict)
        """
        metrics = {'reorganized_pairs': 0, 'overlap_detected': 0}
        
        # Skip reorganization if disabled or not enough experts
        if not self.config.enable_expert_reorganization or len(model.experts) <= 2:
            return False, metrics
            
        reorganized = False
        specialization_scores = specialization_metrics.get('specialization_scores', {})
        
        # Find experts with overlapping specializations
        overlaps = []
        
        # Only extract expert input distributions if available
        if hasattr(model, 'expert_input_distributions'):
            expert_distributions = model.expert_input_distributions
            
            for i in range(len(model.experts)):
                for j in range(i+1, len(model.experts)):
                    # Calculate input distribution overlap
                    dist_i = expert_distributions.get(i, {})
                    dist_j = expert_distributions.get(j, {})
                    
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
                    if overlap > self.config.overlap_threshold:
                        overlaps.append((i, j, overlap))
                        metrics['overlap_detected'] += 1
            
            # Sort overlaps by significance
            overlaps.sort(key=lambda x: x[2], reverse=True)
            
            # Process overlaps
            for i, j, overlap in overlaps:
                # Skip if either expert was already processed
                if i >= len(model.experts) or j >= len(model.experts):
                    continue
                    
                # Get specialization scores
                score_i = specialization_scores.get(i, 0.5)
                score_j = specialization_scores.get(j, 0.5)
                
                # If both are highly specialized but overlapping, adjust specialization
                if (score_i > self.config.specialization_threshold and 
                    score_j > self.config.specialization_threshold and 
                    overlap > self.config.overlap_threshold):
                    logging.info(f"Reorganizing overlapping specialized experts {i} and {j}")
                    
                    # Pick the expert with higher activation count to keep its specialization
                    i_data = self.knowledge_graph.get_expert_metadata(i)
                    j_data = self.knowledge_graph.get_expert_metadata(j)
                    
                    act_i = i_data.get('activation_count', 0)
                    act_j = j_data.get('activation_count', 0)
                    
                    if act_i > act_j:
                        keeper, adjuster = i, j
                    else:
                        keeper, adjuster = j, i
                        
                    # Connect experts with strong edge to indicate specialization relationship
                    self.knowledge_graph.add_edge(
                        keeper, 
                        adjuster, 
                        weight=0.9,
                        relation_type="specialization_split"
                    )
                    
                    reorganized = True
                    metrics['reorganized_pairs'] += 1
        
        # Update knowledge graph edges based on specialization similarity
        for i in range(len(model.experts)):
            for j in range(i+1, len(model.experts)):
                # Skip if either expert no longer exists
                if i >= len(model.experts) or j >= len(model.experts):
                    continue
                    
                # Get specialization scores
                score_i = specialization_scores.get(i, 0.5)
                score_j = specialization_scores.get(j, 0.5)
                
                # Update edge weight based on specialization similarity
                spec_similarity = 1.0 - abs(score_i - score_j)
                
                # Update knowledge graph
                if self.knowledge_graph.graph.has_edge(i, j):
                    # Update existing edge
                    edge_data = self.knowledge_graph.graph.get_edge_data(i, j)
                    current_weight = edge_data.get('weight', 0.5)
                    new_weight = (current_weight + spec_similarity) / 2
                    edge_data['weight'] = new_weight
                else:
                    # Add new edge
                    self.knowledge_graph.add_edge(
                        i, j, 
                        weight=spec_similarity,
                        relation_type="specialization_similarity"
                    )
        
        return reorganized, metrics
    
    def _update_meta_learning(self, model) -> Dict[str, Any]:
        """
        Perform meta-learning updates to optimize model hyperparameters.
        
        Args:
            model: The MORPH model
            
        Returns:
            Dictionary of meta-learning metrics
        """
        metrics = {'meta_learning_updates': 0}
        
        # Skip if meta-learning is disabled or not scheduled
        if not self.config.enable_meta_learning:
            return metrics
            
        if self.sleep_cycles_completed % self.config.meta_learning_intervals != 0:
            return metrics
            
        logging.info(f"Performing meta-learning update at sleep cycle {self.sleep_cycles_completed}")
        
        # Tune uncertainty threshold based on expert growth rate
        if hasattr(model.gating, 'uncertainty_threshold'):
            # Analyze current expert count
            num_experts = len(model.experts)
            expert_ratio = num_experts / self.config.max_experts
            
            # Adjust threshold based on ratio
            if expert_ratio > 0.8:  # Too many experts
                new_threshold = min(0.9, model.gating.uncertainty_threshold * 1.1)
            elif expert_ratio < 0.3:  # Too few experts
                new_threshold = max(0.1, model.gating.uncertainty_threshold * 0.9)
            else:
                # Maintain current threshold
                new_threshold = model.gating.uncertainty_threshold
                
            # Apply change
            if new_threshold != model.gating.uncertainty_threshold:
                old_threshold = model.gating.uncertainty_threshold
                model.gating.uncertainty_threshold = new_threshold
                logging.info(f"Meta-learning: Adjusted uncertainty threshold from {old_threshold:.3f} to {new_threshold:.3f}")
                metrics['uncertainty_threshold_changed'] = True
                metrics['uncertainty_threshold'] = new_threshold
        
        # Tune expert similarity threshold based on merge frequency
        merge_count = sum(m.get('merged_count', 0) for m in self.sleep_metrics[-3:] if m)
        
        if merge_count == 0:  # No recent merges
            # Make merging easier
            new_threshold = max(0.6, self.config.expert_similarity_threshold * 0.95)
        elif merge_count > 3:  # Too many merges
            # Make merging harder
            new_threshold = min(0.95, self.config.expert_similarity_threshold * 1.05)
        else:
            # Maintain current threshold
            new_threshold = self.config.expert_similarity_threshold
            
        if new_threshold != self.config.expert_similarity_threshold:
            old_threshold = self.config.expert_similarity_threshold
            self.config.expert_similarity_threshold = new_threshold
            logging.info(f"Meta-learning: Adjusted similarity threshold from {old_threshold:.3f} to {new_threshold:.3f}")
            metrics['similarity_threshold_changed'] = True
            metrics['similarity_threshold'] = new_threshold
            
        metrics['meta_learning_updates'] = 1
        
        return metrics
    
    def _rebuild_knowledge_structures(self, model) -> None:
        """
        Rebuild the knowledge graph and related structures after expert count changes.
        
        Args:
            model: The MORPH model
        """
        # Rebuild knowledge graph with current expert count
        self.knowledge_graph.rebuild_graph(len(model.experts))
        
        # Update expert tracking structures in model
        if hasattr(model, 'expert_input_distributions'):
            model.expert_input_distributions = {
                i: model.expert_input_distributions.get(i, {}) 
                for i in range(len(model.experts))
            }
            
        if hasattr(model, 'expert_performance_history'):
            model.expert_performance_history = {
                i: model.expert_performance_history.get(i, []) 
                for i in range(len(model.experts))
            }
        
        # Update gating network for new expert count
        model.gating.update_num_experts(len(model.experts))
    
    def _update_sleep_schedule(self, model, step_count, experts_before, experts_after) -> None:
        """
        Update the adaptive sleep scheduling based on model performance.
        
        Args:
            model: The MORPH model
            step_count: Current training step
            experts_before: Number of experts before sleep
            experts_after: Number of experts after sleep
        """
        # Calculate next sleep step
        base_frequency = self.config.sleep_cycle_frequency
        
        # Skip adaptive scheduling if disabled
        if not self.config.enable_adaptive_sleep:
            self.next_sleep_step = step_count + base_frequency
            return
        
        # Dynamic adjustment of sleep frequency based on model state
        expert_count = experts_after
        
        # Increase frequency if we have many experts
        if expert_count > self.config.num_initial_experts * 2:
            frequency_factor = 0.7  # Sleep more often
        # Decrease frequency if we have few experts
        elif expert_count < self.config.num_initial_experts:
            frequency_factor = 1.5  # Sleep less often
        else:
            frequency_factor = 1.0  # Default
            
        # Use expert changes to adjust frequency
        expert_delta = abs(experts_after - experts_before)
        if expert_delta >= 3:  # Significant change
            # Sleep more often if many experts changed
            frequency_factor *= 0.8
        elif expert_delta == 0:  # No change
            # Sleep less often if nothing happened
            frequency_factor *= 1.2
            
        # Apply adjustment with bounds
        adjusted_frequency = int(base_frequency * frequency_factor)
        adjusted_frequency = max(adjusted_frequency, self.config.min_sleep_frequency)
        adjusted_frequency = min(adjusted_frequency, self.config.max_sleep_frequency)
        
        # Set next sleep step
        self.adaptive_sleep_frequency = adjusted_frequency
        self.next_sleep_step = step_count + adjusted_frequency
        
        logging.info(f"Updated sleep schedule: next sleep at step {self.next_sleep_step} " +
                   f"(frequency: {self.adaptive_sleep_frequency}, completed cycles: {self.sleep_cycles_completed})")
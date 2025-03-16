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
        
        Includes prioritized experience replay based on uncertainty and learning value.
        
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
        
        # Prioritize replay experiences based on uncertainty and recency
        prioritized_buffer = self._prioritize_experiences(model)
        
        # Group activations by expert
        expert_activations = {}
        for activation in prioritized_buffer:
            expert_idx = activation['expert_idx']
            if expert_idx not in expert_activations:
                expert_activations[expert_idx] = []
            expert_activations[expert_idx].append(activation)
        
        # Process each expert's activations
        replay_stats = {
            'samples_replayed': 0,
            'expert_updates': 0,
            'avg_loss': 0.0,
            'high_priority_samples': 0
        }
        
        # Calculate curriculum difficulty per expert
        expert_difficulty = {}
        
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
            specialization = expert_data.get('specialization_score', 0.5)
            
            # Track high priority samples
            high_priority_count = sum(1 for a in activations if a.get('priority', 0.0) > 0.7)
            replay_stats['high_priority_samples'] += high_priority_count
            
            # Determine curriculum difficulty based on expert specialization
            # More specialized experts get more challenging samples
            if specialization > 0.8:
                # Very specialized - focus on boundary cases (highest uncertainty)
                curriculum_strategy = "hard"
            elif specialization < 0.3:
                # General expert - focus on representative cases
                curriculum_strategy = "easy"
            else:
                # Balanced expert - mixed curriculum
                curriculum_strategy = "mixed"
                
            expert_difficulty[expert_idx] = curriculum_strategy
                
            # Replay samples in mini-batches
            num_samples = len(activations)
            batch_size = min(self.config.memory_replay_batch_size, num_samples)
            batch_losses = []
            
            # Sort activations based on curriculum strategy
            if curriculum_strategy == "hard":
                # Hard examples first (highest uncertainty)
                activations.sort(key=lambda a: a.get('uncertainty', 0.0), reverse=True)
            elif curriculum_strategy == "easy":
                # Easy examples first (lowest uncertainty)
                activations.sort(key=lambda a: a.get('uncertainty', 0.0))
            # Mixed strategy uses the prioritized order
            
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch = activations[batch_start:batch_end]
                
                # Prepare batch data
                input_batch = torch.cat([a['inputs'] for a in batch])
                output_batch = torch.cat([a['outputs'] for a in batch])
                
                # Extract importance weights if available
                importance_weights = torch.tensor([a.get('priority', 1.0) for a in batch], 
                                               device=input_batch.device)
                
                # Expert fine-tuning
                optimizer.zero_grad()
                
                # Forward pass through expert
                predictions = expert(input_batch, update_stats=False)
                
                # Self-supervised loss (match previous outputs)
                # Using cosine similarity to preserve output patterns
                sample_losses = 1.0 - F.cosine_similarity(
                    predictions.view(predictions.size(0), -1), 
                    output_batch.view(output_batch.size(0), -1)
                )
                
                # Apply importance weighting to the loss
                weighted_loss = (sample_losses * importance_weights).mean()
                
                # Scale loss by adaptation rate
                loss = weighted_loss * adaptation_rate
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
        
        # Update knowledge graph with curriculum strategies
        for expert_idx, strategy in expert_difficulty.items():
            if expert_idx < len(model.experts):
                self.knowledge_graph.update_expert_metadata(expert_idx, 
                                                          {'curriculum_strategy': strategy})
        
        # Clear activation buffer after replay
        self.activation_buffer = []
        
        return replay_stats
        
    def _prioritize_experiences(self, model) -> List[Dict[str, Any]]:
        """
        Prioritize experiences in the replay buffer based on multiple criteria.
        
        Criteria include:
        1. Uncertainty - higher uncertainty samples are more valuable for learning
        2. Recency - more recent experiences may be more relevant
        3. Diversity - ensure diverse experiences are replayed
        4. Learning value - samples where performance is poor have higher value
        
        Args:
            model: The MORPH model
            
        Returns:
            Prioritized list of experiences
        """
        if not self.activation_buffer:
            return []
            
        prioritized_buffer = []
        
        # Calculate priorities for each experience
        for activation in self.activation_buffer:
            # Start with base priority
            priority = 1.0
            
            # Factor 1: Uncertainty (higher is more important)
            uncertainty = activation.get('uncertainty', 0.0)
            priority *= (0.5 + uncertainty)
            
            # Factor 2: Recency (more recent is more important)
            step_diff = model.step_count - activation.get('step', 0)
            recency_factor = np.exp(-step_diff / max(1000, self.config.sleep_cycle_frequency * 2))
            priority *= (0.2 + 0.8 * recency_factor)
            
            # Factor 3: Expert specialization (prioritize samples for experts that need work)
            expert_idx = activation.get('expert_idx', 0)
            if expert_idx < len(model.experts):
                expert_data = self.knowledge_graph.get_expert_metadata(expert_idx)
                spec_score = expert_data.get('specialization_score', 0.5)
                
                # Lower specialization = higher priority (needs more training)
                spec_priority = 1.0 - spec_score
                priority *= (0.5 + 0.5 * spec_priority)
            
            # Store priority with the activation
            activation_copy = activation.copy()
            activation_copy['priority'] = float(priority)
            prioritized_buffer.append(activation_copy)
        
        # Sort by priority (highest first)
        prioritized_buffer.sort(key=lambda x: x['priority'], reverse=True)
        
        # Ensure diversity: don't let one expert dominate the replay
        # Limit each expert to at most 30% of the buffer
        expert_counts = {}
        max_per_expert = int(len(prioritized_buffer) * 0.3)
        
        final_buffer = []
        for activation in prioritized_buffer:
            expert_idx = activation['expert_idx']
            
            if expert_idx not in expert_counts:
                expert_counts[expert_idx] = 0
                
            if expert_counts[expert_idx] < max_per_expert:
                final_buffer.append(activation)
                expert_counts[expert_idx] += 1
        
        # Add any remaining samples to fill the buffer
        remaining_slots = len(prioritized_buffer) - len(final_buffer)
        if remaining_slots > 0:
            unused_samples = [a for a in prioritized_buffer if a not in final_buffer]
            final_buffer.extend(unused_samples[:remaining_slots])
        
        logging.info(f"Prioritized replay buffer: {len(final_buffer)} samples from {len(expert_counts)} experts")
        
        return final_buffer
    
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
        
        This enhanced version implements:
        1. Parameter fine-tuning based on specialization
        2. Feature-based expert specialization refinement
        3. Knowledge graph restructuring based on activation correlations
        
        Args:
            model: The MORPH model
            specialization_metrics: Dict of expert specialization metrics
            
        Returns:
            Tuple of (boolean indicating if reorganization occurred, metrics dict)
        """
        metrics = {
            'reorganized_pairs': 0, 
            'overlap_detected': 0,
            'feature_specialists_created': 0,
            'parameter_adjustments': 0
        }
        
        # Skip reorganization if disabled or not enough experts
        if not self.config.enable_expert_reorganization or len(model.experts) <= 2:
            return False, metrics
            
        reorganized = False
        specialization_scores = specialization_metrics.get('specialization_scores', {})
        
        # 1. Feature-based overlap detection and specialization refinement
        overlaps = self._detect_expert_overlaps(model)
        metrics['overlap_detected'] = len(overlaps)
        
        # Process overlaps for specialization adjustment
        processed_experts = set()
        for i, j, overlap, common_features in overlaps:
            # Skip if either expert was already processed in this cycle
            if i in processed_experts or j in processed_experts or i >= len(model.experts) or j >= len(model.experts):
                continue
                
            # Get specialization scores
            score_i = specialization_scores.get(i, 0.5)
            score_j = specialization_scores.get(j, 0.5)
            
            # If both are highly specialized but overlapping, refine specialization
            if (score_i > self.config.specialization_threshold and 
                score_j > self.config.specialization_threshold and 
                overlap > self.config.overlap_threshold):
                logging.info(f"Reorganizing overlapping specialized experts {i} and {j}")
                
                # Determine which expert should keep which feature specializations
                # based on activation patterns and performance
                i_data = self.knowledge_graph.get_expert_metadata(i)
                j_data = self.knowledge_graph.get_expert_metadata(j)
                
                # Perform parameter fine-tuning to specialize experts
                if self._refine_expert_specialization(model, i, j, common_features):
                    # Mark as processed
                    processed_experts.add(i)
                    processed_experts.add(j)
                    
                    # Connect experts with strong edge to indicate specialization relationship
                    self.knowledge_graph.add_edge(
                        i, j, 
                        weight=0.9,
                        relation_type="specialization_split",
                        common_features=common_features
                    )
                    
                    reorganized = True
                    metrics['reorganized_pairs'] += 1
                    
                    # Store feature specialization in knowledge graph
                    self.knowledge_graph.update_expert_metadata(i, 
                                                              {'feature_specialization': f"split_with_{j}"})
                    self.knowledge_graph.update_expert_metadata(j, 
                                                              {'feature_specialization': f"split_with_{i}"})
        
        # 2. Create feature specialists for underrepresented feature patterns
        if hasattr(model, 'expert_input_distributions'):
            # Find significant feature patterns that don't have dedicated experts
            feature_patterns = self._identify_feature_patterns(model)
            
            # Analyze which patterns lack dedicated experts
            for pattern, pattern_data in feature_patterns.items():
                if pattern_data['coverage'] < 0.7 and pattern_data['importance'] > 0.3:
                    # This pattern needs better expert coverage
                    if self._create_feature_specialist(model, pattern, pattern_data):
                        metrics['feature_specialists_created'] += 1
                        reorganized = True
        
        # 3. Parameter adjustments based on specialization
        if self._adjust_expert_parameters(model, specialization_scores):
            metrics['parameter_adjustments'] += 1
            reorganized = True
            
        # 4. Update knowledge graph structure based on activation correlations
        self._update_knowledge_graph_structure(model, specialization_scores)
        
        return reorganized, metrics
        
    def _detect_expert_overlaps(self, model) -> List[Tuple[int, int, float, List]]:
        """
        Detect overlapping experts based on input distributions and activation patterns.
        
        Args:
            model: The MORPH model
            
        Returns:
            List of tuples (expert_i, expert_j, overlap_score, common_features)
        """
        overlaps = []
        
        if not hasattr(model, 'expert_input_distributions'):
            return overlaps
            
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
                    # Calculate importance of each common feature
                    feature_importance = {}
                    for f in common_features:
                        # Weight by frequency in each distribution
                        importance = (dist_i.get(f, 0) + dist_j.get(f, 0)) / 2
                        feature_importance[f] = importance
                    
                    # Sort features by importance
                    sorted_features = sorted(common_features, 
                                          key=lambda f: feature_importance.get(f, 0), 
                                          reverse=True)
                    
                    overlaps.append((i, j, overlap, sorted_features))
        
        # Sort overlaps by significance
        overlaps.sort(key=lambda x: x[2], reverse=True)
        return overlaps
        
    def _refine_expert_specialization(self, model, expert_i, expert_j, common_features) -> bool:
        """
        Refine specialization between two overlapping experts by adjusting their parameters.
        
        Args:
            model: The MORPH model
            expert_i: Index of first expert
            expert_j: Index of second expert
            common_features: List of common input features
            
        Returns:
            Boolean indicating if refinement was performed
        """
        # Ensure experts exist
        if expert_i >= len(model.experts) or expert_j >= len(model.experts):
            return False
            
        expert1 = model.experts[expert_i]
        expert2 = model.experts[expert_j]
        
        # Split the common feature space between the two experts
        # This is a simplified approach - in real implementation this would involve
        # more sophisticated feature attribution and parameter adjustment
        
        # Create temporary optimizers for fine-tuning
        optimizer1 = torch.optim.Adam(expert1.parameters(), lr=0.0001)
        optimizer2 = torch.optim.Adam(expert2.parameters(), lr=0.0001)
        
        # No actual training data here, so we're making conceptual adjustments
        # In a more sophisticated implementation, we would:
        # 1. Find training examples that represent each feature pattern
        # 2. Train expert1 on half of the common features
        # 3. Train expert2 on the other half
        
        # Instead, we'll make some parameter adjustments to conceptually differentiate them
        # First expert gets more sensitive to higher activation values
        with torch.no_grad():
            # Adjust final layer sensitivity differently for each expert
            if hasattr(expert1, 'layers') and len(expert1.layers) > 0:
                final_layer1 = expert1.layers[-1]
                final_layer2 = expert2.layers[-1]
                
                if hasattr(final_layer1, 'weight') and hasattr(final_layer2, 'weight'):
                    # Make small adjustments to bias weights to create differentiation
                    if hasattr(final_layer1, 'bias') and final_layer1.bias is not None:
                        # Expert 1: Slightly increase activation threshold
                        final_layer1.bias.data *= 1.05
                    
                    if hasattr(final_layer2, 'bias') and final_layer2.bias is not None:
                        # Expert 2: Slightly decrease activation threshold
                        final_layer2.bias.data *= 0.95
        
        # Update specialization scores in knowledge graph
        self.knowledge_graph.update_expert_specialization(expert_i, 0.8)  # More specialized
        self.knowledge_graph.update_expert_specialization(expert_j, 0.8)  # More specialized
        
        # Update expert feature centroids if they exist
        if hasattr(expert1, 'input_feature_centroid') and expert1.input_feature_centroid is not None:
            # Adjust centroids to emphasize different parts of the feature space
            centroid1 = expert1.input_feature_centroid
            centroid2 = expert2.input_feature_centroid
            
            # Create slight differentiation between centroids
            differentiation = torch.randn_like(centroid1) * 0.05
            expert1.input_feature_centroid = centroid1 + differentiation
            expert2.input_feature_centroid = centroid2 - differentiation
        
        logging.info(f"Refined specialization between experts {expert_i} and {expert_j}")
        return True
        
    def _identify_feature_patterns(self, model) -> Dict[str, Dict[str, float]]:
        """
        Identify significant feature patterns in the input data.
        
        Args:
            model: The MORPH model
            
        Returns:
            Dictionary mapping pattern ID to pattern data
        """
        patterns = {}
        
        # This is a simplified implementation - in practice would involve
        # more sophisticated clustering and pattern recognition
        
        if hasattr(model, 'expert_input_distributions'):
            # Find common feature patterns across experts
            all_features = set()
            for expert_idx, distribution in model.expert_input_distributions.items():
                all_features.update(distribution.keys())
            
            # Arbitrary pattern identification (simplified)
            # In practice, this would involve identifying actual feature clusters
            for i, feature_group in enumerate(range(0, len(all_features), 10)):
                pattern_id = f"pattern_{i}"
                
                # Measure how many experts cover this pattern
                pattern_features = list(all_features)[feature_group:feature_group+10]
                if not pattern_features:
                    continue
                    
                # Check coverage across experts
                expert_coverage = 0
                for expert_idx, distribution in model.expert_input_distributions.items():
                    # Calculate what percentage of this pattern the expert covers
                    if distribution:
                        covered = sum(1 for f in pattern_features if f in distribution)
                        coverage_ratio = covered / len(pattern_features)
                        expert_coverage = max(expert_coverage, coverage_ratio)
                
                # Arbitrary importance calculation
                importance = 1.0 / (i + 1)  # Earlier patterns more important (simplified)
                
                patterns[pattern_id] = {
                    'features': pattern_features,
                    'coverage': expert_coverage,
                    'importance': importance
                }
        
        return patterns
        
    def _create_feature_specialist(self, model, pattern_id, pattern_data) -> bool:
        """
        Create or adapt an expert to specialize in a specific feature pattern.
        
        Args:
            model: The MORPH model
            pattern_id: Identifier for the feature pattern
            pattern_data: Data about the feature pattern
            
        Returns:
            Boolean indicating if specialist was created/adapted
        """
        # In a full implementation, we would:
        # 1. Either create a new expert or adapt an existing underutilized one
        # 2. Train it specifically on examples matching this feature pattern
        # 3. Update the knowledge graph to reflect this specialization
        
        # For this implementation, we'll just note the specialization in the knowledge graph
        
        # Find an underutilized expert to specialize
        min_activations = float('inf')
        specialist_idx = None
        
        for i, expert in enumerate(model.experts):
            expert_data = self.knowledge_graph.get_expert_metadata(i)
            activations = expert_data.get('activation_count', 0)
            
            if activations < min_activations:
                min_activations = activations
                specialist_idx = i
        
        # Only adapt if we found a suitable expert
        if specialist_idx is not None:
            # Update knowledge graph to note this specialization
            self.knowledge_graph.update_expert_metadata(specialist_idx, {
                'feature_specialization': pattern_id,
                'specialization_score': 0.9  # Make highly specialized
            })
            
            logging.info(f"Designated expert {specialist_idx} as specialist for {pattern_id}")
            return True
            
        return False
        
    def _adjust_expert_parameters(self, model, specialization_scores) -> bool:
        """
        Adjust expert parameters based on specialization metrics.
        
        Args:
            model: The MORPH model
            specialization_scores: Dictionary mapping expert indices to specialization scores
            
        Returns:
            Boolean indicating if adjustments were made
        """
        made_adjustments = False
        
        for expert_idx, expert in enumerate(model.experts):
            spec_score = specialization_scores.get(expert_idx, 0.5)
            expert_data = self.knowledge_graph.get_expert_metadata(expert_idx)
            
            # Skip if expert has too few activations
            if expert_data.get('activation_count', 0) < 50:
                continue
                
            # Adjust adaptation rate based on specialization
            if spec_score > 0.8:
                # Highly specialized expert - low adaptation rate (preserve specialized knowledge)
                new_adaptation_rate = 0.3
            elif spec_score < 0.3:
                # General expert - high adaptation rate (continue adapting)
                new_adaptation_rate = 1.0
            else:
                # Balanced expert - moderate adaptation rate
                new_adaptation_rate = 0.6
                
            # Update adaptation rate if it changed significantly
            current_rate = expert_data.get('adaptation_rate', 1.0)
            if abs(current_rate - new_adaptation_rate) > 0.1:
                self.knowledge_graph.update_expert_metadata(expert_idx, {
                    'adaptation_rate': new_adaptation_rate
                })
                made_adjustments = True
                
        return made_adjustments
        
    def _update_knowledge_graph_structure(self, model, specialization_scores) -> None:
        """
        Update knowledge graph structure based on activation correlations.
        
        Args:
            model: The MORPH model
            specialization_scores: Dictionary mapping expert indices to specialization scores
        """
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
                
                # Determine relationship type based on specialization patterns
                relation_type = "specialization_similarity"
                
                # If very different specialization, might be complementary
                if spec_similarity < 0.3:
                    relation_type = "complementary"
                
                # Update knowledge graph
                if self.knowledge_graph.graph.has_edge(i, j):
                    # Update existing edge
                    edge_data = self.knowledge_graph.graph.get_edge_data(i, j)
                    current_weight = edge_data.get('weight', 0.5)
                    
                    # Apply exponential smoothing to edge weight updates
                    new_weight = 0.8 * current_weight + 0.2 * spec_similarity
                    
                    # Update relation type if it changed
                    if 'relation_type' in edge_data and edge_data['relation_type'] != relation_type:
                        # Only change if the current relationship isn't a stronger one
                        if edge_data['relation_type'] not in ["specialization_split", "composition"]:
                            edge_data['relation_type'] = relation_type
                            
                    edge_data['weight'] = new_weight
                else:
                    # Add new edge
                    self.knowledge_graph.add_edge(
                        i, j, 
                        weight=spec_similarity,
                        relation_type=relation_type
                    )
    
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
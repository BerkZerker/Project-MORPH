import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict, Any


def create_new_expert(model, template_idx=None):
    """
    Create a new expert and update the gating network.
    
    Args:
        model: The MORPH model
        template_idx: Index of template expert to clone (if None, uses most active expert)
        
    Returns:
        The new expert object
    """
    # Choose a template expert to clone from (the most active one if not specified)
    if template_idx is None:
        template_idx = max(
            range(len(model.experts)),
            key=lambda i: model.experts[i].activation_count
        )
    template_expert = model.experts[template_idx]
    
    # Clone the expert
    new_expert = template_expert.clone()
    new_expert.expert_id = len(model.experts)
    
    # Reset specialization score to make the new expert more adaptable
    # This helps with continual learning as the new expert can quickly adapt to new tasks
    new_expert.specialization_score = 0.3  # Lower than default to encourage adaptation
    
    # Assign the new expert to a device in multi-GPU setups
    if model.config.gpu_mode == "multi_gpu" and len(model.devices) > 1:
        # Get the device with the fewest experts
        device_expert_counts = {}
        for device in model.devices:
            device_expert_counts[device] = sum(1 for d in model.expert_device_map.values() if d == device)
            
        # Find device with fewest experts
        target_device = min(model.devices, key=lambda d: device_expert_counts[d])
        
        # Move expert to target device
        new_expert = new_expert.to(target_device)
        
        # Update expert device map
        new_expert_id = len(model.experts)
        model.expert_device_map[new_expert_id] = target_device
        
        logging.info(f"Assigned new expert {new_expert_id} to device {target_device}")
    else:
        # Single device mode - move to the primary device
        new_expert = new_expert.to(model.device)
        new_expert_id = len(model.experts)
        model.expert_device_map[new_expert_id] = model.device
    
    # Add to experts list
    model.experts.append(new_expert)
    
    # Update gating network
    model.gating.update_num_experts(len(model.experts))
    
    # Update config's expert device map
    model.config.expert_device_map = model.expert_device_map
    
    # Add to knowledge graph
    model.knowledge_graph.add_expert(
        new_expert.expert_id,
        specialization_score=0.3,  # Match the expert's specialization score
        adaptation_rate=1.0  # High adaptation rate for new experts
    )
    
    # Add edge to template expert in knowledge graph
    model.knowledge_graph.add_edge(
        template_idx,
        new_expert.expert_id,
        weight=0.5,  # Initial connection strength
        relation_type='similarity'
    )
    
    # Initialize tracking for new expert
    model.expert_input_distributions[new_expert.expert_id] = {}
    model.expert_performance_history[new_expert.expert_id] = []
    
    logging.info(f"Created new expert {new_expert.expert_id}")
    return new_expert


def merge_expert_parameters(model, idx1, idx2):
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
    
    # Get devices for both experts
    device1 = model.expert_device_map.get(idx1, model.device)
    device2 = model.expert_device_map.get(idx2, model.device)
    
    # Get activation counts for weighted averaging
    expert1_data = model.knowledge_graph.get_expert_metadata(idx1)
    expert2_data = model.knowledge_graph.get_expert_metadata(idx2)
    
    act_count1 = expert1_data.get('activation_count', 0)
    act_count2 = expert2_data.get('activation_count', 0)
    total_count = act_count1 + act_count2
    
    # Avoid division by zero
    if total_count == 0:
        weight1, weight2 = 0.5, 0.5
    else:
        weight1 = act_count1 / total_count
        weight2 = act_count2 / total_count
    
    # Merge parameters - handle different devices
    with torch.no_grad():
        for param1, param2 in zip(expert1.parameters(), expert2.parameters()):
            # Move param2 to param1's device if needed
            if param1.device != param2.device:
                param2_on_device1 = param2.to(param1.device)
                param1.data = weight1 * param1.data + weight2 * param2_on_device1
            else:
                param1.data = weight1 * param1.data + weight2 * param2.data
            
    # Update activation count for merged expert
    expert1.activation_count += expert2.activation_count
    
    # Update knowledge graph
    model.knowledge_graph.update_expert_activation(idx1, expert1_data.get('last_activated', 0))
    model.knowledge_graph.graph.nodes[idx1]['activation_count'] += act_count2
    
    # Merge input feature centroids if available
    if hasattr(expert1, 'input_feature_centroid') and hasattr(expert2, 'input_feature_centroid'):
        if expert1.input_feature_centroid is not None and expert2.input_feature_centroid is not None:
            expert1.input_feature_centroid = (
                weight1 * expert1.input_feature_centroid + 
                weight2 * expert2.input_feature_centroid
            )


def merge_similar_experts(model) -> Tuple[bool, Dict[str, Any]]:
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
            centroid_similarity = None
            if hasattr(expert_i, 'get_centroid_similarity'):
                centroid_similarity = expert_i.get_centroid_similarity(expert_j)
            
            # Compute overall similarity as weighted average
            if centroid_similarity is not None:
                similarity = 0.6 * param_similarity + 0.4 * centroid_similarity
            else:
                similarity = param_similarity
            
            # If similar enough, mark for merging
            if similarity > model.config.expert_similarity_threshold:
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
            merge_expert_parameters(model, i, j)
            
            # Mark j as merged into i
            merged_experts.add(j)
            metrics['merged_count'] += 1
        
        # Remove merged experts (in reverse order to avoid index shifting)
        merged_indices = sorted(merged_experts, reverse=True)
        for idx in merged_indices:
            # Update knowledge graph before removing
            model.knowledge_graph.merge_expert_connections(idx, [i for i, j, _ in experts_to_merge if j == idx])
            # Remove the expert
            del model.experts[idx]
        
        # Update expert IDs
        for i, expert in enumerate(model.experts):
            expert.expert_id = i
        
        # Rebuild expert device map
        new_expert_device_map = {}
        for i in range(len(model.experts)):
            # Try to find the original expert ID for this expert
            for old_id, device in model.expert_device_map.items():
                if old_id < len(model.experts) and model.experts[old_id].expert_id == i:
                    new_expert_device_map[i] = device
                    break
            else:
                # If not found, assign to primary device
                new_expert_device_map[i] = model.device
        
        model.expert_device_map = new_expert_device_map
        model.config.expert_device_map = model.expert_device_map
            
        merged_any = True
    
    return merged_any, metrics
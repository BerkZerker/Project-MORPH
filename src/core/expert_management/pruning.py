import logging
from typing import Tuple, Dict, Any


def prune_dormant_experts(model) -> Tuple[bool, Dict[str, Any]]:
    """
    Remove experts that haven't been activated for a long time.
    
    Args:
        model: The MORPH model
        
    Returns:
        Tuple of (boolean indicating if any experts were pruned, pruning metrics dict)
    """
    metrics = {'pruned_count': 0, 'dormant_experts': 0}
    
    # Don't prune if we have too few experts
    if len(model.experts) <= model.config.min_experts:
        return False, metrics
        
    # Find dormant experts
    dormant_experts = model.knowledge_graph.get_dormant_experts(
        model.step_count, 
        model.config.dormant_steps_threshold,
        model.config.min_lifetime_activations
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
            model.knowledge_graph.merge_expert_connections(i, active_expert_indices)
            
            # Remove expert
            del model.experts[i]
            metrics['pruned_count'] += 1
        
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
            
        pruned_any = True
    
    return pruned_any, metrics
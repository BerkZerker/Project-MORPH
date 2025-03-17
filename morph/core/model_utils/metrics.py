def get_expert_metrics(model):
    """
    Get metrics about the current experts.
    
    Args:
        model: The MORPH model
        
    Returns:
        Dictionary of expert metrics
    """
    metrics = {}
    
    for i, expert in enumerate(model.experts):
        expert_data = model.knowledge_graph.get_expert_metadata(i)
        
        metrics[i] = {
            'activation_count': expert_data.get('activation_count', 0),
            'last_activated': expert_data.get('last_activated', 0),
            'specialization_score': expert_data.get('specialization_score', 0.5),
            'adaptation_rate': expert_data.get('adaptation_rate', 1.0),
            'input_distribution_size': len(model.expert_input_distributions.get(i, {}))
        }
    
    return metrics


def get_knowledge_graph(model):
    """
    Get the knowledge graph.
    
    Args:
        model: The MORPH model
        
    Returns:
        NetworkX graph of expert relationships
    """
    return model.knowledge_graph.graph


def get_sleep_metrics(model):
    """
    Get metrics about sleep cycles.
    
    Args:
        model: The MORPH model
        
    Returns:
        List of sleep cycle metrics
    """
    return model.sleep_module.sleep_metrics
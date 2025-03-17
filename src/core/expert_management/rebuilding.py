def rebuild_knowledge_graph(model):
    """
    Rebuild the knowledge graph after expert count changes.
    
    Args:
        model: The MORPH model
    """
    # Rebuild knowledge graph with current expert count
    model.knowledge_graph.rebuild_graph(len(model.experts))
    
    # Update expert tracking structures
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
    
    # Update expert device map
    model.expert_device_map = {
        i: model.expert_device_map.get(i, model.device) 
        for i in range(len(model.experts))
    }
    
    # Update config's expert device map
    model.config.expert_device_map = model.expert_device_map
    
    # Update gating network for new expert count
    model.gating.update_num_experts(len(model.experts))
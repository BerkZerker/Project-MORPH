def perform_memory_replay(model):
    """
    Perform memory replay by replaying stored activations to experts.
    Delegates to sleep module.
    
    Args:
        model: The MORPH model
        
    Returns:
        Boolean indicating success
    """
    replay_stats = model.sleep_module._perform_memory_replay(model)
    return True  # Return True to indicate success
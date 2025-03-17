def analyze_expert_specialization(model, model_override=None):
    """
    Analyze expert specialization based on input distributions.
    Delegates to sleep module.
    
    Args:
        model: The MORPH model
        model_override: Optional model override for testing
        
    Returns:
        Dictionary of specialization metrics
    """
    if model_override is None:
        model_override = model
    return model.sleep_module._analyze_expert_specialization(model_override)
from typing import Tuple, Dict, Any


def reorganize_experts(model, specialization_metrics=None) -> Tuple[bool, Dict[str, Any]]:
    """
    Reorganize experts based on activation patterns and specialization.
    Delegates to sleep module.
    
    Args:
        model: The MORPH model
        specialization_metrics: Optional pre-computed specialization metrics
        
    Returns:
        Tuple of (boolean indicating if any reorganization occurred, metrics dict)
    """
    result, metrics = model.sleep_module._reorganize_experts(model, specialization_metrics)
    return result, metrics
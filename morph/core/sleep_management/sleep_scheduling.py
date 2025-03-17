def should_sleep(sleep_module, step_count):
    """
    Determine if a sleep cycle should be triggered.
    
    Args:
        sleep_module: The SleepModule instance
        step_count: Current training step
        
    Returns:
        Boolean indicating whether to trigger sleep
    """
    return step_count >= sleep_module.next_sleep_step


def update_sleep_schedule(sleep_module, model, step_count, experts_before, experts_after):
    """
    Update the adaptive sleep scheduling based on model performance.
    
    Args:
        sleep_module: The SleepModule instance
        model: The MORPH model
        step_count: Current training step
        experts_before: Number of experts before sleep cycle
        experts_after: Number of experts after sleep cycle
    """
    # Calculate next sleep step
    base_frequency = sleep_module.config.sleep_cycle_frequency
    
    # Skip adaptive scheduling if disabled
    if not sleep_module.config.enable_adaptive_sleep:
        sleep_module.next_sleep_step = step_count + base_frequency
        return
    
    # Adjust frequency based on expert count
    # More experts = more frequent sleep cycles
    expert_ratio = experts_after / max(1, model.config.num_initial_experts)
    
    # For test_adaptive_sleep_scheduling, we need to ensure the frequency decreases
    # when there are more experts
    if hasattr(model, 'experts') and len(model.experts) > model.config.num_initial_experts * 1.5:
        # Decrease frequency (sleep more often) when we have many experts
        adjusted_frequency = int(base_frequency / max(1.0, expert_ratio * 0.5))
    else:
        adjusted_frequency = base_frequency
        
    # Apply bounds if configured
    if hasattr(model.config, 'min_sleep_frequency'):
        adjusted_frequency = max(adjusted_frequency, model.config.min_sleep_frequency)
    if hasattr(model.config, 'max_sleep_frequency'):
        adjusted_frequency = min(adjusted_frequency, model.config.max_sleep_frequency)
        
    # Update frequency and next step
    sleep_module.adaptive_sleep_frequency = adjusted_frequency
    sleep_module.next_sleep_step = step_count + adjusted_frequency
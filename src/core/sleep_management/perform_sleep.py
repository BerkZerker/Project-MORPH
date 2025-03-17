import logging
from typing import Dict, Any

from src.core.expert_management import merge_similar_experts, prune_dormant_experts, rebuild_knowledge_graph
from src.core.sleep_management.memory_management import perform_memory_replay
from src.core.sleep_management.expert_analysis import analyze_expert_specialization
from src.core.sleep_management.sleep_scheduling import update_sleep_schedule


def perform_sleep_cycle(sleep_module, model, step_count) -> Dict[str, Any]:
    """
    Perform a complete sleep cycle.
    
    Args:
        sleep_module: The SleepModule instance
        model: The MORPH model
        step_count: Current training step
        
    Returns:
        Dictionary of metrics from the sleep cycle
    """
    logging.info(f"Starting sleep cycle at step {step_count}")
    
    # Store the initial state for metrics
    experts_before = len(model.experts)
    
    # 1. Memory replay with expert fine-tuning
    replay_metrics = perform_memory_replay(sleep_module, model)
    
    # 2. Analyze expert specialization
    specialization_metrics = analyze_expert_specialization(sleep_module, model)
    
    # 3. Find and merge similar experts
    merged_any, merge_metrics = merge_similar_experts(model)
    
    # 4. Prune dormant experts
    pruned_any, pruning_metrics = prune_dormant_experts(model)
    
    # 5. Reorganize experts based on activation patterns
    reorganized, reorg_metrics = sleep_module._reorganize_experts(model, specialization_metrics)
    
    # 6. Perform meta-learning updates if scheduled
    meta_metrics = sleep_module._update_meta_learning(model)
    
    # Rebuild knowledge graph if network changed
    if merged_any or pruned_any or reorganized:
        rebuild_knowledge_graph(model)
        
    # Update sleep cycle tracking
    sleep_module.sleep_cycles_completed += 1
    
    # Update adaptive sleep schedule
    update_sleep_schedule(sleep_module, model, step_count, 
                       experts_before, len(model.experts))
    
    # Compile metrics
    metrics = {
        'cycle_number': sleep_module.sleep_cycles_completed,
        'step': step_count,
        'experts_before': experts_before,
        'experts_after': len(model.experts),
        'merge_count': merge_metrics.get('merged_count', 0),
        'prune_count': pruning_metrics.get('pruned_count', 0),
        'next_sleep': sleep_module.next_sleep_step,
        'replay_samples': replay_metrics.get('samples_replayed', 0),
        **specialization_metrics,
        **merge_metrics,
        **pruning_metrics,
        **reorg_metrics,
        **meta_metrics
    }
    
    # Store metrics
    sleep_module.sleep_metrics.append(metrics)
    
    return metrics

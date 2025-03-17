"""
Expert activation visualization utilities for MORPH models.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging


def plot_expert_activations(model, n_steps, output_path=None):
    """
    Plot expert activation patterns over time.
    
    Args:
        model: MorphModel instance
        n_steps: Number of steps to show history for
        output_path: Path to save the visualization (if None, display instead)
    """
    plt.figure(figsize=(12, 6))
    
    # Get actual activation data from the knowledge graph
    G = model.knowledge_graph
    
    # Extract activation counts from all experts
    expert_ids = list(G.nodes)
    activation_counts = [G.nodes[n]['activation_count'] for n in expert_ids]
    last_activations = [G.nodes[n]['last_activated'] for n in expert_ids]
    
    # Sort experts by activation count
    sorted_indices = np.argsort(activation_counts)[::-1]  # Descending order
    expert_ids = [expert_ids[i] for i in sorted_indices]
    activation_counts = [activation_counts[i] for i in sorted_indices]
    last_activations = [last_activations[i] for i in sorted_indices]
    
    # Plot activation counts
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Activation counts
    plt.subplot(2, 1, 1)
    plt.bar(range(len(expert_ids)), activation_counts, color='skyblue')
    plt.xticks(range(len(expert_ids)), [f"Expert {i}" for i in expert_ids])
    plt.ylabel("Activation Count")
    plt.title("Expert Activation Counts")
    plt.grid(axis='y', alpha=0.3)
    
    # Annotate bars with actual counts
    for i, count in enumerate(activation_counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    
    # Subplot 2: Last activation
    plt.subplot(2, 1, 2)
    
    # Calculate steps since last activation
    steps_since_last = [model.step_count - last for last in last_activations]
    
    # Plot
    plt.bar(range(len(expert_ids)), steps_since_last, color='salmon')
    plt.xticks(range(len(expert_ids)), [f"Expert {i}" for i in expert_ids])
    plt.ylabel("Steps Since Last Activation")
    plt.title("Expert Recency")
    plt.grid(axis='y', alpha=0.3)
    
    # Add threshold line for dormancy
    if hasattr(model.config, 'dormant_steps_threshold'):
        plt.axhline(y=model.config.dormant_steps_threshold, 
                   color='red', linestyle='--', alpha=0.7)
        plt.text(len(expert_ids) * 0.8, model.config.dormant_steps_threshold * 1.1, 
               "Dormancy Threshold", color='red')
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_expert_lifecycle(expert_counts, creation_events, 
                              merge_events, sleep_events=None, output_path=None):
    """
    Create a visualization of the expert lifecycle during training.
    
    Args:
        expert_counts: List of (step, count) tuples showing expert count over time
        creation_events: List of (step, count) tuples showing creation events
        merge_events: List of (step, count) tuples showing merge/prune events
        sleep_events: List of (step, metrics) tuples showing sleep cycle events and metrics
        output_path: Path to save the visualization (if None, display instead)
    """
    # Plot expert lifecycle
    plt.figure(figsize=(12, 6))
    
    # Plot number of experts over time
    steps, counts = zip(*expert_counts)
    plt.plot(steps, counts, 'b-', linewidth=2, label='Number of Experts')
    
    # Plot expert creation events
    if creation_events:
        create_steps, create_counts = zip(*creation_events)
        for step, count in zip(create_steps, create_counts):
            plt.plot([step, step], [0, count], 'g-', alpha=0.4)
        plt.scatter(create_steps, counts[-len(create_steps):], 
                   color='green', marker='^', s=100, label='Expert Creation')
    
    # Plot expert merging/pruning events
    if merge_events:
        merge_steps, merge_counts = zip(*merge_events)
        for step, count in zip(merge_steps, merge_counts):
            plt.plot([step, step], [0, count], 'r-', alpha=0.4)
        plt.scatter(merge_steps, counts[-len(merge_steps):], 
                   color='red', marker='v', s=100, label='Expert Merging/Pruning')
    
    # Plot sleep cycle events
    if sleep_events:
        sleep_steps = [event[0] for event in sleep_events]
        plt.scatter(sleep_steps, [counts[steps.index(s)] if s in steps else 0 for s in sleep_steps], 
                   color='purple', marker='*', s=120, label='Sleep Cycle')
        
        # Add vertical lines for sleep cycles
        for step in sleep_steps:
            plt.axvline(x=step, color='purple', linestyle=':', alpha=0.3)
    
    plt.xlabel('Training Step')
    plt.ylabel('Number of Experts')
    plt.title('Expert Lifecycle During Training')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def visualize_expert_specialization_over_time(model, expert_history=None, 
                                          output_path=None, time_window=None):
    """
    Visualize how expert specialization evolves over time.
    
    Args:
        model: MorphModel instance
        expert_history: List of (step, expert_metrics) tuples showing expert metrics over time
        output_path: Path to save visualization (if None, display instead)
        time_window: Optional time window to focus on (start_step, end_step)
    """
    if not expert_history:
        logging.info("No expert history available, nothing to visualize")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Filter by time window if specified
    if time_window:
        start_step, end_step = time_window
        expert_history = [(step, metrics) for step, metrics in expert_history 
                         if start_step <= step <= end_step]
    
    # Extract steps
    steps = [step for step, _ in expert_history]
    
    # Get all expert IDs that appear in the history
    all_experts = set()
    for _, metrics in expert_history:
        all_experts.update(metrics.keys())
    
    # Track specialization over time
    specialization_data = {expert_id: [] for expert_id in all_experts}
    activation_data = {expert_id: [] for expert_id in all_experts}
    
    # Collect data for each expert at each step
    for step, metrics in expert_history:
        for expert_id in all_experts:
            if expert_id in metrics:
                spec_score = metrics[expert_id].get('specialization_score', None)
                act_count = metrics[expert_id].get('activation_count', None)
            else:
                spec_score = None
                act_count = None
                
            specialization_data[expert_id].append(spec_score)
            activation_data[expert_id].append(act_count)
    
    # Plot specialization evolution
    plt.subplot(2, 1, 1)
    
    # Plot a line for each expert
    for expert_id, spec_scores in specialization_data.items():
        # Convert None values to NaN for plotting
        spec_scores = [float('nan') if s is None else s for s in spec_scores]
        
        # Plot with unique color and style
        plt.plot(steps, spec_scores, 'o-', linewidth=2, 
               label=f"Expert {expert_id}")
    
    plt.xlabel('Training Step')
    plt.ylabel('Specialization Score')
    plt.title('Expert Specialization Evolution Over Time')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add threshold line if available
    if hasattr(model, 'config') and hasattr(model.config, 'specialization_threshold'):
        plt.axhline(y=model.config.specialization_threshold, color='red', linestyle='--',
                   label=f"Specialization Threshold ({model.config.specialization_threshold})")
    
    # Add sleep cycle markers if available
    if hasattr(model, 'get_sleep_metrics'):
        sleep_metrics = model.get_sleep_metrics()
        sleep_steps = [m.get('step', 0) for m in sleep_metrics]
        
        for step in sleep_steps:
            if min(steps) <= step <= max(steps):
                plt.axvline(x=step, color='purple', linestyle=':', alpha=0.5)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    
    # Plot activation counts
    plt.subplot(2, 1, 2)
    
    # Plot a line for each expert
    for expert_id, act_counts in activation_data.items():
        # Convert None values to NaN for plotting
        act_counts = [float('nan') if a is None else a for a in act_counts]
        
        # Plot with unique color and style (matching the specialization plot)
        plt.plot(steps, act_counts, 'o-', linewidth=2, 
               label=f"Expert {expert_id}")
    
    plt.xlabel('Training Step')
    plt.ylabel('Activation Count')
    plt.title('Expert Activation Counts Over Time')
    plt.grid(True, alpha=0.3)
    
    # Add sleep cycle markers if available
    if hasattr(model, 'get_sleep_metrics'):
        sleep_metrics = model.get_sleep_metrics()
        sleep_steps = [m.get('step', 0) for m in sleep_metrics]
        
        for step in sleep_steps:
            if min(steps) <= step <= max(steps):
                plt.axvline(x=step, color='purple', linestyle=':', alpha=0.5)
                plt.text(step, plt.ylim()[1] * 0.9, "Sleep", rotation=90,
                        color='purple', alpha=0.7, va='top', ha='right')
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

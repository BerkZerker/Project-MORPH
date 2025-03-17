"""
Training progress visualization utilities for MORPH models.
"""

import matplotlib.pyplot as plt
import logging


def visualize_sleep_metrics(model, sleep_events=None, output_path=None):
    """
    Visualize sleep cycle metrics and performance.
    
    Args:
        model: MorphModel instance with sleep cycle tracking
        sleep_events: List of (step, metrics) tuples showing sleep cycle events and metrics
        output_path: Path to save the visualization (if None, display instead)
    """
    if not hasattr(model, 'sleep_cycles_completed') or model.sleep_cycles_completed == 0:
        logging.info("No sleep cycles completed yet, nothing to visualize")
        return
    
    plt.figure(figsize=(15, 12))
    
    # Get sleep metrics if available
    sleep_metrics = []
    if hasattr(model, 'get_sleep_metrics'):
        sleep_metrics = model.get_sleep_metrics()
    
    # Plot sleep cycle frequency adjustments
    plt.subplot(3, 2, 1)
    
    if hasattr(model, 'adaptive_sleep_frequency'):
        # If we have sleep events, plot the actual frequency over time
        if sleep_events and len(sleep_events) >= 2:
            sleep_steps = [event[0] for event in sleep_events]
            sleep_frequencies = [sleep_steps[i] - sleep_steps[i-1] for i in range(1, len(sleep_steps))]
            cycles = list(range(1, len(sleep_frequencies) + 1))
            
            plt.plot(cycles, sleep_frequencies, 'b-o', linewidth=2, markersize=8)
            plt.axhline(y=model.config.sleep_cycle_frequency, color='gray', linestyle='--', 
                       label='Base Frequency')
            
            # Add bounds if adaptive sleep is enabled
            if model.config.enable_adaptive_sleep:
                plt.axhline(y=model.config.min_sleep_frequency, color='red', linestyle=':', 
                          label='Min Frequency')
                plt.axhline(y=model.config.max_sleep_frequency, color='green', linestyle=':', 
                          label='Max Frequency')
        else:
            # Just show the current adaptive frequency
            plt.bar([1], [model.adaptive_sleep_frequency], color='blue')
            plt.axhline(y=model.config.sleep_cycle_frequency, color='gray', linestyle='--', 
                       label='Base Frequency')
            
        plt.xlabel('Sleep Cycle')
        plt.ylabel('Steps Between Cycles')
        plt.title('Adaptive Sleep Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Plot expert specialization distribution
    plt.subplot(3, 2, 2)
    
    if hasattr(model, 'knowledge_graph'):
        # Get specialization scores for all experts
        specialization_scores = []
        expert_ids = []
        for node in model.knowledge_graph.nodes:
            score = model.knowledge_graph.nodes[node].get('specialization_score', 0.5)
            specialization_scores.append(score)
            expert_ids.append(node)
        
        # Sort by expert ID for consistent display
        sorted_data = sorted(zip(expert_ids, specialization_scores))
        expert_ids, specialization_scores = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot histogram
        plt.hist(specialization_scores, bins=10, range=(0, 1), alpha=0.7, color='blue')
        plt.xlabel('Specialization Score')
        plt.ylabel('Number of Experts')
        plt.title('Expert Specialization Distribution')
        plt.grid(True, alpha=0.3)
    
    # Plot expert adaptation rates
    plt.subplot(3, 2, 3)
    
    if hasattr(model, 'knowledge_graph'):
        # Get adaptation rates for all experts
        adaptation_rates = []
        expert_ids = []
        for node in model.knowledge_graph.nodes:
            rate = model.knowledge_graph.nodes[node].get('adaptation_rate', 1.0)
            adaptation_rates.append(rate)
            expert_ids.append(node)
        
        # Sort by expert ID
        sorted_data = sorted(zip(expert_ids, adaptation_rates))
        expert_ids, adaptation_rates = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot bar chart
        plt.bar(expert_ids, adaptation_rates, color='green', alpha=0.7)
        plt.xlabel('Expert ID')
        plt.ylabel('Adaptation Rate')
        plt.title('Expert Adaptation Rates')
        plt.grid(True, alpha=0.3)
    
    # Plot expert relationships in knowledge graph
    plt.subplot(3, 2, 4)
    
    if hasattr(model, 'knowledge_graph'):
        G = model.knowledge_graph
        
        # Count different relation types
        relation_counts = {}
        
        for e in G.edges:
            edge_data = G.edges[e]
            if 'relation_type' in edge_data:
                rel_type = edge_data['relation_type']
                if rel_type not in relation_counts:
                    relation_counts[rel_type] = 0
                relation_counts[rel_type] += 1
            else:
                # Generic edge
                if 'generic' not in relation_counts:
                    relation_counts['generic'] = 0
                relation_counts['generic'] += 1
        
        # Plot pie chart if we have relations
        if relation_counts:
            labels = list(relation_counts.keys())
            sizes = [relation_counts[k] for k in labels]
            
            # Define nice colors for each relation type
            colors = {
                'similarity': 'blue',
                'specialization': 'green',
                'specialization_split': 'purple',
                'dependency': 'orange',
                'composition': 'red',
                'generic': 'gray',
                'complementary': 'teal'
            }
            
            # Use defined colors or default to a color cycle
            pie_colors = [colors.get(label, 'gray') for label in labels]
            
            plt.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Expert Relationship Types')
        else:
            plt.text(0.5, 0.5, "No relationships defined yet", 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
    
    # Plot memory replay statistics
    plt.subplot(3, 2, 5)
    
    if sleep_metrics:
        # Extract memory replay stats
        cycles = [i+1 for i, _ in enumerate(sleep_metrics)]
        replay_samples = [m.get('samples_replayed', 0) for m in sleep_metrics]
        replay_losses = [m.get('avg_loss', 0) for m in sleep_metrics]
        
        # Plot samples replayed
        ax1 = plt.gca()
        ax1.bar(cycles, replay_samples, color='blue', alpha=0.7)
        ax1.set_xlabel('Sleep Cycle')
        ax1.set_ylabel('Samples Replayed', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title('Memory Replay Statistics')
        
        # Plot average loss on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(cycles, replay_losses, 'r-o', linewidth=2)
        ax2.set_ylabel('Average Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Optional: Add high priority sample info if available
        if any('high_priority_samples' in m for m in sleep_metrics):
            high_priority = [m.get('high_priority_samples', 0) for m in sleep_metrics]
            for i, (c, h) in enumerate(zip(cycles, high_priority)):
                if h > 0:
                    ax1.text(c, replay_samples[i] + 5, f"HP: {h}", ha='center', fontsize=8)
    else:
        plt.text(0.5, 0.5, "No sleep metrics available", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # Plot expert evolution over sleep cycles
    plt.subplot(3, 2, 6)
    
    if sleep_metrics:
        # Extract expert counts before/after sleep
        cycles = [i+1 for i, _ in enumerate(sleep_metrics)]
        experts_before = [m.get('experts_before', 0) for m in sleep_metrics]
        experts_after = [m.get('experts_after', 0) for m in sleep_metrics]
        
        # Plot expert counts
        plt.plot(cycles, experts_before, 'b-o', linewidth=2, label='Before Sleep')
        plt.plot(cycles, experts_after, 'g-s', linewidth=2, label='After Sleep')
        
        # Add merge and prune annotations
        for i, m in enumerate(sleep_metrics):
            merged = m.get('merged_count', 0)
            pruned = m.get('pruned_count', 0)
            
            if merged > 0 or pruned > 0:
                annotation = []
                if merged > 0:
                    annotation.append(f"Merged: {merged}")
                if pruned > 0:
                    annotation.append(f"Pruned: {pruned}")
                    
                plt.annotate('\n'.join(annotation), (cycles[i], experts_after[i]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        plt.xlabel('Sleep Cycle')
        plt.ylabel('Number of Experts')
        plt.title('Expert Evolution During Sleep')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No sleep metrics available", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

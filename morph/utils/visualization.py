import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Dict, List, Any


def visualize_knowledge_graph(model, output_path=None, highlight_dormant=True, 
                        highlight_similar=True, highlight_specialization=True):
    """
    Visualize the knowledge graph of experts.
    
    Args:
        model: MorphModel instance
        output_path: Path to save the visualization (if None, display instead)
        highlight_dormant: Whether to highlight dormant experts
        highlight_similar: Whether to highlight similar experts
        highlight_specialization: Whether to highlight expert specialization
    """
    G = model.knowledge_graph
    
    # Get activation counts for node sizing
    activation_counts = [G.nodes[n]['activation_count'] for n in G.nodes]
    max_count = max(activation_counts) if activation_counts else 1
    
    # Normalize counts for node sizes
    node_sizes = [500 * (count / max_count) + 100 for count in activation_counts]
    
    # Get edge weights
    edge_weights = [G.edges[e]['weight'] * 3 for e in G.edges]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # For reproducible layout
    
    # Determine node colors based on dormancy, specialization, and similarity
    node_colors = []
    node_borders = []
    node_border_widths = []
    
    for node in G.nodes:
        # Default color (blue gradient based on specialization)
        if highlight_specialization and 'specialization_score' in G.nodes[node]:
            spec_score = G.nodes[node]['specialization_score']
            # Color gradient from light blue (general) to dark blue (specialized)
            color = (0.5 - 0.5 * spec_score, 0.5, 0.5 + 0.5 * spec_score)
        else:
            color = 'skyblue'  # Default blue
            
        # Check if dormant
        dormant = False
        if highlight_dormant and 'last_activated' in G.nodes[node]:
            step_diff = model.step_count - G.nodes[node]['last_activated']
            dormant_threshold = getattr(model.config, 'dormant_steps_threshold', float('inf'))
            
            if step_diff > dormant_threshold:
                color = 'lightgray'  # Dormant expert
                dormant = True
        
        # Border color and width
        if dormant:
            node_borders.append('red')
            node_border_widths.append(2.0)
        else:
            # Border based on adaptation rate if available
            if 'adaptation_rate' in G.nodes[node]:
                adapt_rate = G.nodes[node]['adaptation_rate']
                # Higher adaptation = thicker border
                node_border_widths.append(adapt_rate * 3)
                node_borders.append('black')
            else:
                node_borders.append('black')
                node_border_widths.append(1.0)
        
        # Add to color list
        node_colors.append(color)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.9, 
                          node_color=node_colors, edgecolors=node_borders,
                          linewidths=node_border_widths)
    
    # Determine edge colors and styles based on relationship types
    edge_colors = []
    edge_styles = []
    
    for e in G.edges:
        # Default
        color = 'gray'
        style = 'solid'
        
        # Get edge data
        edge_data = G.edges[e]
        
        # Check relation type if available
        if 'relation_type' in edge_data:
            rel_type = edge_data['relation_type']
            
            if rel_type == 'similarity':
                color = 'blue'
            elif rel_type == 'specialization':
                color = 'green'
            elif rel_type == 'specialization_split':
                color = 'purple'
                style = 'dashed'
            elif rel_type == 'dependency':
                color = 'orange'
                style = 'dotted'
            elif rel_type == 'composition':
                color = 'red'
                style = 'dashdot'
        # Otherwise use weight for similarity coloring
        elif highlight_similar:
            weight = edge_data['weight']
            similarity_threshold = getattr(model.config, 'expert_similarity_threshold', 0.8)
            
            if weight > similarity_threshold:
                color = 'red'  # Indicates potential merging
            else:
                # Color gradient based on weight
                intensity = min(1.0, weight / similarity_threshold)
                color = (0.7, 0.7 - 0.5 * intensity, 0.7 - 0.5 * intensity)
        
        edge_colors.append(color)
        edge_styles.append(style)
    
    # Draw edges
    for i, (e, color, style) in enumerate(zip(G.edges, edge_colors, edge_styles)):
        nx.draw_networkx_edges(G, pos, edgelist=[e], width=edge_weights[i], 
                              alpha=0.6, edge_color=color, style=style,
                              connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add metadata to nodes
    node_info = {}
    for node in G.nodes:
        node_data = G.nodes[node]
        info_str = f"Expert {node}\n"
        
        if 'activation_count' in node_data:
            info_str += f"Act: {node_data['activation_count']}\n"
            
        if 'specialization_score' in node_data:
            spec = node_data['specialization_score']
            info_str += f"Spec: {spec:.2f}\n"
            
        if 'adaptation_rate' in node_data:
            adapt = node_data['adaptation_rate']
            info_str += f"Adapt: {adapt:.2f}\n"
            
        if 'last_activated' in node_data:
            steps_ago = model.step_count - node_data['last_activated']
            info_str += f"Last: {steps_ago} steps ago"
            
        node_info[node] = info_str
    
    # Display metadata for important nodes
    if node_info:
        # Find important nodes: most active, most specialized, and most recently modified
        important_nodes = list(sorted(G.nodes, 
                                   key=lambda n: G.nodes[n].get('activation_count', 0), 
                                   reverse=True)[:2])
        
        # Add most specialized
        if highlight_specialization:
            specialized_nodes = sorted(G.nodes, 
                                     key=lambda n: G.nodes[n].get('specialization_score', 0), 
                                     reverse=True)[:2]
            important_nodes.extend(n for n in specialized_nodes if n not in important_nodes)
        
        # Add most recently active
        recent_nodes = sorted(G.nodes, 
                           key=lambda n: G.nodes[n].get('last_activated', 0), 
                           reverse=True)[:2]
        important_nodes.extend(n for n in recent_nodes if n not in important_nodes)
        
        # Limit to at most 5 nodes
        important_nodes = important_nodes[:5]
        
        for node in important_nodes:
            pos_x, pos_y = pos[node]
            plt.text(pos_x + 0.1, pos_y, node_info[node], 
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add title with sleep cycle info
    title = f"Expert Knowledge Graph (Experts: {len(G.nodes)})"
    if hasattr(model, 'sleep_cycles_completed'):
        title += f"\nSleep Cycles: {model.sleep_cycles_completed}"
        if hasattr(model, 'adaptive_sleep_frequency'):
            title += f" (Frequency: {model.adaptive_sleep_frequency} steps)"
    
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Add legend
    legend_y = 0.01
    
    plt.text(0.01, legend_y, "Node size: activation frequency", 
             transform=plt.gca().transAxes)
    legend_y += 0.04
    
    plt.text(0.01, legend_y, "Edge width: connection strength", 
             transform=plt.gca().transAxes)
    legend_y += 0.04
    
    if highlight_dormant:
        plt.text(0.01, legend_y, "Gray nodes: dormant experts", 
               transform=plt.gca().transAxes)
        legend_y += 0.04
    
    if highlight_specialization:
        plt.text(0.01, legend_y, "Node color: darker blue = more specialized", 
               transform=plt.gca().transAxes)
        legend_y += 0.04
        
        plt.text(0.01, legend_y, "Border width: adaptation rate", 
               transform=plt.gca().transAxes)
        legend_y += 0.04
    
    if highlight_similar:
        plt.text(0.01, legend_y, "Red edges: similar experts (candidates for merging)", 
               transform=plt.gca().transAxes)
        legend_y += 0.04
        
    # Add relation type legend
    plt.text(0.5, 0.01, "Edge colors/styles:", 
             transform=plt.gca().transAxes, fontweight='bold')
    legend_y = 0.05
    
    plt.text(0.5, legend_y, "Blue: similarity relation", transform=plt.gca().transAxes)
    legend_y += 0.04
    
    plt.text(0.5, legend_y, "Green: specialization relation", transform=plt.gca().transAxes)
    legend_y += 0.04
    
    plt.text(0.5, legend_y, "Purple dashed: specialization split", transform=plt.gca().transAxes)
    legend_y += 0.04
    
    plt.text(0.5, legend_y, "Orange dotted: dependency relation", transform=plt.gca().transAxes)
    legend_y += 0.04
    
    plt.text(0.5, legend_y, "Red dash-dot: composition relation", transform=plt.gca().transAxes)
    
    # Remove axis
    plt.axis('off')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


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


def visualize_concept_drift_adaptation(drift_metrics, model_metrics, output_path=None):
    """
    Visualize how the model adapts to concept drift over time.
    
    Args:
        drift_metrics: Dictionary of concept drift metrics
        model_metrics: Dictionary of model performance metrics
        output_path: Path to save visualization (if None, display instead)
    """
    if not drift_metrics or 'drift_detected_tasks' not in drift_metrics:
        logging.info("No drift metrics available, nothing to visualize")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot drift magnitude and adaptation rate
    plt.subplot(2, 2, 1)
    
    models = list(model_metrics.keys())
    x = np.arange(len(models))
    width = 0.35
    
    # Extract drift magnitudes and adaptation rates
    drift_magnitudes = []
    adaptation_rates = []
    
    for model_name in models:
        if model_name in drift_metrics:
            model_drift = drift_metrics[model_name]
            drift_magnitudes.append(model_drift.get('avg_drift_magnitude', 0))
            adaptation_rates.append(model_drift.get('avg_adaptation_rate', 0))
        else:
            drift_magnitudes.append(0)
            adaptation_rates.append(0)
    
    # Plot bar chart
    plt.bar(x - width/2, drift_magnitudes, width, label='Drift Magnitude', color='red', alpha=0.7)
    plt.bar(x + width/2, adaptation_rates, width, label='Adaptation Rate', color='green', alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Magnitude / Rate')
    plt.title('Concept Drift Magnitude and Adaptation Rate')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot task performance before and after drift
    plt.subplot(2, 2, 2)
    
    # Extract performance data
    if 'task_performance' in drift_metrics:
        task_ids = sorted(drift_metrics['task_performance'].keys())
        
        for model_name in models:
            if model_name in drift_metrics['task_performance']:
                pre_drift = [drift_metrics['task_performance'][model_name].get(task_id, {})
                             .get('pre_drift_accuracy', 0) for task_id in task_ids]
                post_drift = [drift_metrics['task_performance'][model_name].get(task_id, {})
                              .get('post_drift_accuracy', 0) for task_id in task_ids]
                
                plt.plot(task_ids, pre_drift, 'o--', label=f"{model_name} Pre-Drift")
                plt.plot(task_ids, post_drift, 's-', label=f"{model_name} Post-Drift")
        
        plt.xlabel('Task ID')
        plt.ylabel('Accuracy (%)')
        plt.title('Performance Before and After Drift')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Task performance data not available", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # Plot drift detection timeline
    plt.subplot(2, 1, 2)
    
    if 'drift_timeline' in drift_metrics:
        for model_name in models:
            if model_name in drift_metrics['drift_timeline']:
                timeline = drift_metrics['drift_timeline'][model_name]
                steps = [entry['step'] for entry in timeline]
                magnitudes = [entry['magnitude'] for entry in timeline]
                
                plt.plot(steps, magnitudes, 'o-', linewidth=2, label=model_name)
                
                # Mark points where adaptation occurred
                adaptation_steps = [entry['step'] for entry in timeline 
                                  if entry.get('adaptation_occurred', False)]
                
                if adaptation_steps:
                    adaptation_magnitudes = [timeline[steps.index(step)]['magnitude'] 
                                          for step in adaptation_steps]
                    plt.scatter(adaptation_steps, adaptation_magnitudes, 
                               marker='*', s=150, c='green', 
                               label=f"{model_name} Adaptation")
        
        plt.xlabel('Training Step')
        plt.ylabel('Drift Magnitude')
        plt.title('Concept Drift Timeline')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Drift timeline data not available", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

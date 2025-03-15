import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Dict, List, Any


def visualize_knowledge_graph(model, output_path=None, highlight_dormant=True, 
                        highlight_similar=True):
    """
    Visualize the knowledge graph of experts.
    
    Args:
        model: MorphModel instance
        output_path: Path to save the visualization (if None, display instead)
        highlight_dormant: Whether to highlight dormant experts
        highlight_similar: Whether to highlight similar experts
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
    plt.figure(figsize=(12, 8))
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # For reproducible layout
    
    # Determine node colors based on dormancy and similarity
    node_colors = []
    for i, node in enumerate(G.nodes):
        # Default color
        color = 'skyblue'
        
        if highlight_dormant:
            # Check if dormant (last_activated too long ago)
            if 'last_activated' in G.nodes[node]:
                step_diff = model.step_count - G.nodes[node]['last_activated']
                dormant_threshold = getattr(model.config, 'dormant_steps_threshold', float('inf'))
                
                if step_diff > dormant_threshold:
                    color = 'lightgray'  # Dormant expert
        
        # Add to color list
        node_colors.append(color)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8, 
                          node_color=node_colors, edgecolors='black')
    
    # Determine edge colors based on similarity
    edge_colors = []
    for e in G.edges:
        # Default color
        color = 'gray'
        
        if highlight_similar:
            # Check if edge weight indicates high similarity
            weight = G.edges[e]['weight']
            similarity_threshold = getattr(model.config, 'expert_similarity_threshold', 0.8)
            
            if weight > similarity_threshold:
                color = 'red'  # Indicates potential merging
        
        # Add to color list
        edge_colors.append(color)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, 
                          edge_color=edge_colors, connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add metadata to nodes
    node_info = {}
    for node in G.nodes:
        if 'activation_count' in G.nodes[node] and 'last_activated' in G.nodes[node]:
            act_count = G.nodes[node]['activation_count']
            last_act = G.nodes[node]['last_activated']
            node_info[node] = f"Act: {act_count}\nLast: {last_act}"
    
    # Display metadata for a few important nodes
    if node_info:
        important_nodes = sorted(node_info.keys(), 
                               key=lambda n: G.nodes[n]['activation_count'], 
                               reverse=True)[:3]
        
        for node in important_nodes:
            pos_x, pos_y = pos[node]
            plt.text(pos_x + 0.05, pos_y, node_info[node], 
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add title
    plt.title(f"Expert Knowledge Graph (Experts: {len(G.nodes)})", 
              fontsize=16, fontweight='bold')
    
    # Add legend
    plt.text(0.01, 0.01, "Node size represents activation frequency", 
             transform=plt.gca().transAxes)
    plt.text(0.01, 0.05, "Edge width represents connection strength", 
             transform=plt.gca().transAxes)
    
    if highlight_dormant:
        plt.text(0.01, 0.09, "Gray nodes are dormant experts", 
               transform=plt.gca().transAxes)
    
    if highlight_similar:
        plt.text(0.01, 0.13, "Red edges indicate similar experts (candidates for merging)", 
               transform=plt.gca().transAxes)
    
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
                              merge_events, output_path=None):
    """
    Create a visualization of the expert lifecycle during training.
    
    Args:
        expert_counts: List of (step, count) tuples showing expert count over time
        creation_events: List of (step, count) tuples showing creation events
        merge_events: List of (step, count) tuples showing merge/prune events
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

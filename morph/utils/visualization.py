import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Dict, List, Any


def visualize_knowledge_graph(model, output_path=None):
    """
    Visualize the knowledge graph of experts.
    
    Args:
        model: MorphModel instance
        output_path: Path to save the visualization (if None, display instead)
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
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.8, 
                           node_color='skyblue', edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, 
                           edge_color='gray', connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add title
    plt.title(f"Expert Knowledge Graph (Experts: {len(G.nodes)})", 
              fontsize=16, fontweight='bold')
    
    # Add legend
    plt.text(0.01, 0.01, "Node size represents activation frequency", 
             transform=plt.gca().transAxes)
    plt.text(0.01, 0.05, "Edge width represents similarity", 
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
    # Assuming model has tracked expert activations over time
    # This is a placeholder - actual implementation would depend on data tracking
    
    plt.figure(figsize=(12, 6))
    
    # Simulated data - replace with actual tracking data
    n_experts = len(model.experts)
    expert_ids = list(range(n_experts))
    
    # Generate sample activations
    # In practice, this would come from model's actual tracking
    np.random.seed(42)
    activation_history = np.zeros((n_steps, n_experts))
    for i in range(n_steps):
        # Simulate more recent activations being more relevant
        time_factor = i / n_steps
        activation_history[i] = np.random.rand(n_experts) * time_factor * 10
    
    # Cumulative activations
    cumulative = np.cumsum(activation_history, axis=0)
    
    # Plot
    for i in range(n_experts):
        plt.plot(cumulative[:, i], label=f"Expert {i}")
    
    plt.xlabel("Training Steps")
    plt.ylabel("Cumulative Activations")
    plt.title("Expert Usage Over Time")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

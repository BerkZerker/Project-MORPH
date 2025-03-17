"""
Knowledge graph visualization utilities for MORPH models.
"""

import networkx as nx
import matplotlib.pyplot as plt


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
    G = model.knowledge_graph.graph
    
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

"""
Visualization utilities for benchmark results.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_results(results, title="Continual Learning Benchmark", output_path=None):
    """
    Visualize benchmark results.
    
    Args:
        results: Results from run_benchmark
        title: Plot title
        output_path: Path to save visualization (if None, show instead)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Average accuracy across all tasks
    ax1 = axes[0, 0]
    
    model_names = list(results['final_accuracies'].keys())
    x = np.arange(len(model_names))
    avg_accuracies = [
        sum(accuracies.values()) / len(accuracies) 
        for accuracies in results['final_accuracies'].values()
    ]
    
    bars = ax1.bar(x, avg_accuracies, width=0.6)
    
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Average Accuracy Across All Tasks')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.set_ylim(0, 100)
    
    # Add values on top of bars
    for bar, acc in zip(bars, avg_accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{acc:.1f}%',
            ha='center',
            va='bottom'
        )
    
    # Plot 2: Average forgetting
    ax2 = axes[0, 1]
    
    avg_forgetting = list(results['avg_forgetting'].values())
    
    bars = ax2.bar(x, avg_forgetting, width=0.6, color='orange')
    
    ax2.set_ylabel('Forgetting (%)')
    ax2.set_title('Average Forgetting')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.set_ylim(0, max(avg_forgetting) * 1.2 + 5 if avg_forgetting else 10)
    
    # Add values on top of bars
    for bar, forget in zip(bars, avg_forgetting):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{forget:.1f}%',
            ha='center',
            va='bottom'
        )
    
    # Plot 3: Task accuracy evolution for each model
    ax3 = axes[1, 0]
    
    # Get all tasks
    all_tasks = sorted(list(results['final_accuracies'][model_names[0]].keys()))
    
    # Plot accuracy for each model across tasks
    for model_name in model_names:
        task_accuracies = [results['final_accuracies'][model_name][task_id] for task_id in all_tasks]
        ax3.plot(all_tasks, task_accuracies, 'o-', label=model_name)
    
    ax3.set_xlabel('Task ID')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Final Accuracy by Task')
    ax3.set_xticks(all_tasks)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Plot 4: Forgetting by task for each model
    ax4 = axes[1, 1]
    
    # Get tasks that have forgetting metrics (all except the last one)
    forgetting_tasks = all_tasks[:-1] if len(all_tasks) > 1 else []
    
    if forgetting_tasks:
        for model_name in model_names:
            if model_name in results['forgetting_metrics']:
                task_forgetting = [
                    results['forgetting_metrics'][model_name].get(task_id, 0) 
                    for task_id in forgetting_tasks
                ]
                ax4.plot(forgetting_tasks, task_forgetting, 'o-', label=model_name)
        
        ax4.set_xlabel('Task ID')
        ax4.set_ylabel('Forgetting (%)')
        ax4.set_title('Forgetting by Task')
        ax4.set_xticks(forgetting_tasks)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend()
    else:
        ax4.set_title('Forgetting by Task (Not enough tasks)')
        ax4.set_xlabel('Task ID')
        ax4.set_ylabel('Forgetting (%)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

"""
Performance visualization utilities for MORPH models.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging


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

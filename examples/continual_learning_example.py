import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from tqdm import tqdm
import time
from typing import Dict, List, Tuple

# Add parent directory to path to import morph
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morph.config import MorphConfig
from morph.core.model import MorphModel
from morph.utils.data import get_mnist_dataloaders
from morph.utils.benchmarks import (
    StandardModel,
    EWCModel,
    ContinualLearningBenchmark,
    create_rotating_mnist_tasks
)
from morph.utils.visualization import (
    visualize_knowledge_graph,
    plot_expert_activations,
    visualize_expert_lifecycle,
    visualize_sleep_metrics,
    visualize_expert_specialization_over_time,
    visualize_concept_drift_adaptation
)

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/continual_learning', exist_ok=True)
    
    # Run the advanced continual learning benchmark
    run_continual_learning_benchmark(device)
    
def create_enhanced_morph_model(device):
    """
    Create an enhanced MORPH model with the improved sleep module.
    
    Args:
        device: Device to create the model on
        
    Returns:
        Configured MORPH model
    """
    # Configure the model with enhanced sleep settings
    config = MorphConfig(
        input_size=784,  # 28x28 MNIST images flattened
        expert_hidden_size=256,
        output_size=10,  # 10 MNIST classes
        num_initial_experts=3,
        expert_k=2,  # Number of experts to route to
        
        # Dynamic expert creation settings
        enable_dynamic_experts=True,
        min_experts=2,
        max_experts=15,
        expert_creation_uncertainty_threshold=0.25,
        
        # Enhanced sleep cycle settings
        enable_sleep=True,
        sleep_cycle_frequency=300,  # Base frequency for sleep cycles
        enable_adaptive_sleep=True,  # Enable adaptive sleep scheduling
        min_sleep_frequency=150,     # Minimum steps between sleep cycles
        max_sleep_frequency=600,     # Maximum steps between sleep cycles
        
        # Expert merging and pruning
        expert_similarity_threshold=0.75,  # Similarity threshold for merging
        dormant_steps_threshold=500,  # Steps before considering an expert dormant
        min_lifetime_activations=50,  # Min activations to avoid pruning
        
        # Memory replay settings
        memory_replay_batch_size=32,  # Batch size for memory replay
        memory_buffer_size=1000,      # Maximum size of activation buffer
        replay_learning_rate=0.0001,  # Learning rate for replay fine-tuning
        
        # Expert reorganization
        enable_expert_reorganization=True,  # Whether to reorganize experts
        specialization_threshold=0.7,  # Threshold for considering an expert specialized
        overlap_threshold=0.3,  # Threshold for considering expert overlap significant
        
        # Knowledge graph settings
        knowledge_edge_decay=0.95,
        knowledge_edge_min=0.1,
        
        # Meta-learning
        enable_meta_learning=True,  # Whether to enable meta-learning optimizations
        meta_learning_intervals=2,  # Sleep cycles between meta-learning updates
        
        # Learning rate and batch size
        learning_rate=0.001,
        batch_size=64
    )
    
    # Create the model
    model = MorphModel(config).to(device)
    logging.info(f"Created enhanced MORPH model with {len(model.experts)} initial experts")
    
    return model, config

def run_continual_learning_benchmark(device):
    """
    Run comprehensive continual learning benchmark comparing MORPH with baseline models.
    
    Args:
        device: Device to run on
    """
    # Create tasks (rotated MNIST)
    num_tasks = 5
    tasks = create_rotating_mnist_tasks(num_tasks=num_tasks, samples_per_task=5000, feature_dim=784)
    
    # Setup models for comparison
    logging.info("Setting up models for benchmarking...")
    
    # Enhanced MORPH model
    morph_model, morph_config = create_enhanced_morph_model(device)
    
    # Standard neural network model
    standard_model = StandardModel(
        input_size=784,
        hidden_size=256,
        output_size=10,
        num_layers=2
    ).to(device)
    
    # Elastic Weight Consolidation (EWC) model
    ewc_model = EWCModel(
        input_size=784,
        hidden_size=256,
        output_size=10,
        ewc_lambda=5000
    ).to(device)
    
    # Create optimizers
    morph_optimizer = optim.Adam(
        morph_model.parameters(), 
        lr=morph_config.learning_rate
    )
    
    standard_optimizer = optim.Adam(
        standard_model.parameters(), 
        lr=0.001
    )
    
    ewc_optimizer = optim.Adam(
        ewc_model.parameters(),
        lr=0.001
    )
    
    # Create model dictionary for benchmark
    models = {
        "MORPH": morph_model,
        "Standard NN": standard_model,
        "EWC": ewc_model
    }
    
    optimizers = {
        "MORPH": morph_optimizer,
        "Standard NN": standard_optimizer,
        "EWC": ewc_optimizer
    }
    
    # Create task similarities (simplified example)
    task_similarities = {}
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            # Tasks with smaller rotation differences are more similar
            similarity = 1.0 - (abs(j - i) / num_tasks)
            task_similarities[(i, j)] = similarity
    
    # Set up benchmark with concept drift detection
    benchmark = ContinualLearningBenchmark(
        tasks=tasks,
        input_size=784,
        output_size=10,
        device=device,
        drift_detection=True,
        task_similarities=task_similarities
    )
    
    # Run benchmark with detailed evaluation
    logging.info("Running continual learning benchmark...")
    start_time = time.time()
    
    results = benchmark.run_benchmark(
        models=models,
        optimizers=optimizers,
        epochs_per_task=3,
        batch_size=64,
        detailed_eval=True
    )
    
    elapsed_time = time.time() - start_time
    logging.info(f"Benchmark completed in {elapsed_time:.2f} seconds")
    
    # Generate visualizations from benchmark results
    generate_benchmark_visualizations(models, results)
    
    # Track expert evolution for MORPH (throughout training)
    track_expert_evolution(morph_model, num_tasks)

def track_expert_evolution(model, num_tasks):
    """
    Track and visualize the evolution of experts during training.
    
    Args:
        model: MORPH model
        num_tasks: Number of tasks trained on
    """
    # Visualize expert knowledge graph
    for task_id in range(num_tasks):
        visualize_knowledge_graph(
            model,
            output_path=f"results/continual_learning/knowledge_graph_task_{task_id}.png",
            highlight_dormant=True,
            highlight_similar=True,
            highlight_specialization=True
        )
    
    # Visualize expert activation patterns
    plot_expert_activations(
        model,
        n_steps=model.step_count,
        output_path="results/continual_learning/expert_activations.png"
    )
    
    # Visualize sleep metrics
    visualize_sleep_metrics(
        model,
        output_path="results/continual_learning/sleep_metrics.png"
    )
    
    # If we had tracked expert metrics over time, we would visualize specialization
    # In a real implementation, we would collect this data during training
    expert_history = []
    
    # Extract current expert metrics as a sample point
    if hasattr(model, 'get_expert_metrics'):
        current_metrics = model.get_expert_metrics()
        expert_history.append((model.step_count, current_metrics))
        
        visualize_expert_specialization_over_time(
            model,
            expert_history=expert_history,
            output_path="results/continual_learning/expert_specialization.png"
        )

def generate_benchmark_visualizations(models, results):
    """
    Generate visualizations for benchmark results.
    
    Args:
        models: Dictionary of models used in benchmark
        results: Benchmark results dictionary
    """
    os.makedirs("results/continual_learning/benchmark", exist_ok=True)
    
    # 1. Plot final accuracy comparison
    plt.figure(figsize=(10, 6))
    
    model_names = list(results['final_accuracies'].keys())
    task_ids = sorted(list(results['final_accuracies'][model_names[0]].keys()))
    
    x = np.arange(len(task_ids))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        accuracies = []
        for task_id in task_ids:
            acc = results['final_accuracies'][model_name][task_id]
            # Handle both simple and detailed metrics
            if isinstance(acc, dict):
                acc = acc['accuracy']
            accuracies.append(acc)
        
        plt.bar(x + (i - 1) * width, accuracies, width, label=model_name)
    
    plt.xlabel('Task ID')
    plt.ylabel('Final Accuracy (%)')
    plt.title('Model Performance Comparison Across Tasks')
    plt.xticks(x, [f"Task {i}" for i in task_ids])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/continual_learning/benchmark/final_accuracy.png", dpi=300)
    plt.close()
    
    # 2. Plot forgetting metrics
    plt.figure(figsize=(10, 6))
    
    for model_name in model_names:
        forgetting = results['forgetting_metrics'][model_name]
        task_ids = sorted(list(forgetting.keys()))
        forgetting_values = [forgetting[task_id] for task_id in task_ids]
        
        plt.plot(task_ids, forgetting_values, 'o-', linewidth=2, label=model_name)
    
    plt.xlabel('Task ID')
    plt.ylabel('Forgetting (%)')
    plt.title('Catastrophic Forgetting by Task and Model')
    plt.xticks(task_ids)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/continual_learning/benchmark/forgetting.png", dpi=300)
    plt.close()
    
    # 3. Plot knowledge retention
    plt.figure(figsize=(10, 6))
    
    for model_name in model_names:
        retention = results['retention_metrics'][model_name]
        task_ids = sorted(list(retention.keys()))
        retention_values = [retention[task_id] * 100 for task_id in task_ids]
        
        plt.plot(task_ids, retention_values, 'o-', linewidth=2, label=model_name)
    
    plt.xlabel('Task ID')
    plt.ylabel('Retention (%)')
    plt.title('Knowledge Retention by Task and Model')
    plt.xticks(task_ids)
    plt.legend()
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3)
    plt.savefig("results/continual_learning/benchmark/retention.png", dpi=300)
    plt.close()
    
    # 4. Plot average forgetting comparison
    plt.figure(figsize=(8, 6))
    
    avg_forgetting = results['avg_forgetting']
    
    plt.bar(range(len(model_names)), 
           [avg_forgetting[name] for name in model_names],
           color=['blue', 'orange', 'green'])
    
    plt.xlabel('Model')
    plt.ylabel('Average Forgetting (%)')
    plt.title('Average Catastrophic Forgetting')
    plt.xticks(range(len(model_names)), model_names)
    plt.grid(True, alpha=0.3)
    plt.savefig("results/continual_learning/benchmark/avg_forgetting.png", dpi=300)
    plt.close()
    
    # 5. Plot expert metrics for MORPH
    if 'expert_metrics' in results and 'MORPH' in results['expert_metrics']:
        expert_metrics = results['expert_metrics']['MORPH']
        
        # Expert utilization across tasks
        plt.figure(figsize=(12, 8))
        
        # Plot expert usage per task
        task_ids = sorted(list(expert_metrics['expert_utilization'].keys()))
        
        for task_id in task_ids:
            utilization = expert_metrics['expert_utilization'][task_id]
            expert_ids = sorted(list(utilization.keys()))
            activations = [utilization[expert_id] for expert_id in expert_ids]
            
            plt.plot(expert_ids, activations, 'o-', linewidth=2, 
                   label=f"Task {task_id}")
        
        plt.xlabel('Expert ID')
        plt.ylabel('Activation Count')
        plt.title('Expert Utilization by Task')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("results/continual_learning/benchmark/expert_utilization.png", dpi=300)
        plt.close()
        
        # Expert specialization
        plt.figure(figsize=(12, 8))
        
        for task_id in task_ids:
            specialization = expert_metrics['expert_specialization'][task_id]
            expert_ids = sorted(list(specialization.keys()))
            spec_scores = [specialization[expert_id] for expert_id in expert_ids]
            
            plt.plot(expert_ids, spec_scores, 'o-', linewidth=2, 
                   label=f"Task {task_id}")
        
        plt.xlabel('Expert ID')
        plt.ylabel('Specialization Score')
        plt.title('Expert Specialization by Task')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("results/continual_learning/benchmark/expert_specialization.png", dpi=300)
        plt.close()
    
    # 6. Plot task overlap matrix (how similar the tasks are)
    if 'expert_metrics' in results and 'MORPH' in results['expert_metrics']:
        expert_metrics = results['expert_metrics']['MORPH']
        
        if 'task_expert_overlap' in expert_metrics:
            task_overlap = expert_metrics['task_expert_overlap']
            
            # Create matrix representation of task overlaps
            task_ids = sorted(list(set([t[0] for t in task_overlap.keys()] + 
                                    [t[1] for t in task_overlap.keys()])))
            num_tasks = len(task_ids)
            overlap_matrix = np.zeros((num_tasks, num_tasks))
            
            # Fill diagonal with 1.0 (perfect overlap with self)
            for i in range(num_tasks):
                overlap_matrix[i, i] = 1.0
            
            # Fill rest of matrix
            for (task_i, task_j), overlap in task_overlap.items():
                i = task_ids.index(task_i)
                j = task_ids.index(task_j)
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap  # Mirror
            
            # Plot heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(overlap_matrix, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label='Expert Overlap')
            plt.title('Task Similarity Based on Expert Utilization')
            plt.xlabel('Task ID')
            plt.ylabel('Task ID')
            plt.xticks(range(num_tasks), task_ids)
            plt.yticks(range(num_tasks), task_ids)
            
            # Add text annotations
            for i in range(num_tasks):
                for j in range(num_tasks):
                    plt.text(j, i, f"{overlap_matrix[i, j]:.2f}", 
                           ha='center', va='center', 
                           color='white' if overlap_matrix[i, j] < 0.7 else 'black')
            
            plt.savefig("results/continual_learning/benchmark/task_overlap.png", dpi=300)
            plt.close()
    
    # 7. Plot concept drift metrics if available
    if 'drift_metrics' in results and results['drift_metrics']:
        visualize_concept_drift_adaptation(
            results['drift_metrics'],
            {model_name: results['final_accuracies'][model_name] for model_name in model_names},
            output_path="results/continual_learning/benchmark/concept_drift.png"
        )

if __name__ == "__main__":
    main()
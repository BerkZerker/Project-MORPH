import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add parent directory to path to import morph
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from morph.config import MorphConfig
from morph.core.model import MorphModel
from morph.utils.benchmarks import (
    StandardModel,
    EWCModel,
    ContinualLearningBenchmark,
    create_rotating_mnist_tasks,
    create_split_mnist_tasks,
    create_permuted_mnist_tasks
)


def run_benchmark(benchmark_type="rotating", num_tasks=5):
    """
    Run a continual learning benchmark comparing MORPH against other models.
    
    Args:
        benchmark_type: Type of benchmark to run ("rotating", "split", or "permuted")
        num_tasks: Number of tasks to create
    """
    # Create output directory
    os.makedirs('results/benchmarks', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create tasks based on benchmark type
    if benchmark_type == "rotating":
        logging.info(f"Creating rotating MNIST tasks (n={num_tasks})")
        tasks = create_rotating_mnist_tasks(num_tasks=num_tasks, samples_per_task=2000)
        feature_dim = 784
        output_dim = 10
        benchmark_name = "Rotating MNIST"
    elif benchmark_type == "split":
        logging.info(f"Creating split MNIST tasks (n={num_tasks})")
        tasks = create_split_mnist_tasks(num_tasks=num_tasks)
        feature_dim = 784
        output_dim = 10 // num_tasks
        benchmark_name = "Split MNIST"
    elif benchmark_type == "permuted":
        logging.info(f"Creating permuted MNIST tasks (n={num_tasks})")
        tasks = create_permuted_mnist_tasks(num_tasks=num_tasks, samples_per_task=2000)
        feature_dim = 784
        output_dim = 10
        benchmark_name = "Permuted MNIST"
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    
    # Initialize benchmark
    benchmark = ContinualLearningBenchmark(
        tasks=tasks,
        input_size=feature_dim,
        output_size=output_dim,
        device=device
    )
    
    # Create MORPH model
    morph_config = MorphConfig(
        input_size=feature_dim,
        expert_hidden_size=256,
        output_size=output_dim,
        num_initial_experts=3,
        expert_k=2,
        
        # Dynamic expert creation
        enable_dynamic_experts=True,
        expert_creation_uncertainty_threshold=0.25,
        min_experts=3,
        max_experts=20,
        
        # Sleep cycle
        enable_sleep=True,
        sleep_cycle_frequency=200,
        enable_adaptive_sleep=True,
        min_sleep_frequency=100,
        max_sleep_frequency=500,
        expert_similarity_threshold=0.8,
        
        # Expert reorganization
        enable_expert_reorganization=True,
        
        # Knowledge graph
        knowledge_edge_decay=0.9,
        knowledge_edge_min=0.1,
        
        # Meta-learning
        enable_meta_learning=True,
        meta_learning_intervals=2
    )
    
    morph_model = MorphModel(morph_config).to(device)
    
    # Create standard model (same capacity as MORPH)
    standard_model = StandardModel(
        input_size=feature_dim,
        hidden_size=512,  # Larger to match MORPH capacity
        output_size=output_dim,
        num_layers=3
    ).to(device)
    
    # Create EWC model
    ewc_model = EWCModel(
        input_size=feature_dim,
        hidden_size=512,  # Match standard model capacity
        output_size=output_dim,
        ewc_lambda=5000  # Importance of preserving previous tasks
    ).to(device)
    
    # Create optimizers
    morph_optimizer = optim.Adam(morph_model.parameters(), lr=0.001)
    standard_optimizer = optim.Adam(standard_model.parameters(), lr=0.001)
    ewc_optimizer = optim.Adam(ewc_model.parameters(), lr=0.001)
    
    # Create model and optimizer dictionaries
    models = {
        "MORPH": morph_model,
        "Standard": standard_model,
        "EWC": ewc_model
    }
    
    optimizers = {
        "MORPH": morph_optimizer,
        "Standard": standard_optimizer,
        "EWC": ewc_optimizer
    }
    
    # Run benchmark
    logging.info("Starting benchmark...")
    results = benchmark.run_benchmark(
        models=models,
        optimizers=optimizers,
        epochs_per_task=3,  # Train for 3 epochs per task
        batch_size=64
    )
    
    # Visualize results
    logging.info("Generating visualizations...")
    benchmark.visualize_results(
        results,
        title=f"Continual Learning Benchmark: {benchmark_name} ({num_tasks} tasks)",
        output_path=f"results/benchmarks/{benchmark_type}_benchmark_{num_tasks}tasks.png"
    )
    
    # Print summary
    logging.info("Benchmark summary:")
    logging.info(f"Benchmark type: {benchmark_name}")
    logging.info(f"Number of tasks: {num_tasks}")
    
    # Print average accuracy
    logging.info("Average accuracy across all tasks:")
    for model_name, accuracies in results['final_accuracies'].items():
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        logging.info(f"  {model_name}: {avg_accuracy:.2f}%")
    
    # Print average forgetting
    logging.info("Average forgetting:")
    for model_name, avg_forgetting in results['avg_forgetting'].items():
        logging.info(f"  {model_name}: {avg_forgetting:.2f}%")
    
    # Check final MORPH model structure
    logging.info(f"Final MORPH model structure:")
    logging.info(f"  Initial experts: {morph_config.num_initial_experts}")
    logging.info(f"  Final experts: {len(morph_model.experts)}")
    
    # Save MORPH model
    torch.save(
        morph_model.state_dict(),
        f"results/benchmarks/morph_{benchmark_type}_{num_tasks}tasks.pt"
    )
    logging.info(f"MORPH model saved to results/benchmarks/morph_{benchmark_type}_{num_tasks}tasks.pt")
    
    # Return results for further analysis
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MORPH benchmarks for continual learning")
    parser.add_argument("--type", type=str, default="rotating",
                      choices=["rotating", "split", "permuted"],
                      help="Type of benchmark to run")
    parser.add_argument("--tasks", type=int, default=5,
                      help="Number of tasks to create")
    
    args = parser.parse_args()
    
    run_benchmark(benchmark_type=args.type, num_tasks=args.tasks)
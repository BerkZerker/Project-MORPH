# MORPH Examples

This directory contains example implementations and use cases demonstrating the MORPH architecture.

## Available Examples

### 1. MNIST Classification (`mnist_example.py`)

A basic example showing how to train a MORPH model on the MNIST dataset. This example demonstrates:

- Basic model setup and configuration
- Training and evaluation procedures
- Visualization of expert activations and knowledge graphs
- Sleep cycle functionality

To run this example:

```bash
python mnist_example.py
```

Results will be saved to the `results/` directory, including:
- Knowledge graph visualizations for each epoch
- Expert activation charts
- Training/testing loss and accuracy curves
- The trained model

### 2. Continual Learning (`continual_learning_example.py`)

An example that demonstrates how MORPH handles continual learning scenarios with distribution shifts:

- Sequential task introduction with rotated MNIST digits
- Measurement of catastrophic forgetting
- Expert specialization per task
- Dynamic expert creation as new tasks are introduced

To run this example:

```bash
python continual_learning_example.py
```

Results will be saved to the `results/continual/` directory, including:
- Task performance over time, showing reduced catastrophic forgetting
- Expert dynamics during task transitions
- Knowledge graph evolution
- Comprehensive metrics on expert utilization and task performance

### 3. Performance Benchmarking (`benchmark_example.py`)

An example that benchmarks MORPH against baseline methods for continual learning:

- Compares MORPH against standard networks and Elastic Weight Consolidation (EWC)
- Tests on multiple continual learning scenarios (rotating, split, and permuted MNIST)
- Measures catastrophic forgetting and overall performance
- Visualizes results across tasks and models

To run this example:

```bash
python benchmark_example.py --type rotating --tasks 5
```

Results will be saved to the `results/benchmarks/` directory, including:
- Comparison charts of performance metrics
- Saved model weights for further analysis
- Detailed logging of benchmark results

### 4. Sleep Module (`sleep_module_example.py`)

An example demonstrating the Sleep Module's memory consolidation capabilities:

- Memory replay and knowledge consolidation
- Expert reorganization during sleep cycles
- Analysis of sleep cycle effects on model performance
- Visualization of expert merging and specialization

### 5. GPU Training (`gpu_training_example.py`)

An example demonstrating how to use GPU acceleration for training MORPH models:

- Automatic GPU detection and configuration
- Multi-GPU training with different parallelization strategies
- Mixed precision training for faster performance
- Expert distribution across multiple GPUs
- Performance monitoring and visualization

To run this example:

```bash
# Basic usage with automatic GPU detection
python gpu_training_example.py

# Use multiple GPUs with expert parallel strategy
python gpu_training_example.py --gpu-mode multi_gpu --parallel-strategy expert_parallel

# Enable mixed precision training
python gpu_training_example.py --mixed-precision
```

See [README_GPU.md](README_GPU.md) for detailed GPU usage instructions.

### 6. Multi-Domain Adaptation (Coming Soon)

An example showing how MORPH can adapt to different data domains:

- Training on multiple data types/distributions
- Expert specialization by domain
- Knowledge transfer between domains

## Running Examples

All examples can be run directly from Python. Make sure you've installed the required dependencies:

```bash
pip install -r ../requirements.txt
```

For customizing experiment configurations, modify the `MorphConfig` parameters in each example file.

## Creating Your Own Examples

To implement your own MORPH example:

1. Import the necessary components:
   ```python
   from morph.config import MorphConfig
   from morph.core.model import MorphModel
   ```

2. Configure the model:
   ```python
   config = MorphConfig(
       input_size=YOUR_INPUT_SIZE,
       output_size=YOUR_OUTPUT_SIZE,
       # Other configuration parameters...
   )
   ```

3. Create and train the model:
   ```python
   model = MorphModel(config)
   # Your training loop here...
   ```

4. Access experts and knowledge graph:
   ```python
   # Get number of experts
   num_experts = len(model.experts)
   
   # Access the knowledge graph
   knowledge_graph = model.knowledge_graph
   ```

5. Trigger sleep cycles:
   ```python
   # Manual sleep cycle
   model.sleep()
   ```

See the existing examples for more detailed implementations.

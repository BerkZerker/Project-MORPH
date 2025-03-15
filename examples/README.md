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

### 3. Multi-Domain Adaptation (Coming Soon)

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
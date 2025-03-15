# MORPH API Reference

## Core Modules

### `morph.core.MorphModel`

The main model class that implements the MORPH architecture.

```python
model = MorphModel(config)
```

**Parameters:**
- `config`: Instance of `MorphConfig` with model configuration

**Methods:**
- `forward(x, training=True)`: Forward pass through the model
  - `x`: Input tensor [batch_size, input_size]
  - `training`: Whether in training mode
  - Returns: Output tensor [batch_size, output_size]
  
- `sleep()`: Trigger a sleep cycle for knowledge consolidation
  
- `train_step(batch, optimizer, criterion)`: Perform a single training step
  - `batch`: Tuple of (inputs, targets)
  - `optimizer`: Optimizer to use
  - `criterion`: Loss function
  - Returns: Dictionary with loss and metrics
  
- `evaluate(data_loader, criterion, device)`: Evaluate model on a dataset
  - `data_loader`: DataLoader with evaluation data
  - `criterion`: Loss function
  - `device`: Device to use
  - Returns: Dictionary with evaluation metrics
  
- `get_knowledge_graph()`: Get the knowledge graph
  - Returns: NetworkX graph of expert relationships
  
- `get_expert_metrics()`: Get metrics about the current experts
  - Returns: Dictionary of expert metrics
  
- `get_sleep_metrics()`: Get metrics about sleep cycles
  - Returns: List of sleep cycle metrics

**Properties:**
- `sleep_cycles_completed`: Number of sleep cycles completed
- `adaptive_sleep_frequency`: Current sleep frequency after adaptation
- `next_sleep_step`: Step at which the next sleep cycle will be triggered

### `morph.core.Expert`

Individual expert network that specializes in a particular subset of data.

```python
expert = Expert(input_size, hidden_size, output_size, num_layers=2)
```

**Parameters:**
- `input_size`: Dimension of input features
- `hidden_size`: Size of hidden layers
- `output_size`: Dimension of output features
- `num_layers`: Number of hidden layers

**Methods:**
- `forward(x, update_stats=True)`: Forward pass through the expert
  - `x`: Input tensor
  - `update_stats`: Whether to update expert statistics
  - Returns: Expert output
  
- `clone()`: Create a clone of this expert with same architecture but re-initialized weights
  - Returns: A new Expert instance
  
- `get_parameter_similarity(other_expert)`: Compute cosine similarity between this expert's parameters and another expert
  - `other_expert`: Another Expert instance to compare with
  - Returns: Similarity score between 0 and 1
  
- `get_specialization_score()`: Calculate a specialization score for this expert
  - Returns: Specialization score between 0 and 1
  
- `get_centroid_similarity(other_expert)`: Compute similarity between this expert's input centroid and another expert's
  - `other_expert`: Another Expert instance to compare with
  - Returns: Similarity score between 0 and 1, or None if centroids not available
  
- `update_confidence(loss_value)`: Update the expert's confidence score based on loss values
  - `loss_value`: Loss value from a recent forward pass
  - Returns: Updated confidence score

### `morph.core.GatingNetwork`

Gating network that determines which experts to use for a given input.

```python
gating = GatingNetwork(input_size, num_experts, k=2, routing_type="top_k")
```

**Parameters:**
- `input_size`: Dimension of input features
- `num_experts`: Number of experts to route between
- `k`: Number of experts to activate per input (for top-k routing)
- `routing_type`: Type of routing mechanism ("top_k" or "noisy_top_k")

**Methods:**
- `forward(x, training=True)`: Compute routing probabilities for each expert
  - `x`: Input tensor [batch_size, input_size]
  - `training`: Whether in training mode (affects routing)
  - Returns: Tuple of (routing_weights, expert_indices, uncertainty)
  
- `should_create_new_expert(uncertainty)`: Determine if a new expert should be created based on uncertainty
  - `uncertainty`: Routing uncertainty score
  - Returns: Boolean indicating whether to create a new expert
  
- `update_num_experts(num_experts)`: Update the gating network when number of experts changes
  - `num_experts`: New number of experts

### `morph.core.KnowledgeGraph`

Manages the relationships between experts, concept specialization, and expert similarities.

```python
kg = KnowledgeGraph(config)
```

**Parameters:**
- `config`: Configuration object with knowledge graph parameters

**Methods:**
- `add_expert(expert_id, specialization_score=0.5, adaptation_rate=1.0)`: Add a new expert to the knowledge graph
  - `expert_id`: Unique identifier for the expert
  - `specialization_score`: Initial specialization score (0.0-1.0)
  - `adaptation_rate`: Initial adaptation rate (0.0-1.0)
  
- `add_edge(expert1_id, expert2_id, weight=0.5, relation_type='similarity')`: Add an edge between two experts
  - `expert1_id`: First expert ID
  - `expert2_id`: Second expert ID
  - `weight`: Edge weight (0.0-1.0)
  - `relation_type`: Type of relationship
  
- `update_expert_activation(expert_id, step)`: Update expert activation information
  - `expert_id`: Expert ID to update
  - `step`: Current training step
  
- `update_expert_specialization(expert_id, specialization_score)`: Update expert specialization score
  - `expert_id`: Expert ID to update
  - `specialization_score`: New specialization score (0.0-1.0)
  
- `add_concept(concept_id, embedding, parent_concept=None)`: Add a new concept to the knowledge graph
  - `concept_id`: Unique identifier for the concept
  - `embedding`: Tensor representation of the concept
  - `parent_concept`: Optional parent concept ID for hierarchical organization
  
- `link_expert_to_concept(expert_id, concept_id, strength=1.0)`: Link an expert to a concept
  - `expert_id`: Expert ID
  - `concept_id`: Concept ID
  - `strength`: Strength of association (0.0-1.0)
  
- `get_similar_experts(expert_id, threshold=0.7)`: Get experts similar to the given expert
  - `expert_id`: Expert ID to find similarities for
  - `threshold`: Similarity threshold (0.0-1.0)
  - Returns: List of (expert_id, similarity) tuples for similar experts
  
- `get_expert_centrality(expert_id)`: Get the centrality score of an expert
  - `expert_id`: Expert ID
  - Returns: Centrality score (0.0-1.0)
  
- `find_experts_for_concepts(concept_ids)`: Find experts that specialize in given concepts
  - `concept_ids`: List of concept IDs
  - Returns: List of (expert_id, relevance) tuples sorted by relevance
  
- `get_dormant_experts(current_step, dormancy_threshold, min_activations)`: Find experts that haven't been activated for a long time
  - `current_step`: Current training step
  - `dormancy_threshold`: Steps of inactivity to consider dormant
  - `min_activations`: Minimum lifetime activations to be considered dormant
  - Returns: List of dormant expert IDs
  
- `merge_expert_connections(source_id, target_ids)`: Transfer connections from source expert to target experts
  - `source_id`: Expert ID that will be removed
  - `target_ids`: List of experts to transfer connections to
  
- `get_expert_metadata(expert_id)`: Get all metadata for an expert
  - `expert_id`: Expert ID
  - Returns: Dictionary of expert metadata
  
- `rebuild_graph(expert_count)`: Rebuild the knowledge graph after expert count changes
  - `expert_count`: New expert count

### `morph.core.SleepModule`

Implements the 'sleep' phase of the model for knowledge consolidation.

```python
sleep_module = SleepModule(config, knowledge_graph)
```

**Parameters:**
- `config`: Configuration object with sleep parameters
- `knowledge_graph`: KnowledgeGraph instance

**Methods:**
- `should_sleep(step_count)`: Determine if a sleep cycle should be triggered
  - `step_count`: Current training step
  - Returns: Boolean indicating whether to trigger sleep
  
- `add_to_memory_buffer(activation_data)`: Add activation data to the memory buffer
  - `activation_data`: Dictionary containing activation information
  
- `perform_sleep_cycle(model, step_count)`: Perform a complete sleep cycle
  - `model`: The MORPH model
  - `step_count`: Current training step
  - Returns: Dictionary of metrics from the sleep cycle

## Configuration

### `morph.config.MorphConfig`

Configuration class for the MORPH model.

```python
config = MorphConfig(
    input_size=784,
    expert_hidden_size=256,
    output_size=10,
    num_initial_experts=4,
    ...
)
```

**Parameters:**
- `input_size`: Dimension of input features (default: 784 for MNIST)
- `expert_hidden_size`: Size of expert hidden layers (default: 256)
- `output_size`: Dimension of output features (default: 10 for MNIST)
- `num_initial_experts`: Number of experts to start with (default: 4)
- `expert_k`: Number of experts to route to for each input (default: 2)
- `enable_dynamic_experts`: Whether to enable dynamic expert creation (default: True)
- `expert_creation_uncertainty_threshold`: Threshold for creating new experts (default: 0.3)
- `min_experts`: Minimum number of experts to maintain (default: 2)
- `max_experts`: Maximum number of experts (default: 32)
- `enable_sleep`: Whether to enable sleep cycles (default: True)
- `sleep_cycle_frequency`: Steps between sleep cycles (default: 1000)
- `enable_adaptive_sleep`: Whether to adjust sleep frequency dynamically (default: True)
- `min_sleep_frequency`: Minimum steps between sleep cycles (default: 500)
- `max_sleep_frequency`: Maximum steps between sleep cycles (default: 2000)
- `memory_replay_batch_size`: Batch size for memory replay (default: 32)
- `memory_buffer_size`: Maximum size of activation buffer (default: 2000)
- `replay_learning_rate`: Learning rate for replay fine-tuning (default: 0.0001)
- `enable_expert_reorganization`: Whether to reorganize experts (default: True)
- `specialization_threshold`: Threshold for considering an expert specialized (default: 0.7)
- `overlap_threshold`: Threshold for considering expert overlap significant (default: 0.3)
- `expert_similarity_threshold`: Threshold for merging experts (default: 0.8)
- `dormant_steps_threshold`: Steps of inactivity before pruning (default: 5000)
- `min_lifetime_activations`: Minimum activations to avoid pruning (default: 100)
- `knowledge_edge_decay`: Decay factor for edge weights (default: 0.99)
- `knowledge_edge_min`: Minimum edge weight before pruning (default: 0.1)
- `enable_meta_learning`: Whether to enable meta-learning optimizations (default: True)
- `meta_learning_intervals`: Sleep cycles between meta-learning updates (default: 10)
- `learning_rate`: Learning rate for training (default: 0.001)
- `weight_decay`: Weight decay for training (default: 1e-5)
- `batch_size`: Batch size for training (default: 64)
- `num_epochs`: Number of epochs for training (default: 10)

## Utilities

### `morph.utils.data`

Data handling utilities.

**Functions:**
- `get_mnist_dataloaders(batch_size=64, num_workers=2)`: Get MNIST dataset loaders
  - `batch_size`: Batch size for training/testing
  - `num_workers`: Number of worker processes for data loading
  - Returns: Tuple of (train_loader, test_loader)

**Classes:**
- `ContinualTaskDataset`: Dataset for continual learning with distribution shifts

### `morph.utils.visualization`

Visualization utilities for model analysis.

**Functions:**
- `visualize_knowledge_graph(model, output_path=None, highlight_dormant=True, highlight_similar=True, highlight_specialization=True)`: Visualize the knowledge graph
  - `model`: MorphModel instance
  - `output_path`: Path to save the visualization
  - `highlight_dormant`: Whether to highlight dormant experts
  - `highlight_similar`: Whether to highlight similar experts
  - `highlight_specialization`: Whether to highlight expert specialization
  
- `plot_expert_activations(model, n_steps, output_path=None)`: Plot expert activation patterns
  - `model`: MorphModel instance
  - `n_steps`: Number of steps to show history for
  - `output_path`: Path to save the visualization
  
- `visualize_expert_lifecycle(expert_counts, creation_events, merge_events, sleep_events=None, output_path=None)`: Create a visualization of the expert lifecycle
  - `expert_counts`: List of (step, count) tuples showing expert count over time
  - `creation_events`: List of (step, count) tuples showing creation events
  - `merge_events`: List of (step, count) tuples showing merge/prune events
  - `sleep_events`: List of (step, metrics) tuples showing sleep events
  - `output_path`: Path to save the visualization
  
- `visualize_sleep_metrics(model, sleep_events=None, output_path=None)`: Visualize sleep cycle metrics
  - `model`: MorphModel instance with sleep cycle tracking
  - `sleep_events`: List of (step, metrics) tuples showing sleep events
  - `output_path`: Path to save the visualization
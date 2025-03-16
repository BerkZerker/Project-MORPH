# GPU Training with MORPH

This guide explains how to use GPU acceleration for training and running MORPH models. The implementation automatically detects available GPUs and can distribute the workload across multiple GPUs for improved performance.

## Features

- **Automatic GPU Detection**: Automatically detects available CUDA-capable GPUs
- **Multi-GPU Support**: Distributes workload across multiple GPUs
- **Mixed Precision Training**: Uses FP16 precision for faster training with minimal accuracy loss
- **Automatic Batch Size Optimization**: Determines optimal batch size based on GPU memory
- **Expert Distribution**: Distributes experts across GPUs to maximize parallelism
- **Performance Monitoring**: Tracks and visualizes training performance

## GPU Parallelization Strategies

MORPH supports two main parallelization strategies for multi-GPU setups:

1. **Data Parallel**: Distributes batches across multiple GPUs
   - Each GPU processes a portion of the batch
   - Gradients are synchronized across GPUs
   - Simpler implementation, works well for most cases

2. **Expert Parallel**: Distributes experts across multiple GPUs
   - Each expert is assigned to a specific GPU
   - Inputs are routed to the appropriate GPU based on expert assignment
   - Better for models with many experts
   - More efficient use of GPU memory

## Running the GPU Training Example

The `gpu_training_example.py` script demonstrates how to use GPU acceleration with MORPH. It trains a model on the MNIST dataset and shows performance metrics.

### Command Line Arguments

- `--gpu-mode`: GPU mode to use (`auto`, `cpu`, `single_gpu`, `multi_gpu`)
- `--parallel-strategy`: Parallelization strategy for multi-GPU (`data_parallel`, `expert_parallel`)
- `--mixed-precision`: Enable mixed precision training
- `--batch-size`: Batch size for training
- `--auto-batch-size`: Automatically determine optimal batch size
- `--epochs`: Number of epochs to train
- `--experts`: Number of initial experts

### Example Usage

```bash
# Basic usage with automatic GPU detection
python examples/gpu_training_example.py

# Force CPU-only mode
python examples/gpu_training_example.py --gpu-mode cpu

# Use single GPU with mixed precision
python examples/gpu_training_example.py --gpu-mode single_gpu --mixed-precision

# Use multiple GPUs with expert parallel strategy
python examples/gpu_training_example.py --gpu-mode multi_gpu --parallel-strategy expert_parallel

# Use automatic batch size determination
python examples/gpu_training_example.py --auto-batch-size

# Specify number of experts and epochs
python examples/gpu_training_example.py --experts 16 --epochs 10
```

## Performance Considerations

### Memory Usage

- Each expert requires memory for its parameters and gradients
- The gating network requires memory proportional to the number of experts
- Activation storage for sleep cycles requires additional memory
- Mixed precision training can reduce memory usage by up to 50%

### Optimal Batch Size

- Larger batch sizes generally lead to faster training
- GPU memory limits the maximum batch size
- Use `--auto-batch-size` to automatically determine the optimal batch size
- For multi-GPU setups, the effective batch size is multiplied by the number of GPUs in data parallel mode

### Expert Distribution

- In expert parallel mode, experts are distributed across GPUs to balance the load
- The distribution is based on the number of experts assigned to each GPU
- The gating network always runs on the primary GPU
- Expert creation dynamically assigns new experts to the GPU with the fewest experts

## Visualizing Results

The example script generates several visualizations in the `results/gpu_training/` directory:

- `training_metrics.png`: Training loss, accuracy, and epoch time
- `expert_activations.png`: Expert activation patterns
- `expert_device_distribution.png`: Distribution of experts across devices (multi-GPU only)
- `knowledge_graph_epoch_*.png`: Knowledge graph after each epoch
- `performance_summary.txt`: Summary of training performance

## Configuration Options

The following configuration options in `MorphConfig` control GPU usage:

```python
# GPU Acceleration
device: str = "auto"  # "auto", "cuda", "cuda:n", or "cpu"
enable_mixed_precision: bool = False  # Whether to use mixed precision training
gpu_memory_fraction: float = 0.9  # Fraction of GPU memory to use
gpu_mode: str = "auto"  # "auto", "cpu", "single_gpu", "multi_gpu"
parallel_strategy: str = "data_parallel"  # "data_parallel", "expert_parallel"
auto_batch_size: bool = True  # Whether to automatically determine optimal batch size

# Data Loading Optimization
num_workers: int = -1  # Number of workers for DataLoader (-1 for auto)
pin_memory: bool = True  # Whether to use pinned memory for faster transfer
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce the batch size
2. Enable mixed precision training
3. Reduce the number of experts
4. Use expert parallel strategy for better memory distribution
5. Reduce the expert hidden size

### Slow Training

If training is slower than expected:

1. Ensure you're using CUDA-capable GPUs
2. Enable mixed precision training
3. Increase the batch size (if memory allows)
4. Increase the number of workers for data loading
5. Enable pin memory for faster data transfer

### Multi-GPU Issues

If you encounter issues with multi-GPU training:

1. Ensure all GPUs are of the same model for best performance
2. Try different parallelization strategies
3. Check that CUDA and PyTorch are properly installed
4. Verify that all GPUs are detected with `nvidia-smi`

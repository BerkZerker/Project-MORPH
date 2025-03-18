# GPU Training

This guide explains how to use GPU acceleration for training and running models. The implementation automatically detects available GPUs and can distribute the workload across multiple GPUs for improved performance.

## Features

- **Automatic GPU Detection**: Automatically detects available CUDA-capable GPUs
- **Multi-GPU Support**: Distributes workload across multiple GPUs
- **Mixed Precision Training**: Supports both FP16 and BF16 precision for faster training
- **Automatic Batch Size Optimization**: Determines optimal batch size based on GPU memory
- **Expert Distribution**: Distributes experts across GPUs to maximize parallelism
- **Performance Monitoring**: Tracks and visualizes training performance
- **Gradient Accumulation**: Enables larger effective batch sizes without increasing memory usage
- **Model Compilation**: Uses PyTorch 2.0+ compilation for faster execution
- **Memory Management**: Proactive cache clearing and memory optimization
- **GPU Monitoring**: Real-time monitoring of GPU memory usage

## GPU Parallelization Strategies

The framework supports two main parallelization strategies for multi-GPU setups:

1. **Data Parallel**: Distributes batches across multiple GPUs
   - Each GPU processes a portion of the batch
   - Gradients are synchronized across GPUs
   - Simpler implementation, works well for most cases

2. **Expert Parallel**: Distributes experts across multiple GPUs
   - Each expert is assigned to a specific GPU based on complexity and GPU capability
   - Inputs are routed to the appropriate GPU based on expert assignment
   - Better for models with many experts
   - More efficient use of GPU memory

## Running the GPU Training Example

The `gpu_training_example.py` script demonstrates how to use GPU acceleration. It trains a model on the MNIST dataset and shows performance metrics.

### Command Line Arguments

- `--gpu-mode`: GPU mode to use (`auto`, `cpu`, `single_gpu`, `multi_gpu`)
- `--parallel-strategy`: Parallelization strategy for multi-GPU (`data_parallel`, `expert_parallel`)
- `--mixed-precision`: Enable mixed precision training
- `--precision-type`: Precision type for mixed precision training (`auto`, `fp16`, `bf16`)
- `--batch-size`: Batch size for training
- `--auto-batch-size`: Automatically determine optimal batch size
- `--accumulation-steps`: Number of steps to accumulate gradients over
- `--epochs`: Number of epochs to train
- `--experts`: Number of initial experts
- `--compile`: Use torch.compile() for faster execution (PyTorch 2.0+)
- `--compile-mode`: Compilation mode (`default`, `reduce-overhead`, `max-autotune`)
- `--monitor-gpu`: Enable GPU memory monitoring

### Example Usage

```bash
# Basic usage with automatic GPU detection
python examples/gpu_training_example.py

# Force CPU-only mode
python examples/gpu_training_example.py --gpu-mode cpu

# Use single GPU with mixed precision (FP16)
python examples/gpu_training_example.py --gpu-mode single_gpu --mixed-precision --precision-type fp16

# Use BF16 precision (for Ampere+ GPUs)
python examples/gpu_training_example.py --mixed-precision --precision-type bf16

# Use gradient accumulation for larger effective batch size
python examples/gpu_training_example.py --batch-size 64 --accumulation-steps 4

# Use model compilation with PyTorch 2.0+
python examples/gpu_training_example.py --compile --compile-mode default

# Use multiple GPUs with expert parallel strategy
python examples/gpu_training_example.py --gpu-mode multi_gpu --parallel-strategy expert_parallel

# Use automatic batch size determination
python examples/gpu_training_example.py --auto-batch-size

# Enable GPU memory monitoring
python examples/gpu_training_example.py --monitor-gpu

# Full optimization example
python examples/gpu_training_example.py --mixed-precision --precision-type auto --compile --accumulation-steps 2 --monitor-gpu
```

## Advanced GPU Optimizations

### Mixed Precision Training

Mixed precision training uses lower precision formats (FP16 or BF16) to reduce memory usage and increase computational throughput:

- **FP16 (Half Precision)**: Works on all CUDA GPUs, but requires gradient scaling to prevent underflow
- **BF16 (Brain Float)**: Available on Ampere+ GPUs (compute capability 8.0+), provides better numerical stability
- **Auto Selection**: When using `--precision-type auto`, the system automatically selects the best precision format based on your GPU capabilities

### Gradient Accumulation

Gradient accumulation allows training with larger effective batch sizes without increasing memory usage:

- Accumulates gradients over multiple forward/backward passes before updating weights
- Effective batch size = batch_size × accumulation_steps
- Useful for training with limited GPU memory
- Enables using larger models or batch sizes than would otherwise fit in memory

### Model Compilation

PyTorch 2.0+ includes a just-in-time (JIT) compiler that can significantly accelerate model execution:

- **Default Mode**: Balances compilation time and runtime performance
- **Reduce-Overhead**: Minimizes runtime overhead for faster execution
- **Max-Autotune**: Maximizes performance through autotuning (slower compilation)

Compilation is applied to both experts and the gating network for maximum performance.

### Memory Management

The framework includes several memory optimization techniques:

- **Proactive Cache Clearing**: Clears GPU cache at strategic points to free memory
- **Tensor Memory Optimization**: Ensures tensors are contiguous for better memory efficiency
- **Non-Blocking Transfers**: Uses asynchronous data transfers between CPU and GPU
- **Periodic Cache Clearing**: Clears cache during long evaluations to prevent memory buildup

### GPU Monitoring

Real-time GPU memory monitoring helps identify memory bottlenecks and optimize performance:

- Tracks memory allocation and usage during training
- Logs memory statistics at regular intervals
- Helps identify memory leaks and inefficient operations

## Performance Considerations

### Memory Usage

- Each expert requires memory for its parameters and gradients
- The gating network requires memory proportional to the number of experts
- Activation storage for sleep cycles requires additional memory
- Mixed precision training can reduce memory usage by up to 50%
- Gradient accumulation can reduce peak memory usage

### Optimal Batch Size

- Larger batch sizes generally lead to faster training
- GPU memory limits the maximum batch size
- Use `--auto-batch-size` to automatically determine the optimal batch size
- For multi-GPU setups, the effective batch size is multiplied by the number of GPUs in data parallel mode
- Use gradient accumulation to increase effective batch size without increasing memory usage

### Expert Distribution

- In expert parallel mode, experts are distributed across GPUs to balance the load
- The distribution considers expert complexity and GPU capabilities
- More complex experts are assigned to more powerful GPUs
- The gating network always runs on the primary GPU
- Expert creation dynamically assigns new experts to the GPU with the fewest experts

## Visualizing Results

The example script generates several visualizations in the `results/gpu_training/` directory:

- `training_metrics.png`: Training loss, accuracy, and epoch time
- `expert_activations.png`: Expert activation patterns
- `expert_device_distribution.png`: Distribution of experts across devices (multi-GPU only)
- `knowledge_graph_epoch_*.png`: Knowledge graph after each epoch
- `performance_summary.txt`: Summary of training performance including compilation status

## Configuration Options

The following configuration options in `Config` control GPU usage:

```python
# GPU Acceleration
device: str = "auto"  # "auto", "cuda", "cuda:n", or "cpu"
enable_mixed_precision: bool = False  # Whether to use mixed precision training
precision_type: str = "auto"  # "auto", "fp16", or "bf16"
gpu_memory_fraction: float = 0.9  # Fraction of GPU memory to use
gpu_mode: str = "auto"  # "auto", "cpu", "single_gpu", "multi_gpu"
parallel_strategy: str = "data_parallel"  # "data_parallel", "expert_parallel"
auto_batch_size: bool = True  # Whether to automatically determine optimal batch size
accumulation_steps: int = 1  # Number of steps to accumulate gradients over

# Data Loading Optimization
num_workers: int = -1  # Number of workers for DataLoader (-1 for auto)
pin_memory: bool = True  # Whether to use pinned memory for faster transfer
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce the batch size
2. Enable mixed precision training
3. Use gradient accumulation to maintain effective batch size
4. Clear GPU cache more frequently
5. Reduce the number of experts
6. Use expert parallel strategy for better memory distribution
7. Reduce the expert hidden size

### Slow Training

If training is slower than expected:

1. Ensure you're using CUDA-capable GPUs
2. Enable mixed precision training with appropriate precision type
3. Use model compilation with PyTorch 2.0+
4. Increase the batch size (if memory allows)
5. Optimize the number of workers for data loading
6. Enable pin memory for faster data transfer
7. Use non-blocking transfers for CPU-GPU data movement

### Multi-GPU Issues

If you encounter issues with multi-GPU training:

1. Ensure all GPUs are of the same model for best performance
2. Try different parallelization strategies
3. Check that CUDA and PyTorch are properly installed
4. Verify that all GPUs are detected with `nvidia-smi`
5. Consider expert complexity when distributing experts

### Compilation Issues

If you encounter issues with model compilation:

1. Ensure you're using PyTorch 2.0 or newer
2. Try different compilation modes
3. Compile only specific components (experts or gating)
4. Check for dynamic shapes that might cause compilation issues
5. Verify CUDA compatibility with your PyTorch version

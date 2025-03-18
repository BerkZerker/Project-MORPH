import torch
import logging
import os
import numpy as np
import time
import gc
from typing import Tuple, List, Dict, Optional, Union, Callable


def clear_gpu_cache(device_indices: Optional[List[int]] = None) -> bool:
    """
    Clear GPU cache to free up memory.
    
    Args:
        device_indices: List of GPU indices to clear cache for.
                       If None, clears cache for all available GPUs.
    
    Returns:
        True if cache was cleared, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        if device_indices is None:
            # Clear cache for all GPUs
            torch.cuda.empty_cache()
            return True
        else:
            # Clear cache for specific GPUs
            for idx in device_indices:
                if idx < torch.cuda.device_count():
                    # Store current device
                    current_device = torch.cuda.current_device()
                    # Set device to clear
                    torch.cuda.set_device(idx)
                    # Clear cache
                    torch.cuda.empty_cache()
                    # Restore device
                    torch.cuda.set_device(current_device)
            return True
    except Exception as e:
        logging.warning(f"Failed to clear GPU cache: {e}")
        return False


def detect_available_gpus() -> List[int]:
    """
    Detect and return a list of available GPU indices.
    
    Returns:
        List of available GPU indices
    """
    if not torch.cuda.is_available():
        return []
    
    n_gpus = torch.cuda.device_count()
    return list(range(n_gpus))


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """
    Get memory information for all available GPUs.
    
    Returns:
        Dictionary mapping GPU indices to memory info dictionaries
        with 'total' and 'free' memory in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for gpu_idx in detect_available_gpus():
        try:
            # Get memory info in bytes
            total_memory = torch.cuda.get_device_properties(gpu_idx).total_memory
            # Get allocated memory in bytes
            allocated_memory = torch.cuda.memory_allocated(gpu_idx)
            # Get cached memory in bytes
            cached_memory = torch.cuda.memory_reserved(gpu_idx)
            
            # Calculate free memory
            free_memory = total_memory - allocated_memory - cached_memory
            
            # Convert to GB for readability
            memory_info[gpu_idx] = {
                'total': total_memory / (1024**3),  # Convert to GB
                'free': free_memory / (1024**3),    # Convert to GB
                'used': (allocated_memory + cached_memory) / (1024**3)  # Convert to GB
            }
        except Exception as e:
            logging.warning(f"Failed to get memory info for GPU {gpu_idx}: {e}")
            memory_info[gpu_idx] = {'total': 0.0, 'free': 0.0, 'used': 0.0}
    
    return memory_info


def select_best_gpu() -> Optional[int]:
    """
    Select the GPU with the most available memory.
    
    Returns:
        Index of the GPU with most free memory, or None if no GPUs are available
    """
    if not torch.cuda.is_available():
        return None
    
    memory_info = get_gpu_memory_info()
    if not memory_info:
        return None
    
    # Find GPU with most free memory
    best_gpu = max(memory_info.keys(), key=lambda gpu_idx: memory_info[gpu_idx]['free'])
    return best_gpu


def detect_and_select_gpu() -> Tuple[str, List[torch.device]]:
    """
    Detect available GPU resources and select the optimal configuration.
    
    Returns:
        Tuple of (mode, devices) where:
            - mode: One of 'cpu', 'single_gpu', 'multi_gpu'
            - devices: List of torch.device objects to use
    """
    if not torch.cuda.is_available():
        logging.info("No CUDA-capable GPUs detected, using CPU")
        return "cpu", [torch.device("cpu")]
    
    # Count available GPUs
    gpu_count = torch.cuda.device_count()
    logging.info(f"Detected {gpu_count} CUDA-capable GPUs")
    
    if gpu_count == 0:
        return "cpu", [torch.device("cpu")]
    elif gpu_count == 1:
        return "single_gpu", [torch.device("cuda:0")]
    else:
        # Multi-GPU setup
        return "multi_gpu", [torch.device(f"cuda:{i}") for i in range(gpu_count)]


def estimate_max_batch_size(
    model: torch.nn.Module, 
    input_shape: Tuple[int, ...], 
    device: torch.device,
    max_memory_fraction: float = 0.8
) -> int:
    """
    Estimate the maximum batch size that can fit in GPU memory.
    
    Args:
        model: The PyTorch model
        input_shape: Shape of a single input sample (without batch dimension)
        device: The device to use
        max_memory_fraction: Maximum fraction of GPU memory to use
        
    Returns:
        Estimated maximum batch size
    """
    if device.type != 'cuda':
        # For CPU, return a reasonable default
        return 64
    
    try:
        # Get GPU memory info
        gpu_idx = device.index
        memory_info = get_gpu_memory_info()[gpu_idx]
        free_memory_gb = memory_info['free'] * max_memory_fraction
        
        # Create a sample input to measure memory usage
        sample_batch_size = 4  # Start with a small batch
        sample_input = torch.zeros((sample_batch_size,) + input_shape, device=device)
        
        # Record memory before forward pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(gpu_idx)
        
        # Run forward pass
        with torch.no_grad():
            _ = model(sample_input)
        
        # Get peak memory usage
        peak_memory_bytes = torch.cuda.max_memory_allocated(gpu_idx)
        peak_memory_gb = peak_memory_bytes / (1024**3)
        
        # Calculate memory per sample
        memory_per_sample_gb = peak_memory_gb / sample_batch_size
        
        # Estimate max batch size
        max_batch_size = int(free_memory_gb / memory_per_sample_gb)
        
        # Ensure batch size is at least 1
        max_batch_size = max(1, max_batch_size)
        
        logging.info(f"Estimated max batch size: {max_batch_size} on device {device}")
        return max_batch_size
    
    except Exception as e:
        logging.warning(f"Failed to estimate max batch size: {e}")
        # Return a conservative default
        return 16


def distribute_experts_across_gpus(
    num_experts: int, 
    devices: List[torch.device]
) -> Dict[int, torch.device]:
    """
    Distribute experts evenly across available GPUs.
    
    Args:
        num_experts: Number of experts to distribute
        devices: List of available devices
        
    Returns:
        Dictionary mapping expert indices to devices
    """
    if not devices or all(d.type == 'cpu' for d in devices):
        # All CPU devices, assign all experts to CPU
        return {i: torch.device('cpu') for i in range(num_experts)}
    
    # Filter to only include CUDA devices
    cuda_devices = [d for d in devices if d.type == 'cuda']
    
    if not cuda_devices:
        return {i: torch.device('cpu') for i in range(num_experts)}
    
    # Distribute experts evenly
    expert_devices = {}
    for i in range(num_experts):
        device_idx = i % len(cuda_devices)
        expert_devices[i] = cuda_devices[device_idx]
    
    return expert_devices


def set_gpu_memory_fraction(fraction: float = 0.9) -> None:
    """
    Set the fraction of GPU memory to use.
    
    Args:
        fraction: Fraction of GPU memory to use (0.0 to 1.0)
    """
    if not torch.cuda.is_available():
        return
    
    try:
        # This is a PyTorch 2.0+ feature
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(fraction, device)
        logging.info(f"Set GPU memory fraction to {fraction}")
    except AttributeError:
        logging.warning("PyTorch version does not support setting memory fraction directly")
        # For older PyTorch versions, we can't directly set memory fraction
        pass


def setup_gpu_environment() -> None:
    """
    Set up the GPU environment with optimal settings.
    """
    if not torch.cuda.is_available():
        return
    
    # Enable TF32 precision on Ampere or newer GPUs
    # This provides better performance with minimal accuracy loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmark mode for optimized performance
    torch.backends.cudnn.benchmark = True
    
    # Set reasonable memory fraction
    set_gpu_memory_fraction(0.9)
    
    # Clear GPU cache to start with clean memory
    clear_gpu_cache()
    
    # Run garbage collection to free memory
    gc.collect()
    
    # Log GPU information
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logging.info(f"GPU {i}: {props.name} with {props.total_memory / (1024**3):.2f} GB memory")
        
        # Log compute capability
        logging.info(f"  Compute capability: {props.major}.{props.minor}")
        
        # Log memory bandwidth (estimate) - skipping as memory_clock_rate is not available in all PyTorch versions
        try:
            if hasattr(props, 'memory_clock_rate'):
                memory_clock = props.memory_clock_rate / (10**6)  # MHz
                bus_width = props.memory_bus_width
                bandwidth = 2 * memory_clock * (bus_width / 8) / 1000  # GB/s
                logging.info(f"  Memory bandwidth: ~{bandwidth:.1f} GB/s")
        except Exception as e:
            logging.debug(f"Could not estimate memory bandwidth: {e}")
        
        # Log CUDA cores (estimate based on compute capability)
        if props.major >= 8:  # Ampere or newer
            cores_per_sm = 128
        elif props.major == 7:  # Volta/Turing
            cores_per_sm = 64
        else:  # Pascal or older
            cores_per_sm = 128
        cuda_cores = props.multi_processor_count * cores_per_sm
        logging.info(f"  CUDA cores: ~{cuda_cores}")


def get_optimal_worker_count(batch_size: Optional[int] = None, dataset_size: Optional[int] = None) -> int:
    """
    Get the optimal number of worker processes for data loading based on
    hardware and data characteristics.
    
    Args:
        batch_size: Size of each batch
        dataset_size: Total dataset size
    
    Returns:
        Recommended number of worker processes
    """
    try:
        # Use CPU count as a heuristic
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # GPU-specific optimization
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            # Base workers on GPU:CPU ratio and GPU compute capability
            total_compute_units = 0
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                # Use multiprocessor count as a proxy for GPU power
                total_compute_units += props.multi_processor_count
            
            # Calculate optimal workers (more powerful GPUs need more workers)
            # Use a heuristic scaling factor based on empirical testing
            worker_ratio = min(1.0, total_compute_units / (cpu_count * 24))
            workers = max(1, int(cpu_count * worker_ratio))
            
            # Adjust based on batch size and dataset size if provided
            if batch_size and dataset_size:
                # Fewer workers for small datasets or large batches
                dataset_factor = min(1.0, dataset_size / (batch_size * 1000))
                workers = max(1, int(workers * dataset_factor))
        else:
            # CPU-only mode - use half of CPU cores for workers
            workers = max(1, cpu_count // 2)
        
        # Cap at CPU count - 1 to leave a core for the main process
        workers = min(workers, max(1, cpu_count - 1))
        
        logging.info(f"Using {workers} dataloader workers")
        return workers
    except Exception as e:
        logging.warning(f"Failed to determine optimal worker count: {e}")
        # Default fallback
        return 4


def monitor_gpu_memory(interval: float = 5.0, callback: Optional[Callable] = None) -> None:
    """
    Start a background thread to monitor GPU memory usage.
    
    Args:
        interval: Monitoring interval in seconds
        callback: Optional callback function to call with memory info
    """
    if not torch.cuda.is_available():
        logging.warning("No CUDA-capable GPUs detected, memory monitoring disabled")
        return
    
    import threading
    
    def _monitor_thread():
        while True:
            try:
                memory_info = get_gpu_memory_info()
                
                # Log memory usage
                for gpu_idx, info in memory_info.items():
                    usage_percent = (info['used'] / info['total']) * 100
                    logging.info(f"GPU {gpu_idx}: {info['used']:.2f}/{info['total']:.2f} GB "
                                f"({usage_percent:.1f}% used, {info['free']:.2f} GB free)")
                
                # Call callback if provided
                if callback:
                    callback(memory_info)
                
                # Sleep for the specified interval
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Error in GPU memory monitoring: {e}")
                break
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=_monitor_thread, daemon=True)
    monitor_thread.start()
    logging.info(f"Started GPU memory monitoring (interval: {interval}s)")


def optimize_tensor_memory(tensor: torch.Tensor) -> torch.Tensor:
    """
    Optimize a tensor for memory efficiency.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Memory-optimized tensor
    """
    # Contiguous tensors are more memory-efficient
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    return tensor


def gradient_accumulation_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    accumulation_steps: int,
    current_step: int,
    retain_graph: bool = False
) -> None:
    """
    Perform a gradient accumulation step.
    
    Args:
        loss: Loss tensor
        optimizer: Optimizer
        scaler: Optional gradient scaler for mixed precision
        accumulation_steps: Number of steps to accumulate gradients over
        current_step: Current step index
        retain_graph: Whether to retain computation graph
    """
    # Scale loss by accumulation steps
    scaled_loss = loss / accumulation_steps
    
    # Backward pass
    if scaler is not None:
        # Mixed precision path
        scaler.scale(scaled_loss).backward(retain_graph=retain_graph)
        
        # Only step optimizer and update scaler after accumulation
        if (current_step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        # Standard precision path
        scaled_loss.backward(retain_graph=retain_graph)
        
        # Only step optimizer after accumulation
        if (current_step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

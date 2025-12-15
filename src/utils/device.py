"""
Device management utilities.

Handles GPU detection and device selection for PyTorch.
"""

from typing import Optional, List, Tuple
import torch

from dtypes import Device


def get_device(
    preferred: Optional[str] = None,
    gpu_id: Optional[int] = None
) -> torch.device:
    """
    Get the best available compute device.

    Args:
        preferred: Preferred device string ('cuda', 'cpu', 'mps')
        gpu_id: Specific GPU ID to use (for multi-GPU systems)

    Returns:
        PyTorch device object
    """
    if preferred is not None:
        if preferred == "cuda":
            if torch.cuda.is_available():
                if gpu_id is not None:
                    return torch.device(f"cuda:{gpu_id}")
                return torch.device("cuda")
            else:
                print("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
        elif preferred == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                print("MPS requested but not available, falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    # Auto-detect best device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_available_gpus() -> List[Tuple[int, str, int]]:
    """
    Get list of available GPUs with their properties.

    Returns:
        List of tuples (gpu_id, name, memory_mb)
    """
    gpus = []

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_mb = props.total_memory // (1024 * 1024)
            gpus.append((i, props.name, memory_mb))

    return gpus


def set_random_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device_memory_info(device: torch.device) -> dict:
    """
    Get memory information for a device.

    Args:
        device: PyTorch device

    Returns:
        Dictionary with memory statistics
    """
    info = {"device": str(device)}

    if device.type == "cuda":
        info["allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
        info["cached_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
        info["max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        props = torch.cuda.get_device_properties(device)
        info["total_mb"] = props.total_memory / (1024 * 1024)
        info["name"] = props.name

    return info


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

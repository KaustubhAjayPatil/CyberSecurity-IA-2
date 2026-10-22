"""Reproducibility utilities for ensuring deterministic results."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> int:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value. If None, uses a random seed.
        
    Returns:
        The seed value used
    """
    if seed is None:
        seed = 42

    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def get_device() -> torch.device:
    """Get the appropriate device (CPU or GPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    else:
        device = torch.device("cpu")

    return device

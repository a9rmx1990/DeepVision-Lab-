"""
utils/helpers.py
================
Shared utility functions used across multiple DeepVisionLab modules.

Keep this module lean — only truly cross-cutting helpers belong here.
Domain-specific logic should live in the relevant module (dataset,
models, trainer, etc.).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Parameters
    ----------
    seed:
        Seed value.  Default ``42``.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed — skip


def get_device() -> Any:
    """Return a ``torch.device`` — CUDA if available, otherwise CPU.

    Returns
    -------
    torch.device

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for get_device(). "
            "Install via: pip install torch"
        ) from exc

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: Any) -> Dict[str, int]:
    """Count trainable and total parameters for a PyTorch model.

    Parameters
    ----------
    model:
        ``torch.nn.Module`` instance.

    Returns
    -------
    Dict[str, int]
        Dictionary with ``"trainable"`` and ``"total"`` counts.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def format_time(seconds: float) -> str:
    """Format elapsed time in a human-readable string.

    Parameters
    ----------
    seconds:
        Time in seconds.

    Returns
    -------
    str
        Formatted string like ``"2m 35s"`` or ``"1h 5m 12s"``.
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def is_pytorch_model(model: Any) -> bool:
    """Check if *model* is a PyTorch ``nn.Module``.

    Parameters
    ----------
    model:
        Any object.

    Returns
    -------
    bool
    """
    try:
        import torch.nn as nn
        return isinstance(model, nn.Module)
    except ImportError:
        return False


def is_sklearn_model(model: Any) -> bool:
    """Check if *model* is a scikit-learn ``BaseEstimator``.

    Parameters
    ----------
    model:
        Any object.

    Returns
    -------
    bool
    """
    try:
        from sklearn.base import BaseEstimator
        return isinstance(model, BaseEstimator)
    except ImportError:
        return False

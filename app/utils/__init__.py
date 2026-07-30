"""
app.utils
=========
Shared utilities: logging, seeding, device helpers, and common functions.

Exposes:
- ``setup_logger``: Configure root/named logger with file + console output.
- ``get_logger``: Convenience wrapper for ``logging.getLogger``.
- ``set_seed``: Seed Python, NumPy, and PyTorch for reproducibility.
- ``get_device``: Auto-detect CUDA / CPU device.
- ``count_parameters``: Count trainable / total parameters in a PyTorch model.
- ``format_time``: Format seconds into human-readable string.
- ``is_pytorch_model``: Check if an object is a PyTorch nn.Module.
- ``is_sklearn_model``: Check if an object is a scikit-learn BaseEstimator.
"""

from app.utils.logger import get_logger, setup_logger
from app.utils.helpers import (
    count_parameters,
    format_time,
    get_device,
    is_pytorch_model,
    is_sklearn_model,
    set_seed,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "set_seed",
    "get_device",
    "count_parameters",
    "format_time",
    "is_pytorch_model",
    "is_sklearn_model",
]

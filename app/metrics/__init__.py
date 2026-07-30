"""
app.metrics
===========
Evaluation metrics for classification, regression, and clustering tasks.

Exposes:
- ``MetricsCalculator``: Unified interface for computing task-appropriate metrics.

Usage::

    from app.metrics import MetricsCalculator

    mc = MetricsCalculator()
    results = mc.compute("classification", y_true, y_pred, y_proba=probs)
"""

from app.metrics.calculator import MetricsCalculator

__all__ = [
    "MetricsCalculator",
]

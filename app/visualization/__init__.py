"""
app.visualization
=================
Training and evaluation visualizations using Matplotlib / Seaborn.

All functions accept data as arguments and return ``matplotlib.figure.Figure``
objects.  They do not know about Streamlit, file paths, or any UI framework.

Exposes:
- ``plot_loss_curve``: Train vs. validation loss across epochs.
- ``plot_accuracy_curve``: Train vs. validation accuracy across epochs.
- ``plot_confusion_matrix``: Confusion matrix heatmap.
- ``plot_roc_curve``: Receiver Operating Characteristic curve.
- ``plot_precision_recall_curve``: Precision–Recall curve.
- ``plot_prediction_vs_actual``: Scatter plot for regression tasks.
"""

from app.visualization.plots import (
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_precision_recall_curve,
    plot_prediction_vs_actual,
    plot_roc_curve,
)

__all__ = [
    "plot_loss_curve",
    "plot_accuracy_curve",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_prediction_vs_actual",
]

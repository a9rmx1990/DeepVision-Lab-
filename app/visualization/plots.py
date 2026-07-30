"""
visualization/plots.py
======================
Generate publication-ready training and evaluation plots.

All functions accept data / metrics as arguments and **return**
``matplotlib.figure.Figure`` objects.  They should **not** know about
Streamlit, file I/O paths, or any UI framework — callers decide how to
display or save the returned figure.

Plots defined (per ``skill.md``):

- Loss curve (train vs. validation)
- Accuracy curve (train vs. validation)
- Confusion matrix (heatmap)
- ROC curve (per-class for multiclass)
- Precision–Recall curve (per-class for multiclass)

Usage::

    from app.visualization import plot_loss_curve
    fig = plot_loss_curve(train_losses, val_losses)
    fig.savefig("loss_curve.png")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for servers / CI
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

_DEFAULT_FIGSIZE: Tuple[int, int] = (8, 6)


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    *,
    title: str = "Loss Curve",
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
) -> plt.Figure:
    """Plot training and validation loss across epochs.

    Parameters
    ----------
    train_losses:
        Per-epoch training loss values.
    val_losses:
        Per-epoch validation loss values.
    title:
        Plot title.
    figsize:
        Figure size ``(width, height)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Accuracy curve
# ---------------------------------------------------------------------------

def plot_accuracy_curve(
    train_accuracies: Sequence[float],
    val_accuracies: Sequence[float],
    *,
    title: str = "Accuracy Curve",
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
) -> plt.Figure:
    """Plot training and validation accuracy across epochs.

    Parameters
    ----------
    train_accuracies:
        Per-epoch training accuracy values.
    val_accuracies:
        Per-epoch validation accuracy values.
    title:
        Plot title.
    figsize:
        Figure size ``(width, height)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(train_accuracies) + 1)

    ax.plot(epochs, train_accuracies, label="Train Accuracy", linewidth=2)
    ax.plot(epochs, val_accuracies, label="Val Accuracy", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    class_names: Optional[Sequence[str]] = None,
    *,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
    normalize: bool = False,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground truth labels.
    y_pred:
        Predicted labels.
    class_names:
        Display names for each class.  When ``None``, uses unique labels
        found in *y_true*.
    title:
        Plot title.
    figsize:
        Figure size.
    normalize:
        If ``True``, normalise each row to sum to 1 (shows proportions
        instead of counts).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(y_true, y_pred)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes with no samples
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm.astype(float) / row_sums

    if class_names is None:
        class_names = [str(c) for c in sorted(set(y_true))]

    fig, ax = plt.subplots(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: Any,
    y_proba: Any,
    class_names: Optional[Sequence[str]] = None,
    *,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
) -> plt.Figure:
    """Plot Receiver Operating Characteristic curve(s).

    For binary classification, a single curve is drawn.  For multiclass,
    one curve per class (one-vs-rest) is drawn.

    Parameters
    ----------
    y_true:
        Ground truth labels.
    y_proba:
        Predicted probabilities, shape ``(n_samples, n_classes)``.
    class_names:
        Display names per class.
    title:
        Plot title.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    y_proba = np.asarray(y_proba)
    classes = np.unique(y_true)

    fig, ax = plt.subplots(figsize=figsize)

    if len(classes) == 2:
        proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc_val = roc_auc_score(y_true, proba)
        label = class_names[1] if class_names and len(class_names) > 1 else "Positive"
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc_val:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            try:
                auc_val = roc_auc_score(y_bin[:, i], y_proba[:, i])
            except ValueError:
                auc_val = 0.0
            label = class_names[i] if class_names else str(cls)
            ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall curve
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(
    y_true: Any,
    y_proba: Any,
    class_names: Optional[Sequence[str]] = None,
    *,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
) -> plt.Figure:
    """Plot Precision–Recall curve(s).

    Parameters
    ----------
    y_true:
        Ground truth labels.
    y_proba:
        Predicted probabilities, shape ``(n_samples, n_classes)``.
    class_names:
        Display names per class.
    title:
        Plot title.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve as pr_curve
    from sklearn.preprocessing import label_binarize

    y_proba = np.asarray(y_proba)
    classes = np.unique(y_true)

    fig, ax = plt.subplots(figsize=figsize)

    if len(classes) == 2:
        proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        prec, rec, _ = pr_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        label = class_names[1] if class_names and len(class_names) > 1 else "Positive"
        ax.plot(rec, prec, linewidth=2, label=f"{label} (AP = {ap:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            prec, rec, _ = pr_curve(y_bin[:, i], y_proba[:, i])
            try:
                ap = average_precision_score(y_bin[:, i], y_proba[:, i])
            except ValueError:
                ap = 0.0
            label = class_names[i] if class_names else str(cls)
            ax.plot(rec, prec, linewidth=2, label=f"{label} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regression plots (bonus — useful for regression tasks)
# ---------------------------------------------------------------------------

def plot_prediction_vs_actual(
    y_true: Any,
    y_pred: Any,
    *,
    title: str = "Predicted vs Actual",
    figsize: Tuple[int, int] = _DEFAULT_FIGSIZE,
) -> plt.Figure:
    """Scatter plot of predicted vs actual values for regression tasks.

    Parameters
    ----------
    y_true:
        Ground truth values.
    y_pred:
        Predicted values.
    title:
        Plot title.
    figsize:
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)

    # Perfect prediction line
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Prediction")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

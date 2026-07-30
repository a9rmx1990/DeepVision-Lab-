"""
metrics/calculator.py
=====================
Compute evaluation metrics for classification, regression, and clustering
tasks.

The ``MetricsCalculator`` provides a unified interface so that callers
(trainer, UI, inference) always get a consistent dict of metrics without
reimplementing sklearn metric calls.

Metric set (per ``skill.md``):

=================  ======================================
Task               Metrics
=================  ======================================
Classification     Accuracy, Precision, Recall, F1 Score
Regression         MAE, MSE, RMSE, R²
Clustering         Silhouette Score
=================  ======================================

Usage::

    from app.metrics import MetricsCalculator

    mc = MetricsCalculator()
    results = mc.compute("classification", y_true, y_pred)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.config.constants import (
    TASK_CLASSIFICATION,
    TASK_CLUSTERING,
    TASK_REGRESSION,
    TASK_TIME_SERIES_FORECASTING,
)


class MetricsCalculator:
    """Compute evaluation metrics for all supported ML tasks.

    This class is stateless — all data is passed through method arguments.
    """

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        task: str,
        y_true: Any,
        y_pred: Any,
        y_proba: Optional[Any] = None,
        X: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute metrics appropriate for *task*.

        Parameters
        ----------
        task:
            ML task string (matches ``config/constants.py`` values).
        y_true:
            Ground truth labels / values.  May be ``None`` for clustering.
        y_pred:
            Model predictions or cluster labels.
        y_proba:
            Predicted class probabilities (classification only, optional).
        X:
            Feature matrix (clustering only, needed for silhouette).

        Returns
        -------
        Dict[str, Any]
            Metric name → value mapping.

        Raises
        ------
        ValueError
            If *task* is not recognised.
        """
        task_clean = task.lower().strip()

        if task_clean == TASK_CLASSIFICATION:
            return self.classification_metrics(y_true, y_pred, y_proba)
        if task_clean in (TASK_REGRESSION, TASK_TIME_SERIES_FORECASTING):
            return self.regression_metrics(y_true, y_pred)
        if task_clean == TASK_CLUSTERING:
            return self.clustering_metrics(X, y_pred)

        raise ValueError(
            f"Unknown task '{task}'. Expected one of: "
            f"{TASK_CLASSIFICATION}, {TASK_REGRESSION}, {TASK_CLUSTERING}, "
            f"{TASK_TIME_SERIES_FORECASTING}."
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def classification_metrics(
        y_true: Any,
        y_pred: Any,
        y_proba: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute classification metrics.

        Parameters
        ----------
        y_true:
            Ground truth labels.
        y_pred:
            Predicted labels.
        y_proba:
            Predicted class probabilities (optional).  Used for ROC /
            Precision-Recall curve data.

        Returns
        -------
        Dict[str, Any]
            Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
            ``confusion_matrix``, and optionally ``roc_data`` /
            ``pr_data`` when *y_proba* is provided.
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        # ROC curve data (per-class for multiclass)
        if y_proba is not None:
            metrics["roc_data"] = MetricsCalculator._compute_roc_data(
                y_true, y_proba,
            )
            metrics["pr_data"] = MetricsCalculator._compute_pr_data(
                y_true, y_proba,
            )

        return metrics

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    @staticmethod
    def regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
        """Compute regression metrics.

        Parameters
        ----------
        y_true:
            Ground truth values.
        y_pred:
            Predicted values.

        Returns
        -------
        Dict[str, float]
            Keys: ``mae``, ``mse``, ``rmse``, ``r2``.
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    @staticmethod
    def clustering_metrics(
        X: Any,
        labels: Any,
    ) -> Dict[str, float]:
        """Compute clustering metrics.

        Parameters
        ----------
        X:
            Feature matrix used for clustering.
        labels:
            Cluster assignments (from ``model.predict()`` or
            ``model.labels_``).

        Returns
        -------
        Dict[str, float]
            Keys: ``silhouette``.  Value is ``0.0`` if silhouette cannot
            be computed (fewer than 2 clusters or degenerate input).
        """
        from sklearn.metrics import silhouette_score

        unique_labels = set(labels) if hasattr(labels, '__iter__') else {labels}
        n_labels = len(unique_labels)

        if X is not None and 2 <= n_labels < len(X):
            score = float(silhouette_score(X, labels))
        else:
            score = 0.0

        return {"silhouette": score}

    # ------------------------------------------------------------------
    # Private helpers for curve data
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_roc_data(
        y_true: Any, y_proba: Any,
    ) -> Dict[str, Any]:
        """Compute per-class ROC curve data.

        Returns
        -------
        Dict[str, Any]
            ``{"fpr": {...}, "tpr": {...}, "auc": {...}}`` keyed by
            class label (string).
        """
        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.preprocessing import label_binarize

        y_proba = np.asarray(y_proba)
        classes = np.unique(y_true)

        if len(classes) == 2:
            # Binary: use column 1 probabilities
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc_val = float(roc_auc_score(y_true, proba))
            return {
                "fpr": {str(classes[1]): fpr.tolist()},
                "tpr": {str(classes[1]): tpr.tolist()},
                "auc": {str(classes[1]): auc_val},
            }

        # Multiclass: one-vs-rest
        y_bin = label_binarize(y_true, classes=classes)
        result: Dict[str, Any] = {"fpr": {}, "tpr": {}, "auc": {}}

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            result["fpr"][str(cls)] = fpr.tolist()
            result["tpr"][str(cls)] = tpr.tolist()
            try:
                result["auc"][str(cls)] = float(
                    roc_auc_score(y_bin[:, i], y_proba[:, i])
                )
            except ValueError:
                result["auc"][str(cls)] = 0.0

        return result

    @staticmethod
    def _compute_pr_data(
        y_true: Any, y_proba: Any,
    ) -> Dict[str, Any]:
        """Compute per-class Precision-Recall curve data.

        Returns
        -------
        Dict[str, Any]
            ``{"precision": {...}, "recall": {...}, "ap": {...}}`` keyed
            by class label (string).
        """
        from sklearn.metrics import average_precision_score, precision_recall_curve
        from sklearn.preprocessing import label_binarize

        y_proba = np.asarray(y_proba)
        classes = np.unique(y_true)

        if len(classes) == 2:
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            prec, rec, _ = precision_recall_curve(y_true, proba)
            ap = float(average_precision_score(y_true, proba))
            return {
                "precision": {str(classes[1]): prec.tolist()},
                "recall": {str(classes[1]): rec.tolist()},
                "ap": {str(classes[1]): ap},
            }

        y_bin = label_binarize(y_true, classes=classes)
        result: Dict[str, Any] = {"precision": {}, "recall": {}, "ap": {}}

        for i, cls in enumerate(classes):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            result["precision"][str(cls)] = prec.tolist()
            result["recall"][str(cls)] = rec.tolist()
            try:
                result["ap"][str(cls)] = float(
                    average_precision_score(y_bin[:, i], y_proba[:, i])
                )
            except ValueError:
                result["ap"][str(cls)] = 0.0

        return result

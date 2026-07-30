"""
trainer/validator.py
====================
Validation logic for evaluating models during training.

.. note::
    This is the **training validator** — it evaluates model performance
    on held-out data *during* the training loop.  It is distinct from
    ``dataset/validator.py`` which validates raw dataset integrity
    *before* training.

Provides:
- ``validate_epoch`` — run one PyTorch evaluation pass over a DataLoader.
- ``validate_sklearn`` — evaluate a fitted scikit-learn model on (X, y).

Both methods return a plain ``dict`` of metric names → values so callers
can log, compare, or feed them into ``EarlyStopping`` without coupling
to any specific metrics library.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.config.constants import (
    TASK_CLASSIFICATION,
    TASK_CLUSTERING,
    TASK_REGRESSION,
    TASK_TIME_SERIES_FORECASTING,
)


class TrainingValidator:
    """Evaluate models during training.

    This class is stateless — each method receives all the context it
    needs through its arguments.
    """

    # ------------------------------------------------------------------
    # PyTorch validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_epoch(
        model: Any,
        dataloader: Any,
        criterion: Any,
        device: Any,
        task: str = TASK_CLASSIFICATION,
    ) -> Dict[str, float]:
        """Run one full evaluation pass over a PyTorch ``DataLoader``.

        Parameters
        ----------
        model:
            ``torch.nn.Module`` in eval mode (caller should call
            ``model.eval()`` before invoking this, but we enforce it here
            as a safety net).
        dataloader:
            PyTorch ``DataLoader`` yielding ``(inputs, targets)`` batches.
        criterion:
            Loss function (e.g. ``nn.CrossEntropyLoss``).
        device:
            ``torch.device`` on which tensors should be placed.
        task:
            ML task string (``"classification"``, ``"regression"``,
            ``"time_series_forecasting"``).

        Returns
        -------
        Dict[str, float]
            At minimum ``{"val_loss": ...}``.  For classification tasks
            an ``"val_accuracy"`` key is also included.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for validate_epoch. "
                "Install via: pip install torch"
            ) from exc

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                total += batch_size

                # Classification accuracy
                if task == TASK_CLASSIFICATION:
                    _, predicted = torch.max(outputs, dim=1)
                    correct += (predicted == targets).sum().item()

        avg_loss = running_loss / total if total > 0 else 0.0

        metrics: Dict[str, float] = {"val_loss": avg_loss}

        if task == TASK_CLASSIFICATION:
            metrics["val_accuracy"] = correct / total if total > 0 else 0.0

        return metrics

    # ------------------------------------------------------------------
    # Scikit-learn validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_sklearn(
        model: Any,
        X_val: Any,
        y_val: Any,
        task: str = TASK_CLASSIFICATION,
    ) -> Dict[str, float]:
        """Evaluate a fitted scikit-learn estimator on validation data.

        Parameters
        ----------
        model:
            Fitted ``BaseEstimator`` instance.
        X_val:
            Validation feature matrix (array-like).
        y_val:
            Validation target vector (array-like).  May be ``None`` for
            clustering tasks.
        task:
            ML task string.

        Returns
        -------
        Dict[str, float]
            Task-appropriate metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
            silhouette_score,
        )

        metrics: Dict[str, float] = {}

        if task == TASK_CLASSIFICATION:
            y_pred = model.predict(X_val)
            metrics["val_accuracy"] = float(accuracy_score(y_val, y_pred))
            metrics["val_precision"] = float(
                precision_score(y_val, y_pred, average="weighted", zero_division=0)
            )
            metrics["val_recall"] = float(
                recall_score(y_val, y_pred, average="weighted", zero_division=0)
            )
            metrics["val_f1"] = float(
                f1_score(y_val, y_pred, average="weighted", zero_division=0)
            )

        elif task in (TASK_REGRESSION, TASK_TIME_SERIES_FORECASTING):
            y_pred = model.predict(X_val)
            mse = float(mean_squared_error(y_val, y_pred))
            metrics["val_mse"] = mse
            metrics["val_rmse"] = float(np.sqrt(mse))
            metrics["val_mae"] = float(mean_absolute_error(y_val, y_pred))
            metrics["val_r2"] = float(r2_score(y_val, y_pred))

        elif task == TASK_CLUSTERING:
            labels = model.predict(X_val) if hasattr(model, "predict") else model.labels_
            n_labels = len(set(labels))
            if 2 <= n_labels < len(X_val):
                metrics["val_silhouette"] = float(silhouette_score(X_val, labels))
            else:
                metrics["val_silhouette"] = 0.0

        return metrics

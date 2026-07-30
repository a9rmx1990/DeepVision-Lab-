"""
trainer/trainer.py
==================
Main training orchestrator for DeepVisionLab.

Handles both **scikit-learn** (classical ML) and **PyTorch** (deep learning)
training workflows, delegating to ``EarlyStopping``, ``CheckpointManager``,
and ``TrainingValidator`` for their respective responsibilities.

No Streamlit, no UI state — this module is pure training logic.

Usage::

    from app.models.model_factory import ModelFactory
    from app.trainer import Trainer

    # Scikit-learn
    model = ModelFactory.create("classification", "random_forest", n_estimators=100)
    trainer = Trainer(model=model, task="classification")
    result = trainer.train_sklearn(X_train, y_train, X_val, y_val)

    # PyTorch
    model = ModelFactory.create("classification", "cnn", in_channels=3, num_classes=10)
    trainer = Trainer(model=model, task="classification")
    result = trainer.train_pytorch(train_loader, val_loader)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from app.config.constants import (
    TASK_CLASSIFICATION,
    TASK_CLUSTERING,
    TASK_REGRESSION,
    TASK_TIME_SERIES_FORECASTING,
)
from app.config.settings import Settings
from app.trainer.checkpoint import CheckpointManager
from app.trainer.early_stopping import EarlyStopping
from app.trainer.validator import TrainingValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Container for all outputs of a training run.

    Attributes
    ----------
    train_losses:
        Per-epoch training loss (PyTorch) or single-element list (sklearn).
    val_losses:
        Per-epoch validation loss (PyTorch) or empty list (sklearn).
    train_metrics:
        Per-epoch training metrics, keyed by metric name.
    val_metrics:
        Per-epoch validation metrics, keyed by metric name.
    best_epoch:
        Epoch at which the best validation score was recorded (0-indexed,
        ``-1`` for sklearn).
    best_model_path:
        Absolute path to the saved best model checkpoint.
    training_time_seconds:
        Total wall-clock training time in seconds.
    """

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics: Dict[str, List[float]] = field(default_factory=dict)
    best_epoch: int = -1
    best_model_path: Optional[str] = None
    training_time_seconds: float = 0.0

    def summary(self) -> Dict[str, Any]:
        """Return a concise summary dictionary for logging / display."""
        result: Dict[str, Any] = {
            "best_epoch": self.best_epoch,
            "training_time_seconds": round(self.training_time_seconds, 2),
        }
        if self.val_losses:
            result["best_val_loss"] = min(self.val_losses)
        if self.best_model_path:
            result["best_model_path"] = self.best_model_path

        # Include last-epoch val metrics
        for key, values in self.val_metrics.items():
            if values:
                result[f"final_{key}"] = values[-1]

        return result


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrate model training for classical ML and deep learning.

    Parameters
    ----------
    model:
        An instantiated model — either a scikit-learn ``BaseEstimator``
        or a PyTorch ``nn.Module``.  Typically obtained from
        ``ModelFactory.create()``.
    task:
        ML task string.  One of ``"classification"``, ``"regression"``,
        ``"clustering"``, ``"time_series_forecasting"``.
    settings:
        Project-wide settings.  Defaults to ``Settings()``.
    """

    def __init__(
        self,
        model: Any,
        task: str,
        settings: Optional[Settings] = None,
    ) -> None:
        self.model = model
        self.task = task.lower().strip()
        self._settings = settings or Settings()
        self._validator = TrainingValidator()

    # ------------------------------------------------------------------
    # Scikit-learn training
    # ------------------------------------------------------------------

    def train_sklearn(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        *,
        model_name: str = "sklearn_model",
    ) -> TrainingResult:
        """Fit a scikit-learn model and evaluate on validation data.

        Parameters
        ----------
        X_train:
            Training feature matrix.
        y_train:
            Training target vector.  Pass ``None`` for clustering.
        X_val:
            Validation feature matrix (optional).
        y_val:
            Validation target vector (optional).
        model_name:
            Name used for the checkpoint directory.

        Returns
        -------
        TrainingResult
        """
        result = TrainingResult()
        start = time.time()

        logger.info("Training scikit-learn model (%s) for task '%s'", type(self.model).__name__, self.task)

        # --- Fit ---
        if self.task == TASK_CLUSTERING:
            self.model.fit(X_train)
        else:
            self.model.fit(X_train, y_train)

        # --- Validate ---
        val_metrics: Dict[str, float] = {}
        if X_val is not None:
            val_metrics = self._validator.validate_sklearn(
                self.model, X_val, y_val, task=self.task,
            )
            for key, value in val_metrics.items():
                result.val_metrics.setdefault(key, []).append(value)
            logger.info("Validation metrics: %s", val_metrics)

        # --- Checkpoint ---
        ckpt = CheckpointManager(model_name=model_name, settings=self._settings)
        saved_path = ckpt.save_sklearn(self.model, metrics=val_metrics)
        result.best_model_path = str(saved_path)

        result.training_time_seconds = time.time() - start
        logger.info(
            "Scikit-learn training complete in %.2fs. Model saved to %s",
            result.training_time_seconds,
            saved_path,
        )
        return result

    # ------------------------------------------------------------------
    # PyTorch training
    # ------------------------------------------------------------------

    def train_pytorch(
        self,
        train_loader: Any,
        val_loader: Any,
        *,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        optimizer: Optional[Any] = None,
        criterion: Optional[Any] = None,
        device: Optional[Any] = None,
        patience: Optional[int] = None,
        model_name: str = "pytorch_model",
    ) -> TrainingResult:
        """Run the full PyTorch training loop.

        Parameters
        ----------
        train_loader:
            Training ``DataLoader``.
        val_loader:
            Validation ``DataLoader``.
        epochs:
            Max training epochs.  Default from ``Settings``.
        learning_rate:
            Learning rate.  Default from ``Settings``.
        optimizer:
            Pre-configured optimizer.  Default: ``Adam``.
        criterion:
            Loss function.  Default: auto-selected based on ``self.task``.
        device:
            ``torch.device``.  Default: CUDA if available, else CPU.
        patience:
            Early-stopping patience.  Default from ``Settings``.
        model_name:
            Name used for checkpoint directory.

        Returns
        -------
        TrainingResult
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for train_pytorch. "
                "Install via: pip install torch"
            ) from exc

        # --- Resolve defaults from Settings ---
        epochs = epochs or self._settings.epochs
        learning_rate = learning_rate or self._settings.learning_rate
        patience = patience or self._settings.patience

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if criterion is None:
            criterion = self._auto_criterion(self.task)

        self.model = self.model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        early_stopping = EarlyStopping(patience=patience, mode="min")
        ckpt_manager = CheckpointManager(model_name=model_name, settings=self._settings)
        result = TrainingResult()
        start = time.time()

        logger.info(
            "Starting PyTorch training: task='%s', epochs=%d, lr=%g, device=%s",
            self.task, epochs, learning_rate, device,
        )

        for epoch in range(epochs):
            # --- Train one epoch ---
            train_metrics = self._train_one_epoch(
                train_loader, criterion, optimizer, device,
            )
            train_loss = train_metrics["train_loss"]
            result.train_losses.append(train_loss)

            for key, value in train_metrics.items():
                result.train_metrics.setdefault(key, []).append(value)

            # --- Validate ---
            val_metrics = self._validator.validate_epoch(
                self.model, val_loader, criterion, device, task=self.task,
            )
            val_loss = val_metrics["val_loss"]
            result.val_losses.append(val_loss)

            for key, value in val_metrics.items():
                result.val_metrics.setdefault(key, []).append(value)

            # --- Checkpoint ---
            is_best = val_loss <= early_stopping.best_score
            ckpt_manager.save_pytorch(
                self.model, optimizer, epoch=epoch,
                metrics=val_metrics, is_best=is_best,
            )

            if is_best:
                result.best_epoch = epoch
                result.best_model_path = str(ckpt_manager.best_checkpoint_path)

            # --- Early stopping ---
            early_stopping.step(val_loss)

            # --- Logging ---
            log_parts = [
                f"Epoch [{epoch + 1}/{epochs}]",
                f"train_loss={train_loss:.4f}",
                f"val_loss={val_loss:.4f}",
            ]
            if "val_accuracy" in val_metrics:
                log_parts.append(f"val_acc={val_metrics['val_accuracy']:.4f}")
            if "train_accuracy" in train_metrics:
                log_parts.append(f"train_acc={train_metrics['train_accuracy']:.4f}")
            logger.info(" | ".join(log_parts))

            if early_stopping.should_stop:
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch + 1, patience,
                )
                break

        result.training_time_seconds = time.time() - start

        # Load best model weights back
        if result.best_model_path is not None:
            try:
                ckpt_manager.load_pytorch(self.model, optimizer, load_best=True)
                logger.info("Restored best model from epoch %d", result.best_epoch + 1)
            except FileNotFoundError:
                logger.warning("Could not restore best model checkpoint.")

        logger.info(
            "Training complete in %.2fs. Best epoch: %d",
            result.training_time_seconds,
            result.best_epoch + 1,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        dataloader: Any,
        criterion: Any,
        optimizer: Any,
        device: Any,
    ) -> Dict[str, float]:
        """Run one training epoch and return metrics.

        Returns
        -------
        Dict[str, float]
            Contains at least ``"train_loss"``.  For classification,
            also ``"train_accuracy"``.
        """
        import torch

        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            if self.task == TASK_CLASSIFICATION:
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == targets).sum().item()

        avg_loss = running_loss / total if total > 0 else 0.0
        metrics: Dict[str, float] = {"train_loss": avg_loss}

        if self.task == TASK_CLASSIFICATION:
            metrics["train_accuracy"] = correct / total if total > 0 else 0.0

        return metrics

    @staticmethod
    def _auto_criterion(task: str) -> Any:
        """Auto-select a loss function based on the ML task.

        Returns
        -------
        torch.nn.Module
            ``CrossEntropyLoss`` for classification, ``MSELoss`` for
            regression / time-series.
        """
        import torch.nn as nn

        if task == TASK_CLASSIFICATION:
            return nn.CrossEntropyLoss()
        if task in (TASK_REGRESSION, TASK_TIME_SERIES_FORECASTING):
            return nn.MSELoss()

        raise ValueError(
            f"Cannot auto-select criterion for task '{task}'. "
            "Pass an explicit `criterion` argument."
        )

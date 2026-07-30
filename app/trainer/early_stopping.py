"""
trainer/early_stopping.py
=========================
Early stopping callback to halt training when a monitored metric
stops improving.

Works for both PyTorch and scikit-learn workflows — no framework
dependency.  The caller invokes ``step(metric_value)`` after each
epoch/evaluation round and checks ``should_stop`` to decide whether
to break the training loop.

Usage::

    from app.trainer.early_stopping import EarlyStopping

    es = EarlyStopping(patience=10, mode="min")  # monitor val_loss
    for epoch in range(max_epochs):
        val_loss = train_and_validate(...)
        es.step(val_loss)
        if es.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break
"""

from __future__ import annotations

import math
from typing import Optional

from app.config.constants import DEFAULT_PATIENCE


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    patience:
        Number of consecutive epochs with no improvement after which
        training is stopped.  Default ``DEFAULT_PATIENCE`` (10).
    min_delta:
        Minimum change in the monitored metric to qualify as an
        improvement.  Default ``0.0``.
    mode:
        One of ``"min"`` or ``"max"``.

        - ``"min"`` — training stops when the metric stops *decreasing*
          (e.g. loss).
        - ``"max"`` — training stops when the metric stops *increasing*
          (e.g. accuracy).
    """

    def __init__(
        self,
        patience: int = DEFAULT_PATIENCE,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        # Internal state
        self._best_score: float = math.inf if mode == "min" else -math.inf
        self._counter: int = 0
        self._should_stop: bool = False
        self._best_epoch: int = 0
        self._epoch: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, metric_value: float) -> None:
        """Update the early-stopping state with the latest metric value.

        Parameters
        ----------
        metric_value:
            Current epoch's monitored metric (e.g. validation loss or
            validation accuracy).
        """
        self._epoch += 1
        improved = self._is_improvement(metric_value)

        if improved:
            self._best_score = metric_value
            self._counter = 0
            self._best_epoch = self._epoch
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._should_stop = True

    @property
    def should_stop(self) -> bool:
        """``True`` when the patience budget has been exhausted."""
        return self._should_stop

    @property
    def best_score(self) -> float:
        """Best metric value observed so far."""
        return self._best_score

    @property
    def best_epoch(self) -> int:
        """Epoch number (1-indexed) at which the best score was recorded."""
        return self._best_epoch

    @property
    def counter(self) -> int:
        """Number of epochs since the last improvement."""
        return self._counter

    def reset(self) -> None:
        """Reset all internal state so the instance can be reused."""
        self._best_score = math.inf if self.mode == "min" else -math.inf
        self._counter = 0
        self._should_stop = False
        self._best_epoch = 0
        self._epoch = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        """Return ``True`` if *value* is an improvement over ``_best_score``."""
        if self.mode == "min":
            return value < (self._best_score - self.min_delta)
        return value > (self._best_score + self.min_delta)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EarlyStopping(patience={self.patience}, mode='{self.mode}', "
            f"best={self._best_score:.4f}, counter={self._counter}, "
            f"stop={self._should_stop})"
        )

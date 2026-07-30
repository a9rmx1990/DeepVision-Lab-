"""
trainer/checkpoint.py
=====================
Save and load model checkpoints for both PyTorch (``nn.Module``) and
scikit-learn (``BaseEstimator``) models.

All paths are resolved through ``config/settings.py`` — never hardcode
a path inside this module.

Usage::

    from app.trainer.checkpoint import CheckpointManager

    mgr = CheckpointManager(model_name="resnet18_cifar10")
    mgr.save_pytorch(model, optimizer, epoch=5, metrics={"val_loss": 0.32})
    mgr.load_pytorch(model, optimizer)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app.config.settings import Settings


class CheckpointManager:
    """Manage model checkpoint persistence for PyTorch and scikit-learn models.

    Parameters
    ----------
    model_name:
        A descriptive name used as the sub-directory and file prefix
        (e.g. ``"resnet18_cifar10"``).
    settings:
        Project-wide settings.  Defaults to ``Settings()``.
    save_dir:
        Explicit override for the checkpoint directory.  When ``None``
        (default), uses ``settings.saved_models_dir / model_name``.
    """

    def __init__(
        self,
        model_name: str = "model",
        settings: Optional[Settings] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        self._settings = settings or Settings()
        self.model_name = model_name

        if save_dir is not None:
            self._save_dir = Path(save_dir)
        else:
            self._save_dir = self._settings.saved_models_dir / model_name

        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._best_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def save_dir(self) -> Path:
        """Absolute path to the checkpoint directory."""
        return self._save_dir

    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        """Path to the most-recently saved *best* checkpoint, or ``None``."""
        return self._best_path

    # ------------------------------------------------------------------
    # PyTorch checkpoints
    # ------------------------------------------------------------------

    def save_pytorch(
        self,
        model: Any,
        optimizer: Any = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        *,
        is_best: bool = False,
    ) -> Path:
        """Persist a PyTorch model checkpoint to disk.

        Parameters
        ----------
        model:
            ``torch.nn.Module`` instance.
        optimizer:
            ``torch.optim.Optimizer`` instance (optional).
        epoch:
            Current epoch number (0-indexed).
        metrics:
            Dictionary of metric values to store alongside the model.
        is_best:
            If ``True``, an additional copy is saved as ``best_model.pt``.

        Returns
        -------
        Path
            Absolute path to the saved checkpoint file.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to save PyTorch checkpoints. "
                "Install via: pip install torch"
            ) from exc

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Save epoch checkpoint
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
        filepath = self._save_dir / filename
        torch.save(checkpoint, filepath)

        # Save best model copy
        if is_best:
            best_path = self._save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self._best_path = best_path

        return filepath

    def load_pytorch(
        self,
        model: Any,
        optimizer: Any = None,
        *,
        path: Optional[Path] = None,
        load_best: bool = True,
    ) -> Dict[str, Any]:
        """Restore a PyTorch model (and optionally optimizer) from checkpoint.

        Parameters
        ----------
        model:
            ``torch.nn.Module`` instance to load state into.
        optimizer:
            ``torch.optim.Optimizer`` instance (optional).
        path:
            Explicit checkpoint path.  When ``None``, loads the
            ``best_model.pt`` if ``load_best=True``, else the latest
            epoch checkpoint.
        load_best:
            Whether to prefer ``best_model.pt`` when *path* is ``None``.

        Returns
        -------
        Dict[str, Any]
            The full checkpoint dictionary (epoch, metrics, etc.).

        Raises
        ------
        FileNotFoundError
            If no checkpoint file is found.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to load PyTorch checkpoints. "
                "Install via: pip install torch"
            ) from exc

        if path is None:
            path = self._resolve_pytorch_path(load_best)

        if path is None or not path.exists():
            raise FileNotFoundError(
                f"No checkpoint found in '{self._save_dir}'. "
                "Train a model first."
            )

        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    # ------------------------------------------------------------------
    # Scikit-learn checkpoints
    # ------------------------------------------------------------------

    def save_sklearn(
        self,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Persist a scikit-learn model to disk using ``joblib``.

        Parameters
        ----------
        model:
            Fitted ``BaseEstimator`` instance.
        metrics:
            Dictionary of evaluation metrics to store alongside the model.

        Returns
        -------
        Path
            Absolute path to the saved model file.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib is required to save scikit-learn models. "
                "Install via: pip install joblib"
            ) from exc

        model_path = self._save_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Save metrics alongside
        if metrics:
            meta_path = self._save_dir / "metrics.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat(),
                        "model_name": self.model_name,
                    },
                    f,
                    indent=2,
                )

        self._best_path = model_path
        return model_path

    def load_sklearn(self, path: Optional[Path] = None) -> Any:
        """Load a scikit-learn model from disk.

        Parameters
        ----------
        path:
            Explicit path to the ``.joblib`` file.  Defaults to
            ``<save_dir>/model.joblib``.

        Returns
        -------
        Any
            The loaded scikit-learn estimator.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib is required to load scikit-learn models. "
                "Install via: pip install joblib"
            ) from exc

        if path is None:
            path = self._save_dir / "model.joblib"

        if not path.exists():
            raise FileNotFoundError(
                f"No scikit-learn model found at '{path}'. "
                "Train a model first."
            )

        return joblib.load(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_pytorch_path(self, load_best: bool) -> Optional[Path]:
        """Find the best or latest PyTorch checkpoint in ``_save_dir``."""
        best = self._save_dir / "best_model.pt"
        if load_best and best.exists():
            return best

        # Fallback: find the latest epoch checkpoint
        checkpoints = sorted(self._save_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            return checkpoints[-1]

        return None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CheckpointManager(model_name='{self.model_name}', "
            f"save_dir='{self._save_dir}')"
        )

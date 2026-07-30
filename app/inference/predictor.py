"""
inference/predictor.py
======================
Load trained models and generate predictions on new data.

Supports both **scikit-learn** and **PyTorch** models.  Model loading
delegates to ``CheckpointManager`` so paths are always resolved through
``config/settings.py``.

Usage::

    from app.inference import Predictor

    # Sklearn
    predictor = Predictor(model_name="random_forest_iris", task="classification")
    predictions = predictor.predict_sklearn(X_new)

    # PyTorch
    from app.models import ModelFactory
    model = ModelFactory.create("classification", "cnn", in_channels=3, num_classes=10)
    predictor = Predictor(model_name="cnn_cifar10", task="classification", model=model)
    predictions = predictor.predict_pytorch(image_tensor)
"""

from __future__ import annotations

import logging
from pathlib import Path
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

logger = logging.getLogger(__name__)


class Predictor:
    """Load a trained model and run inference on new data.

    Parameters
    ----------
    model_name:
        Name of the saved model (matches the ``CheckpointManager``
        directory name under ``saved_models/``).
    task:
        ML task string.
    model:
        For PyTorch, the *unloaded* model architecture (same constructor
        args as training) — weights will be loaded from checkpoint.
        For sklearn, leave ``None`` — the entire model is loaded from
        the joblib file.
    settings:
        Project-wide settings.
    """

    def __init__(
        self,
        model_name: str,
        task: str,
        model: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.model_name = model_name
        self.task = task.lower().strip()
        self._settings = settings or Settings()
        self._ckpt = CheckpointManager(
            model_name=model_name, settings=self._settings,
        )
        self._model = model
        self._loaded = False

    # ------------------------------------------------------------------
    # Sklearn inference
    # ------------------------------------------------------------------

    def predict_sklearn(self, X: Any) -> Any:
        """Run inference with a scikit-learn model.

        Parameters
        ----------
        X:
            Feature matrix (array-like).

        Returns
        -------
        Any
            Predictions (labels for classification, values for regression,
            cluster IDs for clustering).
        """
        model = self._load_sklearn_model()
        logger.info("Running sklearn prediction on %d samples", len(X))
        return model.predict(X)

    def predict_sklearn_proba(self, X: Any) -> Optional[Any]:
        """Return class probabilities if supported by the sklearn model.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        Optional[np.ndarray]
            Class probabilities, or ``None`` if the model doesn't
            support ``predict_proba``.
        """
        model = self._load_sklearn_model()
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None

    # ------------------------------------------------------------------
    # PyTorch inference
    # ------------------------------------------------------------------

    def predict_pytorch(
        self,
        inputs: Any,
        *,
        device: Optional[Any] = None,
    ) -> Any:
        """Run inference with a PyTorch model.

        Parameters
        ----------
        inputs:
            Input tensor of shape appropriate for the model (e.g.
            ``(N, C, H, W)`` for CNNs, ``(N, L, F)`` for RNNs).
        device:
            ``torch.device``.  Default: auto-detect.

        Returns
        -------
        torch.Tensor
            Raw model outputs (logits for classification, values for
            regression).
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for predict_pytorch. "
                "Install via: pip install torch"
            ) from exc

        model = self._load_pytorch_model(device)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.to(device)
            outputs = model(inputs)

        return outputs

    def predict_pytorch_classes(
        self,
        inputs: Any,
        *,
        device: Optional[Any] = None,
    ) -> Any:
        """Run classification inference and return predicted class indices.

        Parameters
        ----------
        inputs:
            Input tensor.
        device:
            ``torch.device``.

        Returns
        -------
        torch.Tensor
            Predicted class indices of shape ``(N,)``.
        """
        import torch

        outputs = self.predict_pytorch(inputs, device=device)
        _, predicted = torch.max(outputs, dim=1)
        return predicted

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def predict_from_csv(
        self,
        df: Any,
        preprocessor: Optional[Any] = None,
        target_column: Optional[str] = None,
    ) -> Any:
        """Predict from a pandas DataFrame (sklearn models).

        Parameters
        ----------
        df:
            Input DataFrame.
        preprocessor:
            Fitted ``CSVPreprocessor`` instance for transforming features.
            Pass ``None`` if data is already preprocessed.
        target_column:
            Column to exclude from features (if still present).

        Returns
        -------
        Any
            Predictions.
        """
        import pandas as pd

        features = df.copy()
        if target_column and target_column in features.columns:
            features = features.drop(columns=[target_column])

        if preprocessor is not None:
            features = preprocessor.transform(features, target_column=None)

        return self.predict_sklearn(features)

    def predict_from_image(
        self,
        image_path: Union[str, Path],
        transform: Optional[Any] = None,
        *,
        device: Optional[Any] = None,
    ) -> int:
        """Predict class label for a single image (PyTorch models).

        Parameters
        ----------
        image_path:
            Path to the image file.
        transform:
            Torchvision transform to apply.  Typically from
            ``ImagePreprocessor.eval_transforms()``.
        device:
            ``torch.device``.

        Returns
        -------
        int
            Predicted class index.
        """
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "PyTorch and Pillow are required for predict_from_image. "
                "Install via: pip install torch Pillow"
            ) from exc

        img = Image.open(image_path).convert("RGB")

        if transform is not None:
            img_tensor = transform(img)
        else:
            from torchvision import transforms
            img_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])(img)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        predicted = self.predict_pytorch_classes(img_tensor, device=device)
        return int(predicted.item())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_sklearn_model(self) -> Any:
        """Load and cache the sklearn model."""
        if self._model is None or not self._loaded:
            self._model = self._ckpt.load_sklearn()
            self._loaded = True
            logger.info("Loaded sklearn model from %s", self._ckpt.save_dir)
        return self._model

    def _load_pytorch_model(self, device: Optional[Any] = None) -> Any:
        """Load checkpoint weights into the PyTorch model."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required. Install via: pip install torch"
            ) from exc

        if self._model is None:
            raise ValueError(
                "A PyTorch model architecture must be provided via the "
                "'model' parameter for PyTorch inference."
            )

        if not self._loaded:
            self._ckpt.load_pytorch(self._model, load_best=True)
            self._loaded = True
            logger.info("Loaded PyTorch checkpoint from %s", self._ckpt.save_dir)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = self._model.to(device)
        return self._model

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Predictor(model_name='{self.model_name}', task='{self.task}', "
            f"loaded={self._loaded})"
        )

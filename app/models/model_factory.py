"""
app/models/model_factory.py
============================
Single entry-point factory mapping ML tasks and model names to constructed
model instances (scikit-learn algorithms and PyTorch deep learning architectures).

All model creation in DeepVisionLab flows through ``ModelFactory.create()``.
Never construct models ad-hoc in trainer or UI code.

Usage::

    from app.models.model_factory import ModelFactory

    # Classical ML (scikit-learn)
    clf = ModelFactory.create("classification", "random_forest", n_estimators=100)

    # Deep Learning (PyTorch CNN)
    cnn = ModelFactory.create("classification", "cnn", in_channels=3, num_classes=10)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# Classical ML models (scikit-learn)
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Deep Learning models (PyTorch)
from app.models.cnn import FlexibleCNN, SimpleCNN
from app.models.lstm import SimpleLSTM
from app.models.pretrained import PretrainedCNN
from app.models.rnn import SimpleRNN

# Scikit-learn model builders map
_CLASSICAL_CLASSIFIERS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
}

_CLASSICAL_REGRESSORS = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
}

_CLASSICAL_CLUSTERING = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "hierarchical": AgglomerativeClustering,
}


class ModelFactory:
    """Factory class for instantiating classical ML and deep learning models.

    Methods
    -------
    create(task, model_name, **kwargs):
        Construct and return a model instance matching *task* and *model_name*.
    list_supported_models(task=None):
        List supported model names for a task or all tasks.
    """

    @classmethod
    def create(
        cls,
        task: str,
        model_name: str,
        **kwargs: Any,
    ) -> Any:
        """Construct and return a model instance.

        Parameters
        ----------
        task:
            Target ML task. One of:
            - ``"classification"``
            - ``"regression"``
            - ``"clustering"``
            - ``"time_series_forecasting"``
        model_name:
            Identifier of the architecture / algorithm (case-insensitive).
        **kwargs:
            Hyperparameters forwarded to the model constructor.

        Returns
        -------
        Any
            Instantiated scikit-learn Estimator or PyTorch nn.Module.

        Raises
        ------
        ValueError
            If *task* or *model_name* is not supported.
        """
        task_clean = task.lower().strip()
        name_clean = model_name.lower().strip().replace("-", "_")

        # --- Classical Classification ---
        if task_clean == "classification" and name_clean in _CLASSICAL_CLASSIFIERS:
            model_cls = _CLASSICAL_CLASSIFIERS[name_clean]
            return model_cls(**kwargs)

        # --- Classical Regression ---
        if task_clean == "regression" and name_clean in _CLASSICAL_REGRESSORS:
            model_cls = _CLASSICAL_REGRESSORS[name_clean]
            return model_cls(**kwargs)

        # --- Classical Clustering ---
        if task_clean == "clustering" and name_clean in _CLASSICAL_CLUSTERING:
            model_cls = _CLASSICAL_CLUSTERING[name_clean]
            return model_cls(**kwargs)

        # --- Deep Learning CNN Architecture ---
        if name_clean in ("cnn", "simple_cnn"):
            return SimpleCNN(**kwargs)

        if name_clean in ("flexible_cnn", "custom_cnn"):
            return FlexibleCNN(**kwargs)

        # Pretrained TorchVision architectures
        if name_clean in (
            "resnet", "resnet18", "resnet34", "resnet50",
            "efficientnet", "efficientnet_b0", "efficientnet_b1",
            "mobilenet", "mobilenet_v3_small", "mobilenet_v3_large",
            "vgg16",
        ):
            backbone_name = "resnet18" if name_clean == "resnet" else (
                "efficientnet_b0" if name_clean == "efficientnet" else (
                    "mobilenet_v3_small" if name_clean == "mobilenet" else name_clean
                )
            )
            kwargs["model_name"] = backbone_name
            return PretrainedCNN(**kwargs)

        # --- Deep Learning RNN / GRU Architecture ---
        if name_clean in ("rnn", "gru"):
            if "cell_type" not in kwargs and name_clean == "gru":
                kwargs["cell_type"] = "gru"
            return SimpleRNN(**kwargs)

        # --- Deep Learning LSTM Architecture ---
        if name_clean == "lstm":
            return SimpleLSTM(**kwargs)

        raise ValueError(
            f"Unsupported model '{model_name}' for task '{task}'. "
            f"Available models: {cls.list_supported_models(task_clean)}"
        )

    @classmethod
    def list_supported_models(cls, task: Optional[str] = None) -> Dict[str, List[str]]:
        """Return supported model names grouped by task.

        Parameters
        ----------
        task:
            Optional task filter.

        Returns
        -------
        Dict[str, List[str]]
        """
        all_models = {
            "classification": list(_CLASSICAL_CLASSIFIERS.keys()) + [
                "cnn", "flexible_cnn", "resnet18", "resnet34", "resnet50",
                "efficientnet_b0", "mobilenet_v3_small", "vgg16",
            ],
            "regression": list(_CLASSICAL_REGRESSORS.keys()) + ["rnn", "lstm"],
            "clustering": list(_CLASSICAL_CLUSTERING.keys()),
            "time_series_forecasting": ["rnn", "gru", "lstm"],
        }

        if task is not None:
            t = task.lower().strip()
            return {t: all_models.get(t, [])}

        return all_models

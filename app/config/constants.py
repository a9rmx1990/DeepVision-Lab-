"""
config/constants.py
===================
Immutable, project-wide constants.

All magic numbers, string literals, and fixed configuration values live
here.  Import from this module — never hardcode these values in business
logic or UI code.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Supported file formats
# ---------------------------------------------------------------------------

CSV_EXTENSIONS: tuple[str, ...] = (".csv",)

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
)

# ---------------------------------------------------------------------------
# Dataset pipeline defaults
# ---------------------------------------------------------------------------

DEFAULT_TEST_SIZE:        float = 0.15
DEFAULT_VAL_SIZE:         float = 0.15
DEFAULT_RANDOM_STATE:     int   = 42
DEFAULT_REGRESSION_THRESHOLD: int = 15   # unique values above which → regression
DEFAULT_IMBALANCE_THRESHOLD: float = 10.0
DEFAULT_MISSING_VALUE_THRESHOLD: float = 0.20

# ---------------------------------------------------------------------------
# Preprocessing defaults
# ---------------------------------------------------------------------------

DEFAULT_SCALER:          str   = "standard"   # standard | minmax | robust
DEFAULT_IMPUTE_STRATEGY: str   = "mean"       # mean | median | most_frequent | constant

# ---------------------------------------------------------------------------
# Image defaults
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE:   int   = 32
DEFAULT_EPOCHS:       int   = 50
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_PATIENCE:     int   = 10    # early-stopping patience

# ---------------------------------------------------------------------------
# Supported ML tasks
# ---------------------------------------------------------------------------

TASK_CLASSIFICATION:          str = "classification"
TASK_REGRESSION:              str = "regression"
TASK_CLUSTERING:              str = "clustering"
TASK_TIME_SERIES_FORECASTING: str = "time_series_forecasting"

SUPPORTED_CSV_TASKS: tuple[str, ...] = (
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
    TASK_CLUSTERING,
    TASK_TIME_SERIES_FORECASTING,
)

SUPPORTED_IMAGE_TASKS: tuple[str, ...] = (
    TASK_CLASSIFICATION,
)

# ---------------------------------------------------------------------------
# Supported models (classical ML)
# ---------------------------------------------------------------------------

CLASSICAL_CLASSIFIERS: tuple[str, ...] = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
    "naive_bayes",
)

CLASSICAL_REGRESSORS: tuple[str, ...] = (
    "linear_regression",
    "ridge",
    "lasso",
)

CLASSICAL_CLUSTERING: tuple[str, ...] = (
    "kmeans",
    "dbscan",
    "hierarchical",
)

# Deep learning architectures
DEEP_LEARNING_MODELS: tuple[str, ...] = (
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "resnet",
    "efficientnet",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

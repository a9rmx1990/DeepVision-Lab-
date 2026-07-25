"""
app/dataset
===========
Dataset pipeline (must be followed in this order)::

    Upload → Loader → Validator → Detector → Splitter → Preprocessor → Dataset Object

Each stage feeds into the next — never skip or reorder steps.

Typical usage::

    from app.dataset import DatasetLoader, DatasetValidator, DatasetDetector
    from app.dataset import DatasetSplitter, DatasetPreprocessor

    # 1. Load
    loader = DatasetLoader()
    data   = loader.load("datasets/my_data.csv")

    # 2. Validate  ← must come before detect / split / preprocess
    validator = DatasetValidator()
    report    = validator.validate(data, target_column="label")
    if not report.is_valid:
        raise ValueError(report)

    # 3. Detect (runs on validated data only)
    detector = DatasetDetector()
    dtype    = detector.detect_type(data)
    task     = detector.infer_task(data, target_column="label")

    # 4. Split
    splits = DatasetSplitter().split(data, target_column="label")

    # 5. Preprocess
    pre = DatasetPreprocessor()
    X_train = pre.csv.fit_transform(splits.train.drop(columns=["label"]))
    X_val   = pre.csv.transform(splits.val.drop(columns=["label"]))
"""

from .loader import DatasetLoader
from .validator import DatasetValidator
from .detector import DatasetDetector, DatasetType
from .splitter import DatasetSplitter
from .preprocessing import DatasetPreprocessor

__all__ = [
    "DatasetLoader",
    "DatasetValidator",
    "DatasetDetector",
    "DatasetType",
    "DatasetSplitter",
    "DatasetPreprocessor",
]

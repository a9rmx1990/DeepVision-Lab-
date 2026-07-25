"""
dataset/detector.py
===================
**Stage 3 of the dataset pipeline.**  Receives data that has already been
loaded by ``DatasetLoader`` *and* validated by ``DatasetValidator``.

Pipeline position::

    Loader → Validator → [Detector] → Splitter → Preprocessing

.. warning::
    Never call ``detect_type`` or ``infer_task`` on raw loader output that
    has not passed through ``DatasetValidator`` first.  Unvalidated data
    (missing columns, corrupt rows, wrong types) can cause misleading
    detection results.

Detect whether a loaded, validated dataset is a **CSV** (tabular) dataset
or an **Image** dataset, and — for CSV datasets — infer the ML task
(classification, regression, clustering, time-series).

Detection rules
---------------
* ``DatasetType.IMAGE`` → the original path was a directory containing
  image files organised in sub-folders (ImageFolder convention).
* ``DatasetType.CSV``   → the original path was a ``.csv`` file.

Task inference (CSV only)
-------------------------
* **Time Series**     – target column is a datetime type, or the DataFrame
  contains a monotonically increasing datetime-like index.
* **Clustering**      – no target column is specified.
* **Regression**      – target column is numeric (float / int) with more
  than ``regression_threshold`` unique values.
* **Classification**  – target column is categorical or numeric with few
  unique values.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DatasetType(Enum):
    """High-level type of a dataset."""
    CSV   = auto()
    IMAGE = auto()
    UNKNOWN = auto()


class MLTask(Enum):
    """Inferred ML task for a CSV dataset."""
    CLASSIFICATION         = "classification"
    REGRESSION             = "regression"
    CLUSTERING             = "clustering"
    TIME_SERIES_FORECASTING = "time_series_forecasting"


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ImageSample = Tuple[str, str]   # (absolute_path, label)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DatasetDetector:
    """Detect dataset type and infer ML task.

    Parameters
    ----------
    regression_threshold:
        Minimum number of unique target values for a numeric column to be
        treated as *regression* rather than *classification*.  Default 15.
    """

    def __init__(self, regression_threshold: int = 15) -> None:
        self.regression_threshold = regression_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_type(
        self,
        data: Union[pd.DataFrame, List[ImageSample], None],
        source_path: Optional[Union[str, Path]] = None,
    ) -> DatasetType:
        """Detect whether *data* represents a CSV or Image dataset.

        .. important::
            *data* must come from ``DatasetLoader`` **and** have already
            been passed through ``DatasetValidator``.  This method is
            **Stage 3** of the pipeline (Loader → Validator → Detector).

        Parameters
        ----------
        data:
            Output from ``DatasetLoader.load()`` after successful validation.
        source_path:
            Original path passed to the loader.  Used as a tie-breaker
            when *data* alone is ambiguous.

        Returns
        -------
        DatasetType
        """
        if isinstance(data, pd.DataFrame):
            return DatasetType.CSV

        if isinstance(data, list) and data and isinstance(data[0], tuple):
            return DatasetType.IMAGE

        # Fallback: inspect the source path extension
        if source_path is not None:
            path = Path(source_path)
            if path.is_dir():
                return DatasetType.IMAGE
            if path.suffix.lower() == ".csv":
                return DatasetType.CSV

        return DatasetType.UNKNOWN

    def infer_task(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> MLTask:
        """Infer the ML task from a CSV DataFrame.

        Parameters
        ----------
        df:
            The loaded DataFrame.
        target_column:
            Name of the column to predict.  Pass ``None`` to signal
            an *unsupervised* (clustering) use-case.

        Returns
        -------
        MLTask

        Raises
        ------
        ValueError
            If *target_column* is not present in *df*.
        """
        if target_column is None:
            return MLTask.CLUSTERING

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        series = df[target_column]

        # --- Time Series check -------------------------------------------
        if self._is_datetime_series(series):
            return MLTask.TIME_SERIES_FORECASTING

        # --- Numeric target ----------------------------------------------
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique()
            if n_unique > self.regression_threshold:
                return MLTask.REGRESSION
            return MLTask.CLASSIFICATION

        # --- Categorical / object target ---------------------------------
        return MLTask.CLASSIFICATION

    def detect_image_classes(
        self,
        samples: List[ImageSample],
    ) -> List[str]:
        """Return the sorted list of class labels found in an image dataset.

        Parameters
        ----------
        samples:
            Output from ``DatasetLoader.load_image_dataset()``.

        Returns
        -------
        List[str]
            Sorted unique class names.
        """
        return sorted({label for _, label in samples})

    def summary(
        self,
        data: Union[pd.DataFrame, List[ImageSample]],
        source_path: Optional[Union[str, Path]] = None,
        target_column: Optional[str] = None,
    ) -> dict:
        """Return a human-readable detection summary dictionary.

        Parameters
        ----------
        data:
            Loaded dataset.
        source_path:
            Original source path (optional).
        target_column:
            Target column for CSV task inference (optional).

        Returns
        -------
        dict
            Keys: ``dataset_type``, ``n_samples``, ``task`` (CSV),
            ``classes`` (Image).
        """
        dtype = self.detect_type(data, source_path)
        result: dict = {"dataset_type": dtype.name}

        if dtype == DatasetType.CSV:
            df: pd.DataFrame = data  # type: ignore[assignment]
            result["n_samples"]  = len(df)
            result["n_features"] = len(df.columns)
            result["task"]       = self.infer_task(df, target_column).value

        elif dtype == DatasetType.IMAGE:
            samples: List[ImageSample] = data  # type: ignore[assignment]
            classes = self.detect_image_classes(samples)
            result["n_samples"] = len(samples)
            result["n_classes"] = len(classes)
            result["classes"]   = classes
            result["task"]      = MLTask.CLASSIFICATION.value

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_datetime_series(series: pd.Series) -> bool:
        """Return True if *series* has a datetime or period dtype."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if pd.api.types.is_period_dtype(series):
            return True
        # Try parsing a sample of string values as dates
        if pd.api.types.is_object_dtype(series):
            sample = series.dropna().head(10)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                return True
            except (ValueError, TypeError):
                pass
        return False

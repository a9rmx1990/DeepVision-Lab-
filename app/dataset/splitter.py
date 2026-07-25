"""
dataset/splitter.py
===================
Split datasets into training, validation, and test sets.

Supports:
* **CSV** (``pd.DataFrame``) — stratified or plain random split.
* **Image** (``List[ImageSample]``) — stratified by class label.

Returned splits are named tuples so callers can unpack them or access
individual splits by name without relying on index position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ImageSample = Tuple[str, str]   # (absolute_path, label)


# ---------------------------------------------------------------------------
# Split result containers
# ---------------------------------------------------------------------------

@dataclass
class CSVSplit:
    """Container for a CSV dataset split."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CSVSplit(train={len(self.train)}, "
            f"val={len(self.val)}, test={len(self.test)})"
        )


@dataclass
class ImageSplit:
    """Container for an image dataset split."""
    train: List[ImageSample]
    val: List[ImageSample]
    test: List[ImageSample]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ImageSplit(train={len(self.train)}, "
            f"val={len(self.val)}, test={len(self.test)})"
        )


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

class DatasetSplitter:
    """Split CSV or image datasets into train / val / test subsets.

    Parameters
    ----------
    val_size:
        Fraction of data reserved for validation.  Default ``0.15``.
    test_size:
        Fraction of the **original** data reserved for testing.
        Default ``0.15``.
    random_state:
        Seed for reproducibility.  Default ``42``.
    stratify:
        Whether to perform stratified splitting (preserves class ratios).
        Default ``True``.  Ignored for regression targets.
    """

    def __init__(
        self,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        if not (0 < val_size < 1):
            raise ValueError(f"val_size must be in (0, 1), got {val_size}")
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        if val_size + test_size >= 1.0:
            raise ValueError(
                f"val_size + test_size must be < 1.0, "
                f"got {val_size + test_size}"
            )

        self.val_size     = val_size
        self.test_size    = test_size
        self.random_state = random_state
        self.stratify     = stratify

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        data: Union[pd.DataFrame, List[ImageSample]],
        target_column: Optional[str] = None,
    ) -> Union[CSVSplit, ImageSplit]:
        """Auto-dispatch split based on data type.

        Parameters
        ----------
        data:
            Output from ``DatasetLoader.load()``.
        target_column:
            For CSV data, the name of the label/target column used for
            stratification.  Pass ``None`` to disable stratification.

        Returns
        -------
        CSVSplit | ImageSplit
        """
        if isinstance(data, pd.DataFrame):
            return self.split_csv(data, target_column=target_column)
        if isinstance(data, list):
            return self.split_image_dataset(data)
        raise TypeError(f"Unsupported data type for splitting: {type(data)}")

    def split_csv(
        self,
        df: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
    ) -> CSVSplit:
        """Split a DataFrame into train / val / test.

        Parameters
        ----------
        df:
            Full dataset.
        target_column:
            Column used for stratification.  If ``None`` or the column has
            continuous values, stratification is skipped.

        Returns
        -------
        CSVSplit
        """
        stratify_col = self._resolve_stratify_col(df, target_column)

        # First carve out the test set
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col,
        )

        # Recompute stratify labels for the train_val portion
        stratify_train_val = (
            train_val[target_column]
            if stratify_col is not None and target_column is not None
            else None
        )

        # Val fraction relative to the remaining train+val data
        relative_val_size = self.val_size / (1.0 - self.test_size)

        train, val = train_test_split(
            train_val,
            test_size=relative_val_size,
            random_state=self.random_state,
            stratify=stratify_train_val,
        )

        return CSVSplit(
            train=train.reset_index(drop=True),
            val=val.reset_index(drop=True),
            test=test.reset_index(drop=True),
        )

    def split_image_dataset(
        self,
        samples: List[ImageSample],
    ) -> ImageSplit:
        """Split a list of image samples into train / val / test.

        Stratified by class label when ``self.stratify=True``.

        Parameters
        ----------
        samples:
            Output from ``DatasetLoader.load_image_dataset()``.

        Returns
        -------
        ImageSplit
        """
        paths  = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        stratify_labels = labels if self.stratify else None

        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            paths, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_labels,
        )

        relative_val_size = self.val_size / (1.0 - self.test_size)
        tv_stratify = train_val_labels if self.stratify else None

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=relative_val_size,
            random_state=self.random_state,
            stratify=tv_stratify,
        )

        return ImageSplit(
            train=list(zip(train_paths, train_labels)),
            val=list(zip(val_paths, val_labels)),
            test=list(zip(test_paths, test_labels)),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_stratify_col(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
    ) -> Optional[pd.Series]:
        """Return the stratify Series, or ``None`` if not applicable."""
        if not self.stratify or target_column is None:
            return None
        if target_column not in df.columns:
            return None
        series = df[target_column]
        # Stratify only makes sense for classification-like targets
        if pd.api.types.is_float_dtype(series):
            return None
        return series

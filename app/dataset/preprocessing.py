"""
dataset/preprocessing.py
========================
All preprocessing logic lives here and is reused everywhere it is needed.
**Never** re-implement scaling / encoding / transforms inline in trainer
or UI code — extend this module instead.

CSV preprocessing
-----------------
* Label encoding (``LabelEncoder``)
* One-hot encoding (``pd.get_dummies`` / ``OneHotEncoder``)
* Standard scaling (``StandardScaler``)
* Min-max scaling (``MinMaxScaler``)
* Robust scaling (``RobustScaler``)
* Missing-value imputation (mean / median / mode / constant)

Image preprocessing
-------------------
* Resize
* Normalize (ImageNet mean/std or custom)
* Augmentation (flip, rotate, colour jitter) — training only
* ToTensor conversion

The ``DatasetPreprocessor`` class exposes a ``fit_transform`` /
``transform`` interface so the fitted state can be re-used on validation
and test sets without re-fitting (leak prevention).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ScalerType = Union[StandardScaler, MinMaxScaler, RobustScaler]
Numeric2D  = np.ndarray


# ---------------------------------------------------------------------------
# Scaler factory helper
# ---------------------------------------------------------------------------
_SCALER_MAP: Dict[str, type] = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
}


def _build_scaler(name: str) -> ScalerType:
    """Return a freshly instantiated scikit-learn scaler.

    Parameters
    ----------
    name:
        One of ``"standard"``, ``"minmax"``, or ``"robust"``.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    key = name.lower()
    if key not in _SCALER_MAP:
        raise ValueError(
            f"Unknown scaler '{name}'. Choose from: {list(_SCALER_MAP)}"
        )
    return _SCALER_MAP[key]()


# ---------------------------------------------------------------------------
# CSV Preprocessor
# ---------------------------------------------------------------------------

class CSVPreprocessor:
    """Fit and transform tabular (CSV) data for ML pipelines.

    Parameters
    ----------
    scaler:
        Scaling strategy for numeric columns.  One of ``"standard"``,
        ``"minmax"``, ``"robust"``.  Pass ``None`` to skip scaling.
    impute_strategy:
        Missing-value strategy passed to ``SimpleImputer``.  One of
        ``"mean"``, ``"median"``, ``"most_frequent"``, ``"constant"``.
        Pass ``None`` to skip imputation.
    impute_fill_value:
        Constant fill value when ``impute_strategy="constant"``.
    """

    def __init__(
        self,
        scaler: Optional[str] = "standard",
        impute_strategy: Optional[str] = "mean",
        impute_fill_value: Union[str, float, None] = 0,
    ) -> None:
        self.scaler_name      = scaler
        self.impute_strategy  = impute_strategy
        self.impute_fill_value = impute_fill_value

        # Fitted state — populated during ``fit``
        self._scaler:        Optional[ScalerType]   = None
        self._imputer:       Optional[SimpleImputer] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._ohe:           Optional[OneHotEncoder]  = None
        self._ohe_columns:   List[str] = []
        self._numeric_cols:  List[str] = []
        self._is_fitted:     bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> "CSVPreprocessor":
        """Fit all transformers on *df* (training set only).

        Parameters
        ----------
        df:
            Training DataFrame.
        target_column:
            Column that should **not** be scaled (it is the label).

        Returns
        -------
        self
        """
        feature_df = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df.copy()
        self._numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        # Imputer (numeric only)
        if self.impute_strategy and self._numeric_cols:
            self._imputer = SimpleImputer(
                strategy=self.impute_strategy,
                fill_value=self.impute_fill_value if self.impute_strategy == "constant" else None,
            )
            self._imputer.fit(feature_df[self._numeric_cols])

        # Scaler
        if self.scaler_name and self._numeric_cols:
            self._scaler = _build_scaler(self.scaler_name)
            numeric_data = feature_df[self._numeric_cols].copy()
            if self._imputer is not None:
                numeric_data = pd.DataFrame(
                    self._imputer.transform(numeric_data),
                    columns=self._numeric_cols,
                    index=numeric_data.index,
                )
            self._scaler.fit(numeric_data)

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply fitted transformations to *df*.

        Parameters
        ----------
        df:
            DataFrame to transform (can be train, val, or test).
        target_column:
            Column to exclude from feature scaling.

        Returns
        -------
        pd.DataFrame
            Transformed copy of *df*.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called yet.
        """
        self._assert_fitted()
        result = df.copy()
        feature_cols = [c for c in self._numeric_cols if c in result.columns]

        # Impute
        if self._imputer is not None and feature_cols:
            result[feature_cols] = self._imputer.transform(result[feature_cols])

        # Scale
        if self._scaler is not None and feature_cols:
            result[feature_cols] = self._scaler.transform(result[feature_cols])

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Convenience: ``fit`` then ``transform`` in one call.

        Parameters
        ----------
        df:
            Training DataFrame.
        target_column:
            Column to exclude from feature scaling.

        Returns
        -------
        pd.DataFrame
        """
        return self.fit(df, target_column).transform(df, target_column)

    def label_encode(self, series: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
        """Fit a ``LabelEncoder`` on *series* and return the encoded series.

        Parameters
        ----------
        series:
            Categorical column to encode.

        Returns
        -------
        Tuple[pd.Series, LabelEncoder]
            Encoded series and the fitted encoder (store to inverse-transform later).
        """
        le = LabelEncoder()
        encoded = pd.Series(le.fit_transform(series.astype(str)), index=series.index, name=series.name)
        self._label_encoders[series.name] = le
        return encoded, le

    def one_hot_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        drop_first: bool = False,
    ) -> pd.DataFrame:
        """One-hot encode specified *columns* using ``pd.get_dummies``.

        Parameters
        ----------
        df:
            Input DataFrame.
        columns:
            Categorical columns to encode.
        drop_first:
            Whether to drop the first dummy to avoid multicollinearity.

        Returns
        -------
        pd.DataFrame
            DataFrame with original *columns* replaced by dummy columns.
        """
        return pd.get_dummies(df, columns=columns, drop_first=drop_first)

    def inverse_label_encode(self, series: pd.Series) -> pd.Series:
        """Reverse label encoding for a previously encoded column.

        Parameters
        ----------
        series:
            Encoded column (must have been processed by ``label_encode``).

        Returns
        -------
        pd.Series

        Raises
        ------
        KeyError
            If the column was never label-encoded by this instance.
        """
        if series.name not in self._label_encoders:
            raise KeyError(
                f"No LabelEncoder found for column '{series.name}'. "
                "Call label_encode() first."
            )
        le = self._label_encoders[series.name]
        return pd.Series(le.inverse_transform(series.astype(int)), index=series.index, name=series.name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "CSVPreprocessor has not been fitted. Call fit() or fit_transform() first."
            )


# ---------------------------------------------------------------------------
# Image Preprocessor
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Build torchvision transform pipelines for image datasets.

    Parameters
    ----------
    image_size:
        Target ``(height, width)`` after resizing.
    mean:
        Normalisation mean (per channel).  Defaults to ImageNet mean.
    std:
        Normalisation std (per channel).  Defaults to ImageNet std.
    augment:
        Whether to include training augmentations (flip, rotate, jitter).
    """

    # ImageNet statistics
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std:  Tuple[float, float, float] = IMAGENET_STD,
        augment: bool = True,
    ) -> None:
        self.image_size = image_size
        self.mean       = mean
        self.std        = std
        self.augment    = augment

    def train_transforms(self):
        """Return the torchvision transform pipeline for the training set.

        Includes augmentations when ``self.augment=True``.

        Returns
        -------
        torchvision.transforms.Compose
        """
        try:
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for image preprocessing. "
                "Install it with: pip install torchvision"
            ) from exc

        ops = [transforms.Resize(self.image_size)]

        if self.augment:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.05,
                ),
            ]

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        return transforms.Compose(ops)

    def eval_transforms(self):
        """Return the torchvision transform pipeline for val / test sets.

        No augmentation — deterministic for reproducible evaluation.

        Returns
        -------
        torchvision.transforms.Compose
        """
        try:
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for image preprocessing. "
                "Install it with: pip install torchvision"
            ) from exc

        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])


# ---------------------------------------------------------------------------
# Unified entry-point
# ---------------------------------------------------------------------------

class DatasetPreprocessor:
    """Unified preprocessor that wraps both ``CSVPreprocessor`` and ``ImagePreprocessor``.

    Callers import only this class when they need preprocessing regardless
    of dataset type.

    Parameters
    ----------
    csv_scaler:
        Scaler name for CSV datasets.  See ``CSVPreprocessor``.
    csv_impute_strategy:
        Imputation strategy for CSV datasets.  See ``CSVPreprocessor``.
    image_size:
        Target image size for image datasets.
    augment:
        Whether to apply training augmentations for image datasets.
    """

    def __init__(
        self,
        csv_scaler: Optional[str] = "standard",
        csv_impute_strategy: Optional[str] = "mean",
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
    ) -> None:
        self.csv = CSVPreprocessor(
            scaler=csv_scaler,
            impute_strategy=csv_impute_strategy,
        )
        self.image = ImagePreprocessor(
            image_size=image_size,
            augment=augment,
        )

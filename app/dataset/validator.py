"""
dataset/validator.py
====================
Validate datasets **before** they enter the training pipeline.

CSV validation
--------------
* File is a valid, readable CSV.
* No empty DataFrame (zero rows / zero columns).
* Missing-value report per column.
* Duplicate row detection.
* Target-column presence check (when specified).

Image validation
----------------
* Root directory exists and contains at least one sub-folder (class).
* Each sub-folder contains at least one valid image file.
* Corrupted / unreadable images are reported (not silently skipped).
* Class imbalance warning when the ratio of largest/smallest class > threshold.

All validation results are returned as a ``ValidationReport`` dataclass so
that callers (trainer, UI) can inspect errors without parsing strings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
ImageSample = Tuple[str, str]   # (absolute_path, label)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured result returned by ``DatasetValidator``."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, object] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Record an error and mark the report as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Record a non-fatal warning."""
        self.warnings.append(message)

    def __str__(self) -> str:  # pragma: no cover
        lines = [f"Valid: {self.is_valid}"]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  • {e}" for e in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  ⚠ {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """Validate CSV and image datasets and return a ``ValidationReport``.

    Parameters
    ----------
    imbalance_threshold:
        Ratio of largest-to-smallest class count above which a class-
        imbalance warning is emitted.  Default ``10.0``.
    missing_value_threshold:
        Fraction of missing values per column above which a warning is
        added.  Default ``0.20`` (20 %).
    """

    def __init__(
        self,
        imbalance_threshold: float = 10.0,
        missing_value_threshold: float = 0.20,
    ) -> None:
        self.imbalance_threshold     = imbalance_threshold
        self.missing_value_threshold = missing_value_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        data: Union[pd.DataFrame, List[ImageSample]],
        *,
        target_column: Optional[str] = None,
    ) -> ValidationReport:
        """Dispatch validation based on dataset type.

        Parameters
        ----------
        data:
            Output from ``DatasetLoader.load()``.
        target_column:
            For CSV datasets, the column used as the prediction target.

        Returns
        -------
        ValidationReport
        """
        if isinstance(data, pd.DataFrame):
            return self.validate_csv(data, target_column=target_column)
        if isinstance(data, list):
            return self.validate_image_dataset(data)
        report = ValidationReport()
        report.add_error(f"Unsupported data type: {type(data)}")
        return report

    def validate_csv(
        self,
        df: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
    ) -> ValidationReport:
        """Validate a CSV DataFrame.

        Checks performed:
        * Non-empty (rows and columns present).
        * Target column exists (when supplied).
        * Missing value fractions per column.
        * Duplicate rows.

        Parameters
        ----------
        df:
            Loaded DataFrame.
        target_column:
            Optional name of the target/label column.

        Returns
        -------
        ValidationReport
        """
        report = ValidationReport()

        # Basic shape check
        if df.empty or df.shape[1] == 0:
            report.add_error("Dataset is empty (zero rows or zero columns).")
            return report

        report.info["n_rows"]    = df.shape[0]
        report.info["n_columns"] = df.shape[1]
        report.info["columns"]   = list(df.columns)

        # Target column presence
        if target_column is not None and target_column not in df.columns:
            report.add_error(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Missing values
        missing_fractions: Dict[str, float] = {}
        for col in df.columns:
            frac = df[col].isna().mean()
            if frac > 0:
                missing_fractions[col] = round(float(frac), 4)
                if frac >= self.missing_value_threshold:
                    report.add_warning(
                        f"Column '{col}' has {frac:.1%} missing values."
                    )
        report.info["missing_fractions"] = missing_fractions

        # Duplicate rows
        n_duplicates = int(df.duplicated().sum())
        report.info["n_duplicates"] = n_duplicates
        if n_duplicates > 0:
            report.add_warning(
                f"{n_duplicates} duplicate row(s) detected."
            )

        return report

    def validate_image_dataset(
        self,
        samples: List[ImageSample],
    ) -> ValidationReport:
        """Validate an image dataset.

        Checks performed:
        * At least one sample present.
        * Each image file exists on disk.
        * Each image file can be opened (PIL or file-header check).
        * Class imbalance check.

        Parameters
        ----------
        samples:
            Output from ``DatasetLoader.load_image_dataset()``.

        Returns
        -------
        ValidationReport
        """
        report = ValidationReport()

        if not samples:
            report.add_error("Image dataset is empty — no samples found.")
            return report

        # Class distribution
        class_counts: Dict[str, int] = {}
        for _, label in samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        report.info["n_samples"]      = len(samples)
        report.info["n_classes"]      = len(class_counts)
        report.info["class_counts"]   = class_counts

        if len(class_counts) < 2:
            report.add_warning(
                "Only one class found — at least two classes are needed for classification."
            )

        # Class imbalance
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            if min_count > 0 and (max_count / min_count) > self.imbalance_threshold:
                report.add_warning(
                    f"Severe class imbalance detected "
                    f"(max/min = {max_count}/{min_count} = {max_count/min_count:.1f}×). "
                    "Consider oversampling or weighted loss."
                )

        # File existence + readability
        corrupted: List[str] = []
        missing: List[str] = []

        for img_path, _ in samples:
            if not os.path.isfile(img_path):
                missing.append(img_path)
                continue
            if not self._is_readable_image(img_path):
                corrupted.append(img_path)

        if missing:
            report.add_error(
                f"{len(missing)} image file(s) not found on disk: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if corrupted:
            report.add_error(
                f"{len(corrupted)} corrupted / unreadable image(s): {corrupted[:5]}{'...' if len(corrupted) > 5 else ''}"
            )

        report.info["n_missing"]   = len(missing)
        report.info["n_corrupted"] = len(corrupted)

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_readable_image(path: str) -> bool:
        """Return True if *path* points to a readable image file.

        Tries PIL first (most reliable); falls back to a file-header sniff
        if PIL is not available so the validator never hard-depends on it.
        """
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
            return True
        except ImportError:
            # PIL not available — check magic bytes instead
            return DatasetValidator._check_image_magic(path)
        except Exception:
            return False

    @staticmethod
    def _check_image_magic(path: str) -> bool:
        """Lightweight magic-byte check for common image formats."""
        SIGNATURES: Dict[bytes, str] = {
            b"\xff\xd8\xff": "JPEG",
            b"\x89PNG":      "PNG",
            b"GIF8":         "GIF",
            b"BM":           "BMP",
            b"RIFF":         "WEBP",
        }
        try:
            with open(path, "rb") as f:
                header = f.read(12)
            for sig in SIGNATURES:
                if header.startswith(sig):
                    return True
            return False
        except OSError:
            return False

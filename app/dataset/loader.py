"""
dataset/loader.py
=================
Responsible for loading CSV and image datasets from disk into a
framework-agnostic representation.

- CSV  → ``pandas.DataFrame``
- Image → list of ``(absolute_path: str, label: str)`` tuples built
  by walking a directory that follows the ImageFolder convention::

      root/
        class_a/  img1.jpg  img2.jpg
        class_b/  img3.jpg

All path resolution relies on ``config/settings.py`` — never hardcode paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from app.config.settings import Settings


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ImageSample = Tuple[str, str]           # (absolute_path, label)
LoadResult  = Union[pd.DataFrame, List[ImageSample]]


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------
CSV_EXTENSIONS: Tuple[str, ...] = (".csv",)
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


class DatasetLoader:
    """Load CSV or image datasets from the filesystem.

    Parameters
    ----------
    settings:
        Project-wide settings object.  Defaults to ``Settings()`` which
        reads from ``config/settings.py``.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Union[str, Path]) -> LoadResult:
        """Auto-detect dataset type and delegate to the right loader.

        Parameters
        ----------
        path:
            Path to a ``.csv`` file **or** a root image directory.

        Returns
        -------
        pd.DataFrame
            When loading a CSV dataset.
        List[ImageSample]
            When loading an image dataset (list of ``(path, label)`` tuples).

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not recognised as CSV or image directory.
        """
        path = Path(path)
        self._assert_exists(path)

        if path.is_dir():
            return self.load_image_dataset(path)
        if path.suffix.lower() in CSV_EXTENSIONS:
            return self.load_csv(path)

        raise ValueError(
            f"Cannot infer dataset type from path '{path}'. "
            "Provide a .csv file or a root image directory."
        )

    def load_csv(
        self,
        path: Union[str, Path],
        *,
        encoding: str = "utf-8",
        **pd_kwargs,
    ) -> pd.DataFrame:
        """Load a CSV file into a ``pandas.DataFrame``.

        Parameters
        ----------
        path:
            Path to the CSV file.
        encoding:
            File encoding (default ``"utf-8"``).
        **pd_kwargs:
            Extra keyword arguments forwarded to ``pd.read_csv``.

        Returns
        -------
        pd.DataFrame
        """
        path = Path(path)
        self._assert_exists(path)

        try:
            df = pd.read_csv(path, encoding=encoding, **pd_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV '{path}': {exc}") from exc

        return df

    def load_image_dataset(
        self,
        root: Union[str, Path],
        *,
        extensions: Tuple[str, ...] = IMAGE_EXTENSIONS,
    ) -> List[ImageSample]:
        """Walk an ImageFolder-style directory and return ``(path, label)`` tuples.

        The *label* is the name of the immediate parent sub-directory.

        Parameters
        ----------
        root:
            Root directory containing one sub-folder per class.
        extensions:
            Allowed image file extensions (lower-cased, with leading dot).

        Returns
        -------
        List[ImageSample]
            Sorted list of ``(absolute_image_path, label)`` tuples.

        Raises
        ------
        FileNotFoundError
            If *root* does not exist.
        ValueError
            If no images are found under *root*.
        """
        root = Path(root)
        self._assert_exists(root)

        samples: List[ImageSample] = []

        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in extensions:
                    samples.append((str(file_path.resolve()), label))

        if not samples:
            raise ValueError(
                f"No images found under '{root}'. "
                f"Expected sub-folders with images having extensions: {extensions}"
            )

        return samples

    def preview(self, path: Union[str, Path], n_rows: int = 5) -> pd.DataFrame:
        """Return the first *n_rows* rows of a CSV dataset for quick inspection.

        Parameters
        ----------
        path:
            Path to a CSV file.
        n_rows:
            Number of rows to return (default 5).

        Returns
        -------
        pd.DataFrame
        """
        return self.load_csv(path, nrows=n_rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_exists(path: Path) -> None:
        """Raise ``FileNotFoundError`` if *path* does not exist."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: '{path}'")

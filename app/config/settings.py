"""
config/settings.py
==================
Runtime-configurable settings for DeepVisionLab.

Paths are resolved relative to the **project root** (the directory that
contains ``main.py``).  No business logic or UI code should ever
hardcode a path — they should import ``Settings`` and read from here.

Usage::

    from app.config.settings import Settings

    cfg = Settings()
    print(cfg.datasets_dir)   # Path("/abs/path/to/datasets")
"""

from __future__ import annotations

import os
from pathlib import Path

from app.config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMPUTE_STRATEGY,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SCALER,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def _project_root() -> Path:
    """Resolve the project root directory (parent of ``app/``)."""
    # This file lives at  <root>/app/config/settings.py
    return Path(__file__).resolve().parent.parent.parent


class Settings:
    """Centralised runtime settings.

    All attributes default to sensible values but can be overridden by
    setting matching environment variables, e.g.::

        export DVLAB_DATASETS_DIR=/data/my_datasets

    Parameters
    ----------
    project_root:
        Explicit project root override.  Defaults to auto-detection.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self._root: Path = project_root or _project_root()

    # ------------------------------------------------------------------
    # Directory paths
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root directory."""
        return self._root

    @property
    def datasets_dir(self) -> Path:
        """Directory where raw / uploaded datasets are stored."""
        return self._resolve("DVLAB_DATASETS_DIR", self._root / "datasets")

    @property
    def saved_models_dir(self) -> Path:
        """Directory where trained model artefacts are saved."""
        return self._resolve("DVLAB_SAVED_MODELS_DIR", self._root / "saved_models")

    @property
    def experiments_dir(self) -> Path:
        """Directory for experiment run outputs."""
        return self._resolve("DVLAB_EXPERIMENTS_DIR", self._root / "experiments")

    @property
    def logs_dir(self) -> Path:
        """Directory for training and inference logs."""
        return self._resolve("DVLAB_LOGS_DIR", self._root / "logs")

    # ------------------------------------------------------------------
    # Dataset / split settings
    # ------------------------------------------------------------------

    @property
    def test_size(self) -> float:
        return float(os.environ.get("DVLAB_TEST_SIZE", DEFAULT_TEST_SIZE))

    @property
    def val_size(self) -> float:
        return float(os.environ.get("DVLAB_VAL_SIZE", DEFAULT_VAL_SIZE))

    @property
    def random_state(self) -> int:
        return int(os.environ.get("DVLAB_RANDOM_STATE", DEFAULT_RANDOM_STATE))

    # ------------------------------------------------------------------
    # Preprocessing settings
    # ------------------------------------------------------------------

    @property
    def scaler(self) -> str:
        return os.environ.get("DVLAB_SCALER", DEFAULT_SCALER)

    @property
    def impute_strategy(self) -> str:
        return os.environ.get("DVLAB_IMPUTE_STRATEGY", DEFAULT_IMPUTE_STRATEGY)

    # ------------------------------------------------------------------
    # Image settings
    # ------------------------------------------------------------------

    @property
    def image_size(self) -> tuple[int, int]:
        raw = os.environ.get("DVLAB_IMAGE_SIZE", "")
        if raw:
            try:
                h, w = raw.split(",")
                return int(h.strip()), int(w.strip())
            except ValueError:
                pass
        return DEFAULT_IMAGE_SIZE

    @property
    def imagenet_mean(self) -> tuple[float, float, float]:
        return IMAGENET_MEAN

    @property
    def imagenet_std(self) -> tuple[float, float, float]:
        return IMAGENET_STD

    # ------------------------------------------------------------------
    # Training settings
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return int(os.environ.get("DVLAB_BATCH_SIZE", DEFAULT_BATCH_SIZE))

    @property
    def epochs(self) -> int:
        return int(os.environ.get("DVLAB_EPOCHS", DEFAULT_EPOCHS))

    @property
    def learning_rate(self) -> float:
        return float(os.environ.get("DVLAB_LEARNING_RATE", DEFAULT_LEARNING_RATE))

    @property
    def patience(self) -> int:
        return int(os.environ.get("DVLAB_PATIENCE", DEFAULT_PATIENCE))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(env_var: str, default: Path) -> Path:
        """Return the path from *env_var* if set, else *default*."""
        raw = os.environ.get(env_var, "")
        return Path(raw) if raw else default

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Settings(root={self._root}, "
            f"datasets={self.datasets_dir}, "
            f"saved_models={self.saved_models_dir})"
        )

"""
utils/logger.py
===============
Centralised logging configuration for DeepVisionLab.

All modules should obtain their logger via::

    import logging
    logger = logging.getLogger(__name__)

Then call ``setup_logger()`` once at application start (``main.py``) to
configure the root logger with file + console handlers.

Usage::

    from app.utils.logger import setup_logger

    setup_logger()  # call once at startup
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.config.constants import LOG_DATE_FORMAT, LOG_FORMAT
from app.config.settings import Settings


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> logging.Logger:
    """Configure and return a logger with console and file handlers.

    Parameters
    ----------
    name:
        Logger name.  ``None`` configures the root logger (recommended
        for application startup).
    level:
        Logging level.  Default ``logging.INFO``.
    log_file:
        Explicit log file name (e.g. ``"training.log"``).  When ``None``,
        defaults to ``"deepvisionlab.log"``.
    settings:
        Project-wide settings for resolving ``logs_dir``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    settings = settings or Settings()
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File handler ---
    log_dir = settings.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = log_file or "deepvisionlab.log"
    file_path = log_dir / log_filename

    file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (convenience wrapper).

    Parameters
    ----------
    name:
        Logger name, typically ``__name__``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)

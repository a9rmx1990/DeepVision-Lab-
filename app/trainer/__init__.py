"""
app.trainer
===========
Training pipeline: training loop, validation, checkpointing, and early stopping.

Exposes:
- ``Trainer``: Main orchestrator for scikit-learn and PyTorch training.
- ``TrainingResult``: Dataclass holding training run outputs (losses, metrics, paths).
- ``EarlyStopping``: Callback to halt training when a metric stops improving.
- ``CheckpointManager``: Save / load model checkpoints (PyTorch and sklearn).
- ``TrainingValidator``: Evaluate models during training (per-epoch validation).

Usage::

    from app.trainer import Trainer, TrainingResult
    from app.models.model_factory import ModelFactory

    model = ModelFactory.create("classification", "random_forest")
    trainer = Trainer(model=model, task="classification")
    result = trainer.train_sklearn(X_train, y_train, X_val, y_val)
"""

from app.trainer.checkpoint import CheckpointManager
from app.trainer.early_stopping import EarlyStopping
from app.trainer.trainer import Trainer, TrainingResult
from app.trainer.validator import TrainingValidator

__all__ = [
    "Trainer",
    "TrainingResult",
    "EarlyStopping",
    "CheckpointManager",
    "TrainingValidator",
]

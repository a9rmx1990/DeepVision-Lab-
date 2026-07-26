"""
app.models
==========
Model Architectures (Classical ML & Deep Learning) and Model Factory.

Exposes:
- ``ModelFactory``: Unified factory for building scikit-learn & PyTorch models.
- ``SimpleCNN``, ``FlexibleCNN``, ``ConvBlock``: Custom PyTorch CNN architectures.
- ``PretrainedCNN``: TorchVision transfer learning wrapper (ResNet, EfficientNet, MobileNet, VGG).
- ``SimpleRNN``: PyTorch RNN / GRU model.
- ``SimpleLSTM``: PyTorch LSTM model.
"""

from app.models.cnn import ConvBlock, FlexibleCNN, SimpleCNN
from app.models.lstm import SimpleLSTM
from app.models.model_factory import ModelFactory
from app.models.pretrained import PretrainedCNN
from app.models.rnn import SimpleRNN

__all__ = [
    "ModelFactory",
    "ConvBlock",
    "SimpleCNN",
    "FlexibleCNN",
    "PretrainedCNN",
    "SimpleRNN",
    "SimpleLSTM",
]

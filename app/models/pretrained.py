"""
app/models/pretrained.py
========================
Pretrained Transfer Learning models (ResNet, EfficientNet, MobileNet, VGG)
from torchvision.models for Image Classification tasks in DeepVisionLab.

Modules included:
- ``PretrainedCNN``: Wrapper around torchvision pretrained backbones with custom classification heads
  and support for freezing/unfreezing feature extractor backbones.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _check_torch_available() -> None:
    """Raise ImportError if PyTorch/TorchVision is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch and TorchVision are required for app.models.pretrained module. "
            "Install via: pip install torch torchvision"
        )


# Supported backbones mapped to torchvision factory callables
SUPPORTED_BACKBONES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b1",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "vgg16",
)


if _TORCH_AVAILABLE:

    class PretrainedCNN(nn.Module):
        """Pretrained Transfer Learning Model wrapper using TorchVision.

        Parameters
        ----------
        model_name:
            Name of the pretrained architecture. Choose from:
            "resnet18", "resnet34", "resnet50", "efficientnet_b0", "efficientnet_b1",
            "mobilenet_v3_small", "mobilenet_v3_large", "vgg16".
        num_classes:
            Number of output classes. Default 10.
        pretrained:
            Whether to load default pretrained weights (e.g. ImageNet). Default True.
        freeze_backbone:
            Whether to freeze the parameters of the feature extraction backbone. Default False.
        dropout_rate:
            Dropout probability in the new classification head. Default 0.2.
        """

        def __init__(
            self,
            model_name: str = "resnet18",
            num_classes: int = 10,
            pretrained: bool = True,
            freeze_backbone: bool = False,
            dropout_rate: float = 0.2,
        ) -> None:
            super().__init__()

            name_lower = model_name.lower().replace("-", "_")
            if name_lower not in SUPPORTED_BACKBONES:
                raise ValueError(
                    f"Unsupported backbone '{model_name}'. Choose from: {SUPPORTED_BACKBONES}"
                )

            self.model_name = name_lower
            self.num_classes = num_classes
            self.freeze_backbone = freeze_backbone
            self.dropout_rate = dropout_rate

            # Build backbone with appropriate weights
            self.backbone = self._load_backbone(name_lower, pretrained=pretrained)

            # Modify final classification head
            self._replace_classifier(name_lower, num_classes, dropout_rate)

            # Optionally freeze feature extractor weights
            if freeze_backbone:
                self.set_freeze_backbone(True)

        def _load_backbone(self, name: str, pretrained: bool) -> nn.Module:
            """Instantiate model architecture from torchvision."""
            if name == "resnet18":
                weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
                return tv_models.resnet18(weights=weights)
            if name == "resnet34":
                weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
                return tv_models.resnet34(weights=weights)
            if name == "resnet50":
                weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
                return tv_models.resnet50(weights=weights)
            if name == "efficientnet_b0":
                weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                return tv_models.efficientnet_b0(weights=weights)
            if name == "efficientnet_b1":
                weights = tv_models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
                return tv_models.efficientnet_b1(weights=weights)
            if name == "mobilenet_v3_small":
                weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                return tv_models.mobilenet_v3_small(weights=weights)
            if name == "mobilenet_v3_large":
                weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
                return tv_models.mobilenet_v3_large(weights=weights)
            if name == "vgg16":
                weights = tv_models.VGG16_Weights.DEFAULT if pretrained else None
                return tv_models.vgg16(weights=weights)

            raise ValueError(f"Unhandled backbone: {name}")

        def _replace_classifier(self, name: str, num_classes: int, dropout_rate: float) -> None:
            """Replace original final linear layer with custom classification head."""
            if "resnet" in name:
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, num_classes),
                )
            elif "efficientnet" in name or "mobilenet" in name:
                in_features = self.backbone.classifier[-1].in_features
                classifier_list = list(self.backbone.classifier)
                classifier_list[-1] = nn.Linear(in_features, num_classes)
                self.backbone.classifier = nn.Sequential(*classifier_list)
            elif "vgg" in name:
                in_features = self.backbone.classifier[-1].in_features
                classifier_list = list(self.backbone.classifier)
                classifier_list[-1] = nn.Linear(in_features, num_classes)
                self.backbone.classifier = nn.Sequential(*classifier_list)

        def set_freeze_backbone(self, freeze: bool = True) -> None:
            """Freeze or unfreeze the backbone parameters (excluding final classifier)."""
            self.freeze_backbone = freeze
            for name, param in self.backbone.named_parameters():
                # Keep final classifier layer trainable
                if "fc" in name or "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = not freeze

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through pretrained model."""
            return self.backbone(x)

        def count_parameters(self) -> Dict[str, int]:
            """Return trainable and total parameter counts."""
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return {"trainable": trainable, "total": total}

else:
    class PretrainedCNN:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            _check_torch_available()

"""
app/models/cnn.py
=================
Convolutional Neural Network (CNN) architectures for Image Classification
and spatial feature extraction in DeepVisionLab.

Modules included:
- ``ConvBlock``: Modular 2D Convolution + BatchNorm + Activation + Pooling + Dropout layer.
- ``SimpleCNN``: Lightweight 3-block CNN baseline for fast training and low memory footprints.
- ``FlexibleCNN``: Customizable multi-stage CNN supporting arbitrary channels, block depths,
  activation functions, dropout rates, and classification heads.

All architectures use ``nn.AdaptiveAvgPool2d((1, 1))`` before the final classification head
to seamlessly process arbitrary input spatial dimensions (e.g. 32x32, 64x64, 224x224).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _check_torch_available() -> None:
    """Raise ImportError if PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for app.models.cnn module. "
            "Install it via: pip install torch"
        )


def _get_activation_module(activation_name: str) -> nn.Module:
    """Return activation nn.Module matching *activation_name*."""
    name = activation_name.lower().replace("_", "")
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)

    raise ValueError(
        f"Unsupported activation '{activation_name}'. "
        "Supported: 'relu', 'leaky_relu', 'gelu', 'silu', 'elu'."
    )


if _TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Modular 2D Convolutional Block.

        Sequence: Conv2d -> BatchNorm2d (optional) -> Activation -> MaxPool2d (optional) -> Dropout (optional).

        Parameters
        ----------
        in_channels:
            Number of channels in the input image/tensor.
        out_channels:
            Number of feature maps produced by the convolution.
        kernel_size:
            Size of the convolving kernel. Default 3.
        stride:
            Stride of the convolution. Default 1.
        padding:
            Zero-padding added to both sides of the input. Default 1.
        use_batch_norm:
            Whether to apply BatchNorm2d after convolution. Default True.
        activation:
            Activation function name ("relu", "leaky_relu", "gelu", "silu", "elu"). Default "relu".
        pool:
            Whether to apply 2x2 MaxPool2d after activation. Default True.
        dropout_rate:
            Dropout probability applied at the end of the block. Default 0.0.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            use_batch_norm: bool = True,
            activation: str = "relu",
            pool: bool = True,
            dropout_rate: float = 0.0,
        ) -> None:
            super().__init__()

            layers: List[nn.Module] = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=not use_batch_norm,
                )
            ]

            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(_get_activation_module(activation))

            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            if dropout_rate > 0.0:
                layers.append(nn.Dropout2d(p=dropout_rate))

            self.block = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Pass tensor through convolutional block."""
            return self.block(x)


    class SimpleCNN(nn.Module):
        """Lightweight 3-stage CNN Baseline for image classification.

        Architecture:
        - ConvBlock 1: in_channels -> 32 filters, 2x2 MaxPool
        - ConvBlock 2: 32 -> 64 filters, 2x2 MaxPool
        - ConvBlock 3: 64 -> 128 filters, 2x2 MaxPool
        - AdaptiveAvgPool2d((1, 1))
        - Classifier: Linear(128, 128) -> ReLU -> Dropout -> Linear(128, num_classes)

        Parameters
        ----------
        in_channels:
            Number of input image channels (e.g., 3 for RGB, 1 for grayscale). Default 3.
        num_classes:
            Number of output classes. Default 10.
        dropout_rate:
            Dropout probability in the classifier head. Default 0.2.
        use_batch_norm:
            Whether to use BatchNorm2d in ConvBlocks. Default True.
        """

        def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
            dropout_rate: float = 0.2,
            use_batch_norm: bool = True,
        ) -> None:
            super().__init__()

            self.in_channels = in_channels
            self.num_classes = num_classes
            self.dropout_rate = dropout_rate
            self.use_batch_norm = use_batch_norm

            self.feature_extractor = nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=32,
                    use_batch_norm=use_batch_norm,
                    pool=True,
                ),
                ConvBlock(
                    in_channels=32,
                    out_channels=64,
                    use_batch_norm=use_batch_norm,
                    pool=True,
                ),
                ConvBlock(
                    in_channels=64,
                    out_channels=128,
                    use_batch_norm=use_batch_norm,
                    pool=True,
                ),
            )

            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )

            self._init_weights()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through feature extractor, global pool, and classifier.

            Parameters
            ----------
            x:
                Input tensor of shape ``(N, C, H, W)``.

            Returns
            -------
            torch.Tensor
                Output logits tensor of shape ``(N, num_classes)``.
            """
            features = self.feature_extractor(x)
            pooled = self.global_pool(features)
            logits = self.classifier(pooled)
            return logits

        def _init_weights(self) -> None:
            """Initialize weights with Kaiming Normal for Conv/Linear layers."""
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def count_parameters(self) -> Dict[str, int]:
            """Return trainable and total parameter counts.

            Returns
            -------
            Dict[str, int]
                Dictionary with keys ``trainable`` and ``total``.
            """
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return {"trainable": trainable, "total": total}


    class FlexibleCNN(nn.Module):
        """Highly Configurable Deep CNN Architecture for Image Classification.

        Allows custom channel progressions (e.g. [32, 64, 128, 256]), custom dense heads,
        flexible activations, batch normalization, and pooling strategies.

        Parameters
        ----------
        in_channels:
            Number of input image channels (e.g., 3 for RGB, 1 for Grayscale). Default 3.
        num_classes:
            Number of target classification classes. Default 10.
        channel_list:
            Sequence of output channels for each conv stage. Default (32, 64, 128, 256).
        fc_dims:
            Sequence of hidden layer dimensions in the FC classifier head. Default (256,).
        use_batch_norm:
            Whether to use BatchNorm2d in ConvBlocks. Default True.
        activation:
            Activation function name. Default "relu".
        dropout_rate:
            Dropout probability used in ConvBlocks and FC layers. Default 0.3.
        pool_layers:
            Boolean or sequence of booleans determining if MaxPool2d is applied per conv stage.
            Default True (pools after every stage).
        """

        def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
            channel_list: Sequence[int] = (32, 64, 128, 256),
            fc_dims: Sequence[int] = (256,),
            use_batch_norm: bool = True,
            activation: str = "relu",
            dropout_rate: float = 0.3,
            pool_layers: Union[bool, Sequence[bool]] = True,
        ) -> None:
            super().__init__()

            if not channel_list:
                raise ValueError("channel_list cannot be empty.")

            self.in_channels = in_channels
            self.num_classes = num_classes
            self.channel_list = list(channel_list)
            self.fc_dims = list(fc_dims)
            self.use_batch_norm = use_batch_norm
            self.activation = activation
            self.dropout_rate = dropout_rate

            # Resolve pooling strategy for each block
            if isinstance(pool_layers, bool):
                pool_flags = [pool_layers] * len(channel_list)
            else:
                if len(pool_layers) != len(channel_list):
                    raise ValueError(
                        f"Length of pool_layers ({len(pool_layers)}) must match "
                        f"channel_list ({len(channel_list)})."
                    )
                pool_flags = list(pool_layers)

            # Build Conv Blocks
            conv_stages: List[nn.Module] = []
            curr_in = in_channels

            for curr_out, do_pool in zip(channel_list, pool_flags):
                conv_stages.append(
                    ConvBlock(
                        in_channels=curr_in,
                        out_channels=curr_out,
                        use_batch_norm=use_batch_norm,
                        activation=activation,
                        pool=do_pool,
                        dropout_rate=dropout_rate / 2.0,  # mild spatial dropout
                    )
                )
                curr_in = curr_out

            self.feature_extractor = nn.Sequential(*conv_stages)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            # Build Classifier Head
            classifier_layers: List[nn.Module] = [nn.Flatten()]
            in_dim = channel_list[-1]

            for h_dim in fc_dims:
                classifier_layers.append(nn.Linear(in_dim, h_dim))
                if use_batch_norm:
                    classifier_layers.append(nn.BatchNorm1d(h_dim))
                classifier_layers.append(_get_activation_module(activation))
                if dropout_rate > 0.0:
                    classifier_layers.append(nn.Dropout(p=dropout_rate))
                in_dim = h_dim

            classifier_layers.append(nn.Linear(in_dim, num_classes))
            self.classifier = nn.Sequential(*classifier_layers)

            self._init_weights()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass returning class logits tensor of shape ``(N, num_classes)``."""
            features = self.feature_extractor(x)
            pooled = self.global_pool(features)
            logits = self.classifier(pooled)
            return logits

        def _init_weights(self) -> None:
            """Initialize weights using Kaiming Normal initialization."""
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def count_parameters(self) -> Dict[str, int]:
            """Return trainable and total parameter counts."""
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return {"trainable": trainable, "total": total}

else:
    # Dummy placeholder classes if PyTorch is absent at runtime
    class ConvBlock:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            _check_torch_available()

    class SimpleCNN:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            _check_torch_available()

    class FlexibleCNN:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            _check_torch_available()

"""
app/models/lstm.py
==================
Long Short-Term Memory (LSTM) architectures for sequence classification,
time-series forecasting, and tabular sequence modeling in DeepVisionLab.

Modules included:
- ``SimpleLSTM``: PyTorch nn.LSTM model for time-series forecasting and sequence tasks.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _check_torch_available() -> None:
    """Raise ImportError if PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for app.models.lstm module. "
            "Install via: pip install torch"
        )


if _TORCH_AVAILABLE:

    class SimpleLSTM(nn.Module):
        """Long Short-Term Memory (LSTM) Model for time-series forecasting and sequence tasks.

        Parameters
        ----------
        input_size:
            Number of input features per timestep. Default 1.
        hidden_size:
            Number of features in the hidden state. Default 64.
        num_layers:
            Number of LSTM layers. Default 2.
        output_size:
            Number of output targets (e.g. 1 for regression/forecasting, num_classes for classification). Default 1.
        dropout_rate:
            Dropout probability between LSTM layers and FC head. Default 0.2.
        bidirectional:
            Whether to use bidirectional LSTM. Default False.
        """

        def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 64,
            num_layers: int = 2,
            output_size: int = 1,
            dropout_rate: float = 0.2,
            bidirectional: bool = False,
        ) -> None:
            super().__init__()

            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            dropout = dropout_rate if num_layers > 1 else 0.0

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )

            fc_input_dim = hidden_size * self.num_directions
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            x:
                Input tensor of shape ``(N, L, H_in)`` where N=batch_size, L=sequence_length, H_in=input_size.

            Returns
            -------
            torch.Tensor
                Output predictions of shape ``(N, output_size)``.
            """
            out, (hn, cn) = self.lstm(x)
            # Extract last timestep output
            last_out = out[:, -1, :]
            prediction = self.fc(last_out)
            return prediction

        def count_parameters(self) -> Dict[str, int]:
            """Return trainable and total parameter counts."""
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return {"trainable": trainable, "total": total}

else:
    class SimpleLSTM:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            _check_torch_available()

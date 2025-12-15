"""
CNN-Transformer hybrid model for statistical arbitrage.

Combines convolutional layers for local pattern extraction with
transformer attention for capturing dependencies across assets.
"""

from typing import Optional, List
import torch
import torch.nn as nn

from models.base import BaseModel


class CNNBlock(nn.Module):
    """
    Convolutional block with residual connection.

    Architecture:
        Input -> InstNorm -> Pad -> Conv -> ReLU -> InstNorm -> Pad -> Conv -> ReLU -> (+Input)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        use_normalization: bool = True
    ):
        """
        Initialize CNN block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            use_normalization: Whether to use instance normalization
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_normalization = use_normalization

        # First convolution path
        layers1 = []
        if use_normalization:
            layers1.append(nn.InstanceNorm1d(in_channels))
        layers1.extend([
            nn.ConstantPad1d((kernel_size - 1, 0), 0),  # Left padding for causal conv
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.ReLU(inplace=True)
        ])
        self.conv1 = nn.Sequential(*layers1)

        # Second convolution path
        layers2 = []
        if use_normalization:
            layers2.append(nn.InstanceNorm1d(out_channels))
        layers2.extend([
            nn.ConstantPad1d((kernel_size - 1, 0), 0),
            nn.Conv1d(out_channels, out_channels, kernel_size),
            nn.ReLU(inplace=True)
        ])
        self.conv2 = nn.Sequential(*layers2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (N, C_in, T)

        Returns:
            Output tensor of shape (N, C_out, T)
        """
        out = self.conv1(x)
        out = self.conv2(out)

        # Residual connection: repeat input channels if needed
        if self.in_channels != self.out_channels:
            repeats = self.out_channels // self.in_channels
            x = x.repeat(1, repeats, 1)

        return out + x


class CNNTransformer(BaseModel):
    """
    CNN-Transformer model for generating portfolio weights.

    The model processes time series features through:
    1. CNN blocks for local pattern extraction
    2. Transformer encoder for cross-asset attention
    3. Linear projection to scalar weight per asset

    Input: (N, T) - N assets, T time steps
    Output: (N,) - weight per asset
    """

    is_trainable = True
    is_frictions_model = False

    def __init__(
        self,
        lookback: int,
        filter_numbers: List[int] = None,
        attention_heads: int = 4,
        hidden_units_factor: int = 2,
        dropout: float = 0.25,
        filter_size: int = 2,
        use_convolution: bool = True,
        use_transformer: bool = True,
        random_seed: int = 0,
        **kwargs
    ):
        """
        Initialize CNN-Transformer model.

        Args:
            lookback: Lookback window (sequence length)
            filter_numbers: List of CNN filter sizes [in, out1, out2, ...]
            attention_heads: Number of attention heads in transformer
            hidden_units_factor: Multiplier for FFN hidden dimension
            dropout: Dropout rate
            filter_size: CNN kernel size
            use_convolution: Whether to use CNN layers
            use_transformer: Whether to use transformer layer
            random_seed: Random seed for reproducibility
        """
        super().__init__(lookback=lookback, random_seed=random_seed)

        if filter_numbers is None:
            filter_numbers = [1, 8]

        self.filter_numbers = filter_numbers
        self.attention_heads = attention_heads
        self.hidden_units_factor = hidden_units_factor
        self.dropout = dropout
        self.filter_size = filter_size
        self.use_convolution = use_convolution
        self.use_transformer = use_transformer

        # Determine the dimension after CNN
        self.feature_dim = filter_numbers[-1] if use_convolution else 1

        # CNN blocks
        if use_convolution:
            cnn_blocks = []
            for i in range(len(filter_numbers) - 1):
                cnn_blocks.append(
                    CNNBlock(
                        in_channels=filter_numbers[i],
                        out_channels=filter_numbers[i + 1],
                        kernel_size=filter_size
                    )
                )
            self.cnn = nn.Sequential(*cnn_blocks)
        else:
            self.cnn = None

        # Transformer encoder
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=attention_heads,
                dim_feedforward=hidden_units_factor * self.feature_dim,
                dropout=dropout,
                batch_first=False  # (T, N, D) format
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        else:
            self.transformer = None

        # Output projection: feature_dim -> 1
        self.output_proj = nn.Linear(self.feature_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        old_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute portfolio weights.

        Args:
            x: Input features of shape (N, T)
            old_weights: Not used (for interface compatibility)

        Returns:
            Unnormalized portfolio weights of shape (N,)
        """
        N, T = x.shape

        # Add channel dimension: (N, T) -> (N, 1, T)
        x = x.unsqueeze(1)

        # CNN: (N, 1, T) -> (N, C, T)
        if self.cnn is not None:
            x = self.cnn(x)

        # Reshape for transformer: (N, C, T) -> (T, N, C)
        x = x.permute(2, 0, 1)

        # Transformer: (T, N, C) -> (T, N, C)
        if self.transformer is not None:
            x = self.transformer(x)

        # Take last time step: (T, N, C) -> (N, C)
        x = x[-1]

        # Project to scalar: (N, C) -> (N, 1) -> (N,)
        weights = self.output_proj(x).squeeze(-1)

        return weights

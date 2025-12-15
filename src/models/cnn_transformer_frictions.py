"""
CNN-Transformer model with transaction cost awareness.

A variant of the CNN-Transformer that takes previous portfolio
weights as input to model transaction costs explicitly.
"""

from typing import Optional, List
import torch
import torch.nn as nn

from models.base import BaseModel
from models.cnn_transformer import CNNBlock


class WeightsTransformer(nn.Module):
    """
    Transformer that conditions on previous weights.

    Concatenates old weights to the feature representation
    before applying transformer attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.25
    ):
        """
        Initialize weights-conditioned transformer.

        Args:
            d_model: Model dimension (feature channels + 1 for weights)
            nhead: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # Weight embedding
        self.weight_proj = nn.Linear(1, d_model)

        # Transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )

    def forward(
        self,
        x: torch.Tensor,
        old_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass conditioning on old weights.

        Args:
            x: Features of shape (T, N, D)
            old_weights: Previous weights of shape (N,)

        Returns:
            Output of shape (T, N, D)
        """
        T, N, D = x.shape

        # Embed old weights: (N,) -> (N, D)
        weight_emb = self.weight_proj(old_weights.unsqueeze(-1))

        # Add weight embedding to each time step
        x = x + weight_emb.unsqueeze(0)

        # Apply transformer
        return self.transformer(x)


class CNNTransformerFrictions(BaseModel):
    """
    CNN-Transformer with transaction friction modeling.

    This model takes previous portfolio weights as input to
    explicitly model transaction costs. Requires sequential
    processing (cannot batch across time).

    Input: (N, T) features + (N,) old weights
    Output: (N,) new weights
    """

    is_trainable = True
    is_frictions_model = True

    def __init__(
        self,
        lookback: int,
        filter_numbers: List[int] = None,
        attention_heads: int = 4,
        hidden_units_factor: int = 2,
        dropout: float = 0.25,
        filter_size: int = 2,
        use_convolution: bool = True,
        random_seed: int = 0,
        **kwargs
    ):
        """
        Initialize CNN-Transformer with frictions.

        Args:
            lookback: Lookback window (sequence length)
            filter_numbers: List of CNN filter sizes
            attention_heads: Number of attention heads
            hidden_units_factor: Multiplier for FFN hidden dimension
            dropout: Dropout rate
            filter_size: CNN kernel size
            use_convolution: Whether to use CNN layers
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

        # Weights-conditioned transformer
        self.transformer = WeightsTransformer(
            d_model=self.feature_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_units_factor * self.feature_dim,
            dropout=dropout
        )

        # Output projection
        self.output_proj = nn.Linear(self.feature_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        old_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass conditioning on previous weights.

        Args:
            x: Input features of shape (N, T)
            old_weights: Previous portfolio weights of shape (N,)

        Returns:
            Unnormalized portfolio weights of shape (N,)
        """
        N, T = x.shape

        # Initialize old_weights if not provided
        if old_weights is None:
            old_weights = torch.zeros(N, device=x.device, dtype=x.dtype)

        # Add channel dimension: (N, T) -> (N, 1, T)
        x = x.unsqueeze(1)

        # CNN: (N, 1, T) -> (N, C, T)
        if self.cnn is not None:
            x = self.cnn(x)

        # Reshape for transformer: (N, C, T) -> (T, N, C)
        x = x.permute(2, 0, 1)

        # Weights-conditioned transformer: (T, N, C) -> (T, N, C)
        x = self.transformer(x, old_weights)

        # Take last time step: (T, N, C) -> (N, C)
        x = x[-1]

        # Project to scalar: (N, C) -> (N, 1) -> (N,)
        weights = self.output_proj(x).squeeze(-1)

        return weights

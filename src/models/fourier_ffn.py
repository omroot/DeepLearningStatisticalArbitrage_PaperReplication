"""
Fourier Feedforward Network model.

A simpler baseline model using fully connected layers
on Fourier-transformed features.
"""

from typing import Optional, List
import torch
import torch.nn as nn

from models.base import BaseModel


class FourierFFN(BaseModel):
    """
    Feedforward network for Fourier features.

    A multi-layer perceptron that operates on frequency-domain
    features extracted from time series.

    Input: (N, feature_dim) - Fourier coefficients
    Output: (N,) - weight per asset
    """

    is_trainable = True
    is_frictions_model = False

    def __init__(
        self,
        lookback: int,
        hidden_units: List[int] = None,
        dropout: float = 0.25,
        random_seed: int = 0,
        **kwargs
    ):
        """
        Initialize Fourier FFN model.

        Args:
            lookback: Feature dimension (Fourier coefficients)
            hidden_units: List of hidden layer sizes
            dropout: Dropout rate
            random_seed: Random seed for reproducibility
        """
        super().__init__(lookback=lookback, random_seed=random_seed)

        if hidden_units is None:
            hidden_units = [30, 16, 8, 4]

        self.hidden_units = hidden_units
        self.dropout = dropout

        # Build layers
        layers = []
        in_dim = lookback

        for i, out_dim in enumerate(hidden_units):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        # Output layer
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        old_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute portfolio weights.

        Args:
            x: Input features of shape (N, feature_dim)
            old_weights: Not used (for interface compatibility)

        Returns:
            Unnormalized portfolio weights of shape (N,)
        """
        # (N, feature_dim) -> (N, 1) -> (N,)
        return self.network(x).squeeze(-1)

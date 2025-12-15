"""
Ornstein-Uhlenbeck threshold model.

A non-trainable baseline that generates positions based on
mean-reversion signals from estimated OU parameters.
"""

from typing import Optional
import torch
import torch.nn as nn

from models.base import BaseModel


class OUThreshold(BaseModel):
    """
    Ornstein-Uhlenbeck threshold trading model.

    This is a non-trainable baseline that:
    1. Estimates OU process parameters from cumulative returns
    2. Computes mean-reversion signal: (μ - Y_t) / σ
    3. Takes positions when |signal| > threshold

    Input: (N, 4) - [Y_t, μ, σ, R²] OU parameters
    Output: (N,) - positions in {-1, 0, 1}
    """

    is_trainable = False
    is_frictions_model = False

    def __init__(
        self,
        lookback: int,
        signal_threshold: float = 1.25,
        r2_threshold: float = 0.25,
        use_robust_estimators: bool = False,
        random_seed: int = 0,
        **kwargs
    ):
        """
        Initialize OU threshold model.

        Args:
            lookback: Not used (for interface compatibility)
            signal_threshold: Threshold in std devs for taking positions
            r2_threshold: Minimum R² for valid signal
            use_robust_estimators: Whether to use median/MAD
            random_seed: Random seed (not used, model is deterministic)
        """
        super().__init__(lookback=lookback, random_seed=random_seed)

        self.signal_threshold = signal_threshold
        self.r2_threshold = r2_threshold
        self.use_robust_estimators = use_robust_estimators

        # Register as buffer (not parameter) for state_dict compatibility
        self.register_buffer(
            'threshold_tensor',
            torch.tensor(signal_threshold)
        )

    def forward(
        self,
        x: torch.Tensor,
        old_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute portfolio positions.

        Args:
            x: OU parameters of shape (N, 4) = [Y_t, μ, σ, R²]
            old_weights: Not used

        Returns:
            Portfolio positions of shape (N,) in {-1, 0, 1}
        """
        # Extract OU parameters
        Y_t = x[:, 0]      # Current cumulative return
        mu = x[:, 1]       # Long-term mean
        sigma = x[:, 2]    # Volatility
        r_squared = x[:, 3]  # Regression R²

        # Compute mean-reversion signal
        eps = 1e-8
        signal = (mu - Y_t) / (sigma + eps)

        # Apply R² filter
        valid_r2 = r_squared >= self.r2_threshold

        # Generate positions based on threshold
        positions = torch.zeros_like(signal)

        # Long position: signal > threshold (price below mean)
        positions = torch.where(
            valid_r2 & (signal > self.signal_threshold),
            torch.ones_like(signal),
            positions
        )

        # Short position: signal < -threshold (price above mean)
        positions = torch.where(
            valid_r2 & (signal < -self.signal_threshold),
            -torch.ones_like(signal),
            positions
        )

        return positions

    def get_num_parameters(self) -> int:
        """This model has no trainable parameters."""
        return 0

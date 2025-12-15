"""
Base class for neural network models.

Defines the interface and common functionality for all trading models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for trading models.

    All models must implement the forward pass and declare whether
    they are trainable and whether they handle transaction frictions.
    """

    # Class attributes to be overridden by subclasses
    is_trainable: bool = True
    is_frictions_model: bool = False

    def __init__(
        self,
        lookback: int,
        random_seed: int = 0,
        **kwargs
    ):
        """
        Initialize the base model.

        Args:
            lookback: Lookback window for features
            random_seed: Random seed for reproducibility
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
        self.lookback = lookback
        self.random_seed = random_seed

        # Set random seed
        torch.manual_seed(random_seed)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        old_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute portfolio weights.

        Args:
            x: Input features of shape (N, T) or (N, feature_dim)
               where N is number of assets
            old_weights: Previous weights for frictions models (N,)

        Returns:
            Unnormalized portfolio weights of shape (N,)
        """
        pass

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_summary(self) -> Dict[str, Tuple[Tuple[int, ...], int]]:
        """
        Get summary of model parameters.

        Returns:
            Dict mapping parameter name to (shape, num_elements)
        """
        summary = {}
        for name, param in self.named_parameters():
            summary[name] = (tuple(param.shape), param.numel())
        return summary

    def reset_parameters(self) -> None:
        """
        Reset all model parameters to initial values.

        Override in subclasses for custom initialization.
        """
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def save(self, path: str) -> None:
        """
        Save model state to file.

        Args:
            path: Path to save file
        """
        state = {
            'model_state_dict': self.state_dict(),
            'lookback': self.lookback,
            'random_seed': self.random_seed,
            'model_class': self.__class__.__name__,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "BaseModel":
        """
        Load model from file.

        Args:
            path: Path to saved model
            **kwargs: Additional arguments for model construction

        Returns:
            Loaded model instance
        """
        state = torch.load(path, map_location='cpu')
        model = cls(
            lookback=state['lookback'],
            random_seed=state['random_seed'],
            **kwargs
        )
        model.load_state_dict(state['model_state_dict'])
        return model

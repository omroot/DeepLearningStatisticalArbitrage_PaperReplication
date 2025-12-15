"""
Base class for factor models.

Defines the interface that all factor models must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import torch
import numpy as np


class FactorModel(ABC):
    """
    Abstract base class for factor models.

    Factor models decompose returns into systematic factors and
    idiosyncratic residuals. This class defines the interface for:
    - Fitting the model to training data
    - Computing residuals from returns
    - Computing composition matrices for weight transformation
    """

    def __init__(
        self,
        num_factors: int,
        name: str = "FactorModel"
    ):
        """
        Initialize the factor model.

        Args:
            num_factors: Number of factors to extract
            name: Name identifier for the model
        """
        self.num_factors = num_factors
        self.name = name
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None,
        **kwargs
    ) -> "FactorModel":
        """
        Fit the factor model to data.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Optional characteristics matrix
            **kwargs: Additional model-specific parameters

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def compute_residuals(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute residual returns.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Optional characteristics matrix

        Returns:
            Residual returns of shape (T, N)
        """
        pass

    @abstractmethod
    def compute_factors(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute factor returns.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Optional characteristics matrix

        Returns:
            Factor returns of shape (T, K) where K is num_factors
        """
        pass

    @abstractmethod
    def compute_loadings(
        self,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute factor loadings.

        Args:
            characteristics: Optional characteristics matrix

        Returns:
            Factor loadings of shape (N, K) or (T, N, K)
        """
        pass

    def compute_composition_matrix(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute composition matrix for residual-to-asset weight transformation.

        The composition matrix C maps residual weights w_res to asset weights w:
            w = C @ w_res

        Default implementation returns identity (residuals = assets).

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Optional characteristics matrix

        Returns:
            Composition matrix of shape (T, N_res, N_assets) or (N_res, N_assets)
        """
        T, N = returns.shape
        # Default: identity mapping
        return np.eye(N, dtype=np.float32)

    def save(self, path: Path) -> None:
        """
        Save model parameters to disk.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = self.get_state()
        np.savez(path, **state)

    def load(self, path: Path) -> "FactorModel":
        """
        Load model parameters from disk.

        Args:
            path: Path to saved file

        Returns:
            Self for method chaining
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = dict(np.load(path, allow_pickle=True))
        self.set_state(state)
        self._is_fitted = True

        return self

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get model state for serialization.

        Returns:
            Dictionary of model parameters
        """
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set model state from serialized data.

        Args:
            state: Dictionary of model parameters
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(num_factors={self.num_factors}, fitted={self.is_fitted})"

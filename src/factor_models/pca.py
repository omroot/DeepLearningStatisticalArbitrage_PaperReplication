"""
PCA-based factor model implementation.

Uses Principal Component Analysis to extract latent factors
from the return covariance matrix.
"""

from typing import Optional, Dict, Any
import numpy as np
from scipy import linalg

from factor_models.base import FactorModel


class PCAFactorModel(FactorModel):
    """
    PCA Factor Model.

    Extracts factors as the principal components of the return
    covariance matrix. This is a standard approach for dimensionality
    reduction in asset returns.
    """

    def __init__(self, num_factors: int):
        """
        Initialize PCA factor model.

        Args:
            num_factors: Number of principal components to extract
        """
        super().__init__(num_factors, name="PCA")

        # Model parameters (set after fitting)
        self._loadings: Optional[np.ndarray] = None
        self._mean_returns: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None

    def fit(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None,
        **kwargs
    ) -> "PCAFactorModel":
        """
        Fit PCA to return data.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Not used for PCA
            **kwargs: Additional parameters (not used)

        Returns:
            Self for method chaining
        """
        T, N = returns.shape

        # Handle missing data (zeros)
        # Replace zeros with NaN for proper mean computation
        returns_clean = returns.copy()
        returns_clean[returns_clean == 0] = np.nan

        # Compute mean returns (ignoring NaN)
        self._mean_returns = np.nanmean(returns_clean, axis=0)
        self._mean_returns = np.nan_to_num(self._mean_returns, 0)

        # Center the returns
        returns_centered = returns - self._mean_returns

        # Compute covariance matrix
        # Handle missing data by computing pairwise covariances
        cov_matrix = self._compute_covariance(returns_centered)

        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Extract top K components
        self._loadings = eigenvectors[:, :self.num_factors]  # (N, K)
        self._explained_variance = eigenvalues[:self.num_factors]

        self._is_fitted = True
        return self

    def _compute_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix handling missing data.

        Args:
            returns: Centered returns (T, N)

        Returns:
            Covariance matrix (N, N)
        """
        T, N = returns.shape

        # Create mask for non-missing observations
        valid = (returns != 0)

        # Initialize covariance matrix
        cov = np.zeros((N, N), dtype=np.float64)

        # Compute pairwise covariances
        for i in range(N):
            for j in range(i, N):
                # Find observations where both assets have data
                valid_both = valid[:, i] & valid[:, j]
                n_valid = valid_both.sum()

                if n_valid > 1:
                    cov[i, j] = np.sum(returns[valid_both, i] * returns[valid_both, j]) / (n_valid - 1)
                    cov[j, i] = cov[i, j]

        return cov

    def compute_residuals(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute residual returns after removing factor exposure.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Not used for PCA

        Returns:
            Residual returns of shape (T, N)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing residuals")

        # Center returns
        returns_centered = returns - self._mean_returns

        # Compute factor returns: F = R @ L  (T, K)
        factors = returns_centered @ self._loadings

        # Compute predicted returns: R_hat = F @ L'
        predicted = factors @ self._loadings.T

        # Residuals = actual - predicted
        residuals = returns_centered - predicted

        return residuals

    def compute_factors(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute factor returns.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Not used for PCA

        Returns:
            Factor returns of shape (T, K)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing factors")

        returns_centered = returns - self._mean_returns
        factors = returns_centered @ self._loadings

        return factors

    def compute_loadings(
        self,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get factor loadings.

        Args:
            characteristics: Not used for PCA

        Returns:
            Factor loadings of shape (N, K)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before accessing loadings")

        return self._loadings

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "num_factors": self.num_factors,
            "loadings": self._loadings,
            "mean_returns": self._mean_returns,
            "explained_variance": self._explained_variance,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set model state from serialized data."""
        self.num_factors = int(state["num_factors"])
        self._loadings = state["loadings"]
        self._mean_returns = state["mean_returns"]
        self._explained_variance = state["explained_variance"]

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio for each component.

        Returns:
            Array of explained variance ratios (K,)
        """
        if self._explained_variance is None:
            return None
        total_var = np.sum(self._explained_variance)
        return self._explained_variance / total_var if total_var > 0 else None

"""
IPCA (Instrumented Principal Component Analysis) factor model.

Implements the IPCA methodology for estimating latent factors with
time-varying loadings based on observable characteristics.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from scipy import linalg

from factor_models.base import FactorModel


class IPCAFactorModel(FactorModel):
    """
    Instrumented PCA Factor Model.

    IPCA estimates latent factors where loadings are linear functions
    of observable characteristics:

        β_{i,t} = Z_{i,t} @ Γ

    where Z_{i,t} are characteristics and Γ maps them to loadings.

    Reference: Kelly, Pruitt, Su (2019) - "Characteristics are covariances:
    A unified model of risk and return"
    """

    def __init__(
        self,
        num_factors: int,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ):
        """
        Initialize IPCA factor model.

        Args:
            num_factors: Number of latent factors
            max_iterations: Maximum ALS iterations
            tolerance: Convergence tolerance
        """
        super().__init__(num_factors, name="IPCA")

        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Model parameters (set after fitting)
        self._gamma: Optional[np.ndarray] = None  # (L, K) characteristic -> loading map
        self._factors: Optional[np.ndarray] = None  # (T, K) factor returns
        self._mean_characteristics: Optional[np.ndarray] = None

    def fit(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None,
        **kwargs
    ) -> "IPCAFactorModel":
        """
        Fit IPCA model using Alternating Least Squares.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Characteristics tensor of shape (T, N, L)
            **kwargs: Additional parameters

        Returns:
            Self for method chaining
        """
        if characteristics is None:
            raise ValueError("IPCA requires characteristics data")

        T, N = returns.shape
        _, _, L = characteristics.shape
        K = self.num_factors

        # Handle missing data
        valid_mask = (returns != 0)

        # Center characteristics
        self._mean_characteristics = np.nanmean(characteristics, axis=(0, 1))
        Z = characteristics - self._mean_characteristics

        # Initialize Gamma randomly
        np.random.seed(42)
        self._gamma = np.random.randn(L, K) * 0.01
        self._gamma, _ = linalg.qr(self._gamma)  # Orthogonalize

        prev_loss = float('inf')

        for iteration in range(self.max_iterations):
            # Step 1: Given Gamma, estimate factors F
            # F_t = (Z_t @ Γ)' @ (Z_t @ Γ))^{-1} @ (Z_t @ Γ)' @ R_t
            self._factors = self._estimate_factors(returns, Z, valid_mask)

            # Step 2: Given F, estimate Gamma
            # Γ = (Σ_t Z_t' @ Z_t)^{-1} @ (Σ_t Z_t' @ R_t @ F_t')
            self._gamma = self._estimate_gamma(returns, Z, valid_mask)

            # Orthogonalize Gamma
            self._gamma, _ = linalg.qr(self._gamma)

            # Compute loss (MSE of residuals)
            loss = self._compute_loss(returns, Z, valid_mask)

            # Check convergence
            if abs(prev_loss - loss) / (abs(prev_loss) + 1e-8) < self.tolerance:
                break

            prev_loss = loss

        self._is_fitted = True
        return self

    def _estimate_factors(
        self,
        returns: np.ndarray,
        characteristics: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Estimate factor returns given Gamma.

        Args:
            returns: Returns (T, N)
            characteristics: Centered characteristics (T, N, L)
            valid_mask: Valid observation mask (T, N)

        Returns:
            Factor returns (T, K)
        """
        T, N = returns.shape
        K = self.num_factors

        factors = np.zeros((T, K))

        for t in range(T):
            # Get valid assets at time t
            valid_idx = valid_mask[t]
            if valid_idx.sum() < K:
                continue

            # Loadings for this period: (N_valid, K)
            Z_t = characteristics[t, valid_idx]  # (N_valid, L)
            B_t = Z_t @ self._gamma  # (N_valid, K)

            # Returns for valid assets
            R_t = returns[t, valid_idx]  # (N_valid,)

            # OLS: F_t = (B'B)^{-1} B' R
            BTB = B_t.T @ B_t
            BTR = B_t.T @ R_t

            try:
                factors[t] = linalg.solve(BTB + 1e-6 * np.eye(K), BTR)
            except linalg.LinAlgError:
                factors[t] = linalg.lstsq(B_t, R_t)[0]

        return factors

    def _estimate_gamma(
        self,
        returns: np.ndarray,
        characteristics: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Estimate Gamma given factors.

        Args:
            returns: Returns (T, N)
            characteristics: Centered characteristics (T, N, L)
            valid_mask: Valid observation mask (T, N)

        Returns:
            Gamma matrix (L, K)
        """
        T, N = returns.shape
        _, _, L = characteristics.shape
        K = self.num_factors

        # Accumulate cross-products
        ZTZ = np.zeros((L, L))
        ZTR_F = np.zeros((L, K))

        for t in range(T):
            valid_idx = valid_mask[t]
            if valid_idx.sum() == 0:
                continue

            Z_t = characteristics[t, valid_idx]  # (N_valid, L)
            R_t = returns[t, valid_idx]  # (N_valid,)
            F_t = self._factors[t]  # (K,)

            ZTZ += Z_t.T @ Z_t
            ZTR_F += Z_t.T @ (R_t[:, np.newaxis] * F_t)

        # Solve for Gamma
        try:
            gamma = linalg.solve(ZTZ + 1e-6 * np.eye(L), ZTR_F)
        except linalg.LinAlgError:
            gamma = linalg.lstsq(ZTZ, ZTR_F)[0]

        return gamma

    def _compute_loss(
        self,
        returns: np.ndarray,
        characteristics: np.ndarray,
        valid_mask: np.ndarray
    ) -> float:
        """
        Compute mean squared residuals.

        Args:
            returns: Returns (T, N)
            characteristics: Centered characteristics (T, N, L)
            valid_mask: Valid observation mask (T, N)

        Returns:
            Mean squared error
        """
        T, N = returns.shape
        total_error = 0.0
        total_count = 0

        for t in range(T):
            valid_idx = valid_mask[t]
            if valid_idx.sum() == 0:
                continue

            Z_t = characteristics[t, valid_idx]
            B_t = Z_t @ self._gamma
            R_t = returns[t, valid_idx]
            F_t = self._factors[t]

            predicted = B_t @ F_t
            error = R_t - predicted

            total_error += np.sum(error ** 2)
            total_count += len(error)

        return total_error / max(total_count, 1)

    def compute_residuals(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute residual returns.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Characteristics tensor of shape (T, N, L)

        Returns:
            Residual returns of shape (T, N)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing residuals")
        if characteristics is None:
            raise ValueError("IPCA requires characteristics for residual computation")

        T, N = returns.shape
        Z = characteristics - self._mean_characteristics

        residuals = np.zeros_like(returns)

        for t in range(T):
            B_t = Z[t] @ self._gamma  # (N, K)
            F_t = self._factors[t] if t < len(self._factors) else np.zeros(self.num_factors)
            predicted = B_t @ F_t
            residuals[t] = returns[t] - predicted

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
            characteristics: Characteristics tensor of shape (T, N, L)

        Returns:
            Factor returns of shape (T, K)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing factors")

        if characteristics is not None:
            Z = characteristics - self._mean_characteristics
            valid_mask = (returns != 0)
            return self._estimate_factors(returns, Z, valid_mask)

        return self._factors

    def compute_loadings(
        self,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute factor loadings from characteristics.

        Args:
            characteristics: Characteristics tensor (T, N, L) or (N, L)

        Returns:
            Factor loadings of shape (T, N, K) or (N, K)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing loadings")

        if characteristics is None:
            raise ValueError("IPCA requires characteristics for loadings")

        Z = characteristics - self._mean_characteristics
        return Z @ self._gamma

    def compute_composition_matrix(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute composition matrix for IPCA.

        For IPCA, this maps residual positions to asset positions
        accounting for the factor structure.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Characteristics tensor

        Returns:
            Composition matrix of shape (T, N, N) - identity for direct mapping
        """
        T, N = returns.shape
        # For IPCA residuals, composition is identity (residuals = assets)
        return np.broadcast_to(np.eye(N), (T, N, N)).copy()

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "num_factors": self.num_factors,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "gamma": self._gamma,
            "factors": self._factors,
            "mean_characteristics": self._mean_characteristics,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set model state from serialized data."""
        self.num_factors = int(state["num_factors"])
        self.max_iterations = int(state.get("max_iterations", 1000))
        self.tolerance = float(state.get("tolerance", 1e-6))
        self._gamma = state["gamma"]
        self._factors = state["factors"]
        self._mean_characteristics = state["mean_characteristics"]

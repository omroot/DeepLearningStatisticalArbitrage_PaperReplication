"""
Fama-French factor model implementation.

Uses the standard Fama-French factors (market, SMB, HML, RMW, CMA)
to compute residuals.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
from scipy import linalg

from factor_models.base import FactorModel
from utils.logging import get_logger

logger = get_logger(__name__)


class FamaFrenchFactorModel(FactorModel):
    """
    Fama-French Factor Model.

    Regresses asset returns on Fama-French factors to extract
    idiosyncratic residuals. Supports 3-factor and 5-factor models.
    """

    def __init__(
        self,
        num_factors: int = 5,
        include_momentum: bool = False
    ):
        """
        Initialize Fama-French factor model.

        Args:
            num_factors: Number of factors (3, 5, or 6 with momentum)
            include_momentum: Whether to include momentum factor
        """
        if num_factors not in [3, 5, 6]:
            raise ValueError("num_factors must be 3, 5, or 6")

        super().__init__(num_factors, name="FamaFrench")

        self.include_momentum = include_momentum

        # Model parameters
        self._betas: Optional[np.ndarray] = None  # (N, K) factor loadings
        self._alphas: Optional[np.ndarray] = None  # (N,) intercepts
        self._factor_returns: Optional[np.ndarray] = None  # (T, K) factor returns used

    def fit(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None,
        factor_returns: Optional[np.ndarray] = None,
        **kwargs
    ) -> "FamaFrenchFactorModel":
        """
        Fit Fama-French model by regressing returns on factors.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Not used
            factor_returns: Factor returns of shape (T, K), required
            **kwargs: Additional parameters

        Returns:
            Self for method chaining
        """
        if factor_returns is None:
            raise ValueError("Fama-French model requires factor_returns")

        T, N = returns.shape
        K = factor_returns.shape[1]

        if K < self.num_factors:
            raise ValueError(f"Need at least {self.num_factors} factors, got {K}")

        # Use only the specified number of factors
        F = factor_returns[:, :self.num_factors]
        self._factor_returns = F

        # Add intercept column
        F_with_intercept = np.column_stack([np.ones(T), F])  # (T, K+1)

        # Initialize betas and alphas
        self._betas = np.zeros((N, self.num_factors))
        self._alphas = np.zeros(N)

        # Regress each asset on factors
        for i in range(N):
            # Get valid observations for this asset
            valid = returns[:, i] != 0
            if valid.sum() < self.num_factors + 2:
                continue

            y = returns[valid, i]
            X = F_with_intercept[valid]

            # OLS: (X'X)^{-1} X'y
            try:
                XTX = X.T @ X
                XTy = X.T @ y
                coeffs = linalg.solve(XTX + 1e-8 * np.eye(XTX.shape[0]), XTy)
                self._alphas[i] = coeffs[0]
                self._betas[i] = coeffs[1:]
            except linalg.LinAlgError:
                # Fallback to lstsq
                coeffs = linalg.lstsq(X, y)[0]
                self._alphas[i] = coeffs[0]
                self._betas[i] = coeffs[1:]

        self._is_fitted = True
        return self

    def compute_residuals(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None,
        factor_returns: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute residual returns.

        Args:
            returns: Return matrix of shape (T, N)
            characteristics: Not used
            factor_returns: Optional factor returns for out-of-sample

        Returns:
            Residual returns of shape (T, N)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing residuals")

        T, N = returns.shape

        # Use provided factors or stored factors
        if factor_returns is not None:
            F = factor_returns[:, :self.num_factors]
        else:
            F = self._factor_returns

        if F is None or len(F) != T:
            raise ValueError("Factor returns must match returns length")

        # Predicted returns: R_hat = α + F @ β'
        predicted = self._alphas + F @ self._betas.T  # (T, N)

        # Residuals
        residuals = returns - predicted

        # Keep zeros where original data was missing
        residuals[returns == 0] = 0

        return residuals

    def compute_factors(
        self,
        returns: np.ndarray,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Return the factor returns (these are given, not estimated).

        Args:
            returns: Not used
            characteristics: Not used

        Returns:
            Factor returns of shape (T, K)
        """
        if self._factor_returns is None:
            raise RuntimeError("No factor returns available")
        return self._factor_returns

    def compute_loadings(
        self,
        characteristics: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get factor loadings (betas).

        Args:
            characteristics: Not used

        Returns:
            Factor loadings of shape (N, K)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before accessing loadings")
        return self._betas

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "num_factors": self.num_factors,
            "include_momentum": self.include_momentum,
            "betas": self._betas,
            "alphas": self._alphas,
            "factor_returns": self._factor_returns,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set model state from serialized data."""
        self.num_factors = int(state["num_factors"])
        self.include_momentum = bool(state.get("include_momentum", False))
        self._betas = state["betas"]
        self._alphas = state["alphas"]
        self._factor_returns = state.get("factor_returns")

    @staticmethod
    def download_factors(
        start_date: str,
        end_date: str,
        frequency: str = "daily"
    ) -> np.ndarray:
        """
        Download Fama-French factors from Ken French's website.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: 'daily' or 'monthly'

        Returns:
            Factor returns array of shape (T, 6) for
            [Mkt-RF, SMB, HML, RMW, CMA, Mom]
        """
        try:
            import pandas_datareader.data as web
        except ImportError:
            raise ImportError("pandas_datareader required for downloading FF factors")

        # Download 5 factors
        if frequency == "daily":
            ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench',
                                 start=start_date, end=end_date)[0]
            mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench',
                                 start=start_date, end=end_date)[0]
        else:
            ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench',
                                 start=start_date, end=end_date)[0]
            mom = web.DataReader('F-F_Momentum_Factor', 'famafrench',
                                 start=start_date, end=end_date)[0]

        # Merge and convert to numpy
        ff5 = ff5 / 100  # Convert from percentage
        mom = mom / 100

        # Align dates
        common_dates = ff5.index.intersection(mom.index)
        ff5 = ff5.loc[common_dates]
        mom = mom.loc[common_dates]

        # Combine: [Mkt-RF, SMB, HML, RMW, CMA, Mom]
        factors = np.column_stack([
            ff5['Mkt-RF'].values,
            ff5['SMB'].values,
            ff5['HML'].values,
            ff5['RMW'].values,
            ff5['CMA'].values,
            mom['Mom'].values if 'Mom' in mom.columns else mom.iloc[:, 0].values
        ])

        return factors

"""
Factor model implementations for residual computation.

Supports:
- IPCA (Instrumented Principal Component Analysis)
- PCA (Principal Component Analysis)
- Fama-French factor models
"""

from factor_models.base import FactorModel
from factor_models.ipca import IPCAFactorModel
from factor_models.pca import PCAFactorModel
from factor_models.fama_french import FamaFrenchFactorModel
from factor_models.factory import create_factor_model

__all__ = [
    "FactorModel",
    "IPCAFactorModel",
    "PCAFactorModel",
    "FamaFrenchFactorModel",
    "create_factor_model",
]

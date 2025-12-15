"""
Factory function for creating factor models.
"""

from typing import Optional, Dict, Any

from dtypes import FactorModelType
from factor_models.base import FactorModel
from factor_models.ipca import IPCAFactorModel
from factor_models.pca import PCAFactorModel
from factor_models.fama_french import FamaFrenchFactorModel


def create_factor_model(
    model_type: str,
    num_factors: int,
    **kwargs
) -> FactorModel:
    """
    Create a factor model instance.

    Args:
        model_type: Type of factor model ('IPCA', 'PCA', 'FamaFrench')
        num_factors: Number of factors
        **kwargs: Additional model-specific parameters

    Returns:
        Factor model instance

    Raises:
        ValueError: If model_type is not recognized
    """
    model_type_upper = model_type.upper()

    if model_type_upper == "IPCA":
        return IPCAFactorModel(
            num_factors=num_factors,
            max_iterations=kwargs.get("max_iterations", 1000),
            tolerance=kwargs.get("tolerance", 1e-6)
        )

    elif model_type_upper == "PCA":
        return PCAFactorModel(num_factors=num_factors)

    elif model_type_upper in ["FAMAFRENCH", "FAMA_FRENCH", "FF"]:
        return FamaFrenchFactorModel(
            num_factors=num_factors,
            include_momentum=kwargs.get("include_momentum", False)
        )

    else:
        raise ValueError(f"Unknown factor model type: {model_type}")

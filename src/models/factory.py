"""
Factory function for creating models.
"""

from typing import Dict, Any

from models.base import BaseModel
from models.cnn_transformer import CNNTransformer
from models.fourier_ffn import FourierFFN
from models.ou_threshold import OUThreshold
from models.cnn_transformer_frictions import CNNTransformerFrictions


# Model registry
MODEL_REGISTRY: Dict[str, type] = {
    "CNNTransformer": CNNTransformer,
    "FourierFFN": FourierFFN,
    "OUThreshold": OUThreshold,
    "CNNTransformerFrictions": CNNTransformerFrictions,
}


def create_model(
    model_class: str,
    lookback: int,
    **kwargs
) -> BaseModel:
    """
    Create a model instance from class name.

    Args:
        model_class: Name of model class
        lookback: Lookback window size
        **kwargs: Additional model-specific parameters

    Returns:
        Model instance

    Raises:
        ValueError: If model_class is not recognized
    """
    if model_class not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model class: {model_class}. "
            f"Available: {available}"
        )

    return MODEL_REGISTRY[model_class](lookback=lookback, **kwargs)


def register_model(name: str, model_cls: type) -> None:
    """
    Register a custom model class.

    Args:
        name: Name to register under
        model_cls: Model class (must inherit from BaseModel)
    """
    if not issubclass(model_cls, BaseModel):
        raise TypeError("Model must inherit from BaseModel")
    MODEL_REGISTRY[name] = model_cls


def get_available_models() -> list:
    """
    Get list of available model names.

    Returns:
        List of registered model names
    """
    return list(MODEL_REGISTRY.keys())

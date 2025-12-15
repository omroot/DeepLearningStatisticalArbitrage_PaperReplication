"""
Neural network models for statistical arbitrage.

Includes:
- CNNTransformer: Hybrid CNN-Transformer architecture
- FourierFFN: Feedforward network with Fourier features
- OUThreshold: Non-trainable OU baseline
- CNNTransformerFrictions: Transaction cost-aware model
"""

from models.base import BaseModel
from models.cnn_transformer import CNNTransformer
from models.fourier_ffn import FourierFFN
from models.ou_threshold import OUThreshold
from models.cnn_transformer_frictions import CNNTransformerFrictions
from models.factory import create_model

__all__ = [
    "BaseModel",
    "CNNTransformer",
    "FourierFFN",
    "OUThreshold",
    "CNNTransformerFrictions",
    "create_model",
]

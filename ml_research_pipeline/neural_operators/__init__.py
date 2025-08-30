"""
Neural Operators Module

This module implements neural operators for physics-informed learning,
including Fourier Neural Operators (FNO) and DeepONet architectures.
"""

from .deeponet import PhysicsInformedDeepONet
from .fourier_neural_operator import InverseFourierNeuralOperator

# from .operator_meta_pinn import OperatorMetaPINN

__all__ = [
    "InverseFourierNeuralOperator",
    "PhysicsInformedDeepONet",
    # 'OperatorMetaPINN'
]

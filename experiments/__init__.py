"""
Experiments module for ML research pipeline.

This module contains experimental configurations, training scripts,
and evaluation pipelines for meta-learning PINN research.
"""

__version__ = "0.1.0"
__author__ = "ML Research Team"

# Import key experimental components
from .config import ExperimentConfig
from .runner import ExperimentRunner

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
]
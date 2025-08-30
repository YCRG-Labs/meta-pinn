"""
Hyperparameter optimization module for physics discovery pipeline.

This module provides Bayesian optimization and multi-objective optimization
capabilities for automated hyperparameter tuning across all discovery components.
"""

from .bayesian_optimizer import BayesianOptimizer
from .hyperparameter_framework import HyperparameterOptimizationFramework

__all__ = ["BayesianOptimizer", "HyperparameterOptimizationFramework"]

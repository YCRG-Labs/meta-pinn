"""
ML Research Pipeline for Meta-Learning Physics-Informed Neural Networks.

This package provides a comprehensive framework for meta-learning research
on physics-informed neural networks, including task generation, meta-learning
algorithms, evaluation frameworks, and publication tools.
"""

__version__ = "0.1.0"
__author__ = "ML Research Team"
__email__ = "research@example.com"

# Core imports (will be implemented in subsequent tasks)
from .config import Config, ExperimentConfig
from .utils import set_random_seeds, setup_logging

# Make key classes available at package level
__all__ = [
    "Config",
    "ExperimentConfig",
    "setup_logging",
    "set_random_seeds",
]

# Package metadata
__package_name__ = "ml_research_pipeline"
__description__ = "Meta-Learning Physics-Informed Neural Networks Research Pipeline"
__url__ = "https://github.com/example/ml-research-pipeline"
__license__ = "MIT"

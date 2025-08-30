"""
Meta-learning module for physics-informed neural networks.

This module provides physics-informed meta-learning components including:
- PhysicsRegularizer: Physics-informed loss functions and regularization
- AdaptiveConstraintIntegration: Dynamic constraint weighting
- MetaLearningOptimizer: Physics-constrained optimization
- PhysicsInformedMetaLearner: Complete physics-informed meta-learning system
"""

from .adaptive_constraint_integration import AdaptiveConstraintIntegration
from .meta_learning_optimizer import (
    ConstraintFunction,
    GradientProjectionOptimizer,
    LagrangianOptimizer,
    MetaLearningOptimizer,
    OptimizationConfig,
)
from .physics_informed_meta_learner import (
    MetaLearningResult,
    PhysicsGuidedInitializer,
    PhysicsInformedMetaLearner,
    PhysicsMetaLearningConfig,
    Task,
)
from .physics_regularizer import PhysicsRegularizer

__all__ = [
    "PhysicsRegularizer",
    "AdaptiveConstraintIntegration",
    "MetaLearningOptimizer",
    "GradientProjectionOptimizer",
    "LagrangianOptimizer",
    "OptimizationConfig",
    "ConstraintFunction",
    "PhysicsInformedMetaLearner",
    "PhysicsMetaLearningConfig",
    "Task",
    "MetaLearningResult",
    "PhysicsGuidedInitializer",
]

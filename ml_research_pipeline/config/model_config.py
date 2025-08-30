"""
Model configuration classes for different PINN architectures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Base model configuration."""

    # Architecture
    input_dim: int = 3  # x, y, t for 2D+time problems
    output_dim: int = 3  # u, v, p for velocity and pressure
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128, 128, 128])
    activation: str = "tanh"

    # Initialization
    weight_init: str = "xavier_normal"
    bias_init: str = "zeros"

    # Normalization
    input_normalization: bool = True
    output_normalization: bool = False
    layer_normalization: bool = False
    batch_normalization: bool = False

    # Regularization
    dropout_rate: float = 0.0
    weight_decay: float = 1e-4

    # Physics constraints
    enforce_boundary_conditions: bool = True
    physics_loss_weight: float = 1.0
    adaptive_physics_weight: bool = True

    def get_layer_sizes(self) -> List[int]:
        """Get complete layer size specification."""
        return [self.input_dim] + self.hidden_layers + [self.output_dim]


@dataclass
class MetaPINNConfig(ModelConfig):
    """Configuration for Meta-PINN models."""

    # Meta-learning settings
    meta_lr: float = 0.001
    adapt_lr: float = 0.01
    adaptation_steps: int = 5
    first_order: bool = True  # First-order MAML for efficiency

    # Task adaptation
    n_support: int = 25
    n_query: int = 25
    task_batch_size: int = 4

    # Meta-training
    meta_batch_size: int = 8
    meta_epochs: int = 1000

    # Inner loop optimization
    inner_optimizer: str = "sgd"
    inner_momentum: float = 0.0
    gradient_clipping: Optional[float] = 1.0

    # Outer loop optimization
    outer_optimizer: str = "adam"
    outer_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    outer_eps: float = 1e-8

    # Learning rate scheduling
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    lr_decay_factor: float = 0.1

    # Physics-informed meta-learning
    physics_meta_weight: float = 1.0
    residual_sampling_strategy: str = "uniform"  # uniform, adaptive, importance

    # Regularization for meta-learning
    meta_regularization: float = 0.0
    task_regularization: float = 0.0


@dataclass
class BayesianMetaPINNConfig(MetaPINNConfig):
    """Configuration for Bayesian Meta-PINN models."""

    # Variational parameters
    prior_std: float = 1.0
    posterior_std_init: float = 0.1
    kl_weight: float = 1e-3

    # Uncertainty quantification
    n_samples: int = 100
    uncertainty_type: str = "both"  # epistemic, aleatoric, both

    # Calibration
    calibration_method: str = "isotonic"  # isotonic, platt, temperature
    calibration_split: float = 0.2

    # Monte Carlo settings
    mc_dropout: bool = False
    mc_dropout_rate: float = 0.1

    # Variational inference
    vi_optimizer: str = "adam"
    vi_lr: float = 0.001
    vi_epochs: int = 100


@dataclass
class NeuralOperatorConfig(ModelConfig):
    """Configuration for Neural Operator models."""

    # Operator type
    operator_type: str = "fno"  # fno, deeponet, graph_neural_operator

    # Fourier Neural Operator settings
    modes: int = 12
    width: int = 64
    n_layers: int = 4

    # DeepONet settings
    branch_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    trunk_layers: List[int] = field(default_factory=lambda: [128, 128, 128])

    # Input/Output handling
    input_resolution: List[int] = field(default_factory=lambda: [64, 64])
    output_resolution: List[int] = field(default_factory=lambda: [64, 64])

    # Physics integration
    physics_informed: bool = True
    operator_physics_weight: float = 0.1


@dataclass
class EnsembleConfig(BaseConfig):
    """Configuration for ensemble methods."""

    # Ensemble settings
    n_models: int = 5
    ensemble_method: str = "average"  # average, weighted, stacking

    # Model diversity
    diversity_method: str = "random_init"  # random_init, bagging, boosting
    diversity_strength: float = 1.0

    # Training strategy
    parallel_training: bool = True
    shared_features: bool = False

    # Uncertainty aggregation
    uncertainty_aggregation: str = "mean"  # mean, max, epistemic_aleatoric

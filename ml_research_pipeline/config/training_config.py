"""
Training configuration for meta-learning PINN experiments.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training procedures."""

    # Basic training settings
    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001

    # Optimizer settings
    optimizer: str = "adam"
    momentum: float = 0.9
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 1e-4

    # Learning rate scheduling
    lr_scheduler: str = "cosine"
    lr_warmup_epochs: int = 10
    lr_decay_epochs: List[int] = field(default_factory=lambda: [300, 600, 900])
    lr_decay_factor: float = 0.1
    min_lr: float = 1e-6

    # Loss function settings
    loss_function: str = "mse"
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"data": 1.0, "physics": 1.0, "boundary": 1.0}
    )

    # Physics loss settings
    physics_loss_type: str = "residual"  # residual, energy, variational
    residual_sampling: str = "uniform"  # uniform, adaptive, importance
    n_residual_points: int = 1000

    # Adaptive weighting
    adaptive_weights: bool = True
    weight_update_frequency: int = 100
    weight_adaptation_method: str = "gradnorm"  # gradnorm, uncertainty, loss_ratio

    # Gradient settings
    gradient_clipping: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1

    # Early stopping
    early_stopping: bool = True
    patience: int = 100
    min_delta: float = 1e-6
    monitor_metric: str = "validation_loss"

    # Validation settings
    validation_frequency: int = 10
    validation_split: float = 0.2

    # Checkpointing
    save_best_model: bool = True
    save_last_model: bool = True
    checkpoint_frequency: int = 100

    # Mixed precision training
    mixed_precision: bool = False
    amp_opt_level: str = "O1"

    # Curriculum learning
    curriculum_learning: bool = False
    curriculum_schedule: str = "linear"  # linear, exponential, step
    curriculum_epochs: int = 200


@dataclass
class MetaTrainingConfig(TrainingConfig):
    """Configuration for meta-learning training."""

    # Meta-learning specific settings
    meta_batch_size: int = 4
    n_support: int = 25
    n_query: int = 25
    adaptation_steps: int = 5

    # Inner loop settings
    inner_lr: float = 0.01
    inner_optimizer: str = "sgd"
    inner_momentum: float = 0.0

    # Outer loop settings
    outer_lr: float = 0.001
    outer_optimizer: str = "adam"

    # Task sampling
    task_sampling_strategy: str = "uniform"  # uniform, curriculum, adaptive
    task_difficulty_schedule: str = "linear"

    # Meta-validation
    meta_validation_tasks: int = 100
    meta_validation_frequency: int = 50

    # MAML specific
    first_order: bool = True
    allow_unused: bool = True
    allow_nograd: bool = True


@dataclass
class DistributedTrainingConfig(BaseConfig):
    """Configuration for distributed training."""

    # Distributed settings
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0

    # Multi-GPU settings
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    sync_bn: bool = True
    find_unused_parameters: bool = False

    # Communication settings
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = True

    # Load balancing
    balance_workload: bool = True
    dynamic_loss_scaling: bool = True

    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 100
    auto_resume: bool = True


@dataclass
class OptimizationConfig(BaseConfig):
    """Advanced optimization configuration."""

    # Second-order methods
    use_second_order: bool = False
    hessian_approximation: str = "lbfgs"  # lbfgs, bfgs, newton

    # Line search
    line_search: bool = False
    line_search_method: str = "strong_wolfe"

    # Trust region methods
    trust_region: bool = False
    trust_region_radius: float = 1.0

    # Natural gradients
    natural_gradients: bool = False
    fisher_information_method: str = "empirical"

    # Constrained optimization
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    constraint_penalty: float = 1.0

    # Multi-objective optimization
    multi_objective: bool = False
    objective_weights: Dict[str, float] = field(default_factory=dict)
    pareto_optimization: bool = False

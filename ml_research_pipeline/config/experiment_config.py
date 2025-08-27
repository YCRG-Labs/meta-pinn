"""
Experiment configuration for meta-learning PINN research.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .base_config import BaseConfig


@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration for meta-learning experiments."""
    
    # Experiment metadata
    name: str = "meta_pinn_experiment"
    description: str = "Meta-learning PINN experiment"
    version: str = "0.1.0"
    author: str = "ML Research Team"
    
    # Experiment settings
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    
    # Output settings
    output_dir: str = "experiments/outputs"
    save_checkpoints: bool = True
    checkpoint_interval: int = 100
    log_interval: int = 10
    
    # Evaluation settings
    eval_interval: int = 50
    save_predictions: bool = True
    generate_plots: bool = True
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False
    
    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # Resource limits
    max_memory_gb: Optional[float] = None
    max_time_hours: Optional[float] = None
    
    # Experiment tags and metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.distributed and self.world_size <= 1:
            raise ValueError("world_size must be > 1 for distributed training")
        
        if self.rank >= self.world_size:
            raise ValueError("rank must be < world_size")


@dataclass 
class HyperparameterConfig(BaseConfig):
    """Configuration for hyperparameter optimization."""
    
    # Optimization method
    method: str = "random"  # random, grid, bayesian
    n_trials: int = 100
    timeout: Optional[float] = None
    
    # Search space definition
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Pruning settings
    enable_pruning: bool = True
    pruning_warmup_steps: int = 5
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ["validation_loss"])
    directions: List[str] = field(default_factory=lambda: ["minimize"])
    
    # Resource allocation
    n_jobs: int = 1
    memory_limit: Optional[str] = None
    
    def add_hyperparameter(self, name: str, param_type: str, **kwargs):
        """Add hyperparameter to search space.
        
        Args:
            name: Parameter name
            param_type: Type of parameter (float, int, categorical, etc.)
            **kwargs: Parameter-specific arguments (low, high, choices, etc.)
        """
        self.search_space[name] = {
            "type": param_type,
            **kwargs
        }


@dataclass
class BenchmarkConfig(BaseConfig):
    """Configuration for benchmark experiments."""
    
    # Benchmark suite settings
    benchmark_name: str = "fluid_dynamics_suite"
    problem_types: List[str] = field(default_factory=lambda: [
        "cavity_flow", "channel_flow", "cylinder_flow", "thermal_convection"
    ])
    
    # Task generation
    n_train_tasks: int = 1000
    n_val_tasks: int = 200
    n_test_tasks: int = 200
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        "parameter_accuracy", "adaptation_speed", "physics_consistency", 
        "computational_efficiency", "uncertainty_calibration"
    ])
    
    # Statistical analysis
    confidence_level: float = 0.95
    n_bootstrap_samples: int = 1000
    multiple_comparison_correction: str = "bonferroni"
    
    # Method comparison
    baseline_methods: List[str] = field(default_factory=lambda: [
        "standard_pinn", "transfer_learning_pinn", "fourier_neural_operator", "deeponet"
    ])
    
    # Reporting
    generate_latex_tables: bool = True
    generate_publication_plots: bool = True
    statistical_significance_threshold: float = 0.05
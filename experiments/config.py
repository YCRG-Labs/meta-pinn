"""
Experiment configuration and management.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
import json

from ml_research_pipeline.config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig


@dataclass
class ExperimentRunner:
    """Experiment runner for managing and executing experiments."""
    
    config: ExperimentConfig
    output_dir: Path
    
    def __post_init__(self):
        """Initialize experiment runner."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
    
    def save_config(self):
        """Save experiment configuration."""
        config_path = self.output_dir / "config.yaml"
        self.config.to_yaml(config_path)
    
    def load_config(self, config_path: Path) -> ExperimentConfig:
        """Load experiment configuration."""
        return ExperimentConfig.from_yaml(config_path)


def create_default_configs() -> Dict[str, Any]:
    """Create default configuration templates."""
    
    # Default experiment config
    experiment_config = ExperimentConfig(
        name="meta_pinn_baseline",
        description="Baseline meta-learning PINN experiment",
        seed=42,
        output_dir="experiments/outputs/baseline"
    )
    
    # Default model config  
    model_config = ModelConfig(
        input_dim=3,
        output_dim=3,
        hidden_layers=[128, 128, 128, 128],
        activation="tanh"
    )
    
    # Default training config
    training_config = TrainingConfig(
        epochs=1000,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam"
    )
    
    # Default data config
    data_config = DataConfig(
        n_train_tasks=1000,
        n_val_tasks=200,
        n_test_tasks=200
    )
    
    return {
        "experiment": experiment_config,
        "model": model_config,
        "training": training_config,
        "data": data_config
    }


def save_default_configs(config_dir: Path = Path("configs")):
    """Save default configuration templates."""
    config_dir.mkdir(parents=True, exist_ok=True)
    
    configs = create_default_configs()
    
    for name, config in configs.items():
        config_path = config_dir / f"{name}_default.yaml"
        config.to_yaml(config_path)
        print(f"Saved default {name} config to {config_path}")


if __name__ == "__main__":
    # Create default configuration files
    save_default_configs()
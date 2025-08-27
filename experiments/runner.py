"""
Experiment runner for executing meta-learning PINN experiments.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.distributed as dist

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_research_pipeline.config import ExperimentConfig
from ml_research_pipeline.utils import setup_logging, set_random_seeds, get_logger


class ExperimentRunner:
    """Main experiment runner class."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = None
        self.device = None
        self.distributed = False
        
        # Setup experiment
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Set up experiment environment."""
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = output_dir / "experiment.log"
        setup_logging(
            level=self.config.log_level,
            log_file=log_file,
            use_colors=True
        )
        self.logger = get_logger(self.__class__.__name__)
        
        # Set random seeds
        set_random_seeds(
            self.config.seed,
            deterministic=self.config.deterministic
        )
        
        # Setup device
        self._setup_device()
        
        # Setup distributed training if needed
        if self.config.distributed:
            self._setup_distributed()
        
        # Log experiment info
        self.logger.info(f"Starting experiment: {self.config.name}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Distributed: {self.distributed}")
    
    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available() and self.config.device == "cuda":
            self.device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if not dist.is_available():
            self.logger.warning("Distributed training not available")
            return
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        self.distributed = True
        self.logger.info(f"Initialized distributed training: rank {self.config.rank}/{self.config.world_size}")
    
    def run(self) -> Dict[str, Any]:
        """Run the experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        try:
            self.logger.info("Starting experiment execution")
            
            # Save configuration
            config_path = Path(self.config.output_dir) / "config.yaml"
            self.config.to_yaml(config_path)
            
            # Placeholder for actual experiment logic
            # This will be implemented in subsequent tasks
            results = {
                "status": "completed",
                "config": self.config.to_dict(),
                "device": str(self.device),
                "distributed": self.distributed
            }
            
            self.logger.info("Experiment completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            raise
        
        finally:
            # Cleanup distributed training
            if self.distributed and dist.is_initialized():
                dist.destroy_process_group()
    
    def save_results(self, results: Dict[str, Any]):
        """Save experiment results.
        
        Args:
            results: Results dictionary to save
        """
        output_dir = Path(self.config.output_dir)
        
        # Save as JSON
        results_path = output_dir / "results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")


def main():
    """Main entry point for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run meta-learning PINN experiment")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment configuration file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Override output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config = config.update(output_dir=args.output_dir)
    
    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run()
    runner.save_results(results)


if __name__ == "__main__":
    main()
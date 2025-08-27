"""
Comprehensive experiment management and reproducibility system.
"""

import os
import json
import hashlib
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import yaml

import torch
import numpy as np

from .random_utils import set_random_seeds, get_random_state, generate_experiment_seed
from .logging_utils import get_logger, LoggerMixin
from ..config import ExperimentConfig


@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking and reproducibility."""
    
    # Basic experiment info
    experiment_id: str
    name: str
    description: str
    version: str
    timestamp: str
    
    # Reproducibility info
    seed: int
    python_version: str
    torch_version: str
    numpy_version: str
    cuda_version: Optional[str]
    
    # System info
    hostname: str
    platform: str
    cpu_count: int
    gpu_count: int
    gpu_names: List[str]
    
    # Git info (if available)
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: Optional[bool]
    
    # Configuration hash
    config_hash: str
    
    # Status tracking
    status: str = "initialized"  # initialized, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Results summary
    final_metrics: Optional[Dict[str, float]] = None
    best_metrics: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentMetadata':
        """Create from dictionary."""
        return cls(**data)


class ExperimentManager(LoggerMixin):
    """Comprehensive experiment management system."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        base_output_dir: Union[str, Path] = "experiments",
        enable_versioning: bool = True,
        enable_git_tracking: bool = True
    ):
        """Initialize experiment manager.
        
        Args:
            config: Experiment configuration
            base_output_dir: Base directory for all experiments
            enable_versioning: Whether to enable experiment versioning
            enable_git_tracking: Whether to track git information
        """
        self.config = config
        self.base_output_dir = Path(base_output_dir)
        self.enable_versioning = enable_versioning
        self.enable_git_tracking = enable_git_tracking
        
        # Generate unique experiment ID
        self.experiment_id = self._generate_experiment_id()
        
        # Setup experiment directory
        self.experiment_dir = self._setup_experiment_directory()
        
        # Initialize metadata
        self.metadata = self._create_metadata()
        
        # Setup reproducibility
        self._setup_reproducibility()
        
        # Save initial state
        self._save_initial_state()
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        # Use timestamp + config hash for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{self.config.name}_{timestamp}_{config_hash}"
    
    def _setup_experiment_directory(self) -> Path:
        """Setup experiment directory structure."""
        # Create base directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific directory
        exp_dir = self.base_output_dir / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "config", "checkpoints", "logs", "plots", "results", 
            "code_snapshot", "data", "models", "metrics"
        ]
        
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=True)
        
        self.log_info(f"Created experiment directory: {exp_dir}")
        return exp_dir
    
    def _create_metadata(self) -> ExperimentMetadata:
        """Create experiment metadata."""
        import platform
        import sys
        
        # System information
        hostname = platform.node()
        platform_info = platform.platform()
        cpu_count = os.cpu_count()
        
        # GPU information
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_names = []
        if gpu_count > 0:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        
        # Version information
        python_version = sys.version
        torch_version = torch.__version__
        numpy_version = np.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else None
        
        # Git information
        git_info = self._get_git_info() if self.enable_git_tracking else {}
        
        # Configuration hash
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        return ExperimentMetadata(
            experiment_id=self.experiment_id,
            name=self.config.name,
            description=self.config.description,
            version=self.config.version,
            timestamp=datetime.now().isoformat(),
            seed=self.config.seed,
            python_version=python_version,
            torch_version=torch_version,
            numpy_version=numpy_version,
            cuda_version=cuda_version,
            hostname=hostname,
            platform=platform_info,
            cpu_count=cpu_count,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            git_commit=git_info.get('commit'),
            git_branch=git_info.get('branch'),
            git_dirty=git_info.get('dirty'),
            config_hash=config_hash
        )
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information."""
        try:
            import subprocess
            
            # Get current commit
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Get current branch
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Check if repository is dirty
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            dirty = len(status) > 0
            
            return {
                'commit': commit,
                'branch': branch,
                'dirty': dirty
            }
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log_warning("Could not retrieve git information")
            return {}
    
    def _setup_reproducibility(self):
        """Setup reproducibility guarantees."""
        # Set deterministic seed
        experiment_seed = generate_experiment_seed(
            self.config.seed, 
            self.experiment_id
        )
        
        set_random_seeds(
            experiment_seed,
            deterministic=self.config.deterministic
        )
        
        # Save random state
        self.initial_random_state = get_random_state()
        
        # Set environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = str(experiment_seed)
        
        # Additional PyTorch settings
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        
        self.log_info(f"Set experiment seed: {experiment_seed}")
    
    def _save_initial_state(self):
        """Save initial experiment state."""
        # Save configuration
        config_path = self.experiment_dir / "config" / "experiment_config.yaml"
        self.config.to_yaml(config_path)
        
        # Save metadata
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save random state
        random_state_path = self.experiment_dir / "config" / "random_state.pkl"
        with open(random_state_path, 'wb') as f:
            pickle.dump(self.initial_random_state, f)
        
        # Create code snapshot
        if self.enable_versioning:
            self._create_code_snapshot()
        
        self.log_info("Saved initial experiment state")
    
    def _create_code_snapshot(self):
        """Create snapshot of current code state."""
        code_snapshot_dir = self.experiment_dir / "code_snapshot"
        
        # Find project root (directory containing setup.py or pyproject.toml)
        current_dir = Path.cwd()
        project_root = current_dir
        
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
                project_root = parent
                break
        
        # Copy Python files
        for py_file in project_root.rglob("*.py"):
            # Skip __pycache__ and .git directories
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            
            # Skip experiment output directories
            if "experiments" in str(py_file) and "outputs" in str(py_file):
                continue
            
            # Create relative path structure
            rel_path = py_file.relative_to(project_root)
            dest_path = code_snapshot_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(py_file, dest_path)
        
        # Copy configuration files
        config_files = ["setup.py", "pyproject.toml", "requirements.txt", "environment.yml"]
        for config_file in config_files:
            src_path = project_root / config_file
            if src_path.exists():
                shutil.copy2(src_path, code_snapshot_dir / config_file)
        
        self.log_info(f"Created code snapshot in {code_snapshot_dir}")
    
    def start_experiment(self):
        """Mark experiment as started."""
        self.metadata.status = "running"
        self.metadata.start_time = datetime.now().isoformat()
        self._update_metadata()
        self.log_info("Experiment started")
    
    def complete_experiment(
        self, 
        final_metrics: Optional[Dict[str, float]] = None,
        best_metrics: Optional[Dict[str, float]] = None
    ):
        """Mark experiment as completed."""
        end_time = datetime.now()
        self.metadata.status = "completed"
        self.metadata.end_time = end_time.isoformat()
        
        if self.metadata.start_time:
            start_time = datetime.fromisoformat(self.metadata.start_time)
            self.metadata.duration_seconds = (end_time - start_time).total_seconds()
        
        if final_metrics:
            self.metadata.final_metrics = final_metrics
        
        if best_metrics:
            self.metadata.best_metrics = best_metrics
        
        self._update_metadata()
        self.log_info("Experiment completed successfully")
    
    def fail_experiment(self, error_message: str):
        """Mark experiment as failed."""
        self.metadata.status = "failed"
        self.metadata.end_time = datetime.now().isoformat()
        
        # Save error information
        error_path = self.experiment_dir / "error.txt"
        with open(error_path, 'w') as f:
            f.write(f"Experiment failed at: {self.metadata.end_time}\n")
            f.write(f"Error: {error_message}\n")
        
        self._update_metadata()
        self.log_error(f"Experiment failed: {error_message}")
    
    def _update_metadata(self):
        """Update metadata file."""
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
    
    def save_checkpoint(
        self, 
        model_state: Dict[str, Any], 
        optimizer_state: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint_dir = self.experiment_dir / "checkpoints"
        
        checkpoint_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'epoch': epoch,
            'metrics': metrics,
            'experiment_id': self.experiment_id,
            'config_hash': self.metadata.config_hash,
            'random_state': get_random_state()
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best checkpoint if specified
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint_data, best_path)
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint_data, latest_path)
        
        self.log_debug(f"Saved checkpoint for epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.experiment_dir / "checkpoints" / "latest_checkpoint.pth"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Verify checkpoint compatibility
        if checkpoint_data.get('config_hash') != self.metadata.config_hash:
            self.log_warning("Checkpoint config hash mismatch - may be incompatible")
        
        self.log_info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data
    
    def save_metrics(self, metrics: Dict[str, float], step: int):
        """Save training metrics."""
        metrics_dir = self.experiment_dir / "metrics"
        
        # Append to metrics log
        metrics_log_path = metrics_dir / "metrics.jsonl"
        
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        with open(metrics_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        return {
            'metadata': self.metadata.to_dict(),
            'config': self.config.to_dict(),
            'experiment_dir': str(self.experiment_dir),
            'files': {
                'config': str(self.experiment_dir / "config" / "experiment_config.yaml"),
                'metadata': str(self.experiment_dir / "metadata.json"),
                'logs': str(self.experiment_dir / "logs"),
                'checkpoints': str(self.experiment_dir / "checkpoints"),
                'results': str(self.experiment_dir / "results")
            }
        }
    
    @classmethod
    def load_experiment(cls, experiment_dir: Union[str, Path]) -> 'ExperimentManager':
        """Load existing experiment."""
        experiment_dir = Path(experiment_dir)
        
        # Load metadata
        metadata_path = experiment_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = ExperimentMetadata.from_dict(metadata_dict)
        
        # Load configuration
        config_path = experiment_dir / "config" / "experiment_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = ExperimentConfig.from_yaml(config_path)
        
        # Create manager instance
        manager = cls.__new__(cls)
        manager.config = config
        manager.experiment_dir = experiment_dir
        manager.experiment_id = metadata.experiment_id
        manager.metadata = metadata
        manager.enable_versioning = True
        manager.enable_git_tracking = True
        
        return manager


class ExperimentRegistry(LoggerMixin):
    """Registry for tracking multiple experiments."""
    
    def __init__(self, registry_dir: Union[str, Path] = "experiments"):
        """Initialize experiment registry.
        
        Args:
            registry_dir: Directory containing experiments
        """
        self.registry_dir = Path(registry_dir)
        self.registry_file = self.registry_dir / "experiment_registry.json"
        
        # Load existing registry
        self.experiments = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load experiment registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save experiment registry."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def register_experiment(self, manager: ExperimentManager):
        """Register an experiment."""
        self.experiments[manager.experiment_id] = {
            'name': manager.config.name,
            'description': manager.config.description,
            'timestamp': manager.metadata.timestamp,
            'status': manager.metadata.status,
            'experiment_dir': str(manager.experiment_dir),
            'config_hash': manager.metadata.config_hash
        }
        self._save_registry()
        self.log_info(f"Registered experiment: {manager.experiment_id}")
    
    def list_experiments(
        self, 
        status: Optional[str] = None,
        name_pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [exp for exp in experiments if exp['status'] == status]
        
        if name_pattern:
            experiments = [exp for exp in experiments if name_pattern in exp['name']]
        
        return experiments
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment information."""
        return self.experiments.get(experiment_id)
    
    def cleanup_failed_experiments(self):
        """Remove failed experiments from registry."""
        failed_experiments = [
            exp_id for exp_id, exp_data in self.experiments.items()
            if exp_data['status'] == 'failed'
        ]
        
        for exp_id in failed_experiments:
            del self.experiments[exp_id]
        
        self._save_registry()
        self.log_info(f"Cleaned up {len(failed_experiments)} failed experiments")
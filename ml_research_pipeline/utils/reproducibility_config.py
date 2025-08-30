"""
Reproducibility configuration and environment management system.
"""

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .logging_utils import LoggerMixin


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility settings."""

    # Random seeds
    global_seed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42
    python_seed: int = 42

    # Deterministic settings
    torch_deterministic: bool = True
    torch_benchmark: bool = False
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    # Environment variables
    pythonhashseed: Optional[str] = None
    cuda_launch_blocking: bool = True

    # Precision settings
    torch_float32_matmul_precision: str = "high"  # "highest", "high", "medium"

    # Validation settings
    validation_tolerance: float = 1e-6
    cross_platform_tolerance: float = 1e-5

    # Hardware constraints
    require_cuda: bool = False
    min_cuda_version: Optional[str] = None
    required_torch_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReproducibilityConfig":
        """Create from dictionary."""
        return cls(**data)

    def to_json(self, file_path: Path):
        """Save to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, file_path: Path) -> "ReproducibilityConfig":
        """Load from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ReproducibilityEnvironmentManager(LoggerMixin):
    """Manager for reproducibility environment setup and validation."""

    def __init__(self, config: Optional[ReproducibilityConfig] = None):
        """Initialize reproducibility environment manager.

        Args:
            config: Reproducibility configuration
        """
        self.config = config or ReproducibilityConfig()
        self.environment_snapshot = None
        self.applied_settings = {}

    def setup_reproducible_environment(self) -> Dict[str, Any]:
        """Setup reproducible environment based on configuration.

        Returns:
            Dictionary of applied settings
        """
        self.log_info("Setting up reproducible environment")

        # Take snapshot of current environment
        self.environment_snapshot = self._capture_environment_snapshot()

        applied_settings = {}

        # Set random seeds
        self._set_random_seeds(applied_settings)

        # Configure PyTorch settings
        self._configure_torch_settings(applied_settings)

        # Set environment variables
        self._set_environment_variables(applied_settings)

        # Validate hardware requirements
        self._validate_hardware_requirements()

        self.applied_settings = applied_settings
        self.log_info("Reproducible environment setup complete")

        return applied_settings

    def _set_random_seeds(self, applied_settings: Dict[str, Any]):
        """Set all random seeds."""
        # Python hash seed (must be set before importing modules)
        if self.config.pythonhashseed is not None:
            os.environ["PYTHONHASHSEED"] = str(self.config.pythonhashseed)
            applied_settings["PYTHONHASHSEED"] = self.config.pythonhashseed
        elif "PYTHONHASHSEED" not in os.environ:
            os.environ["PYTHONHASHSEED"] = str(self.config.python_seed)
            applied_settings["PYTHONHASHSEED"] = self.config.python_seed

        # PyTorch seed
        torch.manual_seed(self.config.torch_seed)
        applied_settings["torch_seed"] = self.config.torch_seed

        # NumPy seed
        np.random.seed(self.config.numpy_seed)
        applied_settings["numpy_seed"] = self.config.numpy_seed

        # CUDA seed (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.torch_seed)
            torch.cuda.manual_seed_all(self.config.torch_seed)
            applied_settings["cuda_seed"] = self.config.torch_seed

    def _configure_torch_settings(self, applied_settings: Dict[str, Any]):
        """Configure PyTorch deterministic settings."""
        # Deterministic algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(self.config.torch_deterministic)
            applied_settings["torch_deterministic_algorithms"] = (
                self.config.torch_deterministic
            )

        # Backend settings
        torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        applied_settings["cudnn_deterministic"] = self.config.cudnn_deterministic
        applied_settings["cudnn_benchmark"] = self.config.cudnn_benchmark

        # Benchmark mode
        if hasattr(torch.backends.cudnn, "benchmark"):
            torch.backends.cudnn.benchmark = self.config.torch_benchmark
            applied_settings["torch_benchmark"] = self.config.torch_benchmark

        # Float32 matmul precision
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(
                self.config.torch_float32_matmul_precision
            )
            applied_settings["float32_matmul_precision"] = (
                self.config.torch_float32_matmul_precision
            )

    def _set_environment_variables(self, applied_settings: Dict[str, Any]):
        """Set environment variables for reproducibility."""
        env_vars = {}

        # CUDA launch blocking
        if self.config.cuda_launch_blocking:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            env_vars["CUDA_LAUNCH_BLOCKING"] = "1"

        # Additional reproducibility environment variables
        reproducibility_env_vars = {
            "TF_DETERMINISTIC_OPS": "1",  # For TensorFlow compatibility
            "TF_CUDNN_DETERMINISTIC": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # For CUDA deterministic operations
        }

        for key, value in reproducibility_env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
                env_vars[key] = value

        applied_settings["environment_variables"] = env_vars

    def _validate_hardware_requirements(self):
        """Validate hardware requirements for reproducibility."""
        # Check CUDA availability
        if self.config.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available")

        # Check CUDA version
        if self.config.min_cuda_version and torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version and cuda_version < self.config.min_cuda_version:
                raise RuntimeError(
                    f"CUDA version {cuda_version} is below minimum required {self.config.min_cuda_version}"
                )

        # Check PyTorch version
        if self.config.required_torch_version:
            torch_version = torch.__version__
            if torch_version != self.config.required_torch_version:
                self.log_warning(
                    f"PyTorch version {torch_version} differs from required {self.config.required_torch_version}"
                )

    def _capture_environment_snapshot(self) -> Dict[str, Any]:
        """Capture snapshot of current environment."""
        snapshot = {
            "timestamp": torch.utils.data.get_worker_info(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
            },
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else None
                ),
                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if torch.cuda.is_available()
                    else None
                ),
                "device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
            },
            "numpy": {"version": np.__version__},
            "environment_variables": dict(os.environ),
            "git_info": self._get_git_info(),
        }

        # Add GPU information
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_info.append(
                    {
                        "device_id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(
                            i
                        ).total_memory,
                        "compute_capability": torch.cuda.get_device_properties(i).major,
                    }
                )
            snapshot["torch"]["gpu_info"] = gpu_info

        return snapshot

    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information."""
        try:
            # Get current commit
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

            # Get current branch
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

            # Check if repository is dirty
            status = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            dirty = len(status) > 0

            # Get remote URL
            try:
                remote_url = (
                    subprocess.check_output(
                        ["git", "config", "--get", "remote.origin.url"],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError:
                remote_url = None

            return {
                "commit": commit,
                "branch": branch,
                "dirty": dirty,
                "remote_url": remote_url,
            }

        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"error": "Git information not available"}

    def validate_environment_consistency(
        self, reference_snapshot: Dict[str, Any], strict_mode: bool = False
    ) -> Dict[str, Any]:
        """Validate current environment against reference snapshot.

        Args:
            reference_snapshot: Reference environment snapshot
            strict_mode: Whether to use strict validation

        Returns:
            Validation results
        """
        current_snapshot = self._capture_environment_snapshot()

        validation_results = {
            "overall_consistent": True,
            "differences": {},
            "warnings": [],
            "errors": [],
        }

        # Check critical components
        critical_checks = [
            ("torch.version", "PyTorch version"),
            ("numpy.version", "NumPy version"),
            ("platform.python_version", "Python version"),
            ("platform.system", "Operating system"),
        ]

        for path, description in critical_checks:
            ref_value = self._get_nested_value(reference_snapshot, path)
            curr_value = self._get_nested_value(current_snapshot, path)

            if ref_value != curr_value:
                validation_results["differences"][path] = {
                    "reference": ref_value,
                    "current": curr_value,
                    "description": description,
                }

                if strict_mode:
                    validation_results["errors"].append(
                        f"{description} mismatch: {ref_value} -> {curr_value}"
                    )
                    validation_results["overall_consistent"] = False
                else:
                    validation_results["warnings"].append(
                        f"{description} difference: {ref_value} -> {curr_value}"
                    )

        # Check CUDA consistency
        if (
            reference_snapshot["torch"]["cuda_available"]
            != current_snapshot["torch"]["cuda_available"]
        ):
            validation_results["differences"]["cuda_availability"] = {
                "reference": reference_snapshot["torch"]["cuda_available"],
                "current": current_snapshot["torch"]["cuda_available"],
                "description": "CUDA availability",
            }
            validation_results["errors"].append("CUDA availability changed")
            validation_results["overall_consistent"] = False

        # Check GPU configuration
        if (
            reference_snapshot["torch"]["cuda_available"]
            and current_snapshot["torch"]["cuda_available"]
        ):

            ref_gpu_count = reference_snapshot["torch"]["device_count"]
            curr_gpu_count = current_snapshot["torch"]["device_count"]

            if ref_gpu_count != curr_gpu_count:
                validation_results["differences"]["gpu_count"] = {
                    "reference": ref_gpu_count,
                    "current": curr_gpu_count,
                    "description": "GPU count",
                }
                validation_results["warnings"].append(
                    f"GPU count changed: {ref_gpu_count} -> {curr_gpu_count}"
                )

        return validation_results

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def generate_reproducibility_checklist(self) -> List[Dict[str, Any]]:
        """Generate reproducibility checklist for current environment.

        Returns:
            List of checklist items with status
        """
        checklist = []

        # Check random seeds
        checklist.append(
            {
                "item": "Random seeds set",
                "status": (
                    "pass"
                    if self.applied_settings.get("torch_seed") is not None
                    else "fail"
                ),
                "details": f"Torch seed: {self.applied_settings.get('torch_seed')}",
            }
        )

        # Check deterministic algorithms
        checklist.append(
            {
                "item": "Deterministic algorithms enabled",
                "status": (
                    "pass"
                    if self.applied_settings.get("torch_deterministic_algorithms")
                    else "fail"
                ),
                "details": f"Deterministic: {self.applied_settings.get('torch_deterministic_algorithms')}",
            }
        )

        # Check CUDNN settings
        checklist.append(
            {
                "item": "CUDNN deterministic mode",
                "status": (
                    "pass"
                    if self.applied_settings.get("cudnn_deterministic")
                    else "fail"
                ),
                "details": f"CUDNN deterministic: {self.applied_settings.get('cudnn_deterministic')}",
            }
        )

        # Check environment variables
        env_vars = self.applied_settings.get("environment_variables", {})
        checklist.append(
            {
                "item": "Environment variables set",
                "status": "pass" if env_vars else "warning",
                "details": f"Set variables: {list(env_vars.keys())}",
            }
        )

        # Check git status
        if self.environment_snapshot and "git_info" in self.environment_snapshot:
            git_info = self.environment_snapshot["git_info"]
            if "error" not in git_info:
                checklist.append(
                    {
                        "item": "Git repository clean",
                        "status": "pass" if not git_info.get("dirty") else "warning",
                        "details": f"Branch: {git_info.get('branch')}, Dirty: {git_info.get('dirty')}",
                    }
                )

        # Check hardware consistency
        if torch.cuda.is_available():
            checklist.append(
                {
                    "item": "CUDA available and configured",
                    "status": "pass",
                    "details": f"CUDA version: {torch.version.cuda}, Devices: {torch.cuda.device_count()}",
                }
            )
        else:
            checklist.append(
                {
                    "item": "CPU-only mode",
                    "status": "info",
                    "details": "CUDA not available - using CPU only",
                }
            )

        return checklist

    def save_environment_snapshot(self, file_path: Path):
        """Save current environment snapshot to file.

        Args:
            file_path: Output file path
        """
        if self.environment_snapshot is None:
            self.environment_snapshot = self._capture_environment_snapshot()

        snapshot_data = {
            "config": self.config.to_dict(),
            "applied_settings": self.applied_settings,
            "environment_snapshot": self.environment_snapshot,
            "checklist": self.generate_reproducibility_checklist(),
        }

        with open(file_path, "w") as f:
            json.dump(snapshot_data, f, indent=2, default=str)

        self.log_info(f"Saved environment snapshot to {file_path}")

    @classmethod
    def load_environment_snapshot(
        cls, file_path: Path
    ) -> "ReproducibilityEnvironmentManager":
        """Load environment manager from snapshot file.

        Args:
            file_path: Snapshot file path

        Returns:
            Loaded environment manager
        """
        with open(file_path, "r") as f:
            snapshot_data = json.load(f)

        config = ReproducibilityConfig.from_dict(snapshot_data["config"])
        manager = cls(config)
        manager.applied_settings = snapshot_data["applied_settings"]
        manager.environment_snapshot = snapshot_data["environment_snapshot"]

        return manager


def create_default_reproducibility_config() -> ReproducibilityConfig:
    """Create default reproducibility configuration.

    Returns:
        Default reproducibility configuration
    """
    return ReproducibilityConfig(
        global_seed=42,
        torch_seed=42,
        numpy_seed=42,
        python_seed=42,
        torch_deterministic=True,
        torch_benchmark=False,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        pythonhashseed="42",
        cuda_launch_blocking=True,
        torch_float32_matmul_precision="high",
        validation_tolerance=1e-6,
        cross_platform_tolerance=1e-5,
    )


def create_strict_reproducibility_config() -> ReproducibilityConfig:
    """Create strict reproducibility configuration for maximum determinism.

    Returns:
        Strict reproducibility configuration
    """
    return ReproducibilityConfig(
        global_seed=42,
        torch_seed=42,
        numpy_seed=42,
        python_seed=42,
        torch_deterministic=True,
        torch_benchmark=False,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
        pythonhashseed="42",
        cuda_launch_blocking=True,
        torch_float32_matmul_precision="highest",
        validation_tolerance=1e-8,
        cross_platform_tolerance=1e-6,
        require_cuda=False,
        required_torch_version=torch.__version__,
    )

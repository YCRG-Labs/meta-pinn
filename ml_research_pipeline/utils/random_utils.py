"""
Random seed utilities for reproducible experiments.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_random_seeds(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Try to enable deterministic algorithms, but handle cases where it's not supported
        try:
            torch.use_deterministic_algorithms(True)
        except (RuntimeError, AttributeError) as e:
            # Some operations don't have deterministic implementations
            # or the function might not exist in older PyTorch versions
            # Fall back to just setting cudnn deterministic
            import warnings

            warnings.warn(
                f"Could not enable all deterministic algorithms: {e}. "
                "Using cudnn.deterministic=True only.",
                UserWarning,
            )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_random_state() -> dict:
    """Get current random state for all generators.

    Returns:
        Dictionary containing random states
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def set_random_state(state: dict) -> None:
    """Restore random state for all generators.

    Args:
        state: Dictionary containing random states
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class RandomStateManager:
    """Context manager for temporary random state changes."""

    def __init__(self, seed: Optional[int] = None, deterministic: bool = True):
        """Initialize random state manager.

        Args:
            seed: Temporary seed to use (if None, uses current state)
            deterministic: Whether to use deterministic algorithms
        """
        self.seed = seed
        self.deterministic = deterministic
        self.saved_state = None
        self.saved_deterministic = None

    def __enter__(self):
        """Save current state and set new seed if provided."""
        # Save current state
        self.saved_state = get_random_state()
        self.saved_deterministic = torch.backends.cudnn.deterministic

        # Set new seed if provided
        if self.seed is not None:
            set_random_seeds(self.seed, self.deterministic)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous random state."""
        if self.saved_state is not None:
            set_random_state(self.saved_state)

        # Restore deterministic setting
        if self.saved_deterministic is not None:
            torch.backends.cudnn.deterministic = self.saved_deterministic


def generate_experiment_seed(base_seed: int, experiment_id: str) -> int:
    """Generate a deterministic seed for an experiment.

    Args:
        base_seed: Base random seed
        experiment_id: Unique experiment identifier

    Returns:
        Deterministic seed for the experiment
    """
    # Use hash of experiment_id combined with base_seed
    experiment_hash = hash(experiment_id) % (2**31)
    return (base_seed + experiment_hash) % (2**31)

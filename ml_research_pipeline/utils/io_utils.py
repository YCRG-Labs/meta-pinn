"""
Input/Output utilities for checkpointing and data management.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml


def save_checkpoint(
    state: Dict[str, Any], filepath: Union[str, Path], is_best: bool = False
) -> None:
    """Save model checkpoint.

    Args:
        state: State dictionary containing model, optimizer, etc.
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        best_path = filepath.parent / "best_model.pth"
        torch.save(state, best_path)


def load_checkpoint(
    filepath: Union[str, Path], device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        device: Device to load checkpoint on

    Returns:
        State dictionary
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save data to YAML file.

    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load data from YAML file.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save data using pickle.

    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data using pickle.

    Args:
        filepath: Input file path

    Returns:
        Loaded data
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def ensure_dir(dirpath: Union[str, Path]) -> Path:
    """Ensure directory exists.

    Args:
        dirpath: Directory path

    Returns:
        Path object
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def get_file_size(filepath: Union[str, Path]) -> int:
    """Get file size in bytes.

    Args:
        filepath: File path

    Returns:
        File size in bytes
    """
    return os.path.getsize(filepath)


def list_files(
    directory: Union[str, Path], pattern: str = "*", recursive: bool = False
) -> list:
    """List files in directory.

    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively

    Returns:
        List of file paths
    """
    directory = Path(directory)

    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))

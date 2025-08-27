"""
Base configuration classes and utilities.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration class with serialization capabilities."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> 'BaseConfig':
        """Update configuration with new values."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """Merge with another configuration."""
        config_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Deep merge dictionaries
        def deep_merge(dict1, dict2):
            result = dict1.copy()
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(config_dict, other_dict)
        return self.__class__.from_dict(merged_dict)


class Config:
    """Configuration manager for handling multiple configuration types."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry of configuration classes
        self._config_registry = {}
    
    def register_config(self, name: str, config_class: type):
        """Register a configuration class."""
        self._config_registry[name] = config_class
    
    def load_config(self, name: str, filename: Optional[str] = None) -> BaseConfig:
        """Load configuration by name.
        
        Args:
            name: Name of registered configuration class
            filename: Optional filename, defaults to {name}.yaml
            
        Returns:
            Loaded configuration instance
        """
        if name not in self._config_registry:
            raise ValueError(f"Configuration '{name}' not registered")
        
        config_class = self._config_registry[name]
        
        if filename is None:
            filename = f"{name}.yaml"
        
        filepath = self.config_dir / filename
        
        if filepath.exists():
            return config_class.from_yaml(filepath)
        else:
            # Return default configuration
            return config_class()
    
    def save_config(self, config: BaseConfig, name: str, filename: Optional[str] = None):
        """Save configuration.
        
        Args:
            config: Configuration instance to save
            name: Name for the configuration
            filename: Optional filename, defaults to {name}.yaml
        """
        if filename is None:
            filename = f"{name}.yaml"
        
        filepath = self.config_dir / filename
        config.to_yaml(filepath)
    
    def list_configs(self) -> list:
        """List available configuration files."""
        config_files = []
        for filepath in self.config_dir.glob("*.yaml"):
            config_files.append(filepath.stem)
        for filepath in self.config_dir.glob("*.json"):
            config_files.append(filepath.stem)
        return sorted(config_files)


# Global configuration manager instance
config_manager = Config()
"""
Tests for reproducibility configuration system.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import numpy as np

from ml_research_pipeline.utils.reproducibility_config import (
    ReproducibilityConfig,
    ReproducibilityEnvironmentManager,
    create_default_reproducibility_config,
    create_strict_reproducibility_config
)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Create test reproducibility configuration."""
    return ReproducibilityConfig(
        global_seed=123,
        torch_seed=123,
        numpy_seed=123,
        python_seed=123,
        torch_deterministic=True,
        cudnn_deterministic=True,
        validation_tolerance=1e-8
    )


class TestReproducibilityConfig:
    """Test reproducibility configuration."""
    
    def test_config_creation(self):
        """Test configuration creation with default values."""
        config = ReproducibilityConfig()
        
        assert config.global_seed == 42
        assert config.torch_seed == 42
        assert config.numpy_seed == 42
        assert config.python_seed == 42
        assert config.torch_deterministic is True
        assert config.cudnn_deterministic is True
        assert config.validation_tolerance == 1e-6
    
    def test_config_serialization(self, test_config, temp_dir):
        """Test configuration serialization and deserialization."""
        # Test to_dict
        config_dict = test_config.to_dict()
        
        assert config_dict["global_seed"] == 123
        assert config_dict["torch_deterministic"] is True
        assert config_dict["validation_tolerance"] == 1e-8
        
        # Test from_dict
        restored_config = ReproducibilityConfig.from_dict(config_dict)
        
        assert restored_config.global_seed == test_config.global_seed
        assert restored_config.torch_deterministic == test_config.torch_deterministic
        assert restored_config.validation_tolerance == test_config.validation_tolerance
        
        # Test JSON serialization
        json_file = temp_dir / "config.json"
        test_config.to_json(json_file)
        
        assert json_file.exists()
        
        loaded_config = ReproducibilityConfig.from_json(json_file)
        assert loaded_config.global_seed == test_config.global_seed
        assert loaded_config.torch_deterministic == test_config.torch_deterministic
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = create_default_reproducibility_config()
        
        assert isinstance(config, ReproducibilityConfig)
        assert config.global_seed == 42
        assert config.torch_deterministic is True
        assert config.cudnn_deterministic is True
        assert config.validation_tolerance == 1e-6
    
    def test_strict_config_creation(self):
        """Test strict configuration creation."""
        config = create_strict_reproducibility_config()
        
        assert isinstance(config, ReproducibilityConfig)
        assert config.torch_deterministic is True
        assert config.torch_float32_matmul_precision == "highest"
        assert config.validation_tolerance == 1e-8
        assert config.cross_platform_tolerance == 1e-6
        assert config.required_torch_version == torch.__version__


class TestReproducibilityEnvironmentManager:
    """Test reproducibility environment manager."""
    
    def test_manager_initialization(self, test_config):
        """Test environment manager initialization."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        assert manager.config == test_config
        assert manager.environment_snapshot is None
        assert manager.applied_settings == {}
    
    def test_manager_initialization_with_default_config(self):
        """Test environment manager initialization with default config."""
        manager = ReproducibilityEnvironmentManager()
        
        assert isinstance(manager.config, ReproducibilityConfig)
        assert manager.config.global_seed == 42
    
    def test_reproducible_environment_setup(self, test_config):
        """Test reproducible environment setup."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        # Setup environment
        applied_settings = manager.setup_reproducible_environment()
        
        # Check that settings were applied
        assert "torch_seed" in applied_settings
        assert "numpy_seed" in applied_settings
        assert "torch_deterministic_algorithms" in applied_settings
        assert "cudnn_deterministic" in applied_settings
        
        # Check actual PyTorch settings
        assert torch.backends.cudnn.deterministic == test_config.cudnn_deterministic
        
        # Check environment snapshot was created
        assert manager.environment_snapshot is not None
        assert "platform" in manager.environment_snapshot
        assert "torch" in manager.environment_snapshot
        assert "numpy" in manager.environment_snapshot
    
    def test_random_seed_setting(self, test_config):
        """Test random seed setting."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        applied_settings = {}
        manager._set_random_seeds(applied_settings)
        
        # Check that seeds were set
        assert applied_settings["torch_seed"] == test_config.torch_seed
        assert applied_settings["numpy_seed"] == test_config.numpy_seed
        
        # Verify actual seed values by generating random numbers
        torch.manual_seed(test_config.torch_seed)
        torch_value1 = torch.randn(1).item()
        
        torch.manual_seed(test_config.torch_seed)
        torch_value2 = torch.randn(1).item()
        
        assert torch_value1 == torch_value2  # Should be identical with same seed
        
        np.random.seed(test_config.numpy_seed)
        numpy_value1 = np.random.randn()
        
        np.random.seed(test_config.numpy_seed)
        numpy_value2 = np.random.randn()
        
        assert numpy_value1 == numpy_value2  # Should be identical with same seed
    
    def test_torch_settings_configuration(self, test_config):
        """Test PyTorch settings configuration."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        applied_settings = {}
        manager._configure_torch_settings(applied_settings)
        
        # Check applied settings
        assert applied_settings["cudnn_deterministic"] == test_config.cudnn_deterministic
        assert applied_settings["cudnn_benchmark"] == test_config.cudnn_benchmark
        
        # Check actual PyTorch backend settings
        assert torch.backends.cudnn.deterministic == test_config.cudnn_deterministic
        assert torch.backends.cudnn.benchmark == test_config.cudnn_benchmark
    
    def test_environment_variables_setting(self, test_config):
        """Test environment variables setting."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        # Store original environment
        original_env = dict(os.environ)
        
        try:
            applied_settings = {}
            manager._set_environment_variables(applied_settings)
            
            # Check that environment variables were set
            if test_config.cuda_launch_blocking:
                assert os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"
                assert "CUDA_LAUNCH_BLOCKING" in applied_settings["environment_variables"]
            
            # Check reproducibility-related environment variables
            assert os.environ.get("TF_DETERMINISTIC_OPS") == "1"
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    @patch('subprocess.check_output')
    def test_git_info_collection(self, mock_subprocess, test_config):
        """Test git information collection."""
        # Mock git commands
        mock_subprocess.side_effect = [
            b'abc123def456\n',  # git rev-parse HEAD
            b'main\n',          # git rev-parse --abbrev-ref HEAD
            b'M file.py\n',     # git status --porcelain
            b'https://github.com/user/repo.git\n'  # git config --get remote.origin.url
        ]
        
        manager = ReproducibilityEnvironmentManager(test_config)
        git_info = manager._get_git_info()
        
        assert git_info["commit"] == "abc123def456"
        assert git_info["branch"] == "main"
        assert git_info["dirty"] is True
        assert git_info["remote_url"] == "https://github.com/user/repo.git"
    
    def test_git_info_collection_failure(self, test_config):
        """Test git information collection when git is not available."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        with patch('subprocess.check_output', side_effect=FileNotFoundError):
            git_info = manager._get_git_info()
            
            assert "error" in git_info
            assert git_info["error"] == "Git information not available"
    
    def test_hardware_requirements_validation(self, test_config):
        """Test hardware requirements validation."""
        # Test with CUDA not required
        test_config.require_cuda = False
        manager = ReproducibilityEnvironmentManager(test_config)
        
        # Should not raise exception
        manager._validate_hardware_requirements()
        
        # Test with CUDA required but not available
        if not torch.cuda.is_available():
            test_config.require_cuda = True
            manager = ReproducibilityEnvironmentManager(test_config)
            
            with pytest.raises(RuntimeError, match="CUDA is required but not available"):
                manager._validate_hardware_requirements()
    
    def test_environment_snapshot_capture(self, test_config):
        """Test environment snapshot capture."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        snapshot = manager._capture_environment_snapshot()
        
        # Check snapshot structure
        assert "platform" in snapshot
        assert "torch" in snapshot
        assert "numpy" in snapshot
        assert "environment_variables" in snapshot
        assert "git_info" in snapshot
        
        # Check platform info
        platform_info = snapshot["platform"]
        assert "system" in platform_info
        assert "python_version" in platform_info
        
        # Check torch info
        torch_info = snapshot["torch"]
        assert "version" in torch_info
        assert "cuda_available" in torch_info
        
        # Check numpy info
        numpy_info = snapshot["numpy"]
        assert "version" in numpy_info
    
    def test_environment_consistency_validation(self, test_config):
        """Test environment consistency validation."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        # Create reference snapshot
        reference_snapshot = manager._capture_environment_snapshot()
        
        # Validate against itself (should be consistent)
        validation_results = manager.validate_environment_consistency(
            reference_snapshot, strict_mode=False
        )
        
        assert validation_results["overall_consistent"] is True
        assert len(validation_results["differences"]) == 0
        assert len(validation_results["errors"]) == 0
        
        # Test with modified snapshot (should detect differences)
        modified_snapshot = reference_snapshot.copy()
        modified_snapshot["torch"]["version"] = "different_version"
        
        validation_results = manager.validate_environment_consistency(
            modified_snapshot, strict_mode=True
        )
        
        assert validation_results["overall_consistent"] is False
        assert len(validation_results["differences"]) > 0
        assert len(validation_results["errors"]) > 0
    
    def test_reproducibility_checklist_generation(self, test_config):
        """Test reproducibility checklist generation."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        # Setup environment first
        manager.setup_reproducible_environment()
        
        # Generate checklist
        checklist = manager.generate_reproducibility_checklist()
        
        assert isinstance(checklist, list)
        assert len(checklist) > 0
        
        # Check checklist items
        checklist_items = [item["item"] for item in checklist]
        
        assert "Random seeds set" in checklist_items
        assert "Deterministic algorithms enabled" in checklist_items
        assert "CUDNN deterministic mode" in checklist_items
        assert "Environment variables set" in checklist_items
        
        # Check that all items have required fields
        for item in checklist:
            assert "item" in item
            assert "status" in item
            assert "details" in item
            assert item["status"] in ["pass", "fail", "warning", "info"]
    
    def test_environment_snapshot_save_load(self, test_config, temp_dir):
        """Test environment snapshot save and load."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        # Setup environment and save snapshot
        manager.setup_reproducible_environment()
        
        snapshot_file = temp_dir / "environment_snapshot.json"
        manager.save_environment_snapshot(snapshot_file)
        
        assert snapshot_file.exists()
        
        # Load snapshot and verify
        loaded_manager = ReproducibilityEnvironmentManager.load_environment_snapshot(snapshot_file)
        
        assert loaded_manager.config.global_seed == test_config.global_seed
        assert loaded_manager.config.torch_deterministic == test_config.torch_deterministic
        assert loaded_manager.applied_settings is not None
        assert loaded_manager.environment_snapshot is not None
        
        # Check snapshot file content
        with open(snapshot_file, 'r') as f:
            snapshot_data = json.load(f)
        
        assert "config" in snapshot_data
        assert "applied_settings" in snapshot_data
        assert "environment_snapshot" in snapshot_data
        assert "checklist" in snapshot_data
    
    def test_nested_value_retrieval(self, test_config):
        """Test nested value retrieval from dictionary."""
        manager = ReproducibilityEnvironmentManager(test_config)
        
        test_data = {
            "level1": {
                "level2": {
                    "level3": "target_value"
                },
                "simple": "simple_value"
            },
            "root": "root_value"
        }
        
        # Test successful retrieval
        assert manager._get_nested_value(test_data, "level1.level2.level3") == "target_value"
        assert manager._get_nested_value(test_data, "level1.simple") == "simple_value"
        assert manager._get_nested_value(test_data, "root") == "root_value"
        
        # Test missing keys
        assert manager._get_nested_value(test_data, "level1.missing") is None
        assert manager._get_nested_value(test_data, "missing.key") is None
        assert manager._get_nested_value(test_data, "level1.level2.level3.missing") is None
    
    def test_full_reproducibility_workflow(self, test_config, temp_dir):
        """Test complete reproducibility workflow."""
        # Create manager and setup environment
        manager = ReproducibilityEnvironmentManager(test_config)
        applied_settings = manager.setup_reproducible_environment()
        
        # Verify settings were applied
        assert len(applied_settings) > 0
        assert manager.environment_snapshot is not None
        
        # Generate checklist
        checklist = manager.generate_reproducibility_checklist()
        
        # Check that most items pass
        passed_items = [item for item in checklist if item["status"] == "pass"]
        assert len(passed_items) >= len(checklist) // 2  # At least half should pass
        
        # Save snapshot
        snapshot_file = temp_dir / "full_workflow_snapshot.json"
        manager.save_environment_snapshot(snapshot_file)
        
        # Load and verify
        loaded_manager = ReproducibilityEnvironmentManager.load_environment_snapshot(snapshot_file)
        
        # Validate consistency
        validation_results = loaded_manager.validate_environment_consistency(
            manager.environment_snapshot, strict_mode=False
        )
        
        assert validation_results["overall_consistent"] is True
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_specific_functionality(self, test_config):
        """Test CUDA-specific functionality when CUDA is available."""
        test_config.require_cuda = True
        test_config.cuda_launch_blocking = True
        
        manager = ReproducibilityEnvironmentManager(test_config)
        
        # Should not raise exception when CUDA is available
        manager._validate_hardware_requirements()
        
        # Setup environment
        applied_settings = manager.setup_reproducible_environment()
        
        # Check CUDA-specific settings
        assert "cuda_seed" in applied_settings
        
        # Check environment snapshot includes GPU info
        snapshot = manager.environment_snapshot
        assert snapshot["torch"]["cuda_available"] is True
        assert snapshot["torch"]["device_count"] > 0
        assert "gpu_info" in snapshot["torch"]
        
        # Verify CUDA environment variables
        assert os.environ.get("CUDA_LAUNCH_BLOCKING") == "1"
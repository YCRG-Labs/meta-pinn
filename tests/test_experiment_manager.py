"""
Tests for experiment management system.
"""

import pytest
import json
import time
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from ml_research_pipeline.utils.experiment_manager import (
    ExperimentManager,
    ExperimentMetadata,
    ExperimentRegistry
)
from ml_research_pipeline.config import ExperimentConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config():
    """Create test experiment configuration."""
    return ExperimentConfig(
        name="test_experiment",
        description="Test experiment for validation",
        version="1.0.0",
        seed=42,
        output_dir="test_output",
        deterministic=True
    )


class TestExperimentMetadata:
    """Test experiment metadata."""
    
    def test_metadata_creation(self):
        """Test metadata creation and serialization."""
        metadata = ExperimentMetadata(
            experiment_id="test_exp_001",
            name="test_experiment",
            description="Test description",
            version="1.0.0",
            timestamp="2024-01-01T00:00:00",
            seed=42,
            python_version="3.8.0",
            torch_version="2.0.0",
            numpy_version="1.24.0",
            cuda_version="11.8",
            hostname="test_host",
            platform="Linux",
            cpu_count=8,
            gpu_count=1,
            gpu_names=["Tesla V100"],
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            config_hash="hash123"
        )
        
        # Test serialization
        metadata_dict = metadata.to_dict()
        assert metadata_dict['experiment_id'] == "test_exp_001"
        assert metadata_dict['name'] == "test_experiment"
        assert metadata_dict['seed'] == 42
        
        # Test deserialization
        restored_metadata = ExperimentMetadata.from_dict(metadata_dict)
        assert restored_metadata.experiment_id == metadata.experiment_id
        assert restored_metadata.name == metadata.name
        assert restored_metadata.seed == metadata.seed


class TestExperimentManager:
    """Test experiment manager."""
    
    def test_experiment_manager_initialization(self, test_config, temp_dir):
        """Test experiment manager initialization."""
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_versioning=True,
            enable_git_tracking=False  # Disable git for testing
        )
        
        # Check basic properties
        assert manager.config == test_config
        assert manager.base_output_dir == temp_dir
        assert manager.experiment_id is not None
        assert manager.experiment_dir.exists()
        
        # Check directory structure
        expected_subdirs = [
            "config", "checkpoints", "logs", "plots", "results",
            "code_snapshot", "data", "models", "metrics"
        ]
        
        for subdir in expected_subdirs:
            assert (manager.experiment_dir / subdir).exists()
        
        # Check metadata
        assert manager.metadata.name == test_config.name
        assert manager.metadata.seed == test_config.seed
        assert manager.metadata.status == "initialized"
    
    def test_experiment_id_generation(self, test_config, temp_dir):
        """Test unique experiment ID generation."""
        manager1 = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        # Wait a bit to ensure different timestamp
        time.sleep(0.1)
        
        manager2 = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        # IDs should be different
        assert manager1.experiment_id != manager2.experiment_id
    
    def test_experiment_lifecycle(self, test_config, temp_dir):
        """Test complete experiment lifecycle."""
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        # Start experiment
        manager.start_experiment()
        assert manager.metadata.status == "running"
        assert manager.metadata.start_time is not None
        
        # Complete experiment
        final_metrics = {"accuracy": 0.95, "loss": 0.05}
        best_metrics = {"accuracy": 0.97, "loss": 0.03}
        
        manager.complete_experiment(final_metrics, best_metrics)
        assert manager.metadata.status == "completed"
        assert manager.metadata.end_time is not None
        assert manager.metadata.duration_seconds is not None
        assert manager.metadata.final_metrics == final_metrics
        assert manager.metadata.best_metrics == best_metrics
    
    def test_experiment_failure(self, test_config, temp_dir):
        """Test experiment failure handling."""
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        manager.start_experiment()
        
        # Fail experiment
        error_message = "Test error occurred"
        manager.fail_experiment(error_message)
        
        assert manager.metadata.status == "failed"
        assert manager.metadata.end_time is not None
        
        # Check error file
        error_file = manager.experiment_dir / "error.txt"
        assert error_file.exists()
        
        with open(error_file, 'r') as f:
            content = f.read()
            assert error_message in content
    
    def test_checkpoint_management(self, test_config, temp_dir):
        """Test checkpoint saving and loading."""
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        # Mock model and optimizer states
        model_state = {"layer1.weight": [1, 2, 3], "layer1.bias": [0.1]}
        optimizer_state = {"state": {}, "param_groups": []}
        metrics = {"loss": 0.5, "accuracy": 0.8}
        
        # Save checkpoint
        manager.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            epoch=10,
            metrics=metrics,
            is_best=True
        )
        
        # Check checkpoint files exist
        checkpoint_dir = manager.experiment_dir / "checkpoints"
        assert (checkpoint_dir / "checkpoint_epoch_10.pth").exists()
        assert (checkpoint_dir / "best_checkpoint.pth").exists()
        assert (checkpoint_dir / "latest_checkpoint.pth").exists()
        
        # Load checkpoint
        loaded_checkpoint = manager.load_checkpoint()
        
        assert loaded_checkpoint['model_state_dict'] == model_state
        assert loaded_checkpoint['optimizer_state_dict'] == optimizer_state
        assert loaded_checkpoint['epoch'] == 10
        assert loaded_checkpoint['metrics'] == metrics
        assert loaded_checkpoint['experiment_id'] == manager.experiment_id
    
    def test_metrics_saving(self, test_config, temp_dir):
        """Test metrics saving."""
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        # Save metrics
        metrics = {"loss": 0.5, "accuracy": 0.8, "f1_score": 0.75}
        manager.save_metrics(metrics, step=100)
        
        # Check metrics file
        metrics_file = manager.experiment_dir / "metrics" / "metrics.jsonl"
        assert metrics_file.exists()
        
        # Read and verify metrics
        with open(metrics_file, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            
            assert data['step'] == 100
            assert data['loss'] == 0.5
            assert data['accuracy'] == 0.8
            assert data['f1_score'] == 0.75
            assert 'timestamp' in data
    
    def test_experiment_summary(self, test_config, temp_dir):
        """Test experiment summary generation."""
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        summary = manager.get_experiment_summary()
        
        assert 'metadata' in summary
        assert 'config' in summary
        assert 'experiment_dir' in summary
        assert 'files' in summary
        
        # Check file paths
        files = summary['files']
        assert 'config' in files
        assert 'metadata' in files
        assert 'logs' in files
        assert 'checkpoints' in files
        assert 'results' in files
    
    def test_experiment_loading(self, test_config, temp_dir):
        """Test loading existing experiment."""
        # Create and save experiment
        original_manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        original_manager.start_experiment()
        original_manager.complete_experiment({"accuracy": 0.9})
        
        # Load experiment
        loaded_manager = ExperimentManager.load_experiment(original_manager.experiment_dir)
        
        assert loaded_manager.experiment_id == original_manager.experiment_id
        assert loaded_manager.config.name == original_manager.config.name
        assert loaded_manager.metadata.status == "completed"
        assert loaded_manager.metadata.final_metrics == {"accuracy": 0.9}
    
    @patch('subprocess.check_output')
    def test_git_info_collection(self, mock_subprocess, test_config, temp_dir):
        """Test git information collection."""
        # Mock git commands
        mock_subprocess.side_effect = [
            b'abc123def456\n',  # git rev-parse HEAD
            b'main\n',          # git rev-parse --abbrev-ref HEAD
            b'M file.py\n'      # git status --porcelain
        ]
        
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=True
        )
        
        assert manager.metadata.git_commit == 'abc123def456'
        assert manager.metadata.git_branch == 'main'
        assert manager.metadata.git_dirty is True
    
    def test_code_snapshot_creation(self, test_config, temp_dir):
        """Test code snapshot creation."""
        # Create some Python files in a temporary project structure
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        
        # Create setup.py to mark as project root
        (project_dir / "setup.py").write_text("# Setup file")
        
        # Create some Python files
        (project_dir / "main.py").write_text("print('Hello')")
        
        src_dir = project_dir / "src"
        src_dir.mkdir()
        (src_dir / "module.py").write_text("def func(): pass")
        
        # Change to project directory
        import os
        original_cwd = os.getcwd()
        os.chdir(project_dir)
        
        try:
            manager = ExperimentManager(
                config=test_config,
                base_output_dir=temp_dir / "experiments",
                enable_versioning=True,
                enable_git_tracking=False
            )
            
            # Check code snapshot
            code_snapshot_dir = manager.experiment_dir / "code_snapshot"
            assert code_snapshot_dir.exists()
            assert (code_snapshot_dir / "setup.py").exists()
            assert (code_snapshot_dir / "main.py").exists()
            assert (code_snapshot_dir / "src" / "module.py").exists()
            
        finally:
            os.chdir(original_cwd)


class TestExperimentRegistry:
    """Test experiment registry."""
    
    def test_registry_initialization(self, temp_dir):
        """Test registry initialization."""
        registry = ExperimentRegistry(registry_dir=temp_dir)
        
        assert registry.registry_dir == temp_dir
        assert registry.experiments == {}
    
    def test_experiment_registration(self, test_config, temp_dir):
        """Test experiment registration."""
        registry = ExperimentRegistry(registry_dir=temp_dir)
        
        # Create experiment manager
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        # Register experiment
        registry.register_experiment(manager)
        
        assert manager.experiment_id in registry.experiments
        
        experiment_info = registry.experiments[manager.experiment_id]
        assert experiment_info['name'] == test_config.name
        assert experiment_info['description'] == test_config.description
        assert experiment_info['status'] == "initialized"
        
        # Check registry file
        registry_file = temp_dir / "experiment_registry.json"
        assert registry_file.exists()
    
    def test_experiment_listing(self, test_config, temp_dir):
        """Test experiment listing with filtering."""
        registry = ExperimentRegistry(registry_dir=temp_dir)
        
        # Create multiple experiments
        configs = [
            test_config,
            test_config.update(name="experiment_2"),
            test_config.update(name="different_experiment")
        ]
        
        managers = []
        for config in configs:
            manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / "experiments",
                enable_git_tracking=False
            )
            managers.append(manager)
            registry.register_experiment(manager)
        
        # Test listing all experiments
        all_experiments = registry.list_experiments()
        assert len(all_experiments) == 3
        
        # Test filtering by name pattern
        filtered_experiments = registry.list_experiments(name_pattern="experiment")
        assert len(filtered_experiments) == 3  # All contain "experiment"
        
        filtered_experiments = registry.list_experiments(name_pattern="different")
        assert len(filtered_experiments) == 1
        
        # Test filtering by status
        managers[0].start_experiment()
        managers[0].complete_experiment()
        registry.register_experiment(managers[0])  # Update status
        
        completed_experiments = registry.list_experiments(status="completed")
        assert len(completed_experiments) == 1
    
    def test_registry_persistence(self, test_config, temp_dir):
        """Test registry persistence across instances."""
        # Create first registry instance
        registry1 = ExperimentRegistry(registry_dir=temp_dir)
        
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        
        registry1.register_experiment(manager)
        
        # Create second registry instance (should load existing data)
        registry2 = ExperimentRegistry(registry_dir=temp_dir)
        
        assert manager.experiment_id in registry2.experiments
        assert len(registry2.experiments) == 1
    
    def test_cleanup_failed_experiments(self, test_config, temp_dir):
        """Test cleanup of failed experiments."""
        registry = ExperimentRegistry(registry_dir=temp_dir)
        
        # Create experiments with different statuses
        manager1 = ExperimentManager(
            config=test_config.update(name="success_exp"),
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        manager1.start_experiment()
        manager1.complete_experiment()
        
        manager2 = ExperimentManager(
            config=test_config.update(name="failed_exp"),
            base_output_dir=temp_dir / "experiments",
            enable_git_tracking=False
        )
        manager2.start_experiment()
        manager2.fail_experiment("Test failure")
        
        # Register both
        registry.register_experiment(manager1)
        registry.register_experiment(manager2)
        
        assert len(registry.experiments) == 2
        
        # Cleanup failed experiments
        registry.cleanup_failed_experiments()
        
        assert len(registry.experiments) == 1
        assert manager1.experiment_id in registry.experiments
        assert manager2.experiment_id not in registry.experiments


class TestExperimentManagerIntegration:
    """Integration tests for experiment manager."""
    
    def test_full_experiment_workflow(self, test_config, temp_dir):
        """Test complete experiment workflow."""
        # Initialize manager
        manager = ExperimentManager(
            config=test_config,
            base_output_dir=temp_dir,
            enable_git_tracking=False
        )
        
        # Start experiment
        manager.start_experiment()
        
        # Simulate training loop
        for epoch in range(3):
            # Save some metrics
            metrics = {
                "loss": 1.0 - epoch * 0.3,
                "accuracy": 0.5 + epoch * 0.2
            }
            manager.save_metrics(metrics, step=epoch * 100)
            
            # Save checkpoint
            model_state = {"epoch": epoch, "weights": [1, 2, 3]}
            optimizer_state = {"lr": 0.01}
            
            manager.save_checkpoint(
                model_state=model_state,
                optimizer_state=optimizer_state,
                epoch=epoch,
                metrics=metrics,
                is_best=(epoch == 2)  # Last epoch is best
            )
        
        # Complete experiment
        final_metrics = {"final_loss": 0.1, "final_accuracy": 0.95}
        manager.complete_experiment(final_metrics)
        
        # Verify all components
        assert manager.metadata.status == "completed"
        assert manager.metadata.final_metrics == final_metrics
        
        # Check files exist
        assert (manager.experiment_dir / "config" / "experiment_config.yaml").exists()
        assert (manager.experiment_dir / "metadata.json").exists()
        assert (manager.experiment_dir / "metrics" / "metrics.jsonl").exists()
        assert (manager.experiment_dir / "checkpoints" / "best_checkpoint.pth").exists()
        
        # Test loading
        loaded_manager = ExperimentManager.load_experiment(manager.experiment_dir)
        assert loaded_manager.experiment_id == manager.experiment_id
        assert loaded_manager.metadata.status == "completed"
    
    def test_experiment_registry_integration(self, test_config, temp_dir):
        """Test integration between manager and registry."""
        registry = ExperimentRegistry(registry_dir=temp_dir)
        
        # Create multiple experiments
        managers = []
        for i in range(3):
            config = test_config.update(name=f"experiment_{i}")
            manager = ExperimentManager(
                config=config,
                base_output_dir=temp_dir / "experiments",
                enable_git_tracking=False
            )
            managers.append(manager)
            registry.register_experiment(manager)
        
        # Run experiments with different outcomes
        managers[0].start_experiment()
        managers[0].complete_experiment({"accuracy": 0.9})
        
        managers[1].start_experiment()
        managers[1].fail_experiment("Test failure")
        
        # managers[2] remains initialized
        
        # Update registry
        for manager in managers:
            registry.register_experiment(manager)
        
        # Test registry queries
        all_experiments = registry.list_experiments()
        assert len(all_experiments) == 3
        
        completed_experiments = registry.list_experiments(status="completed")
        assert len(completed_experiments) == 1
        
        failed_experiments = registry.list_experiments(status="failed")
        assert len(failed_experiments) == 1
        
        initialized_experiments = registry.list_experiments(status="initialized")
        assert len(initialized_experiments) == 1
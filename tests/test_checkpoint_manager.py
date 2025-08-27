"""
Integration tests for checkpoint management system.

Tests checkpoint saving/loading, training interruption/resuming,
and automatic checkpoint scheduling and cleanup.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import os
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

from ml_research_pipeline.core.checkpoint_manager import (
    CheckpointManager, AutoCheckpointer, create_checkpoint_manager
)
from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.config.model_config import MetaPINNConfig


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestCheckpointManager:
    """Test cases for CheckpointManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model(self):
        """Create simple test model."""
        return SimpleModel()
    
    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer for test model."""
        return torch.optim.Adam(model.parameters(), lr=0.001)
    
    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager."""
        return CheckpointManager(
            checkpoint_dir=temp_dir,
            max_checkpoints=3,
            save_frequency=10,
            save_best=True
        )
    
    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(temp_dir)
        
        assert manager.checkpoint_dir == Path(temp_dir)
        assert manager.checkpoint_dir.exists()
        assert manager.max_checkpoints == 5  # default
        assert manager.save_frequency == 100  # default
        assert manager.save_best is True
        assert manager.metric_name == "val_loss"
        assert manager.metric_mode == "min"
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_save_checkpoint_basic(self, mock_is_main, checkpoint_manager, model, optimizer):
        """Test basic checkpoint saving."""
        metrics = {'train_loss': 0.5, 'val_loss': 0.3}
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics=metrics
        )
        
        assert checkpoint_path != ""
        assert Path(checkpoint_path).exists()
        
        # Check checkpoint contents
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint['epoch'] == 1
        assert checkpoint['step'] == 100
        assert checkpoint['metrics'] == metrics
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=False)
    def test_save_checkpoint_non_main_process(self, mock_is_main, checkpoint_manager, model, optimizer):
        """Test checkpoint saving on non-main process (should be no-op)."""
        metrics = {'train_loss': 0.5}
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics=metrics
        )
        
        # Should return empty string and not create files
        assert checkpoint_path == ""
        checkpoints = list(checkpoint_manager.checkpoint_dir.glob("*.pth"))
        assert len(checkpoints) == 0
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_save_best_model(self, mock_is_main, checkpoint_manager, model, optimizer):
        """Test saving best model based on metrics."""
        # Save first checkpoint
        metrics1 = {'val_loss': 0.5}
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics=metrics1
        )
        
        # Save better checkpoint
        metrics2 = {'val_loss': 0.3}
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=2,
            step=200,
            metrics=metrics2
        )
        
        # Check best model was saved
        best_path = checkpoint_manager.checkpoint_dir / "best_model.pth"
        assert best_path.exists()
        
        best_checkpoint = torch.load(best_path)
        assert best_checkpoint['epoch'] == 2
        assert best_checkpoint['metrics']['val_loss'] == 0.3
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_checkpoint_cleanup(self, mock_is_main, checkpoint_manager, model, optimizer):
        """Test automatic cleanup of old checkpoints."""
        # Save more checkpoints than max_checkpoints
        for i in range(5):
            metrics = {'val_loss': 0.5 - i * 0.1}
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=i + 1,
                step=(i + 1) * 100,
                metrics=metrics
            )
            time.sleep(0.01)  # Ensure different timestamps
        
        # Should only keep max_checkpoints (3)
        checkpoints = list(checkpoint_manager.checkpoint_dir.glob("checkpoint_*.pth"))
        assert len(checkpoints) == 3
    
    def test_load_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test checkpoint loading."""
        # Create a checkpoint first
        original_state = model.state_dict()
        metrics = {'train_loss': 0.5, 'val_loss': 0.3}
        
        with patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True):
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                step=500,
                metrics=metrics,
                extra_state={'custom_data': 'test'}
            )
        
        # Modify model state
        for param in model.parameters():
            param.data.fill_(0.0)
        
        # Load checkpoint
        loaded_info = checkpoint_manager.load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path
        )
        
        # Verify loaded information
        assert loaded_info['epoch'] == 5
        assert loaded_info['step'] == 500
        assert loaded_info['metrics'] == metrics
        assert loaded_info['extra_state']['custom_data'] == 'test'
        
        # Verify model state was restored
        loaded_state = model.state_dict()
        for key in original_state:
            torch.testing.assert_close(loaded_state[key], original_state[key])
    
    def test_load_best_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test loading best checkpoint."""
        with patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True):
            # Save multiple checkpoints
            for i in range(3):
                metrics = {'val_loss': 0.5 - i * 0.1}
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=i + 1,
                    step=(i + 1) * 100,
                    metrics=metrics
                )
        
        # Load best checkpoint
        loaded_info = checkpoint_manager.load_checkpoint(
            model=model,
            optimizer=optimizer,
            load_best=True
        )
        
        # Should load the checkpoint with lowest val_loss (epoch 3)
        assert loaded_info['epoch'] == 3
        assert loaded_info['metrics']['val_loss'] == 0.2
    
    def test_get_latest_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test getting latest checkpoint."""
        # No checkpoints initially
        assert checkpoint_manager.get_latest_checkpoint() is None
        
        with patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True):
            # Save checkpoints
            paths = []
            for i in range(3):
                path = checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=i + 1,
                    step=(i + 1) * 100,
                    metrics={'val_loss': 0.5}
                )
                paths.append(path)
                time.sleep(0.01)  # Ensure different timestamps
        
        # Get latest should return the last saved
        latest = checkpoint_manager.get_latest_checkpoint()
        assert str(latest) == paths[-1]
    
    def test_list_checkpoints(self, checkpoint_manager, model, optimizer):
        """Test listing all checkpoints."""
        with patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True):
            # Save multiple checkpoints
            for i in range(3):
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=i + 1,
                    step=(i + 1) * 100,
                    metrics={'val_loss': 0.5 - i * 0.1}
                )
        
        checkpoints = checkpoint_manager.list_checkpoints()
        
        assert len(checkpoints) == 3
        
        # Check sorting (should be by epoch/step)
        for i, checkpoint in enumerate(checkpoints):
            assert checkpoint['epoch'] == i + 1
            assert checkpoint['step'] == (i + 1) * 100
            assert 'path' in checkpoint
            assert 'timestamp' in checkpoint
            assert 'size_mb' in checkpoint
    
    def test_should_save_checkpoint(self, checkpoint_manager):
        """Test checkpoint save frequency logic."""
        # save_frequency = 10 in fixture
        assert checkpoint_manager.should_save_checkpoint(10) is True
        assert checkpoint_manager.should_save_checkpoint(20) is True
        assert checkpoint_manager.should_save_checkpoint(15) is False
        assert checkpoint_manager.should_save_checkpoint(7) is False
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_cleanup_all_checkpoints(self, mock_is_main, checkpoint_manager, model, optimizer):
        """Test cleaning up all checkpoints."""
        # Save some checkpoints
        for i in range(3):
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=i + 1,
                step=(i + 1) * 100,
                metrics={'val_loss': 0.5}
            )
        
        # Verify checkpoints exist
        checkpoints = list(checkpoint_manager.checkpoint_dir.glob("*.pth"))
        assert len(checkpoints) > 0
        
        # Cleanup all
        checkpoint_manager.cleanup_all_checkpoints()
        
        # Verify all removed
        checkpoints = list(checkpoint_manager.checkpoint_dir.glob("*.pth"))
        assert len(checkpoints) == 0
    
    def test_load_nonexistent_checkpoint(self, checkpoint_manager, model):
        """Test loading non-existent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(
                model=model,
                checkpoint_path="nonexistent.pth"
            )


class TestAutoCheckpointer:
    """Test cases for AutoCheckpointer class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager."""
        return CheckpointManager(temp_dir, max_checkpoints=5)
    
    @pytest.fixture
    def auto_checkpointer(self, checkpoint_manager):
        """Create auto checkpointer."""
        return AutoCheckpointer(
            checkpoint_manager=checkpoint_manager,
            save_every_n_steps=100,
            save_every_n_epochs=5,
            save_on_interrupt=False  # Disable for testing
        )
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleModel()
    
    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer."""
        return torch.optim.Adam(model.parameters())
    
    def test_initialization(self, checkpoint_manager):
        """Test auto checkpointer initialization."""
        auto_checkpointer = AutoCheckpointer(
            checkpoint_manager=checkpoint_manager,
            save_every_n_steps=50,
            save_every_n_epochs=10
        )
        
        assert auto_checkpointer.checkpoint_manager is checkpoint_manager
        assert auto_checkpointer.save_every_n_steps == 50
        assert auto_checkpointer.save_every_n_epochs == 10
        assert auto_checkpointer.last_save_step == 0
        assert auto_checkpointer.last_save_epoch == 0
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_step_based_saving(self, mock_is_main, auto_checkpointer, model, optimizer):
        """Test automatic saving based on step frequency."""
        metrics = {'val_loss': 0.5}
        
        # Should not save at step 50
        saved = auto_checkpointer.step(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=50,
            metrics=metrics
        )
        assert saved is False
        
        # Should save at step 100
        saved = auto_checkpointer.step(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics=metrics
        )
        assert saved is True
        
        # Should save at step 200
        saved = auto_checkpointer.step(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=200,
            metrics=metrics
        )
        assert saved is True
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_epoch_based_saving(self, mock_is_main, auto_checkpointer, model, optimizer):
        """Test automatic saving based on epoch frequency."""
        metrics = {'val_loss': 0.5}
        
        # Should not save at epoch 3
        saved = auto_checkpointer.step(
            model=model,
            optimizer=optimizer,
            epoch=3,
            step=50,
            metrics=metrics
        )
        assert saved is False
        
        # Should save at epoch 5
        saved = auto_checkpointer.step(
            model=model,
            optimizer=optimizer,
            epoch=5,
            step=60,
            metrics=metrics
        )
        assert saved is True
        
        # Should save at epoch 10
        saved = auto_checkpointer.step(
            model=model,
            optimizer=optimizer,
            epoch=10,
            step=70,
            metrics=metrics
        )
        assert saved is True


class TestCheckpointManagerIntegration:
    """Integration tests for checkpoint manager with MetaPINN."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self):
        """Create MetaPINN config."""
        return MetaPINNConfig(
            input_dim=2,
            hidden_layers=[32, 32],
            output_dim=1,
            meta_lr=0.001,
            adapt_lr=0.01
        )
    
    @pytest.fixture
    def meta_pinn(self, config):
        """Create MetaPINN model."""
        return MetaPINN(config)
    
    @pytest.fixture
    def optimizer(self, meta_pinn):
        """Create optimizer for MetaPINN."""
        return torch.optim.Adam(meta_pinn.parameters())
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_meta_pinn_checkpoint_save_load(self, mock_is_main, temp_dir, meta_pinn, optimizer, config):
        """Test saving and loading MetaPINN checkpoints."""
        checkpoint_manager = CheckpointManager(temp_dir)
        
        # Save checkpoint
        metrics = {'meta_loss': 0.5, 'adaptation_steps': 5}
        extra_state = {'meta_epoch': 10, 'task_count': 100}
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=meta_pinn,
            optimizer=optimizer,
            epoch=10,
            step=1000,
            metrics=metrics,
            extra_state=extra_state
        )
        
        # Modify model state
        for param in meta_pinn.parameters():
            param.data.fill_(0.0)
        
        # Load checkpoint
        loaded_info = checkpoint_manager.load_checkpoint(
            model=meta_pinn,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path
        )
        
        # Verify loaded information
        assert loaded_info['epoch'] == 10
        assert loaded_info['step'] == 1000
        assert loaded_info['metrics'] == metrics
        assert loaded_info['extra_state'] == extra_state
        
        # Verify model config was saved
        checkpoint = torch.load(checkpoint_path)
        assert 'model_config' in checkpoint
    
    def test_create_checkpoint_manager_factory(self, temp_dir, config):
        """Test checkpoint manager factory function."""
        manager = create_checkpoint_manager(
            checkpoint_dir=temp_dir,
            config=config,
            max_checkpoints=10
        )
        
        assert isinstance(manager, CheckpointManager)
        assert manager.checkpoint_dir == Path(temp_dir)
        assert manager.max_checkpoints == 10
        
        # Should adjust save frequency based on config
        expected_frequency = max(1, config.meta_epochs // 20)
        assert manager.save_frequency == expected_frequency


class TestCheckpointManagerErrorHandling:
    """Test error handling in checkpoint manager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_corrupted_checkpoint_loading(self, temp_dir):
        """Test handling of corrupted checkpoint files."""
        checkpoint_manager = CheckpointManager(temp_dir)
        model = SimpleModel()
        
        # Create corrupted checkpoint file
        corrupted_path = Path(temp_dir) / "corrupted.pth"
        with open(corrupted_path, 'w') as f:
            f.write("corrupted data")
        
        # Should raise error when loading
        with pytest.raises(Exception):
            checkpoint_manager.load_checkpoint(
                model=model,
                checkpoint_path=corrupted_path
            )
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_save_checkpoint_permission_error(self, mock_is_main, temp_dir):
        """Test handling of permission errors during save."""
        checkpoint_manager = CheckpointManager(temp_dir)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Make directory read-only (on Unix systems)
        if os.name != 'nt':  # Skip on Windows
            os.chmod(temp_dir, 0o444)
            
            with pytest.raises(Exception):
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=1,
                    step=100,
                    metrics={'loss': 0.5}
                )
            
            # Restore permissions for cleanup
            os.chmod(temp_dir, 0o755)


if __name__ == '__main__':
    pytest.main([__file__])
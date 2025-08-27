"""
Unit tests for distributed training functionality.

Tests gradient synchronization, task distribution, and multi-GPU coordination
for meta-learning physics-informed neural networks.
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from collections import OrderedDict

from ml_research_pipeline.core.distributed_meta_pinn import (
    DistributedMetaPINN, DistributedTrainingManager, create_distributed_meta_pinn
)
from ml_research_pipeline.config.model_config import MetaPINNConfig
from ml_research_pipeline.utils.distributed_utils import (
    setup_distributed, cleanup_distributed, get_world_size, get_rank
)


class TestDistributedMetaPINN:
    """Test cases for DistributedMetaPINN class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return MetaPINNConfig(
            input_dim=2,
            hidden_layers=[64, 64, 64],
            output_dim=3,
            meta_lr=0.001,
            adapt_lr=0.01,
            adaptation_steps=5,
            physics_loss_weight=1.0
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        batch_size = 32
        return {
            'config': {
                'task_type': 'linear_viscosity',
                'reynolds': 100.0,
                'viscosity_params': {'slope': 0.1, 'intercept': 0.01}
            },
            'support_set': {
                'coords': torch.randn(batch_size, 2),
                'data': torch.randn(batch_size, 3)
            },
            'query_set': {
                'coords': torch.randn(batch_size, 2),
                'data': torch.randn(batch_size, 3)
            }
        }
    
    def test_single_process_initialization(self, config):
        """Test initialization in single-process mode."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            assert not model.is_distributed
            assert isinstance(model.model, type(model.meta_pinn))
            assert model.world_size == 1
            assert model.rank == 0
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=2)
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0)
    def test_distributed_initialization(self, mock_rank, mock_world_size, mock_device_count, mock_cuda, config):
        """Test initialization in distributed mode."""
        with patch('torch.nn.parallel.DistributedDataParallel') as mock_ddp:
            mock_ddp.return_value = Mock()
            
            model = DistributedMetaPINN(config)
            
            assert model.is_distributed
            assert model.world_size == 2
            assert model.rank == 0
            mock_ddp.assert_called_once()
    
    def test_module_property(self, config):
        """Test module property access."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            # In single process mode, module should return the model directly
            assert model.module is model.model
    
    def test_forward_pass(self, config):
        """Test forward pass through distributed model."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            x = torch.randn(10, 2)
            output = model(x)
            
            assert output.shape == (10, 3)
            assert output.dtype == torch.float32
    
    def test_task_distribution_single_process(self, config):
        """Test task distribution in single process mode."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            tasks = [{'id': i} for i in range(5)]
            local_tasks = model._distribute_tasks(tasks)
            
            # In single process, should get all tasks
            assert len(local_tasks) == 5
            assert local_tasks == tasks
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=3)
    @patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=1)
    def test_task_distribution_multi_process(self, mock_rank, mock_world_size, config):
        """Test task distribution across multiple processes."""
        model = DistributedMetaPINN(config)
        
        tasks = [{'id': i} for i in range(10)]
        local_tasks = model._distribute_tasks(tasks)
        
        # Process 1 should get tasks 1, 4, 7
        expected_ids = [1, 4, 7]
        actual_ids = [task['id'] for task in local_tasks]
        assert actual_ids == expected_ids
    
    def test_gradient_synchronization_single_process(self, config):
        """Test gradient synchronization in single process mode."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            # Create dummy gradients
            for param in model.module.parameters():
                param.grad = torch.randn_like(param)
            
            original_grads = [param.grad.clone() for param in model.module.parameters()]
            
            # Synchronization should be no-op in single process
            model._synchronize_meta_gradients()
            
            for orig_grad, param in zip(original_grads, model.module.parameters()):
                torch.testing.assert_close(param.grad, orig_grad)
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=2)
    @patch('ml_research_pipeline.utils.distributed_utils.all_reduce')
    def test_gradient_synchronization_multi_process(self, mock_all_reduce, mock_world_size, config):
        """Test gradient synchronization across multiple processes."""
        model = DistributedMetaPINN(config)
        
        # Create dummy gradients
        for param in model.module.parameters():
            param.grad = torch.randn_like(param)
        
        model._synchronize_meta_gradients()
        
        # Should call all_reduce for each parameter with gradients
        num_params_with_grad = sum(1 for p in model.module.parameters() if p.grad is not None)
        assert mock_all_reduce.call_count == num_params_with_grad
    
    def test_metrics_aggregation_single_process(self, config):
        """Test metrics aggregation in single process mode."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            local_metrics = {'meta_loss': 0.5, 'num_tasks': 4}
            aggregated = model._aggregate_metrics(local_metrics)
            
            assert aggregated == local_metrics
    
    @patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=2)
    @patch('ml_research_pipeline.utils.distributed_utils.all_reduce')
    def test_metrics_aggregation_multi_process(self, mock_all_reduce, mock_world_size, config):
        """Test metrics aggregation across multiple processes."""
        model = DistributedMetaPINN(config)
        
        # Mock all_reduce to simulate summing across processes
        def mock_reduce(tensor, op):
            tensor *= 2  # Simulate sum from 2 processes
            return tensor
        
        mock_all_reduce.side_effect = mock_reduce
        
        local_metrics = {'meta_loss': 0.5, 'num_tasks': 2}
        aggregated = model._aggregate_metrics(local_metrics)
        
        # Should average the loss and sum the tasks
        assert aggregated['meta_loss'] == 0.5  # (0.5 * 2) / (2 * 2)
        assert aggregated['num_tasks'] == 4    # 2 * 2
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_checkpoint_saving_main_process(self, mock_is_main, config):
        """Test checkpoint saving on main process."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                filepath = tmp.name
            
            try:
                model.save_checkpoint(filepath, epoch=10, optimizer_state={})
                
                # Check that file was created
                assert os.path.exists(filepath)
                
                # Load and verify contents
                checkpoint = torch.load(filepath)
                assert 'model_state_dict' in checkpoint
                assert 'config' in checkpoint
                assert checkpoint['epoch'] == 10
                
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=False)
    def test_checkpoint_saving_non_main_process(self, mock_is_main, config):
        """Test checkpoint saving on non-main process (should be no-op)."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=2):
            model = DistributedMetaPINN(config)
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                filepath = tmp.name
            
            # Remove the file to test that it's not created
            os.unlink(filepath)
            
            model.save_checkpoint(filepath, epoch=10)
            
            # File should not be created on non-main process
            assert not os.path.exists(filepath)
    
    def test_checkpoint_loading(self, config):
        """Test checkpoint loading."""
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            # Save a checkpoint first
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                filepath = tmp.name
            
            try:
                original_state = model.module.state_dict()
                model.save_checkpoint(filepath, epoch=5)
                
                # Modify model state
                for param in model.module.parameters():
                    param.data.fill_(0.0)
                
                # Load checkpoint
                checkpoint = model.load_checkpoint(filepath)
                
                # Verify state was restored
                loaded_state = model.module.state_dict()
                for key in original_state:
                    torch.testing.assert_close(loaded_state[key], original_state[key])
                
                assert checkpoint['epoch'] == 5
                
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)


class TestDistributedTrainingManager:
    """Test cases for DistributedTrainingManager class."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = DistributedTrainingManager(
            backend="gloo",
            init_method="tcp://localhost:12345",
            timeout_minutes=15
        )
        
        assert manager.backend == "gloo"
        assert manager.init_method == "tcp://localhost:12345"
        assert not manager.is_initialized
    
    @patch('ml_research_pipeline.utils.distributed_utils.setup_distributed', return_value=True)
    def test_setup_success(self, mock_setup):
        """Test successful setup."""
        manager = DistributedTrainingManager()
        
        result = manager.setup()
        
        assert result is True
        assert manager.is_initialized is True
        mock_setup.assert_called_once()
    
    @patch('ml_research_pipeline.utils.distributed_utils.setup_distributed', return_value=False)
    def test_setup_failure(self, mock_setup):
        """Test setup failure."""
        manager = DistributedTrainingManager()
        
        result = manager.setup()
        
        assert result is False
        assert manager.is_initialized is False
    
    @patch('ml_research_pipeline.utils.distributed_utils.setup_distributed', side_effect=Exception("Setup failed"))
    def test_setup_exception(self, mock_setup):
        """Test setup with exception."""
        manager = DistributedTrainingManager()
        
        result = manager.setup()
        
        assert result is False
        assert manager.is_initialized is False
    
    @patch('ml_research_pipeline.utils.distributed_utils.cleanup_distributed')
    def test_cleanup(self, mock_cleanup):
        """Test cleanup."""
        manager = DistributedTrainingManager()
        manager.is_initialized = True
        
        manager.cleanup()
        
        mock_cleanup.assert_called_once()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with patch.object(DistributedTrainingManager, 'setup', return_value=True) as mock_setup, \
             patch.object(DistributedTrainingManager, 'cleanup') as mock_cleanup:
            
            with DistributedTrainingManager() as manager:
                assert manager is not None
            
            mock_setup.assert_called_once()
            mock_cleanup.assert_called_once()
    
    @patch('ml_research_pipeline.utils.distributed_utils.barrier')
    def test_barrier(self, mock_barrier):
        """Test barrier synchronization."""
        manager = DistributedTrainingManager()
        manager.is_initialized = True
        
        manager.barrier()
        
        mock_barrier.assert_called_once()
    
    @patch('ml_research_pipeline.utils.distributed_utils.is_main_process', return_value=True)
    def test_is_main_process(self, mock_is_main):
        """Test main process check."""
        manager = DistributedTrainingManager()
        
        result = manager.is_main_process()
        
        assert result is True
        mock_is_main.assert_called_once()


class TestDistributedUtilsIntegration:
    """Integration tests for distributed utilities."""
    
    def test_create_distributed_meta_pinn(self):
        """Test factory function for creating distributed MetaPINN."""
        config = MetaPINNConfig(
            input_dim=2,
            hidden_layers=[32, 32],
            output_dim=3
        )
        
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = create_distributed_meta_pinn(config)
            
            assert isinstance(model, DistributedMetaPINN)
            assert model.config == config
    
    def test_gradient_synchronization_consistency(self):
        """Test that gradient synchronization maintains consistency."""
        config = MetaPINNConfig(
            input_dim=2,
            hidden_layers=[16, 16],
            output_dim=1
        )
        
        with patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=1):
            model = DistributedMetaPINN(config)
            
            # Create identical gradients
            target_grad = torch.tensor(0.5)
            for param in model.module.parameters():
                param.grad = torch.full_like(param, 0.5)
            
            # Synchronize (should be no-op in single process)
            model._synchronize_meta_gradients()
            
            # Check gradients are unchanged
            for param in model.module.parameters():
                assert torch.allclose(param.grad, torch.full_like(param, 0.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDistributedTrainingGPU:
    """GPU-specific tests for distributed training."""
    
    def test_device_assignment(self):
        """Test proper device assignment in multi-GPU setup."""
        config = MetaPINNConfig(
            input_dim=2,
            hidden_dims=[32],
            output_dim=1
        )
        
        with patch('torch.cuda.device_count', return_value=2), \
             patch('ml_research_pipeline.utils.distributed_utils.get_world_size', return_value=2), \
             patch('ml_research_pipeline.utils.distributed_utils.get_rank', return_value=0), \
             patch('torch.nn.parallel.DistributedDataParallel') as mock_ddp:
            
            mock_ddp.return_value = Mock()
            
            model = DistributedMetaPINN(config, device_ids=[0])
            
            # Should use GPU 0
            assert hasattr(model, 'device')
            assert model.device.type == 'cuda'
            assert model.device.index == 0


if __name__ == '__main__':
    pytest.main([__file__])
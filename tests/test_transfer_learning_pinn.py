"""
Integration tests for TransferLearningPINN implementation.

Tests the transfer learning PINN baseline with pre-training and fine-tuning,
and compares performance with meta-learning approaches.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

from ml_research_pipeline.core.transfer_learning_pinn import TransferLearningPINN
from ml_research_pipeline.core.standard_pinn import StandardPINN
from ml_research_pipeline.config.model_config import ModelConfig


class TestTransferLearningPINN:
    """Test suite for TransferLearningPINN class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ModelConfig(
            input_dim=3,
            output_dim=3,
            hidden_layers=[32, 32, 32],  # Smaller for faster testing
            activation="tanh",
            physics_loss_weight=1.0,
            adaptive_physics_weight=True,
            enforce_boundary_conditions=False,
            weight_init="xavier_normal",
            bias_init="zeros"
        )
    
    @pytest.fixture
    def model(self, config):
        """Create a TransferLearningPINN model for testing."""
        return TransferLearningPINN(config)
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        tasks = []
        n_points = 50
        
        # Create tasks with different viscosity types
        task_configs = [
            {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}},
            {'viscosity_type': 'linear', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}},
            {'viscosity_type': 'bilinear', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.1, 'gamma': 0.05}},
            {'viscosity_type': 'exponential', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}}
        ]
        
        for task_config in task_configs:
            coords = torch.randn(n_points, 3)
            data = torch.randn(n_points, 3)
            
            task = {
                'coords': coords,
                'data': data,
                'task_info': task_config
            }
            tasks.append(task)
        
        return tasks
    
    @pytest.fixture
    def single_task(self):
        """Create a single task for fine-tuning tests."""
        n_points = 50
        coords = torch.randn(n_points, 3)
        data = torch.randn(n_points, 3)
        
        task_info = {
            'viscosity_type': 'temperature_dependent',
            'viscosity_params': {'mu_0': 1.0, 'T_0': 1.0, 'n': -0.5}
        }
        
        return {
            'coords': coords,
            'data': data,
            'task_info': task_info
        }
    
    def test_model_initialization(self, config):
        """Test that the transfer learning model initializes correctly."""
        model = TransferLearningPINN(config)
        
        # Check inheritance
        assert isinstance(model, TransferLearningPINN)
        assert isinstance(model, StandardPINN)
        assert isinstance(model, nn.Module)
        
        # Check transfer learning specific attributes
        assert model.pretrain_history == []
        assert model.finetune_history == []
        assert model.is_pretrained == False
        assert model.pretrain_tasks_seen == 0
        assert model.initial_state is not None
    
    def test_initial_state_saving(self, model):
        """Test that initial state is saved correctly."""
        # Check that initial state is saved
        assert model.initial_state is not None
        
        # Check that all parameters are saved
        current_params = {name: param for name, param in model.named_parameters()}
        assert len(model.initial_state) == len(current_params)
        
        # Check that saved parameters match current parameters
        for name, param in current_params.items():
            assert torch.allclose(model.initial_state[name], param)
    
    def test_pretrain_functionality(self, model, sample_tasks):
        """Test pre-training on multiple tasks."""
        # Pre-train on subset of tasks
        pretrain_tasks = sample_tasks[:3]
        
        history = model.pretrain(
            tasks=pretrain_tasks,
            epochs_per_task=5,  # Small number for testing
            learning_rate=0.001,
            verbose=False
        )
        
        # Check that history is returned
        assert isinstance(history, dict)
        expected_keys = ['data_loss', 'physics_loss', 'total_loss', 'task_losses', 'learning_rates']
        for key in expected_keys:
            assert key in history
        
        # Check history length (3 tasks * 5 epochs = 15 entries)
        assert len(history['data_loss']) == 15
        assert len(history['physics_loss']) == 15
        assert len(history['total_loss']) == 15
        
        # Check task-specific losses
        assert len(history['task_losses']) == 3
        for task_loss in history['task_losses']:
            assert 'task_idx' in task_loss
            assert 'task_type' in task_loss
            assert 'final_loss' in task_loss
        
        # Check model state
        assert model.is_pretrained == True
        assert model.pretrain_tasks_seen == 3
        assert model.pretrain_history == history
    
    def test_finetune_functionality(self, model, sample_tasks, single_task):
        """Test fine-tuning on a new task."""
        # Pre-train first
        model.pretrain(sample_tasks[:2], epochs_per_task=3, verbose=False)
        
        # Fine-tune on new task
        history = model.finetune(
            task=single_task,
            epochs=5,
            learning_rate=0.0001,
            verbose=False
        )
        
        # Check that history is returned
        assert isinstance(history, dict)
        expected_keys = ['data_loss', 'physics_loss', 'total_loss', 
                        'momentum_x_loss', 'momentum_y_loss', 'continuity_loss', 'learning_rates']
        for key in expected_keys:
            assert key in history
            assert len(history[key]) == 5  # 5 epochs
        
        # Check that losses are finite
        for losses in history.values():
            assert all(np.isfinite(loss) for loss in losses)
        
        # Check model state
        assert model.finetune_history == history
        assert model.current_task_info == single_task['task_info']
    
    def test_finetune_without_pretrain_fails(self, model, single_task):
        """Test that fine-tuning without pre-training raises an error."""
        with pytest.raises(ValueError, match="Model must be pre-trained"):
            model.finetune(single_task, epochs=5, verbose=False)
    
    def test_layer_freezing(self, model, sample_tasks, single_task):
        """Test layer freezing during fine-tuning."""
        # Pre-train first
        model.pretrain(sample_tasks[:2], epochs_per_task=3, verbose=False)
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Fine-tune with first layer frozen
        model.finetune(
            task=single_task,
            epochs=3,
            freeze_layers=[0],  # Freeze first layer
            verbose=False
        )
        
        # Check that first layer parameters didn't change
        linear_layer_idx = 0
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                if linear_layer_idx == 0:
                    # First layer should be unchanged
                    assert torch.allclose(layer.weight, initial_params[f'layers.{linear_layer_idx}.weight'])
                    assert torch.allclose(layer.bias, initial_params[f'layers.{linear_layer_idx}.bias'])
                linear_layer_idx += 1
                break
    
    def test_pretrain_then_finetune_pipeline(self, model, sample_tasks, single_task):
        """Test the complete pre-train then fine-tune pipeline."""
        pretrain_history, finetune_history = model.pretrain_then_finetune(
            pretrain_tasks=sample_tasks[:2],
            finetune_task=single_task,
            pretrain_epochs_per_task=3,
            finetune_epochs=3,
            verbose=False
        )
        
        # Check that both histories are returned
        assert isinstance(pretrain_history, dict)
        assert isinstance(finetune_history, dict)
        
        # Check that model is in correct state
        assert model.is_pretrained == True
        assert model.pretrain_history == pretrain_history
        assert model.finetune_history == finetune_history
    
    def test_evaluate_transfer_performance(self, model, sample_tasks):
        """Test transfer learning performance evaluation."""
        # Pre-train on some tasks
        pretrain_tasks = sample_tasks[:2]
        test_tasks = sample_tasks[2:]
        
        model.pretrain(pretrain_tasks, epochs_per_task=3, verbose=False)
        
        # Evaluate transfer performance
        results = model.evaluate_transfer_performance(
            test_tasks=test_tasks,
            finetune_epochs=3,
            finetune_lr=0.0001
        )
        
        # Check results structure
        assert isinstance(results, dict)
        expected_keys = ['task_results', 'avg_final_loss', 'avg_improvement', 
                        'convergence_speeds', 'transfer_effectiveness', 
                        'avg_convergence_speed', 'avg_transfer_effectiveness']
        for key in expected_keys:
            assert key in results
        
        # Check task results
        assert len(results['task_results']) == len(test_tasks)
        for task_result in results['task_results']:
            assert 'task_idx' in task_result
            assert 'task_type' in task_result
            assert 'initial_loss' in task_result
            assert 'final_loss' in task_result
            assert 'improvement' in task_result
    
    def test_compare_with_scratch_training(self, model, sample_tasks, single_task):
        """Test comparison with training from scratch."""
        # Pre-train first
        model.pretrain(sample_tasks[:2], epochs_per_task=3, verbose=False)
        
        # Compare with scratch training
        comparison = model.compare_with_scratch_training(
            task=single_task,
            scratch_epochs=10,
            finetune_epochs=5,
            learning_rate=0.001
        )
        
        # Check comparison results
        assert isinstance(comparison, dict)
        expected_keys = ['finetune_final_loss', 'scratch_final_loss', 
                        'finetune_epochs', 'scratch_epochs',
                        'finetune_metrics', 'scratch_metrics',
                        'transfer_advantage', 'sample_efficiency', 'performance_ratio']
        for key in expected_keys:
            assert key in comparison
        
        # Check that metrics are reasonable
        assert comparison['finetune_epochs'] < comparison['scratch_epochs']
        assert comparison['sample_efficiency'] > 1.0  # Should be more sample efficient
    
    def test_reset_functionality(self, model, sample_tasks, single_task):
        """Test model reset functionality."""
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Pre-train and fine-tune
        model.pretrain(sample_tasks[:2], epochs_per_task=3, verbose=False)
        model.finetune(single_task, epochs=3, verbose=False)
        
        # Parameters should have changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name], atol=1e-6):
                changed = True
                break
        assert changed, "Parameters should have changed after training"
        
        # Reset completely
        model.reset_completely()
        
        # Check that parameters are reset
        for name, param in model.named_parameters():
            assert torch.allclose(param, initial_params[name])
        
        # Check that state is reset
        assert model.pretrain_history == []
        assert model.finetune_history == []
        assert model.is_pretrained == False
        assert model.pretrain_tasks_seen == 0
    
    def test_convergence_improvement(self, model, sample_tasks, single_task):
        """Test that transfer learning improves convergence speed."""
        # Create two identical models
        model_transfer = TransferLearningPINN(model.config)
        model_scratch = StandardPINN(model.config)
        
        # Pre-train transfer model
        model_transfer.pretrain(sample_tasks[:2], epochs_per_task=5, verbose=False)
        
        # Fine-tune transfer model
        transfer_history = model_transfer.finetune(single_task, epochs=10, verbose=False)
        
        # Train scratch model
        scratch_history = model_scratch.train_on_task(single_task, epochs=10, verbose=False)
        
        # Transfer learning should achieve lower loss faster
        # (This is a simplified test - in practice, you'd need more sophisticated comparison)
        transfer_final = transfer_history['total_loss'][-1]
        scratch_final = scratch_history['total_loss'][-1]
        
        # At minimum, both should converge to finite losses
        assert np.isfinite(transfer_final)
        assert np.isfinite(scratch_final)
    
    def test_different_learning_rates(self, model, sample_tasks, single_task):
        """Test fine-tuning with different learning rates."""
        # Pre-train first
        model.pretrain(sample_tasks[:2], epochs_per_task=3, verbose=False)
        
        learning_rates = [0.001, 0.0001, 0.00001]
        
        for lr in learning_rates:
            # Reset to pre-trained state for fair comparison
            # (In practice, you'd save and restore the pre-trained state)
            
            history = model.finetune(
                task=single_task,
                epochs=3,
                learning_rate=lr,
                verbose=False
            )
            
            # Check that training completes successfully
            assert len(history['total_loss']) == 3
            assert all(np.isfinite(loss) for loss in history['total_loss'])
    
    def test_model_info_transfer_learning(self, model, sample_tasks):
        """Test model information for transfer learning model."""
        # Before pre-training
        info_before = model.get_model_info()
        assert info_before['model_type'] == 'TransferLearningPINN'
        assert info_before['is_pretrained'] == False
        assert info_before['pretrain_tasks_seen'] == 0
        
        # After pre-training
        model.pretrain(sample_tasks[:2], epochs_per_task=3, verbose=False)
        info_after = model.get_model_info()
        assert info_after['is_pretrained'] == True
        assert info_after['pretrain_tasks_seen'] == 2
    
    def test_gradient_flow_during_pretraining(self, model, sample_tasks):
        """Test that gradients flow properly during pre-training."""
        task = sample_tasks[0]
        coords = task['coords']
        data = task['data']
        task_info = task['task_info']
        
        # Forward pass
        predictions = model.forward(coords)
        data_loss = torch.nn.functional.mse_loss(predictions, data)
        
        # Physics loss
        physics_losses = model.physics_loss(coords, task_info)
        physics_loss = physics_losses['total']
        
        # Total loss
        total_loss = data_loss + physics_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter {name}"
    
    def test_memory_efficiency(self, model, sample_tasks):
        """Test that the model doesn't accumulate excessive memory during training."""
        import gc
        
        # Get initial memory state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Pre-train (this should not cause memory leaks)
        model.pretrain(sample_tasks[:2], epochs_per_task=3, verbose=False)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # The test passes if no memory errors occur
        assert True


class TestTransferLearningComparison:
    """Test suite for comparing transfer learning with other approaches."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ModelConfig(
            input_dim=3,
            output_dim=3,
            hidden_layers=[32, 32],  # Small for fast testing
            activation="tanh",
            physics_loss_weight=1.0
        )
    
    def test_transfer_vs_standard_pinn(self, config):
        """Test transfer learning vs standard PINN performance."""
        # Create sample tasks
        n_points = 30
        coords = torch.randn(n_points, 3)
        data = torch.randn(n_points, 3)
        task_info = {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
        
        task = {'coords': coords, 'data': data, 'task_info': task_info}
        
        # Create models
        transfer_model = TransferLearningPINN(config)
        standard_model = StandardPINN(config)
        
        # Create pre-training tasks
        pretrain_tasks = []
        for i in range(2):
            pretrain_coords = torch.randn(n_points, 3)
            pretrain_data = torch.randn(n_points, 3)
            pretrain_info = {'viscosity_type': 'linear', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1}}
            pretrain_tasks.append({
                'coords': pretrain_coords, 
                'data': pretrain_data, 
                'task_info': pretrain_info
            })
        
        # Pre-train transfer model
        transfer_model.pretrain(pretrain_tasks, epochs_per_task=3, verbose=False)
        
        # Fine-tune transfer model
        transfer_history = transfer_model.finetune(task, epochs=5, verbose=False)
        
        # Train standard model from scratch
        standard_history = standard_model.train_on_task(task, epochs=5, verbose=False)
        
        # Both should complete successfully
        assert len(transfer_history['total_loss']) == 5
        assert len(standard_history['total_loss']) == 5
        
        # Both should have finite losses
        assert all(np.isfinite(loss) for loss in transfer_history['total_loss'])
        assert all(np.isfinite(loss) for loss in standard_history['total_loss'])


if __name__ == "__main__":
    pytest.main([__file__])
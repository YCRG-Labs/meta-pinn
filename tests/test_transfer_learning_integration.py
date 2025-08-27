"""
Integration tests for TransferLearningPINN comparing with meta-learning approaches.

This test focuses on the core transfer learning functionality and comparison
with meta-learning without getting into complex physics computations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List

from ml_research_pipeline.core.transfer_learning_pinn import TransferLearningPINN
from ml_research_pipeline.core.standard_pinn import StandardPINN
from ml_research_pipeline.config.model_config import ModelConfig


class TestTransferLearningIntegration:
    """Integration tests for transfer learning PINN."""
    
    @pytest.fixture
    def config(self):
        """Create a simple test configuration."""
        return ModelConfig(
            input_dim=3,
            output_dim=3,
            hidden_layers=[16, 16],  # Very small for fast testing
            activation="tanh",
            physics_loss_weight=0.1,  # Reduce physics weight to avoid numerical issues
            adaptive_physics_weight=False,  # Disable adaptive weighting
            enforce_boundary_conditions=False,
            weight_init="xavier_normal",
            bias_init="zeros"
        )
    
    def test_transfer_learning_pipeline(self, config):
        """Test the complete transfer learning pipeline."""
        model = TransferLearningPINN(config)
        
        # Create simple tasks with smaller data
        n_points = 20
        pretrain_tasks = []
        
        for i in range(2):
            coords = torch.randn(n_points, 3) * 0.5  # Smaller coordinates
            data = torch.randn(n_points, 3) * 0.1    # Smaller target values
            task_info = {
                'viscosity_type': 'constant',
                'viscosity_params': {'mu_0': 1.0}
            }
            pretrain_tasks.append({
                'coords': coords,
                'data': data,
                'task_info': task_info
            })
        
        # Pre-train
        pretrain_history = model.pretrain(
            tasks=pretrain_tasks,
            epochs_per_task=3,
            learning_rate=0.001,
            verbose=False
        )
        
        # Check pre-training completed
        assert model.is_pretrained
        assert len(pretrain_history['data_loss']) == 6  # 2 tasks * 3 epochs
        
        # Create fine-tuning task
        finetune_coords = torch.randn(n_points, 3) * 0.5
        finetune_data = torch.randn(n_points, 3) * 0.1
        finetune_task = {
            'coords': finetune_coords,
            'data': finetune_data,
            'task_info': {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
        }
        
        # Fine-tune
        finetune_history = model.finetune(
            task=finetune_task,
            epochs=3,
            learning_rate=0.0001,
            verbose=False
        )
        
        # Check fine-tuning completed
        assert len(finetune_history['data_loss']) == 3
        
        # Evaluate
        metrics = model.evaluate_on_task(finetune_task)
        assert 'data_loss' in metrics
        assert 'parameter_accuracy' in metrics
    
    def test_transfer_vs_scratch_comparison(self, config):
        """Test comparison between transfer learning and training from scratch."""
        # Create models
        transfer_model = TransferLearningPINN(config)
        scratch_model = StandardPINN(config)
        
        # Create simple tasks
        n_points = 15
        
        # Pre-training tasks
        pretrain_tasks = []
        for i in range(2):
            coords = torch.randn(n_points, 3) * 0.3
            data = torch.randn(n_points, 3) * 0.1
            task_info = {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
            pretrain_tasks.append({'coords': coords, 'data': data, 'task_info': task_info})
        
        # Test task
        test_coords = torch.randn(n_points, 3) * 0.3
        test_data = torch.randn(n_points, 3) * 0.1
        test_task = {
            'coords': test_coords,
            'data': test_data,
            'task_info': {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
        }
        
        # Pre-train transfer model
        transfer_model.pretrain(pretrain_tasks, epochs_per_task=2, verbose=False)
        
        # Fine-tune transfer model
        transfer_history = transfer_model.finetune(test_task, epochs=3, verbose=False)
        
        # Train scratch model
        scratch_history = scratch_model.train_on_task(test_task, epochs=3, verbose=False)
        
        # Both should complete successfully
        assert len(transfer_history['data_loss']) == 3
        assert len(scratch_history['data_loss']) == 3
        
        # Both should have reasonable final losses
        transfer_final = transfer_history['data_loss'][-1]
        scratch_final = scratch_history['data_loss'][-1]
        
        assert transfer_final > 0 and transfer_final < 100  # Reasonable range
        assert scratch_final > 0 and scratch_final < 100   # Reasonable range
    
    def test_layer_freezing_functionality(self, config):
        """Test that layer freezing works correctly."""
        model = TransferLearningPINN(config)
        
        # Create simple pre-training task
        n_points = 10
        coords = torch.randn(n_points, 3) * 0.2
        data = torch.randn(n_points, 3) * 0.1
        task_info = {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
        
        pretrain_task = {'coords': coords, 'data': data, 'task_info': task_info}
        finetune_task = {'coords': coords, 'data': data, 'task_info': task_info}
        
        # Pre-train
        model.pretrain([pretrain_task], epochs_per_task=2, verbose=False)
        
        # Store parameters before fine-tuning
        params_before = {}
        for name, param in model.named_parameters():
            params_before[name] = param.clone().detach()
        
        # Fine-tune with first layer frozen
        model.finetune(finetune_task, epochs=2, freeze_layers=[0], verbose=False)
        
        # Check that first layer didn't change much
        first_layer_weight_name = 'layers.0.weight'
        first_layer_bias_name = 'layers.0.bias'
        
        if first_layer_weight_name in params_before:
            weight_before = params_before[first_layer_weight_name]
            weight_after = dict(model.named_parameters())[first_layer_weight_name]
            
            # Should be exactly the same (frozen)
            assert torch.allclose(weight_before, weight_after, atol=1e-6)
    
    def test_model_state_management(self, config):
        """Test model state management (reset, etc.)."""
        model = TransferLearningPINN(config)
        
        # Store initial state
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Create and run a simple task
        n_points = 10
        coords = torch.randn(n_points, 3) * 0.2
        data = torch.randn(n_points, 3) * 0.1
        task_info = {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
        task = {'coords': coords, 'data': data, 'task_info': task_info}
        
        # Pre-train
        model.pretrain([task], epochs_per_task=2, verbose=False)
        assert model.is_pretrained
        
        # Reset completely
        model.reset_completely()
        
        # Check state is reset
        assert not model.is_pretrained
        assert model.pretrain_tasks_seen == 0
        assert len(model.pretrain_history) == 0
        
        # Check parameters are reset
        for name, param in model.named_parameters():
            assert torch.allclose(param, initial_params[name])
    
    def test_evaluation_metrics(self, config):
        """Test evaluation metrics computation."""
        model = TransferLearningPINN(config)
        
        # Create simple task
        n_points = 10
        coords = torch.randn(n_points, 3) * 0.2
        data = torch.randn(n_points, 3) * 0.1
        task_info = {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
        task = {'coords': coords, 'data': data, 'task_info': task_info}
        
        # Pre-train and fine-tune
        model.pretrain([task], epochs_per_task=1, verbose=False)
        model.finetune(task, epochs=1, verbose=False)
        
        # Evaluate
        metrics = model.evaluate_on_task(task)
        
        # Check metrics are present and reasonable
        assert 'data_loss' in metrics
        assert 'physics_residual' in metrics
        assert 'parameter_accuracy' in metrics
        assert 'prediction_mse' in metrics
        assert 'l2_error' in metrics
        
        # All metrics should be non-negative numbers
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert value >= 0
            assert np.isfinite(value)
    
    def test_model_info(self, config):
        """Test model information retrieval."""
        model = TransferLearningPINN(config)
        
        # Before pre-training
        info = model.get_model_info()
        assert info['model_type'] == 'TransferLearningPINN'
        assert not info['is_pretrained']
        assert info['pretrain_tasks_seen'] == 0
        
        # After pre-training
        n_points = 5
        coords = torch.randn(n_points, 3) * 0.1
        data = torch.randn(n_points, 3) * 0.1
        task_info = {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
        task = {'coords': coords, 'data': data, 'task_info': task_info}
        
        model.pretrain([task], epochs_per_task=1, verbose=False)
        
        info = model.get_model_info()
        assert info['is_pretrained']
        assert info['pretrain_tasks_seen'] == 1


if __name__ == "__main__":
    pytest.main([__file__])
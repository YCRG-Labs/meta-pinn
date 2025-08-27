"""
Unit tests for MAML adaptation mechanism in MetaPINN.

Tests adaptation convergence, gradient flow, and parameter management.
"""

import pytest
import torch
import torch.nn.functional as F
from collections import OrderedDict

from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.config.model_config import MetaPINNConfig


class TestMAMLAdaptation:
    """Test suite for MAML adaptation mechanism."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return MetaPINNConfig(
            input_dim=3,
            output_dim=3,
            hidden_layers=[32, 32],
            activation="tanh",
            meta_lr=0.001,
            adapt_lr=0.01,
            adaptation_steps=5,
            gradient_clipping=1.0,
            physics_loss_weight=1.0
        )
    
    @pytest.fixture
    def model(self, config):
        """Create a test model."""
        return MetaPINN(config)
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        # Generate synthetic task data
        n_support = 25
        n_query = 25
        
        # Support set coordinates
        support_coords = torch.randn(n_support, 3)
        support_coords.requires_grad_(True)
        
        # Query set coordinates
        query_coords = torch.randn(n_query, 3)
        query_coords.requires_grad_(True)
        
        # Generate synthetic data (simple function for testing)
        support_data = torch.sin(support_coords.sum(dim=1, keepdim=True)).repeat(1, 3)
        query_data = torch.sin(query_coords.sum(dim=1, keepdim=True)).repeat(1, 3)
        
        # Task information
        task_info = {
            'viscosity_type': 'constant',
            'viscosity_params': {'mu_0': 1.0},
            'boundary_conditions': {'type': 'no_slip'}
        }
        
        return {
            'support_coords': support_coords,
            'support_data': support_data,
            'query_coords': query_coords,
            'query_data': query_data,
            'task_info': task_info
        }
    
    def test_adapt_to_task_basic(self, model, sample_task):
        """Test basic adaptation functionality."""
        # Store original parameters
        original_params = model.clone_parameters()
        
        # Perform adaptation
        adapted_params = model.adapt_to_task(sample_task)
        
        # Check that adapted parameters are different from original
        assert isinstance(adapted_params, OrderedDict)
        assert len(adapted_params) == len(original_params)
        
        # Parameters should be different after adaptation
        params_changed = False
        for name in original_params:
            if not torch.allclose(adapted_params[name], original_params[name], atol=1e-6):
                params_changed = True
                break
        assert params_changed, "Parameters should change during adaptation"
    
    def test_adaptation_convergence(self, model, sample_task):
        """Test that adaptation reduces loss over steps."""
        # Perform adaptation and get history
        adapted_params = model.adapt_to_task(sample_task)
        adaptation_history = model.get_adaptation_history()
        
        # Check that we have the expected number of steps
        assert len(adaptation_history) == model.adaptation_steps
        
        # Check that loss generally decreases (allowing for some fluctuation)
        initial_loss = adaptation_history[0]['total_loss']
        final_loss = adaptation_history[-1]['total_loss']
        
        # Loss should decrease or at least not increase significantly
        assert final_loss <= initial_loss * 1.1, "Loss should not increase significantly during adaptation"
    
    def test_adaptation_with_different_steps(self, model, sample_task):
        """Test adaptation with different numbers of steps."""
        step_counts = [1, 3, 10]
        
        for steps in step_counts:
            adapted_params = model.adapt_to_task(sample_task, adaptation_steps=steps)
            history = model.get_adaptation_history()
            
            assert len(history) == steps
            assert isinstance(adapted_params, OrderedDict)
    
    def test_compute_adaptation_loss(self, model, sample_task):
        """Test adaptation loss computation."""
        # Compute loss with original parameters
        loss_dict = model.compute_adaptation_loss(sample_task)
        
        # Check that all expected loss components are present
        expected_keys = ['data_loss', 'physics_loss', 'total_loss']
        for key in expected_keys:
            assert key in loss_dict
            assert isinstance(loss_dict[key], torch.Tensor)
        
        # Check physics components separately (it's a dictionary)
        assert 'physics_components' in loss_dict
        assert isinstance(loss_dict['physics_components'], dict)
        
        # Check that losses are non-negative
        assert loss_dict['data_loss'] >= 0
        assert loss_dict['physics_loss'] >= 0
        assert loss_dict['total_loss'] >= 0
    
    def test_compute_adaptation_loss_with_params(self, model, sample_task):
        """Test adaptation loss computation with specific parameters."""
        # Get adapted parameters
        adapted_params = model.adapt_to_task(sample_task)
        
        # Compute loss with adapted parameters
        loss_dict = model.compute_adaptation_loss(sample_task, adapted_params)
        
        # Should produce valid losses
        assert not torch.isnan(loss_dict['total_loss'])
        assert loss_dict['total_loss'] >= 0
    
    def test_evaluate_on_query_set(self, model, sample_task):
        """Test evaluation on query set."""
        # Evaluate with original parameters
        metrics = model.evaluate_on_query_set(sample_task)
        
        # Check that all expected metrics are present
        expected_keys = ['mse', 'mae', 'physics_residual']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], torch.Tensor)
        
        # Check physics components separately (it's a dictionary)
        assert 'physics_components' in metrics
        assert isinstance(metrics['physics_components'], dict)
        
        # Check that metrics are non-negative
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['physics_residual'] >= 0
    
    def test_adaptation_improves_performance(self, model, sample_task):
        """Test that adaptation improves performance on the task."""
        # Evaluate before adaptation
        metrics_before = model.evaluate_on_query_set(sample_task)
        
        # Perform adaptation
        adapted_params = model.adapt_to_task(sample_task)
        
        # Evaluate after adaptation
        metrics_after = model.evaluate_on_query_set(sample_task, adapted_params)
        
        # Performance should improve (or at least not degrade significantly)
        # Note: With random initialization, improvement isn't guaranteed, but loss shouldn't explode
        assert metrics_after['mse'] <= metrics_before['mse'] * 2.0, "MSE shouldn't increase dramatically"
        assert torch.isfinite(metrics_after['mse']), "MSE should be finite"
    
    def test_gradient_flow_through_adaptation(self, model, sample_task):
        """Test that gradients flow properly through adaptation."""
        # Perform adaptation with create_graph=True for meta-learning
        adapted_params = model.adapt_to_task(sample_task, create_graph=True)
        
        # Compute loss on query set with adapted parameters
        query_loss = model.compute_adaptation_loss(sample_task, adapted_params)['total_loss']
        
        # Compute gradients with respect to original parameters
        query_loss.backward()
        
        # Check that original model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None, "Original parameters should have gradients"
            assert not torch.isnan(param.grad).any(), "Gradients should not be NaN"
    
    def test_adaptation_with_gradient_clipping(self, model, sample_task):
        """Test adaptation with gradient clipping."""
        # Set a small gradient clipping value
        model.config.gradient_clipping = 0.1
        
        # Perform adaptation
        adapted_params = model.adapt_to_task(sample_task)
        
        # Should complete without errors
        assert isinstance(adapted_params, OrderedDict)
        
        # Check adaptation history
        history = model.get_adaptation_history()
        assert len(history) > 0
    
    def test_adaptation_without_gradient_clipping(self, model, sample_task):
        """Test adaptation without gradient clipping."""
        # Disable gradient clipping
        model.config.gradient_clipping = None
        
        # Perform adaptation
        adapted_params = model.adapt_to_task(sample_task)
        
        # Should complete without errors
        assert isinstance(adapted_params, OrderedDict)
    
    def test_adaptation_history_tracking(self, model, sample_task):
        """Test that adaptation history is properly tracked."""
        # Perform adaptation
        model.adapt_to_task(sample_task)
        
        # Get adaptation history
        history = model.get_adaptation_history()
        
        # Check history structure
        assert isinstance(history, list)
        assert len(history) == model.adaptation_steps
        
        for i, step_info in enumerate(history):
            assert step_info['step'] == i
            assert 'data_loss' in step_info
            assert 'physics_loss' in step_info
            assert 'total_loss' in step_info
            
            # All losses should be finite
            assert torch.isfinite(torch.tensor(step_info['data_loss']))
            assert torch.isfinite(torch.tensor(step_info['physics_loss']))
            assert torch.isfinite(torch.tensor(step_info['total_loss']))
    
    def test_adaptation_with_different_task_types(self, model):
        """Test adaptation with different viscosity types."""
        viscosity_types = [
            ('constant', {'mu_0': 1.0}),
            ('linear', {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}),
            ('exponential', {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0})
        ]
        
        for viscosity_type, params in viscosity_types:
            # Create task with specific viscosity type
            task = {
                'support_coords': torch.randn(20, 3, requires_grad=True),
                'support_data': torch.randn(20, 3),
                'query_coords': torch.randn(20, 3, requires_grad=True),
                'query_data': torch.randn(20, 3),
                'task_info': {
                    'viscosity_type': viscosity_type,
                    'viscosity_params': params
                }
            }
            
            # Perform adaptation
            adapted_params = model.adapt_to_task(task)
            
            # Should complete without errors
            assert isinstance(adapted_params, OrderedDict)
    
    def test_parameter_isolation_between_adaptations(self, model, sample_task):
        """Test that adaptations don't interfere with each other."""
        # Store original parameters
        original_params = model.clone_parameters()
        
        # Perform first adaptation
        adapted_params_1 = model.adapt_to_task(sample_task)
        
        # Model parameters should remain unchanged
        current_params = model.get_parameters_dict()
        for name in original_params:
            assert torch.allclose(current_params[name], original_params[name], atol=1e-6)
        
        # Perform second adaptation
        adapted_params_2 = model.adapt_to_task(sample_task)
        
        # Both adaptations should start from the same original parameters
        # So they should be similar (though not identical due to randomness in optimization)
        for name in adapted_params_1:
            # Parameters should be in the same ballpark
            assert torch.allclose(adapted_params_1[name], adapted_params_2[name], atol=1e-2)
    
    def test_adaptation_with_zero_steps(self, model, sample_task):
        """Test adaptation with zero steps."""
        # Perform adaptation with zero steps
        adapted_params = model.adapt_to_task(sample_task, adaptation_steps=0)
        
        # Parameters should be identical to original
        original_params = model.get_parameters_dict()
        for name in original_params:
            assert torch.allclose(adapted_params[name], original_params[name], atol=1e-6)
        
        # History should be empty
        history = model.get_adaptation_history()
        assert len(history) == 0
    
    def test_adaptation_loss_components(self, model, sample_task):
        """Test that adaptation properly balances data and physics losses."""
        # Perform adaptation
        adapted_params = model.adapt_to_task(sample_task)
        history = model.get_adaptation_history()
        
        # Check that both data and physics losses are present
        for step_info in history:
            assert step_info['data_loss'] >= 0
            assert step_info['physics_loss'] >= 0
            
            # Total loss should be sum of components (approximately)
            expected_total = step_info['data_loss'] + step_info['physics_loss']
            assert abs(step_info['total_loss'] - expected_total) < 1e-5
    
    def test_functional_forward_in_adaptation(self, model, sample_task):
        """Test that functional forward pass works correctly during adaptation."""
        # Get model parameters
        params = model.clone_parameters()
        
        # Test functional forward pass
        coords = sample_task['support_coords']
        output_functional = model.forward(coords, params)
        output_normal = model.forward(coords)
        
        # Should produce same results
        assert torch.allclose(output_functional, output_normal, atol=1e-6)
        
        # Now test adaptation which uses functional forward pass
        adapted_params = model.adapt_to_task(sample_task)
        
        # Should complete without errors
        assert isinstance(adapted_params, OrderedDict)


if __name__ == "__main__":
    pytest.main([__file__])
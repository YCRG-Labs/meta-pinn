"""
Unit tests for StandardPINN implementation.

Tests the standard PINN baseline model for single-task training capability,
evaluation methods, and training convergence.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from ml_research_pipeline.core.standard_pinn import StandardPINN
from ml_research_pipeline.config.model_config import ModelConfig


class TestStandardPINN:
    """Test suite for StandardPINN class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ModelConfig(
            input_dim=3,
            output_dim=3,
            hidden_layers=[64, 64, 64],
            activation="tanh",
            physics_loss_weight=1.0,
            adaptive_physics_weight=True,
            enforce_boundary_conditions=False,
            weight_init="xavier_normal",
            bias_init="zeros"
        )
    
    @pytest.fixture
    def model(self, config):
        """Create a StandardPINN model for testing."""
        return StandardPINN(config)
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        # Generate sample coordinates and data
        n_points = 100
        coords = torch.randn(n_points, 3)  # [x, y, t]
        data = torch.randn(n_points, 3)    # [u, v, p]
        
        task_info = {
            'viscosity_type': 'constant',
            'viscosity_params': {'mu_0': 1.0},
            'boundary_conditions': {}
        }
        
        return {
            'coords': coords,
            'data': data,
            'task_info': task_info
        }
    
    def test_model_initialization(self, config):
        """Test that the model initializes correctly."""
        model = StandardPINN(config)
        
        # Check that model is created
        assert isinstance(model, StandardPINN)
        assert isinstance(model, nn.Module)
        
        # Check configuration is stored
        assert model.config == config
        
        # Check network architecture
        assert len(model.layers) > 0
        
        # Check physics parameters
        assert model.physics_loss_weight == config.physics_loss_weight
        assert model.adaptive_physics_weight == config.adaptive_physics_weight
        
        # Check training state initialization
        assert model.training_history == []
        assert model.current_task_info is None
    
    def test_network_architecture(self, model, config):
        """Test that the network architecture is built correctly."""
        layer_sizes = config.get_layer_sizes()
        
        # Count linear layers
        linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
        assert len(linear_layers) == len(layer_sizes) - 1
        
        # Check layer dimensions
        for i, layer in enumerate(linear_layers):
            assert layer.in_features == layer_sizes[i]
            assert layer.out_features == layer_sizes[i + 1]
    
    def test_forward_pass(self, model):
        """Test the forward pass functionality."""
        batch_size = 32
        input_dim = model.config.input_dim
        output_dim = model.config.output_dim
        
        # Test forward pass
        x = torch.randn(batch_size, input_dim)
        output = model.forward(x)
        
        # Check output shape
        assert output.shape == (batch_size, output_dim)
        
        # Check output is finite
        assert torch.isfinite(output).all()
    
    def test_activation_functions(self, config):
        """Test different activation functions."""
        activations = ["tanh", "relu", "gelu", "swish", "sin"]
        
        for activation in activations:
            config.activation = activation
            model = StandardPINN(config)
            
            x = torch.randn(10, config.input_dim)
            output = model.forward(x)
            
            assert output.shape == (10, config.output_dim)
            assert torch.isfinite(output).all()
    
    def test_physics_loss_computation(self, model, sample_task):
        """Test physics loss computation."""
        coords = sample_task['coords']
        task_info = sample_task['task_info']
        
        # Compute physics loss
        physics_losses = model.physics_loss(coords, task_info)
        
        # Check that all expected components are present
        expected_components = ['momentum_x', 'momentum_y', 'continuity', 'total_pde', 'total']
        for component in expected_components:
            assert component in physics_losses
            assert isinstance(physics_losses[component], torch.Tensor)
            assert physics_losses[component].numel() == 1  # Scalar loss
            assert torch.isfinite(physics_losses[component])
    
    def test_viscosity_computation(self, model):
        """Test viscosity computation for different types."""
        coords = torch.randn(50, 3)
        
        # Test constant viscosity
        task_info = {
            'viscosity_type': 'constant',
            'viscosity_params': {'mu_0': 2.0}
        }
        viscosity = model._compute_viscosity(coords, task_info)
        assert torch.allclose(viscosity, torch.full_like(viscosity, 2.0))
        
        # Test linear viscosity
        task_info = {
            'viscosity_type': 'linear',
            'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.2}
        }
        viscosity = model._compute_viscosity(coords, task_info)
        assert viscosity.shape == (50, 1)
        assert torch.isfinite(viscosity).all()
        
        # Test exponential viscosity
        task_info = {
            'viscosity_type': 'exponential',
            'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}
        }
        viscosity = model._compute_viscosity(coords, task_info)
        assert viscosity.shape == (50, 1)
        assert torch.isfinite(viscosity).all()
        assert (viscosity > 0).all()  # Viscosity should be positive
    
    def test_derivative_computation(self, model):
        """Test automatic differentiation for derivatives."""
        coords = torch.randn(20, 3, requires_grad=True)
        
        # Forward pass
        output = model.forward(coords)
        u = output[:, 0:1]
        
        # Compute derivatives
        derivatives = model._compute_derivatives(u, coords)
        
        # Check that derivatives are computed
        assert 'x' in derivatives
        assert 'y' in derivatives
        assert 't' in derivatives
        
        for deriv in derivatives.values():
            assert deriv.shape == (20, 1)
            assert torch.isfinite(deriv).all()
    
    def test_train_on_task(self, model, sample_task):
        """Test single-task training functionality."""
        # Train for a few epochs
        history = model.train_on_task(sample_task, epochs=10, verbose=False)
        
        # Check that history is returned
        assert isinstance(history, dict)
        expected_keys = ['data_loss', 'physics_loss', 'total_loss', 
                        'momentum_x_loss', 'momentum_y_loss', 'continuity_loss']
        for key in expected_keys:
            assert key in history
            assert len(history[key]) == 10  # 10 epochs
        
        # Check that losses are finite
        for losses in history.values():
            assert all(np.isfinite(loss) for loss in losses)
        
        # Check that training history is stored
        assert model.training_history == history
        assert model.current_task_info == sample_task['task_info']
    
    def test_training_convergence(self, model, sample_task):
        """Test that training reduces loss over time."""
        # Train for more epochs to see convergence
        history = model.train_on_task(sample_task, epochs=50, verbose=False)
        
        # Check that total loss generally decreases
        total_losses = history['total_loss']
        initial_loss = np.mean(total_losses[:5])
        final_loss = np.mean(total_losses[-5:])
        
        # Loss should decrease (allowing for some fluctuation)
        assert final_loss < initial_loss * 1.1  # Allow 10% tolerance
    
    def test_evaluate_on_task(self, model, sample_task):
        """Test task evaluation functionality."""
        # Train briefly first
        model.train_on_task(sample_task, epochs=5, verbose=False)
        
        # Evaluate on the same task
        metrics = model.evaluate_on_task(sample_task)
        
        # Check that metrics are returned
        expected_metrics = ['data_loss', 'physics_residual', 'parameter_accuracy', 
                           'prediction_mse', 'l2_error']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert np.isfinite(metrics[metric])
            assert metrics[metric] >= 0  # All metrics should be non-negative
    
    def test_adaptive_physics_weight(self, model):
        """Test adaptive physics weight computation."""
        # Test with low residuals
        low_residuals = {
            'momentum_x': torch.tensor(1e-6),
            'momentum_y': torch.tensor(1e-6),
            'continuity': torch.tensor(1e-6)
        }
        weight = model.compute_adaptive_physics_weight(low_residuals, 1.0)
        assert weight == 1.0  # Should return base weight for low residuals
        
        # Test with high residuals
        high_residuals = {
            'momentum_x': torch.tensor(1e-2),
            'momentum_y': torch.tensor(1e-3),
            'continuity': torch.tensor(1e-3)
        }
        weight = model.compute_adaptive_physics_weight(high_residuals, 1.0)
        assert weight > 1.0  # Should increase weight for high residuals
        assert weight <= 10.0  # Should be capped
    
    def test_parameter_count(self, model, config):
        """Test parameter counting functionality."""
        param_count = model.count_parameters()
        
        # Calculate expected parameter count
        layer_sizes = config.get_layer_sizes()
        expected_count = 0
        for i in range(len(layer_sizes) - 1):
            # Weights + biases
            expected_count += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        
        assert param_count == expected_count
    
    def test_model_info(self, model, config):
        """Test model information retrieval."""
        info = model.get_model_info()
        
        # Check that all expected information is present
        expected_keys = ['model_type', 'input_dim', 'output_dim', 'hidden_layers', 
                        'activation', 'total_parameters', 'physics_loss_weight', 
                        'adaptive_physics_weight']
        for key in expected_keys:
            assert key in info
        
        # Check specific values
        assert info['model_type'] == 'StandardPINN'
        assert info['input_dim'] == config.input_dim
        assert info['output_dim'] == config.output_dim
        assert info['hidden_layers'] == config.hidden_layers
        assert info['activation'] == config.activation
    
    def test_reset_parameters(self, model, sample_task):
        """Test parameter reset functionality."""
        # Train the model first
        model.train_on_task(sample_task, epochs=5, verbose=False)
        
        # Store initial state
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Reset parameters
        model.reset_parameters()
        
        # Check that parameters have changed (re-initialized)
        for name, param in model.named_parameters():
            # Parameters should be different after reset (with high probability)
            assert not torch.allclose(param, initial_params[name], atol=1e-6)
        
        # Check that training state is reset
        assert model.training_history == []
        assert model.current_task_info is None
    
    def test_different_task_types(self, model):
        """Test training on different viscosity task types."""
        n_points = 50
        coords = torch.randn(n_points, 3)
        data = torch.randn(n_points, 3)
        
        task_types = [
            {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}},
            {'viscosity_type': 'linear', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1}},
            {'viscosity_type': 'bilinear', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.1, 'gamma': 0.05}},
            {'viscosity_type': 'exponential', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1}}
        ]
        
        for task_info in task_types:
            task = {
                'coords': coords,
                'data': data,
                'task_info': task_info
            }
            
            # Reset model for each task
            model.reset_parameters()
            
            # Train on task
            history = model.train_on_task(task, epochs=5, verbose=False)
            
            # Check that training completes successfully
            assert len(history['total_loss']) == 5
            assert all(np.isfinite(loss) for loss in history['total_loss'])
    
    def test_gradient_flow(self, model, sample_task):
        """Test that gradients flow properly during training."""
        coords = sample_task['coords']
        data = sample_task['data']
        task_info = sample_task['task_info']
        
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
    
    def test_batch_processing(self, model):
        """Test that the model handles different batch sizes correctly."""
        batch_sizes = [1, 16, 32, 100]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, model.config.input_dim)
            output = model.forward(x)
            
            assert output.shape == (batch_size, model.config.output_dim)
            assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__])
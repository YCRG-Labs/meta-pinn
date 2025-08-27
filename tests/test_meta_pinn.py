"""
Unit tests for MetaPINN class.

Tests network initialization, forward pass functionality, and parameter management.
"""

import pytest
import torch
import torch.nn as nn
from collections import OrderedDict

from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.config.model_config import MetaPINNConfig


class TestMetaPINN:
    """Test suite for MetaPINN class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return MetaPINNConfig(
            input_dim=3,
            output_dim=3,
            hidden_layers=[64, 64, 64],
            activation="tanh",
            meta_lr=0.001,
            adapt_lr=0.01,
            adaptation_steps=5
        )
    
    @pytest.fixture
    def model(self, config):
        """Create a test model."""
        return MetaPINN(config)
    
    def test_model_initialization(self, model, config):
        """Test that model initializes correctly."""
        # Check that model is an instance of nn.Module
        assert isinstance(model, nn.Module)
        
        # Check configuration is stored
        assert model.config == config
        
        # Check meta-learning parameters
        assert model.meta_lr == config.meta_lr
        assert model.adapt_lr == config.adapt_lr
        assert model.adaptation_steps == config.adaptation_steps
        
        # Check network architecture
        expected_layers = len(config.get_layer_sizes()) - 1
        linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
        assert len(linear_layers) == expected_layers
        
    def test_network_architecture(self, model, config):
        """Test that network architecture matches configuration."""
        layer_sizes = config.get_layer_sizes()
        linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
        
        # Check input and output dimensions
        assert linear_layers[0].in_features == config.input_dim
        assert linear_layers[-1].out_features == config.output_dim
        
        # Check hidden layer dimensions
        for i, layer in enumerate(linear_layers):
            assert layer.in_features == layer_sizes[i]
            assert layer.out_features == layer_sizes[i + 1]
    
    def test_forward_pass_shape(self, model, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 32
        x = torch.randn(batch_size, config.input_dim)
        
        # Test forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, config.output_dim)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_forward_pass_deterministic(self, model, config):
        """Test that forward pass is deterministic."""
        torch.manual_seed(42)
        x = torch.randn(16, config.input_dim)
        
        # Two forward passes should give same result
        output1 = model(x)
        output2 = model(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_functional_forward_pass(self, model, config):
        """Test functional forward pass with external parameters."""
        batch_size = 16
        x = torch.randn(batch_size, config.input_dim)
        
        # Get model parameters
        params = model.get_parameters_dict()
        
        # Test functional forward pass
        output_functional = model(x, params)
        output_normal = model(x)
        
        # Should produce same results
        assert torch.allclose(output_functional, output_normal, atol=1e-6)
    
    def test_parameter_management(self, model):
        """Test parameter cloning and management."""
        # Test parameter dictionary
        params_dict = model.get_parameters_dict()
        assert isinstance(params_dict, OrderedDict)
        assert len(params_dict) > 0
        
        # Test parameter cloning
        cloned_params = model.clone_parameters()
        assert isinstance(cloned_params, OrderedDict)
        assert len(cloned_params) == len(params_dict)
        
        # Check that cloned parameters are different objects but same values
        for name in params_dict:
            assert cloned_params[name] is not params_dict[name]  # Different objects
            assert torch.allclose(cloned_params[name], params_dict[name])  # Same values
    
    def test_parameter_counting(self, model, config):
        """Test parameter counting functionality."""
        param_count = model.count_parameters()
        
        # Calculate expected parameter count
        layer_sizes = config.get_layer_sizes()
        expected_count = 0
        for i in range(len(layer_sizes) - 1):
            # Weight matrix + bias vector
            expected_count += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        
        assert param_count == expected_count
    
    def test_layer_info(self, model, config):
        """Test layer information extraction."""
        layer_info = model.get_layer_info()
        
        # Should have information for all linear layers
        linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
        linear_info = [info for info in layer_info if info['type'] == 'Linear']
        assert len(linear_info) == len(linear_layers)
        
        # Check layer information accuracy
        layer_sizes = config.get_layer_sizes()
        for i, info in enumerate(linear_info):
            assert info['input_size'] == layer_sizes[i]
            assert info['output_size'] == layer_sizes[i + 1]
    
    def test_different_activations(self, config):
        """Test different activation functions."""
        activations = ["tanh", "relu", "gelu", "swish", "sin"]
        
        for activation in activations:
            config.activation = activation
            model = MetaPINN(config)
            
            x = torch.randn(16, config.input_dim)
            output = model(x)
            
            # Should produce valid output
            assert output.shape == (16, config.output_dim)
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()
    
    def test_normalization_options(self, config):
        """Test different normalization options."""
        # Test input normalization
        config.input_normalization = True
        model = MetaPINN(config)
        
        x = torch.randn(16, config.input_dim) * 100  # Large values
        output = model(x)
        assert not torch.isnan(output).any()
        
        # Test layer normalization
        config.layer_normalization = True
        model = MetaPINN(config)
        
        x = torch.randn(16, config.input_dim)
        output = model(x)
        assert not torch.isnan(output).any()
    
    def test_dropout_functionality(self, config):
        """Test dropout functionality."""
        config.dropout_rate = 0.5
        model = MetaPINN(config)
        
        x = torch.randn(16, config.input_dim)
        
        # Training mode should apply dropout
        model.train()
        output1 = model(x)
        output2 = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2, atol=1e-6)
        
        # Eval mode should be deterministic
        model.eval()
        output3 = model(x)
        output4 = model(x)
        
        assert torch.allclose(output3, output4, atol=1e-6)
    
    def test_gradient_flow(self, model, config):
        """Test that gradients flow through the network."""
        x = torch.randn(16, config.input_dim, requires_grad=True)
        output = model(x)
        
        # Compute a simple loss
        loss = output.sum()
        loss.backward()
        
        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_model_representation(self, model):
        """Test model string representation."""
        repr_str = repr(model)
        
        # Should contain key information
        assert "MetaPINN" in repr_str
        assert "layers=" in repr_str
        assert "activation=" in repr_str
        assert "total_parameters=" in repr_str
        assert "meta_lr=" in repr_str
        assert "adapt_lr=" in repr_str
    
    def test_batch_size_independence(self, model, config):
        """Test that model works with different batch sizes."""
        batch_sizes = [1, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, config.input_dim)
            output = model(x)
            
            assert output.shape == (batch_size, config.output_dim)
            assert not torch.isnan(output).any()
    
    def test_device_compatibility(self, model, config):
        """Test model device compatibility."""
        x = torch.randn(16, config.input_dim)
        
        # Test CPU
        model_cpu = model.cpu()
        x_cpu = x.cpu()
        output_cpu = model_cpu(x_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = x.cuda()
            output_gpu = model_gpu(x_gpu)
            assert output_gpu.device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__])
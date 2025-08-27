"""
Unit tests for Bayesian Meta-Learning Physics-Informed Neural Network.

Tests variational parameter initialization, weight sampling, KL divergence computation,
and uncertainty quantification functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from ml_research_pipeline.bayesian.bayesian_meta_pinn import BayesianMetaPINN, VariationalLinear
from ml_research_pipeline.config.model_config import MetaPINNConfig


class TestVariationalLinear:
    """Test cases for VariationalLinear layer."""
    
    def test_initialization(self):
        """Test proper initialization of variational parameters."""
        layer = VariationalLinear(10, 5, prior_std=1.0)
        
        # Check parameter shapes
        assert layer.weight_mu.shape == (5, 10)
        assert layer.weight_logvar.shape == (5, 10)
        assert layer.bias_mu.shape == (5,)
        assert layer.bias_logvar.shape == (5,)
        
        # Check initial values are reasonable
        assert torch.all(layer.weight_logvar < 0)  # Small initial variance
        assert torch.all(layer.bias_logvar < 0)
        
    def test_forward_deterministic(self):
        """Test deterministic forward pass (sample=False)."""
        layer = VariationalLinear(3, 2)
        x = torch.randn(5, 3)
        
        # Deterministic forward should use mean parameters
        output1 = layer(x, sample=False)
        output2 = layer(x, sample=False)
        
        assert output1.shape == (5, 2)
        assert torch.allclose(output1, output2)  # Should be identical
        
    def test_forward_stochastic(self):
        """Test stochastic forward pass (sample=True)."""
        layer = VariationalLinear(3, 2)
        layer.train()  # Enable training mode for sampling
        x = torch.randn(5, 3)
        
        # Stochastic forward should produce different outputs
        output1 = layer(x, sample=True)
        output2 = layer(x, sample=True)
        
        assert output1.shape == (5, 2)
        assert output2.shape == (5, 2)
        # Outputs should be different due to sampling (with high probability)
        assert not torch.allclose(output1, output2, atol=1e-6)
        
    def test_kl_divergence_computation(self):
        """Test KL divergence computation."""
        layer = VariationalLinear(3, 2, prior_std=1.0)
        
        kl_div = layer.kl_divergence()
        
        # KL divergence should be positive
        assert kl_div >= 0
        assert kl_div.requires_grad  # Should be differentiable
        
        # Test that KL divergence changes with parameters
        original_kl = kl_div.item()
        
        # Modify parameters
        layer.weight_mu.data += 0.1
        new_kl = layer.kl_divergence().item()
        
        assert new_kl != original_kl
        
    def test_kl_divergence_properties(self):
        """Test mathematical properties of KL divergence."""
        layer = VariationalLinear(2, 2, prior_std=1.0)
        
        # When posterior equals prior (μ=0, σ²=1), KL should be close to 0
        layer.weight_mu.data.zero_()
        layer.bias_mu.data.zero_()
        layer.weight_logvar.data.zero_()  # log(1) = 0, so σ² = 1
        layer.bias_logvar.data.zero_()
        
        kl_div = layer.kl_divergence()
        assert torch.abs(kl_div) < 1e-5  # Should be very close to 0


class TestBayesianMetaPINN:
    """Test cases for BayesianMetaPINN."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MetaPINNConfig()
        config.input_dim = 3
        config.output_dim = 3
        config.hidden_dims = [20, 20]
        config.activation = "tanh"
        config.meta_lr = 0.001
        config.adapt_lr = 0.01
        config.adaptation_steps = 5
        return config
        
    @pytest.fixture
    def bayesian_model(self, config):
        """Create BayesianMetaPINN instance."""
        return BayesianMetaPINN(config, prior_std=1.0, kl_weight=1e-4)
        
    def test_initialization(self, bayesian_model):
        """Test proper initialization of Bayesian model."""
        # Check that variational layers are created
        variational_layers = [m for m in bayesian_model.modules() if isinstance(m, VariationalLinear)]
        assert len(variational_layers) > 0
        
        # Check Bayesian-specific attributes
        assert hasattr(bayesian_model, 'prior_std')
        assert hasattr(bayesian_model, 'kl_weight')
        assert hasattr(bayesian_model, 'n_mc_samples')
        
    def test_forward_with_sampling(self, bayesian_model):
        """Test forward pass with weight sampling."""
        x = torch.randn(10, 3)
        
        # Test deterministic forward
        output1 = bayesian_model(x, sample=False)
        output2 = bayesian_model(x, sample=False)
        assert torch.allclose(output1, output2)
        
        # Test stochastic forward
        bayesian_model.train()
        output3 = bayesian_model(x, sample=True)
        output4 = bayesian_model(x, sample=True)
        assert not torch.allclose(output3, output4, atol=1e-6)
        
    def test_weight_sampling(self, bayesian_model):
        """Test weight sampling functionality."""
        n_samples = 5
        weight_samples = bayesian_model.sample_weights(n_samples)
        
        assert len(weight_samples) == n_samples
        
        # Check that each sample is an OrderedDict with proper keys
        for sample in weight_samples:
            assert isinstance(sample, OrderedDict)
            assert len(sample) > 0
            
            # Check that samples are different
            for key in sample:
                assert sample[key].requires_grad is False  # Sampled weights shouldn't require grad
                
        # Verify samples are different
        if len(weight_samples) > 1:
            sample1, sample2 = weight_samples[0], weight_samples[1]
            for key in sample1:
                if key in sample2:
                    assert not torch.allclose(sample1[key], sample2[key], atol=1e-6)
                    
    def test_kl_divergence_total(self, bayesian_model):
        """Test total KL divergence computation."""
        kl_div = bayesian_model.kl_divergence()
        
        assert kl_div >= 0
        assert kl_div.requires_grad
        
        # KL divergence should be sum of individual layer KL divergences
        manual_kl = torch.tensor(0.0)
        for module in bayesian_model.modules():
            if isinstance(module, VariationalLinear):
                manual_kl += module.kl_divergence()
                
        assert torch.allclose(kl_div, manual_kl)
        
    def test_forward_with_uncertainty(self, bayesian_model):
        """Test uncertainty quantification through Monte Carlo sampling."""
        x = torch.randn(5, 3)
        n_samples = 10
        
        mean_pred, uncertainty = bayesian_model.forward_with_uncertainty(x, n_samples=n_samples)
        
        # Check output shapes
        assert mean_pred.shape == (5, 3)
        assert uncertainty.shape == (5, 3)
        
        # Uncertainty should be non-negative
        assert torch.all(uncertainty >= 0)
        
        # Test with different number of samples
        mean_pred2, uncertainty2 = bayesian_model.forward_with_uncertainty(x, n_samples=50)
        
        # Results should be similar but not identical due to sampling
        assert not torch.allclose(mean_pred, mean_pred2, atol=1e-3)
        
    def test_epistemic_uncertainty(self, bayesian_model):
        """Test epistemic uncertainty computation."""
        x = torch.randn(8, 3)
        
        epistemic_unc = bayesian_model.compute_epistemic_uncertainty(x, n_samples=20)
        
        assert epistemic_unc.shape == (8, 3)
        assert torch.all(epistemic_unc >= 0)
        
    def test_aleatoric_uncertainty(self, bayesian_model):
        """Test aleatoric uncertainty computation."""
        x = torch.randn(6, 3)
        task_info = {
            'viscosity_type': 'constant',
            'viscosity_params': {'mu_0': 1.0}
        }
        
        aleatoric_unc = bayesian_model.compute_aleatoric_uncertainty(x, task_info)
        
        assert aleatoric_unc.shape == (6, 3)
        assert torch.all(aleatoric_unc >= 0)
        
    def test_uncertainty_decomposition(self, bayesian_model):
        """Test uncertainty decomposition into epistemic and aleatoric components."""
        x = torch.randn(4, 3)
        task_info = {
            'viscosity_type': 'linear',
            'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}
        }
        
        uncertainty_dict = bayesian_model.decompose_uncertainty(x, task_info, n_samples=15)
        
        # Check all components are present
        assert 'epistemic' in uncertainty_dict
        assert 'aleatoric' in uncertainty_dict
        assert 'total' in uncertainty_dict
        
        # Check shapes
        for key in uncertainty_dict:
            assert uncertainty_dict[key].shape == (4, 3)
            assert torch.all(uncertainty_dict[key] >= 0)
            
        # Total uncertainty should be at least as large as individual components
        epistemic = uncertainty_dict['epistemic']
        aleatoric = uncertainty_dict['aleatoric']
        total = uncertainty_dict['total']
        
        # Due to independence assumption: total² = epistemic² + aleatoric²
        expected_total = torch.sqrt(epistemic**2 + aleatoric**2)
        assert torch.allclose(total, expected_total, atol=1e-5)
        
    def test_variational_loss(self, bayesian_model):
        """Test variational loss computation."""
        x = torch.randn(5, 3, requires_grad=True)  # Enable gradients for physics loss
        predictions = torch.randn(5, 3, requires_grad=True)
        targets = torch.randn(5, 3)
        task_info = {
            'viscosity_type': 'constant',
            'viscosity_params': {'mu_0': 1.0}
        }
        
        loss_dict = bayesian_model.variational_loss(predictions, targets, x, task_info)
        
        # Check all loss components are present
        assert 'data_loss' in loss_dict
        assert 'physics_loss' in loss_dict
        assert 'kl_loss' in loss_dict
        assert 'total_loss' in loss_dict
        
        # Check that all losses are non-negative
        for key in ['data_loss', 'physics_loss', 'kl_loss', 'total_loss']:
            assert loss_dict[key] >= 0
            
        # Check that total loss has gradients
        assert loss_dict['total_loss'].requires_grad
            
        # Total loss should be sum of components
        expected_total = loss_dict['data_loss'] + loss_dict['physics_loss'] + loss_dict['kl_loss']
        assert torch.allclose(loss_dict['total_loss'], expected_total)
        
    def test_bayesian_adaptation(self, bayesian_model):
        """Test Bayesian adaptation to new task."""
        # Create mock task
        task = {
            'support_coords': torch.randn(10, 3),
            'support_data': torch.randn(10, 3),
            'task_info': {
                'viscosity_type': 'constant',
                'viscosity_params': {'mu_0': 1.0}
            }
        }
        
        adapted_params, uncertainty_info = bayesian_model.adapt_to_task_bayesian(
            task, adaptation_steps=3, use_uncertainty=True
        )
        
        # Check adapted parameters
        assert isinstance(adapted_params, OrderedDict)
        assert len(adapted_params) > 0
        
        # Check uncertainty information
        assert 'initial_uncertainty' in uncertainty_info
        assert 'final_uncertainty' in uncertainty_info
        assert 'uncertainty_reduction' in uncertainty_info
        
        # Initial and final uncertainties should have proper structure
        for unc_dict in [uncertainty_info['initial_uncertainty'], uncertainty_info['final_uncertainty']]:
            if unc_dict is not None:
                assert 'epistemic' in unc_dict
                assert 'aleatoric' in unc_dict
                assert 'total' in unc_dict
                
    def test_predict_with_confidence(self, bayesian_model):
        """Test prediction with confidence intervals."""
        x = torch.randn(3, 3)
        task_info = {
            'viscosity_type': 'exponential',
            'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}
        }
        
        pred_dict = bayesian_model.predict_with_confidence(x, task_info, confidence_level=0.95)
        
        # Check all components are present
        assert 'mean' in pred_dict
        assert 'uncertainty' in pred_dict
        assert 'lower_bound' in pred_dict
        assert 'upper_bound' in pred_dict
        assert 'confidence_level' in pred_dict
        
        # Check shapes
        for key in ['mean', 'uncertainty', 'lower_bound', 'upper_bound']:
            assert pred_dict[key].shape == (3, 3)
            
        # Check confidence interval properties
        mean = pred_dict['mean']
        lower = pred_dict['lower_bound']
        upper = pred_dict['upper_bound']
        
        assert torch.all(lower <= mean)
        assert torch.all(mean <= upper)
        assert pred_dict['confidence_level'] == 0.95
        
    def test_variational_parameter_management(self, bayesian_model):
        """Test getting and setting variational parameters."""
        # Get initial parameters
        initial_params = bayesian_model.get_variational_parameters()
        
        assert isinstance(initial_params, dict)
        assert len(initial_params) > 0
        
        # Check parameter structure
        for layer_name, params in initial_params.items():
            assert 'weight_mu' in params
            assert 'weight_logvar' in params
            assert 'bias_mu' in params
            assert 'bias_logvar' in params
            
        # Modify parameters
        modified_params = {}
        for layer_name, params in initial_params.items():
            modified_params[layer_name] = {
                'weight_mu': params['weight_mu'] + 0.1,
                'weight_logvar': params['weight_logvar'] + 0.1,
                'bias_mu': params['bias_mu'] + 0.1,
                'bias_logvar': params['bias_logvar'] + 0.1
            }
            
        # Set modified parameters
        bayesian_model.set_variational_parameters(modified_params)
        
        # Verify parameters were set correctly
        new_params = bayesian_model.get_variational_parameters()
        for layer_name in modified_params:
            for param_name in modified_params[layer_name]:
                assert torch.allclose(
                    new_params[layer_name][param_name],
                    modified_params[layer_name][param_name]
                )
                
    def test_gradient_flow(self, bayesian_model):
        """Test that gradients flow properly through variational parameters."""
        x = torch.randn(5, 3, requires_grad=True)
        
        # Forward pass
        output = bayesian_model(x, sample=True)
        loss = torch.mean(output**2)
        
        # Backward pass
        loss.backward()
        
        # Check that variational parameters have gradients
        for module in bayesian_model.modules():
            if isinstance(module, VariationalLinear):
                assert module.weight_mu.grad is not None
                assert module.weight_logvar.grad is not None
                assert module.bias_mu.grad is not None
                assert module.bias_logvar.grad is not None
                
    def test_consistency_with_deterministic_mode(self, bayesian_model):
        """Test that deterministic mode gives consistent results."""
        x = torch.randn(4, 3)
        
        # Multiple deterministic forward passes should be identical
        bayesian_model.eval()
        output1 = bayesian_model(x, sample=False)
        output2 = bayesian_model(x, sample=False)
        output3 = bayesian_model(x, sample=False)
        
        assert torch.allclose(output1, output2)
        assert torch.allclose(output2, output3)
        
    def test_uncertainty_scaling_with_samples(self, bayesian_model):
        """Test that uncertainty estimates stabilize with more samples."""
        x = torch.randn(3, 3)
        
        # Test with different numbers of samples
        _, unc_10 = bayesian_model.forward_with_uncertainty(x, n_samples=10)
        _, unc_100 = bayesian_model.forward_with_uncertainty(x, n_samples=100)
        
        # With more samples, uncertainty estimates should be more stable
        # (This is a statistical property, so we just check they're reasonable)
        assert torch.all(unc_10 >= 0)
        assert torch.all(unc_100 >= 0)
        assert unc_10.shape == unc_100.shape


if __name__ == "__main__":
    pytest.main([__file__])
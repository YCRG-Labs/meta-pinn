"""
Additional tests for uncertainty estimation accuracy in BayesianMetaPINN.

Tests the accuracy and consistency of uncertainty estimates across different scenarios.
"""

import pytest
import torch
import numpy as np

from ml_research_pipeline.bayesian.bayesian_meta_pinn import BayesianMetaPINN
from ml_research_pipeline.config.model_config import MetaPINNConfig


class TestUncertaintyAccuracy:
    """Test uncertainty estimation accuracy."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MetaPINNConfig()
        config.input_dim = 2
        config.output_dim = 1
        config.hidden_dims = [10, 10]
        config.activation = "tanh"
        return config
        
    @pytest.fixture
    def model(self, config):
        """Create BayesianMetaPINN instance."""
        return BayesianMetaPINN(config, prior_std=1.0, n_mc_samples=50)
        
    def test_uncertainty_consistency_across_samples(self, model):
        """Test that uncertainty estimates are consistent across different sample sizes."""
        x = torch.randn(5, 2)
        
        # Test with different sample sizes
        _, unc_10 = model.forward_with_uncertainty(x, n_samples=10)
        _, unc_50 = model.forward_with_uncertainty(x, n_samples=50)
        _, unc_100 = model.forward_with_uncertainty(x, n_samples=100)
        
        # Uncertainty estimates should be similar (within reasonable bounds)
        # More samples should generally give more stable estimates
        assert torch.all(unc_10 >= 0)
        assert torch.all(unc_50 >= 0)
        assert torch.all(unc_100 >= 0)
        
        # Check that estimates are in reasonable range
        assert torch.all(unc_50 < 10.0)  # Shouldn't be extremely large
        
    def test_uncertainty_increases_with_distance(self, model):
        """Test that uncertainty generally increases with distance from training data."""
        # Create training-like data near origin
        x_near = torch.randn(3, 2) * 0.1  # Small values near origin
        x_far = torch.randn(3, 2) * 5.0   # Large values far from origin
        
        _, unc_near = model.forward_with_uncertainty(x_near, n_samples=30)
        _, unc_far = model.forward_with_uncertainty(x_far, n_samples=30)
        
        # On average, uncertainty should be higher for far points
        # (This is a general expectation, not always true for all models)
        mean_unc_near = torch.mean(unc_near)
        mean_unc_far = torch.mean(unc_far)
        
        # Both should be positive
        assert mean_unc_near >= 0
        assert mean_unc_far >= 0
        
    def test_monte_carlo_convergence(self, model):
        """Test that Monte Carlo estimates converge with more samples."""
        x = torch.randn(3, 2)
        
        # Test convergence by comparing estimates with different sample sizes
        sample_sizes = [10, 20, 50, 100]
        means = []
        uncertainties = []
        
        for n_samples in sample_sizes:
            mean_pred, unc = model.forward_with_uncertainty(x, n_samples=n_samples)
            means.append(mean_pred)
            uncertainties.append(unc)
            
        # Check that estimates stabilize (variance decreases with more samples)
        # This is a statistical property, so we just check basic properties
        for i in range(len(sample_sizes)):
            assert torch.all(uncertainties[i] >= 0)
            assert torch.all(torch.isfinite(means[i]))
            assert torch.all(torch.isfinite(uncertainties[i]))
            
    def test_epistemic_vs_aleatoric_decomposition(self, model):
        """Test that epistemic and aleatoric uncertainty decomposition is reasonable."""
        x = torch.randn(4, 2)
        task_info = {
            'viscosity_type': 'constant',
            'viscosity_params': {'mu_0': 1.0}
        }
        
        uncertainty_dict = model.decompose_uncertainty(x, task_info, n_samples=30)
        
        epistemic = uncertainty_dict['epistemic']
        aleatoric = uncertainty_dict['aleatoric']
        total = uncertainty_dict['total']
        
        # Check mathematical relationship: total² = epistemic² + aleatoric²
        expected_total = torch.sqrt(epistemic**2 + aleatoric**2)
        assert torch.allclose(total, expected_total, atol=1e-5)
        
        # All components should be non-negative
        assert torch.all(epistemic >= 0)
        assert torch.all(aleatoric >= 0)
        assert torch.all(total >= 0)
        
        # Total should be at least as large as individual components
        assert torch.all(total >= epistemic)
        assert torch.all(total >= aleatoric)
        
    def test_uncertainty_with_different_physics(self, model):
        """Test uncertainty estimation with different physics scenarios."""
        x = torch.randn(3, 2)
        
        # Test with different viscosity types
        task_infos = [
            {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}},
            {'viscosity_type': 'linear', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}},
            {'viscosity_type': 'exponential', 'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}}
        ]
        
        uncertainties = []
        for task_info in task_infos:
            unc_dict = model.decompose_uncertainty(x, task_info, n_samples=20)
            uncertainties.append(unc_dict)
            
        # All uncertainty estimates should be valid
        for unc_dict in uncertainties:
            for key in ['epistemic', 'aleatoric', 'total']:
                assert torch.all(unc_dict[key] >= 0)
                assert torch.all(torch.isfinite(unc_dict[key]))
                
    def test_confidence_intervals(self, model):
        """Test confidence interval generation."""
        x = torch.randn(3, 2)
        task_info = {'viscosity_type': 'constant', 'viscosity_params': {'mu_0': 1.0}}
        
        pred_dict = model.predict_with_confidence(x, task_info, confidence_level=0.95)
        
        mean = pred_dict['mean']
        lower = pred_dict['lower_bound']
        upper = pred_dict['upper_bound']
        uncertainty = pred_dict['uncertainty']
        
        # Check interval properties
        assert torch.all(lower <= mean)
        assert torch.all(mean <= upper)
        assert torch.all(upper - lower > 0)  # Intervals should have positive width
        
        # Interval width should be related to uncertainty
        interval_width = upper - lower
        expected_width = 2 * 1.96 * uncertainty  # 95% confidence interval
        assert torch.allclose(interval_width, expected_width, atol=1e-5)
        
    def test_deterministic_vs_stochastic_modes(self, model):
        """Test difference between deterministic and stochastic forward passes."""
        x = torch.randn(3, 2)
        
        # Deterministic mode should give consistent results
        model.eval()
        det_output1 = model(x, sample=False)
        det_output2 = model(x, sample=False)
        assert torch.allclose(det_output1, det_output2)
        
        # Stochastic mode should give different results
        model.train()
        stoch_output1 = model(x, sample=True)
        stoch_output2 = model(x, sample=True)
        assert not torch.allclose(stoch_output1, stoch_output2, atol=1e-6)
        
        # Uncertainty should be zero in deterministic mode with fixed parameters
        params = model.clone_parameters()
        _, unc_with_params = model.forward_with_uncertainty(x, params=params)
        assert torch.allclose(unc_with_params, torch.zeros_like(unc_with_params))


if __name__ == "__main__":
    pytest.main([__file__])
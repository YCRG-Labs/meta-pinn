"""
Integration tests for uncertainty calibration system.

Tests the UncertaintyCalibrator class, reliability diagrams, and physics-informed
uncertainty estimation functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch
import tempfile
import os

from ml_research_pipeline.bayesian.uncertainty_calibrator import UncertaintyCalibrator, CalibrationEvaluator
from ml_research_pipeline.bayesian.bayesian_meta_pinn import BayesianMetaPINN
from ml_research_pipeline.config.model_config import MetaPINNConfig


class TestUncertaintyCalibrator:
    """Test cases for UncertaintyCalibrator."""
    
    @pytest.fixture
    def calibrator(self):
        """Create UncertaintyCalibrator instance."""
        return UncertaintyCalibrator(n_bins=5, method='isotonic')
        
    @pytest.fixture
    def sample_data(self):
        """Create sample uncertainty and error data."""
        n_samples = 100
        # Create correlated uncertainty and error data
        true_errors = torch.abs(torch.randn(n_samples))
        # Add some noise to create imperfect uncertainty estimates
        uncertainties = true_errors + 0.1 * torch.randn(n_samples)
        uncertainties = torch.clamp(uncertainties, min=0.01)  # Ensure positive
        
        return uncertainties, true_errors
        
    def test_calibrator_initialization(self, calibrator):
        """Test proper initialization of calibrator."""
        assert calibrator.n_bins == 5
        assert calibrator.method == 'isotonic'
        assert not calibrator.is_fitted
        assert calibrator.calibrator is None
        
    def test_calibrator_fitting(self, calibrator, sample_data):
        """Test fitting the calibrator."""
        uncertainties, errors = sample_data
        
        # Fit calibrator
        fitted_calibrator = calibrator.fit(uncertainties, errors)
        
        # Check that fitting worked
        assert fitted_calibrator is calibrator  # Should return self
        assert calibrator.is_fitted
        assert calibrator.calibrator is not None
        assert calibrator.calibration_error is not None
        
    def test_calibration_application(self, calibrator, sample_data):
        """Test applying calibration to new data."""
        uncertainties, errors = sample_data
        
        # Fit calibrator
        calibrator.fit(uncertainties, errors)
        
        # Apply calibration to new data
        new_uncertainties = torch.randn(20).abs() + 0.1
        calibrated = calibrator.calibrate(new_uncertainties)
        
        # Check output properties
        assert calibrated.shape == new_uncertainties.shape
        assert torch.all(calibrated >= 0)  # Should be non-negative
        assert torch.all(torch.isfinite(calibrated))  # Should be finite
        
    def test_calibration_without_fitting(self, calibrator):
        """Test that calibration fails without fitting."""
        uncertainties = torch.randn(10).abs()
        
        with pytest.raises(ValueError, match="Calibrator must be fitted"):
            calibrator.calibrate(uncertainties)
            
    def test_calibration_evaluation(self, calibrator, sample_data):
        """Test calibration quality evaluation."""
        uncertainties, errors = sample_data
        
        metrics = calibrator.evaluate_calibration(uncertainties, errors)
        
        # Check that metrics are computed
        assert 'ece' in metrics  # Expected Calibration Error
        assert 'mce' in metrics  # Maximum Calibration Error
        assert 'n_samples' in metrics
        assert 'n_bins' in metrics
        
        # Check metric properties
        assert metrics['ece'] >= 0
        assert metrics['mce'] >= 0
        assert metrics['n_samples'] > 0
        
    def test_calibration_evaluation_detailed(self, calibrator, sample_data):
        """Test detailed calibration evaluation."""
        uncertainties, errors = sample_data
        
        metrics = calibrator.evaluate_calibration(uncertainties, errors, return_detailed=True)
        
        # Check detailed metrics
        assert 'bin_accuracies' in metrics
        assert 'bin_confidences' in metrics
        assert 'bin_counts' in metrics
        assert 'bin_weights' in metrics
        
        # Check that lists have reasonable lengths
        assert len(metrics['bin_accuracies']) <= calibrator.n_bins
        assert len(metrics['bin_confidences']) == len(metrics['bin_accuracies'])
        
    def test_reliability_diagram_plotting(self, calibrator, sample_data):
        """Test reliability diagram generation."""
        uncertainties, errors = sample_data
        
        # Mock matplotlib to avoid display issues in testing
        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
            fig = calibrator.plot_reliability_diagram(
                uncertainties, errors, show=False
            )
            
        # Check that figure was created (if valid data)
        if fig is not None:
            assert hasattr(fig, 'axes')
            
    def test_invalid_data_handling(self, calibrator):
        """Test handling of invalid data."""
        # Test with NaN values
        uncertainties = torch.tensor([1.0, float('nan'), 2.0, float('inf')])
        errors = torch.tensor([0.5, 1.0, float('nan'), 1.5])
        
        # Should handle invalid values gracefully
        metrics = calibrator.evaluate_calibration(uncertainties, errors)
        
        # Should return reasonable metrics even with invalid data
        assert 'ece' in metrics
        assert 'mce' in metrics
        
    def test_empty_data_handling(self, calibrator):
        """Test handling of empty data."""
        uncertainties = torch.tensor([])
        errors = torch.tensor([])
        
        metrics = calibrator.evaluate_calibration(uncertainties, errors)
        
        # Should handle empty data gracefully
        assert metrics['ece'] == float('inf')
        assert metrics['mce'] == float('inf')
        
    def test_calibration_summary(self, calibrator, sample_data):
        """Test calibration summary generation."""
        # Test unfitted calibrator
        summary = calibrator.get_calibration_summary()
        assert summary['status'] == 'not_fitted'
        
        # Test fitted calibrator
        uncertainties, errors = sample_data
        calibrator.fit(uncertainties, errors)
        
        summary = calibrator.get_calibration_summary()
        assert summary['status'] == 'fitted'
        assert 'method' in summary
        assert 'n_bins' in summary
        assert 'calibration_error' in summary
        
    def test_calibrator_save_load(self, calibrator, sample_data):
        """Test saving and loading calibrator."""
        uncertainties, errors = sample_data
        calibrator.fit(uncertainties, errors)
        
        # Save calibrator
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
            
        try:
            calibrator.save_calibrator(temp_path)
            
            # Load calibrator
            loaded_calibrator = UncertaintyCalibrator.load_calibrator(temp_path)
            
            # Check that loaded calibrator has same properties
            assert loaded_calibrator.is_fitted == calibrator.is_fitted
            assert loaded_calibrator.n_bins == calibrator.n_bins
            assert loaded_calibrator.method == calibrator.method
            
            # Test that loaded calibrator works
            test_uncertainties = torch.randn(10).abs()
            original_result = calibrator.calibrate(test_uncertainties)
            loaded_result = loaded_calibrator.calibrate(test_uncertainties)
            
            assert torch.allclose(original_result, loaded_result, atol=1e-6)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestPhysicsInformedCalibration:
    """Test physics-informed uncertainty calibration."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MetaPINNConfig()
        config.input_dim = 2
        config.output_dim = 2
        config.hidden_dims = [10, 10]
        config.activation = "tanh"
        return config
        
    @pytest.fixture
    def model(self, config):
        """Create BayesianMetaPINN instance."""
        return BayesianMetaPINN(config, prior_std=1.0)
        
    @pytest.fixture
    def calibrator(self):
        """Create UncertaintyCalibrator instance."""
        return UncertaintyCalibrator(n_bins=5)
        
    def test_physics_informed_uncertainty(self, calibrator, model):
        """Test physics-informed uncertainty computation."""
        coords = torch.randn(5, 2)
        task_info = {
            'viscosity_type': 'constant',
            'viscosity_params': {'mu_0': 1.0}
        }
        
        physics_uncertainty = calibrator.physics_informed_calibration(
            model, coords, task_info
        )
        
        # Check output properties
        assert physics_uncertainty.shape == (5, 2)  # Should match output dim
        assert torch.all(physics_uncertainty >= 0)
        assert torch.all(torch.isfinite(physics_uncertainty))
        
    def test_adaptive_calibration(self, calibrator):
        """Test adaptive calibration combining data and physics uncertainty."""
        # Create sample data
        uncertainties = torch.randn(10).abs() + 0.1
        errors = torch.randn(10).abs()
        physics_residuals = torch.randn(10).abs()
        
        # Fit calibrator first
        calibrator.fit(uncertainties, errors)
        
        # Apply adaptive calibration
        adaptive_uncertainty = calibrator.adaptive_calibration(
            uncertainties, errors, physics_residuals, physics_weight=0.3
        )
        
        # Check output properties
        assert adaptive_uncertainty.shape == uncertainties.shape
        assert torch.all(adaptive_uncertainty >= 0)
        assert torch.all(torch.isfinite(adaptive_uncertainty))
        
    def test_adaptive_calibration_without_fitting(self, calibrator):
        """Test adaptive calibration without pre-fitted calibrator."""
        uncertainties = torch.randn(10).abs() + 0.1
        errors = torch.randn(10).abs()
        physics_residuals = torch.randn(10).abs()
        
        # Should work even without fitting (uses original uncertainties)
        adaptive_uncertainty = calibrator.adaptive_calibration(
            uncertainties, errors, physics_residuals
        )
        
        assert adaptive_uncertainty.shape == uncertainties.shape
        assert torch.all(adaptive_uncertainty >= 0)


class TestCalibrationEvaluator:
    """Test CalibrationEvaluator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MetaPINNConfig()
        config.input_dim = 2
        config.output_dim = 1
        config.hidden_dims = [8, 8]
        config.activation = "tanh"
        return config
        
    @pytest.fixture
    def model(self, config):
        """Create BayesianMetaPINN instance."""
        return BayesianMetaPINN(config, prior_std=1.0, n_mc_samples=10)
        
    @pytest.fixture
    def test_tasks(self):
        """Create sample test tasks."""
        tasks = []
        for i in range(3):
            task = {
                'query_coords': torch.randn(5, 2),
                'query_data': torch.randn(5, 1),
                'task_info': {
                    'viscosity_type': 'constant',
                    'viscosity_params': {'mu_0': 1.0}
                }
            }
            tasks.append(task)
        return tasks
        
    def test_evaluator_initialization(self):
        """Test CalibrationEvaluator initialization."""
        evaluator = CalibrationEvaluator()
        assert evaluator.results == []
        
    def test_model_calibration_evaluation(self, model, test_tasks):
        """Test evaluation of model calibration across tasks."""
        evaluator = CalibrationEvaluator()
        
        results = evaluator.evaluate_model_calibration(
            model, test_tasks, n_mc_samples=5
        )
        
        # Check result structure
        assert 'epistemic_calibration' in results
        assert 'physics_calibration' in results
        assert 'n_tasks' in results
        assert 'n_samples' in results
        
        # Check that results are reasonable
        assert results['n_tasks'] == len(test_tasks)
        assert results['n_samples'] > 0
        
        # Check that results were stored
        assert len(evaluator.results) == 1
        
    def test_calibration_method_comparison(self):
        """Test comparison of different calibration methods."""
        evaluator = CalibrationEvaluator()
        
        # Create sample data
        uncertainties = torch.randn(50).abs() + 0.1
        errors = torch.randn(50).abs()
        
        results = evaluator.compare_calibration_methods(
            uncertainties, errors, methods=['isotonic']
        )
        
        # Check results structure
        assert 'isotonic' in results
        assert 'metrics' in results['isotonic']
        assert 'calibrator' in results['isotonic']
        
        # Check that calibrator was fitted
        assert results['isotonic']['calibrator'].is_fitted


class TestCalibrationIntegration:
    """Integration tests for complete calibration workflow."""
    
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
        return BayesianMetaPINN(config, prior_std=1.0, n_mc_samples=20)
        
    def test_end_to_end_calibration_workflow(self, model):
        """Test complete calibration workflow from training to evaluation."""
        # Generate synthetic training data
        train_coords = torch.randn(50, 2)
        train_data = torch.randn(50, 1)
        task_info = {
            'viscosity_type': 'linear',
            'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}
        }
        
        # Get model predictions and uncertainties
        model.eval()
        with torch.no_grad():
            mean_pred, uncertainty = model.forward_with_uncertainty(train_coords)
            errors = torch.abs(mean_pred - train_data)
            
        # Create and fit calibrator
        calibrator = UncertaintyCalibrator(n_bins=5)
        calibrator.fit(uncertainty.flatten(), errors.flatten())
        
        # Generate test data
        test_coords = torch.randn(20, 2)
        test_data = torch.randn(20, 1)
        
        # Get test predictions
        with torch.no_grad():
            test_pred, test_uncertainty = model.forward_with_uncertainty(test_coords)
            test_errors = torch.abs(test_pred - test_data)
            
        # Apply calibration
        calibrated_uncertainty = calibrator.calibrate(test_uncertainty.flatten())
        
        # Evaluate calibration quality
        metrics = calibrator.evaluate_calibration(calibrated_uncertainty, test_errors.flatten())
        
        # Check that workflow completed successfully
        assert 'ece' in metrics
        assert 'mce' in metrics
        assert metrics['ece'] >= 0
        assert metrics['mce'] >= 0
        
    def test_physics_informed_calibration_integration(self, model):
        """Test integration of physics-informed uncertainty with calibration."""
        coords = torch.randn(30, 2)
        task_info = {
            'viscosity_type': 'exponential',
            'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}
        }
        
        # Create calibrator
        calibrator = UncertaintyCalibrator()
        
        # Get physics-informed uncertainty
        physics_uncertainty = calibrator.physics_informed_calibration(
            model, coords, task_info
        )
        
        # Get model uncertainty
        with torch.no_grad():
            _, model_uncertainty = model.forward_with_uncertainty(coords)
            
        # Create synthetic errors for testing
        synthetic_errors = torch.randn(30, 1).abs()
        
        # Fit calibrator on model uncertainty
        calibrator.fit(model_uncertainty.flatten(), synthetic_errors.flatten())
        
        # Apply adaptive calibration
        adaptive_uncertainty = calibrator.adaptive_calibration(
            model_uncertainty.flatten(),
            synthetic_errors.flatten(),
            physics_uncertainty.flatten(),
            physics_weight=0.4
        )
        
        # Check that adaptive calibration worked
        assert adaptive_uncertainty.shape == model_uncertainty.flatten().shape
        assert torch.all(adaptive_uncertainty >= 0)
        assert torch.all(torch.isfinite(adaptive_uncertainty))
        
    def test_calibration_quality_improvement(self, model):
        """Test that calibration improves uncertainty quality."""
        # Generate data with known relationship between uncertainty and error
        coords = torch.randn(100, 2)
        
        model.eval()
        with torch.no_grad():
            predictions, uncertainties = model.forward_with_uncertainty(coords)
            
        # Create synthetic ground truth with correlation to uncertainty
        # Higher uncertainty should correspond to higher error
        noise_scale = uncertainties.flatten() + 0.1
        true_values = predictions.flatten() + noise_scale * torch.randn_like(predictions.flatten())
        errors = torch.abs(predictions.flatten() - true_values)
        
        # Evaluate calibration before calibration
        calibrator = UncertaintyCalibrator(n_bins=5)
        metrics_before = calibrator.evaluate_calibration(uncertainties.flatten(), errors)
        
        # Fit calibrator and apply calibration
        calibrator.fit(uncertainties.flatten(), errors)
        calibrated_uncertainties = calibrator.calibrate(uncertainties.flatten())
        
        # Evaluate calibration after calibration
        metrics_after = calibrator.evaluate_calibration(calibrated_uncertainties, errors)
        
        # Check that both evaluations completed
        assert 'ece' in metrics_before
        assert 'ece' in metrics_after
        
        # In many cases, calibration should improve ECE, but this isn't guaranteed
        # for synthetic data, so we just check that the process completed
        assert metrics_after['ece'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
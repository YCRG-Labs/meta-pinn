"""
Unit tests for sample complexity analysis.
"""

import pytest
import numpy as np
import math
from theory.sample_complexity import (
    SampleComplexityAnalyzer, 
    ComplexityParameters,
    EmpiricalValidator,
    compute_physics_constraint_benefit,
    compute_meta_learning_benefit
)


class TestSampleComplexityAnalyzer:
    """Test sample complexity theoretical bounds."""
    
    @pytest.fixture
    def complexity_params(self):
        """Standard complexity parameters for testing."""
        return ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
    
    @pytest.fixture
    def analyzer(self, complexity_params):
        """Sample complexity analyzer instance."""
        return SampleComplexityAnalyzer(complexity_params)
    
    def test_traditional_bound_computation(self, analyzer):
        """Test traditional learning bound computation."""
        bound = analyzer.compute_traditional_bound(network_width=64, network_depth=3)
        
        # Bound should be positive and finite
        assert bound > 0
        assert math.isfinite(bound)
        
        # Bound should increase with network size
        larger_bound = analyzer.compute_traditional_bound(network_width=128, network_depth=3)
        assert larger_bound > bound
        
    def test_physics_informed_bound_computation(self, analyzer):
        """Test physics-informed learning bound computation."""
        bound = analyzer.compute_physics_informed_bound(network_width=64, network_depth=3)
        
        # Bound should be positive and finite
        assert bound > 0
        assert math.isfinite(bound)
        
        # Should be better than traditional bound
        traditional_bound = analyzer.compute_traditional_bound(network_width=64, network_depth=3)
        assert bound < traditional_bound
        
    def test_meta_learning_bound_computation(self, analyzer):
        """Test meta-learning bound computation."""
        bound = analyzer.compute_meta_learning_bound(
            n_tasks=100, n_support=10, network_width=64, network_depth=3
        )
        
        # Bound should be positive and finite
        assert bound > 0
        assert math.isfinite(bound)
        
        # Should improve with more tasks
        fewer_tasks_bound = analyzer.compute_meta_learning_bound(
            n_tasks=10, n_support=10, network_width=64, network_depth=3
        )
        assert bound < fewer_tasks_bound
        
    def test_improvement_factor_computation(self, analyzer):
        """Test improvement factor calculation."""
        factor = analyzer.compute_improvement_factor(network_width=64, network_depth=3)
        
        # Should show improvement (factor > 1)
        assert factor > 1
        assert math.isfinite(factor)
        
        # Should increase with stronger physics constraints
        strong_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.8,  # Stronger constraints
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        strong_analyzer = SampleComplexityAnalyzer(strong_params)
        strong_factor = strong_analyzer.compute_improvement_factor(network_width=64, network_depth=3)
        assert strong_factor > factor
        
    def test_comprehensive_analysis(self, analyzer):
        """Test comprehensive sample complexity analysis."""
        bounds = analyzer.analyze_sample_complexity(
            network_width=64, network_depth=3, n_tasks=100
        )
        
        # Check all fields are present and valid
        assert bounds.physics_informed_bound > 0
        assert bounds.traditional_bound > 0
        assert bounds.improvement_factor > 1
        assert 0 < bounds.confidence_level < 1
        assert len(bounds.assumptions) > 0
        
        # Physics-informed should be better
        assert bounds.physics_informed_bound < bounds.traditional_bound
        
    def test_parameter_sensitivity(self):
        """Test sensitivity to different parameters."""
        base_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        
        # Test dimension sensitivity
        high_dim_params = ComplexityParameters(
            dimension=10,  # Higher dimension
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        
        base_analyzer = SampleComplexityAnalyzer(base_params)
        high_dim_analyzer = SampleComplexityAnalyzer(high_dim_params)
        
        base_bound = base_analyzer.compute_traditional_bound(64, 3)
        high_dim_bound = high_dim_analyzer.compute_traditional_bound(64, 3)
        
        # Higher dimension should require more samples
        assert high_dim_bound > base_bound
        
    def test_noise_level_impact(self):
        """Test impact of noise level on bounds."""
        low_noise_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.01,  # Low noise
            confidence_delta=0.05,
            approximation_error=0.01
        )
        
        high_noise_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.5,  # High noise
            confidence_delta=0.05,
            approximation_error=0.01
        )
        
        low_noise_analyzer = SampleComplexityAnalyzer(low_noise_params)
        high_noise_analyzer = SampleComplexityAnalyzer(high_noise_params)
        
        low_noise_bound = low_noise_analyzer.compute_traditional_bound(64, 3)
        high_noise_bound = high_noise_analyzer.compute_traditional_bound(64, 3)
        
        # Higher noise should require more samples
        assert high_noise_bound > low_noise_bound


class TestEmpiricalValidator:
    """Test empirical validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Empirical validator instance."""
        return EmpiricalValidator()
    
    @pytest.fixture
    def sample_data(self, validator):
        """Generate sample empirical data."""
        # Simulate learning curves: error = A * n^(-β)
        A_traditional = 10.0
        beta_traditional = 0.4
        A_physics = 5.0
        beta_physics = 0.6
        
        sample_sizes = [50, 100, 200, 500, 1000, 2000]
        
        for n in sample_sizes:
            # Add some noise to make it realistic
            noise = np.random.normal(0, 0.01)
            
            traditional_error = A_traditional * (n ** (-beta_traditional)) + noise
            physics_error = A_physics * (n ** (-beta_physics)) + noise
            
            validator.record_empirical_result("traditional", n, traditional_error)
            validator.record_empirical_result("physics_informed", n, physics_error)
            
        return validator
    
    def test_empirical_data_recording(self, validator):
        """Test recording of empirical results."""
        validator.record_empirical_result("test_method", 100, 0.1)
        validator.record_empirical_result("test_method", 200, 0.05)
        
        assert "test_method" in validator.empirical_results
        assert len(validator.empirical_results["test_method"]["default"]["samples"]) == 2
        assert len(validator.empirical_results["test_method"]["default"]["errors"]) == 2
        
    def test_empirical_curve_fitting(self, sample_data):
        """Test fitting of empirical learning curves."""
        A, beta = sample_data.fit_empirical_curve("traditional")
        
        # Should recover reasonable parameters
        assert A > 0
        assert 0.1 < beta < 1.0  # Reasonable convergence rate
        
        # Physics-informed should have better convergence rate
        A_physics, beta_physics = sample_data.fit_empirical_curve("physics_informed")
        assert beta_physics > beta  # Faster convergence
        
    def test_theoretical_validation(self, sample_data):
        """Test validation against theoretical predictions."""
        complexity_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        analyzer = SampleComplexityAnalyzer(complexity_params)
        
        validation_results = sample_data.validate_theoretical_prediction(
            analyzer, network_width=64, network_depth=3, method="traditional"
        )
        
        # Check validation metrics
        assert "convergence_rate_error" in validation_results
        assert "average_bound_error" in validation_results
        assert "empirical_rate" in validation_results
        assert "theoretical_rate" in validation_results
        
        # Validation errors should be reasonable
        assert validation_results["convergence_rate_error"] < 1.0  # Within 100%
        assert validation_results["average_bound_error"] < 2.0  # Within 200%
        
    def test_missing_data_handling(self, validator):
        """Test handling of missing empirical data."""
        with pytest.raises(ValueError):
            validator.fit_empirical_curve("nonexistent_method")
            
        with pytest.raises(ValueError):
            validator.validate_theoretical_prediction(
                None, 64, 3, "nonexistent_method"
            )


class TestUtilityFunctions:
    """Test utility functions for sample complexity analysis."""
    
    def test_physics_constraint_benefit(self):
        """Test physics constraint benefit computation."""
        # No constraints should give no benefit
        no_benefit = compute_physics_constraint_benefit(0.0, 10)
        assert no_benefit == 1.0
        
        # Strong constraints should give significant benefit
        strong_benefit = compute_physics_constraint_benefit(0.8, 10)
        assert strong_benefit > 1.0
        
        # Benefit should increase with constraint strength
        weak_benefit = compute_physics_constraint_benefit(0.2, 10)
        assert strong_benefit > weak_benefit
        
    def test_meta_learning_benefit(self):
        """Test meta-learning benefit computation."""
        # No tasks should give no benefit
        no_benefit = compute_meta_learning_benefit(1, 0.5)
        assert no_benefit > 0
        
        # More tasks should give more benefit
        few_tasks_benefit = compute_meta_learning_benefit(10, 0.5)
        many_tasks_benefit = compute_meta_learning_benefit(100, 0.5)
        assert many_tasks_benefit > few_tasks_benefit
        
        # Higher task similarity should give more benefit
        low_sim_benefit = compute_meta_learning_benefit(100, 0.2)
        high_sim_benefit = compute_meta_learning_benefit(100, 0.8)
        assert high_sim_benefit > low_sim_benefit


class TestBoundAccuracy:
    """Test accuracy of theoretical bounds."""
    
    def test_bound_monotonicity(self):
        """Test that bounds behave monotonically with parameters."""
        base_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        analyzer = SampleComplexityAnalyzer(base_params)
        
        # Bounds should increase with network size
        small_bound = analyzer.compute_traditional_bound(32, 2)
        large_bound = analyzer.compute_traditional_bound(128, 4)
        assert large_bound > small_bound
        
        # Bounds should decrease with stronger physics constraints
        weak_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.1,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        weak_analyzer = SampleComplexityAnalyzer(weak_params)
        
        weak_bound = weak_analyzer.compute_physics_informed_bound(64, 3)
        strong_bound = analyzer.compute_physics_informed_bound(64, 3)
        assert weak_bound > strong_bound
        
    def test_bound_scaling(self):
        """Test proper scaling of bounds with problem parameters."""
        params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        analyzer = SampleComplexityAnalyzer(params)
        
        # Test error scaling
        base_bound = analyzer.compute_traditional_bound(64, 3)
        
        # Halving target error should roughly quadruple sample requirement
        tight_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=1.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.005  # Half the error
        )
        tight_analyzer = SampleComplexityAnalyzer(tight_params)
        tight_bound = tight_analyzer.compute_traditional_bound(64, 3)
        
        # Should be roughly 4x larger (within factor of 2 due to log terms)
        assert 2.0 < tight_bound / base_bound < 8.0


if __name__ == "__main__":
    pytest.main([__file__])
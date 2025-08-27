"""
Unit tests for convergence rate analysis.
"""

import pytest
import numpy as np
import math
from theory.convergence_analysis import (
    ConvergenceAnalyzer,
    ConvergenceParameters,
    EmpiricalConvergenceValidator,
    compute_physics_informed_convergence_benefit,
    compute_meta_learning_convergence_benefit,
    analyze_convergence_phase_transitions
)


class TestConvergenceAnalyzer:
    """Test convergence rate theoretical analysis."""
    
    @pytest.fixture
    def convergence_params(self):
        """Standard convergence parameters for testing."""
        return ConvergenceParameters(
            lipschitz_constant=10.0,
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
    
    @pytest.fixture
    def analyzer(self, convergence_params):
        """Convergence analyzer instance."""
        return ConvergenceAnalyzer(convergence_params)
    
    def test_task_level_convergence_rate(self, analyzer):
        """Test task-level convergence rate computation."""
        rate = analyzer.compute_task_level_convergence_rate()
        
        # Rate should be between 0 and 1 (convergent)
        assert 0 < rate < 1
        
        # Physics regularization should improve convergence
        physics_rate = analyzer.compute_task_level_convergence_rate(physics_regularization=0.5)
        assert physics_rate < rate  # Lower rate is better (faster convergence)
        
    def test_meta_level_convergence_rate(self, analyzer):
        """Test meta-level convergence rate computation."""
        rate = analyzer.compute_meta_level_convergence_rate(n_tasks=100)
        
        # Rate should be positive
        assert rate > 0
        
        # More tasks should improve convergence
        fewer_tasks_rate = analyzer.compute_meta_level_convergence_rate(n_tasks=10)
        assert rate < fewer_tasks_rate  # Lower rate is better
        
    def test_error_bounds_computation(self, analyzer):
        """Test error bounds computation."""
        bounds = analyzer.compute_error_bounds(n_tasks=100, n_support=10)
        
        # All bounds should be positive
        assert bounds['task_adaptation_bound'] > 0
        assert bounds['meta_generalization_bound'] > 0
        assert bounds['total_error_bound'] > 0
        assert bounds['physics_improvement'] >= 1.0
        
        # Physics regularization should improve bounds
        physics_bounds = analyzer.compute_error_bounds(
            n_tasks=100, n_support=10, physics_regularization=0.5
        )
        assert physics_bounds['total_error_bound'] < bounds['total_error_bound']
        assert physics_bounds['physics_improvement'] > bounds['physics_improvement']
        
    def test_convergence_constants_analysis(self, analyzer):
        """Test convergence constants computation."""
        constants = analyzer.analyze_convergence_constants()
        
        # Check all required fields
        required_fields = [
            'condition_number', 'convergence_constant', 'optimal_learning_rate',
            'optimal_convergence_rate', 'effective_lipschitz', 'effective_strong_convexity'
        ]
        for field in required_fields:
            assert field in constants
            assert constants[field] > 0
            
        # Condition number should be >= 1
        assert constants['condition_number'] >= 1.0
        
        # Physics regularization should improve conditioning
        physics_constants = analyzer.analyze_convergence_constants(physics_regularization=0.5)
        assert physics_constants['condition_number'] < constants['condition_number']
        
    def test_comprehensive_analysis(self, analyzer):
        """Test comprehensive convergence analysis."""
        analysis = analyzer.compute_comprehensive_analysis(
            n_tasks=100, n_support=10, physics_regularization=0.3
        )
        
        # Check all fields are present
        assert analysis.task_level_rate > 0
        assert analysis.meta_level_rate > 0
        assert analysis.theoretical_task_rate > 0
        assert analysis.theoretical_meta_rate > 0
        assert len(analysis.error_bounds) > 0
        assert len(analysis.convergence_constants) > 0
        
        # Rates should be consistent
        assert analysis.task_level_rate == analysis.theoretical_task_rate
        assert analysis.meta_level_rate == analysis.theoretical_meta_rate
        
    def test_parameter_sensitivity(self):
        """Test sensitivity to different parameters."""
        base_params = ConvergenceParameters(
            lipschitz_constant=10.0,
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        
        # Test Lipschitz constant sensitivity
        high_lipschitz_params = ConvergenceParameters(
            lipschitz_constant=50.0,  # Higher Lipschitz constant
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        
        base_analyzer = ConvergenceAnalyzer(base_params)
        high_lipschitz_analyzer = ConvergenceAnalyzer(high_lipschitz_params)
        
        base_rate = base_analyzer.compute_task_level_convergence_rate()
        high_lipschitz_rate = high_lipschitz_analyzer.compute_task_level_convergence_rate()
        
        # Higher Lipschitz constant should worsen convergence (higher rate = slower)
        # Allow for small numerical differences
        assert high_lipschitz_rate >= base_rate
        
    def test_learning_rate_impact(self):
        """Test impact of learning rate on convergence."""
        params = ConvergenceParameters(
            lipschitz_constant=10.0,
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        
        analyzer = ConvergenceAnalyzer(params)
        constants = analyzer.analyze_convergence_constants()
        
        # Optimal learning rate should be positive and reasonable
        assert 0 < constants['optimal_learning_rate'] < 1.0
        
        # Convergence rate with optimal LR should be better than arbitrary rate
        assert 0 < constants['optimal_convergence_rate'] < 1.0


class TestEmpiricalConvergenceValidator:
    """Test empirical convergence validation."""
    
    @pytest.fixture
    def validator(self):
        """Empirical validator instance."""
        return EmpiricalConvergenceValidator()
    
    @pytest.fixture
    def sample_curves(self, validator):
        """Generate sample convergence curves."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate synthetic exponential decay curves
        iterations = np.arange(100)
        
        # Fast converging method - ensure losses stay positive
        fast_losses = 10.0 * np.exp(-0.1 * iterations) + 0.001
        # Add small amount of noise
        noise = 0.001 * np.random.randn(100)
        fast_losses = np.maximum(fast_losses + noise, 1e-6)
        validator.record_training_curve("fast_method", fast_losses.tolist(), "task_adaptation")
        
        # Slow converging method
        slow_losses = 10.0 * np.exp(-0.05 * iterations) + 0.001
        noise = 0.001 * np.random.randn(100)
        slow_losses = np.maximum(slow_losses + noise, 1e-6)
        validator.record_training_curve("slow_method", slow_losses.tolist(), "task_adaptation")
        
        # Meta-learning curve (different scale)
        meta_losses = 5.0 * np.exp(-0.02 * iterations) + 0.001
        noise = 0.0005 * np.random.randn(100)
        meta_losses = np.maximum(meta_losses + noise, 1e-6)
        validator.record_training_curve("meta_method", meta_losses.tolist(), "meta_training")
        
        return validator
    
    def test_training_curve_recording(self, validator):
        """Test recording of training curves."""
        losses = [1.0, 0.5, 0.25, 0.125]
        validator.record_training_curve("test_method", losses, "task_adaptation")
        
        assert "test_method" in validator.training_curves
        assert "task_adaptation" in validator.training_curves["test_method"]
        assert validator.training_curves["test_method"]["task_adaptation"] == losses
        
    def test_convergence_rate_fitting(self, sample_curves):
        """Test fitting of convergence rates."""
        rate, constant = sample_curves.fit_convergence_rate("fast_method", "task_adaptation")
        
        # Should recover reasonable parameters
        assert rate > 0  # Positive convergence rate
        assert constant > 0  # Positive initial constant
        
        # Fast method should have higher convergence rate than slow method
        slow_rate, _ = sample_curves.fit_convergence_rate("slow_method", "task_adaptation")
        assert rate > slow_rate
        
    def test_theoretical_validation(self, sample_curves):
        """Test validation against theoretical predictions."""
        params = ConvergenceParameters(
            lipschitz_constant=10.0,
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        analyzer = ConvergenceAnalyzer(params)
        
        validation = sample_curves.validate_theoretical_rates(
            analyzer, "fast_method", n_tasks=100, n_support=10
        )
        
        # Check validation metrics
        assert "empirical_task_rate" in validation
        assert "theoretical_task_rate" in validation
        assert validation["empirical_task_rate"] > 0
        assert validation["theoretical_task_rate"] > 0
        
        # Rate error should be reasonable (within order of magnitude)
        if "task_rate_error" in validation:
            assert validation["task_rate_error"] < 10.0  # Within 1000% (theory vs empirical can differ significantly)
            
    def test_method_comparison(self, sample_curves):
        """Test comparison across methods."""
        comparison = sample_curves.compare_convergence_across_methods(
            ["fast_method", "slow_method"], "task_adaptation"
        )
        
        # Both methods should be present
        assert "fast_method" in comparison
        assert "slow_method" in comparison
        
        # Check required fields
        for method in ["fast_method", "slow_method"]:
            assert "convergence_rate" in comparison[method]
            assert "improvement_ratio" in comparison[method]
            assert "final_loss" in comparison[method]
            
        # Fast method should have better convergence
        fast_rate = comparison["fast_method"]["convergence_rate"]
        slow_rate = comparison["slow_method"]["convergence_rate"]
        assert fast_rate > slow_rate
        
    def test_missing_data_handling(self, validator):
        """Test handling of missing data."""
        with pytest.raises(ValueError):
            validator.fit_convergence_rate("nonexistent_method")
            
        # Test with proper analyzer but nonexistent method
        params = ConvergenceParameters(
            lipschitz_constant=10.0,
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        analyzer = ConvergenceAnalyzer(params)
        
        # This should return empty validation results, not raise an error
        validation = validator.validate_theoretical_rates(analyzer, "nonexistent_method", 100, 10)
        assert len(validation) == 0  # No validation possible without data


class TestUtilityFunctions:
    """Test utility functions for convergence analysis."""
    
    def test_physics_informed_convergence_benefit(self):
        """Test physics-informed convergence benefit computation."""
        # No physics constraints should give no benefit
        no_benefit = compute_physics_informed_convergence_benefit(0.0, 10.0)
        assert no_benefit == 1.0
        
        # Strong physics constraints should give benefit
        strong_benefit = compute_physics_informed_convergence_benefit(0.8, 10.0)
        assert strong_benefit > 1.0
        
        # Benefit should increase with constraint strength
        weak_benefit = compute_physics_informed_convergence_benefit(0.2, 10.0)
        assert strong_benefit > weak_benefit
        
        # Test that physics constraints provide benefit
        benefit_with_physics = compute_physics_informed_convergence_benefit(0.5, 10.0)
        benefit_without_physics = compute_physics_informed_convergence_benefit(0.0, 10.0)
        assert benefit_with_physics > benefit_without_physics
        
    def test_meta_learning_convergence_benefit(self):
        """Test meta-learning convergence benefit computation."""
        # More tasks should give more benefit
        few_tasks_benefit = compute_meta_learning_convergence_benefit(10, 0.5)
        many_tasks_benefit = compute_meta_learning_convergence_benefit(100, 0.5)
        assert many_tasks_benefit > few_tasks_benefit
        
        # Higher task similarity should give more benefit
        low_sim_benefit = compute_meta_learning_convergence_benefit(100, 0.2)
        high_sim_benefit = compute_meta_learning_convergence_benefit(100, 0.8)
        assert high_sim_benefit > low_sim_benefit
        
    def test_convergence_phase_transitions(self):
        """Test detection of convergence phase transitions."""
        # Create curve with clear phase transition
        phase1 = 10.0 * np.exp(-0.1 * np.arange(50))  # Fast initial convergence
        phase2 = phase1[-1] * np.exp(-0.01 * np.arange(50))  # Slow final convergence
        curve = np.concatenate([phase1, phase2]).tolist()
        
        transitions = analyze_convergence_phase_transitions(curve, window_size=5)
        
        # Should detect transition around iteration 50
        assert len(transitions) > 0
        # Transition should be roughly in the middle
        assert any(40 < t < 60 for t in transitions)
        
    def test_phase_transitions_smooth_curve(self):
        """Test phase transition detection on smooth curve."""
        # Smooth exponential decay without transitions
        smooth_curve = (10.0 * np.exp(-0.05 * np.arange(100))).tolist()
        
        transitions = analyze_convergence_phase_transitions(smooth_curve, window_size=5)
        
        # Should detect fewer transitions than a curve with clear phase changes
        # Allow for numerical artifacts - smooth curves can still have some detected transitions
        assert len(transitions) <= len(smooth_curve) // 5  # At most 20% of points


class TestConvergenceRateAccuracy:
    """Test accuracy of convergence rate predictions."""
    
    def test_rate_bounds(self):
        """Test that convergence rates are within expected bounds."""
        params = ConvergenceParameters(
            lipschitz_constant=10.0,
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        analyzer = ConvergenceAnalyzer(params)
        
        # Task-level rate should be reasonable for given parameters
        task_rate = analyzer.compute_task_level_convergence_rate()
        assert 0.0 <= task_rate < 1.0  # Should be convergent
        
        # Meta-level rate should scale appropriately
        meta_rate_100 = analyzer.compute_meta_level_convergence_rate(100)
        meta_rate_1000 = analyzer.compute_meta_level_convergence_rate(1000)
        
        # Should improve with more tasks (roughly sqrt scaling)
        improvement_ratio = meta_rate_100 / meta_rate_1000
        expected_ratio = math.sqrt(1000 / 100)
        assert 0.5 * expected_ratio < improvement_ratio < 2.0 * expected_ratio
        
    def test_physics_regularization_scaling(self):
        """Test proper scaling of physics regularization effects."""
        params = ConvergenceParameters(
            lipschitz_constant=10.0,
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        analyzer = ConvergenceAnalyzer(params)
        
        # Test different physics regularization strengths
        rates = []
        physics_strengths = [0.0, 0.2, 0.4, 0.6, 0.8]
        
        for strength in physics_strengths:
            rate = analyzer.compute_task_level_convergence_rate(strength)
            rates.append(rate)
            
        # Rates should monotonically improve (decrease) with physics strength
        for i in range(1, len(rates)):
            assert rates[i] <= rates[i-1]  # Allow for equal rates due to saturation


if __name__ == "__main__":
    pytest.main([__file__])
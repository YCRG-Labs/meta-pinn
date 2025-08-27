"""
Unit tests for evaluation metrics and statistical analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ml_research_pipeline.evaluation.metrics import (
    EvaluationMetrics,
    StatisticalAnalysis,
    MetricResult,
    StatisticalTest
)


class TestMetricResult:
    """Test MetricResult data structure."""
    
    def test_metric_result_creation(self):
        """Test creating metric result."""
        result = MetricResult(
            name="test_metric",
            value=0.85,
            std=0.1,
            confidence_interval=(0.75, 0.95),
            metadata={"n_samples": 100}
        )
        
        assert result.name == "test_metric"
        assert result.value == 0.85
        assert result.std == 0.1
        assert result.confidence_interval == (0.75, 0.95)
        assert result.metadata["n_samples"] == 100
    
    def test_metric_result_default_metadata(self):
        """Test metric result with default metadata."""
        result = MetricResult(name="test", value=1.0)
        assert result.metadata == {}


class TestStatisticalTest:
    """Test StatisticalTest data structure."""
    
    def test_statistical_test_creation(self):
        """Test creating statistical test result."""
        test = StatisticalTest(
            test_name="t-test",
            statistic=2.5,
            p_value=0.01,
            effect_size=0.8,
            interpretation="Significant difference"
        )
        
        assert test.test_name == "t-test"
        assert test.statistic == 2.5
        assert test.p_value == 0.01
        assert test.effect_size == 0.8
        assert test.interpretation == "Significant difference"


class TestEvaluationMetrics:
    """Test evaluation metrics computation."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return {
            'method1': [
                {
                    'parameter_accuracy': 0.85,
                    'adaptation_steps': 10,
                    'computation_time': 5.0,
                    'physics_residual': 1e-4,
                    'total_error': 0.1
                },
                {
                    'parameter_accuracy': 0.90,
                    'adaptation_steps': 8,
                    'computation_time': 4.5,
                    'physics_residual': 2e-4,
                    'total_error': 0.08
                }
            ],
            'method2': [
                {
                    'parameter_accuracy': 0.75,
                    'adaptation_steps': 15,
                    'computation_time': 8.0,
                    'physics_residual': 5e-4,
                    'total_error': 0.15
                },
                {
                    'parameter_accuracy': 0.80,
                    'adaptation_steps': 12,
                    'computation_time': 7.0,
                    'physics_residual': 3e-4,
                    'total_error': 0.12
                }
            ]
        }
    
    def test_evaluation_metrics_initialization(self):
        """Test evaluation metrics initialization."""
        metrics = EvaluationMetrics()
        assert len(metrics.metric_registry) > 0
        assert 'parameter_accuracy' in metrics.metric_registry
        assert 'adaptation_speed' in metrics.metric_registry
    
    def test_compute_parameter_accuracy(self, sample_results):
        """Test parameter accuracy computation."""
        metrics = EvaluationMetrics()
        result = metrics._compute_parameter_accuracy(sample_results)
        
        assert isinstance(result, MetricResult)
        assert result.name == 'parameter_accuracy'
        assert 0.0 <= result.value <= 1.0
        assert result.std is not None
        assert result.metadata['n_samples'] == 4
    
    def test_compute_adaptation_speed(self, sample_results):
        """Test adaptation speed computation."""
        metrics = EvaluationMetrics()
        result = metrics._compute_adaptation_speed(sample_results)
        
        assert isinstance(result, MetricResult)
        assert result.name == 'adaptation_speed'
        assert result.value > 0
        assert result.metadata['n_samples'] == 4
    
    def test_compute_computational_efficiency(self, sample_results):
        """Test computational efficiency computation."""
        metrics = EvaluationMetrics()
        result = metrics._compute_computational_efficiency(sample_results)
        
        assert isinstance(result, MetricResult)
        assert result.name == 'computational_efficiency'
        assert result.value > 0  # Efficiency should be positive
        assert 'mean_time' in result.metadata
    
    def test_compute_physics_consistency(self, sample_results):
        """Test physics consistency computation."""
        metrics = EvaluationMetrics()
        result = metrics._compute_physics_consistency(sample_results)
        
        assert isinstance(result, MetricResult)
        assert result.name == 'physics_consistency'
        assert result.value > 0  # Consistency should be positive
        assert 'mean_residual' in result.metadata
    
    def test_compute_convergence_rate(self, sample_results):
        """Test convergence rate computation."""
        metrics = EvaluationMetrics()
        result = metrics._compute_convergence_rate(sample_results)
        
        assert isinstance(result, MetricResult)
        assert result.name == 'convergence_rate'
        assert result.value >= 0
        assert result.metadata['n_samples'] == 4
    
    def test_compute_generalization_error(self, sample_results):
        """Test generalization error computation."""
        metrics = EvaluationMetrics()
        result = metrics._compute_generalization_error(sample_results)
        
        assert isinstance(result, MetricResult)
        assert result.name == 'generalization_error'
        assert result.value >= 0
        assert result.metadata['n_samples'] == 4
    
    def test_compute_all_metrics(self, sample_results):
        """Test computing all metrics."""
        metrics = EvaluationMetrics()
        all_results = metrics.compute_all_metrics(sample_results)
        
        assert isinstance(all_results, dict)
        assert len(all_results) > 0
        
        for metric_name, result in all_results.items():
            assert isinstance(result, MetricResult)
            assert result.name == metric_name
    
    def test_empty_results(self):
        """Test handling of empty results."""
        metrics = EvaluationMetrics()
        empty_results = {'method1': []}
        
        result = metrics._compute_parameter_accuracy(empty_results)
        assert result.value == 0.0
        assert 'error' in result.metadata
    
    def test_invalid_values(self):
        """Test handling of invalid values."""
        metrics = EvaluationMetrics()
        invalid_results = {
            'method1': [
                {'parameter_accuracy': float('nan')},
                {'parameter_accuracy': float('inf')},
                {'parameter_accuracy': -1.0}  # This should be filtered out in real usage
            ]
        }
        
        result = metrics._compute_parameter_accuracy(invalid_results)
        # Should handle invalid values gracefully
        assert isinstance(result, MetricResult)


class TestStatisticalAnalysis:
    """Test statistical analysis functionality."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample results for statistical testing."""
        np.random.seed(42)  # For reproducible tests
        
        return {
            'method1': [
                {'parameter_accuracy': 0.85 + 0.05 * np.random.randn()}
                for _ in range(20)
            ],
            'method2': [
                {'parameter_accuracy': 0.75 + 0.05 * np.random.randn()}
                for _ in range(20)
            ]
        }
    
    def test_statistical_analysis_initialization(self):
        """Test statistical analysis initialization."""
        analysis = StatisticalAnalysis(alpha=0.05)
        assert analysis.alpha == 0.05
    
    def test_compare_methods(self, sample_results):
        """Test method comparison."""
        analysis = StatisticalAnalysis()
        comparisons = analysis.compare_methods(sample_results, 'parameter_accuracy')
        
        assert isinstance(comparisons, dict)
        assert 'method1' in comparisons
        assert 'method2' in comparisons
        
        # Check that we have pairwise comparisons
        assert 'method2' in comparisons['method1']
        assert 'method1' in comparisons['method2']
        
        # Check statistical test results
        test_result = comparisons['method1']['method2']
        assert isinstance(test_result, StatisticalTest)
        assert test_result.p_value >= 0.0
        assert test_result.effect_size is not None
    
    def test_cohens_d_computation(self):
        """Test Cohen's d effect size computation."""
        analysis = StatisticalAnalysis()
        
        values1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        values2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        cohens_d = analysis._compute_cohens_d(values1, values2)
        
        # Should be negative since values2 > values1
        assert cohens_d < 0
        assert abs(cohens_d) > 0  # Should have some effect
    
    def test_cohens_d_zero_std(self):
        """Test Cohen's d with zero standard deviation."""
        analysis = StatisticalAnalysis()
        
        values1 = [1.0, 1.0, 1.0]
        values2 = [2.0, 2.0, 2.0]
        
        cohens_d = analysis._compute_cohens_d(values1, values2)
        assert cohens_d == 0.0  # Should handle zero std gracefully
    
    def test_interpret_test_result(self):
        """Test interpretation of statistical test results."""
        analysis = StatisticalAnalysis(alpha=0.05)
        
        # Significant result with large effect
        interpretation = analysis._interpret_test_result(0.01, 1.0, "method1", "method2")
        assert "significant" in interpretation
        assert "large" in interpretation
        
        # Non-significant result
        interpretation = analysis._interpret_test_result(0.10, 0.1, "method1", "method2")
        assert "not significant" in interpretation
        assert "negligible" in interpretation
    
    def test_difference_confidence_interval(self):
        """Test confidence interval computation for difference in means."""
        analysis = StatisticalAnalysis()
        
        values1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        values2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        
        ci = analysis._compute_difference_ci(values1, values2)
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < upper bound
    
    def test_power_analysis(self):
        """Test power analysis computation."""
        analysis = StatisticalAnalysis()
        
        power_results = analysis.perform_power_analysis(
            effect_size=0.5,
            alpha=0.05,
            power=0.8
        )
        
        assert isinstance(power_results, dict)
        assert 'required_sample_size' in power_results
        assert 'achieved_powers' in power_results
        assert power_results['effect_size'] == 0.5
        assert power_results['alpha'] == 0.05
        assert power_results['desired_power'] == 0.8
        
        # Required sample size should be reasonable for medium effect
        assert 10 < power_results['required_sample_size'] < 1000
    
    def test_power_analysis_zero_effect(self):
        """Test power analysis with zero effect size."""
        analysis = StatisticalAnalysis()
        
        power_results = analysis.perform_power_analysis(effect_size=0.0)
        
        assert power_results['required_sample_size'] == float('inf')
    
    def test_multiple_comparison_correction_bonferroni(self):
        """Test Bonferroni correction."""
        analysis = StatisticalAnalysis()
        
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        corrected = analysis.multiple_comparison_correction(p_values, 'bonferroni')
        
        assert len(corrected) == len(p_values)
        # Bonferroni correction should multiply by number of tests
        assert corrected[0] == 0.01 * 5
        assert all(c >= p for c, p in zip(corrected, p_values))
    
    def test_multiple_comparison_correction_holm(self):
        """Test Holm correction."""
        analysis = StatisticalAnalysis()
        
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        corrected = analysis.multiple_comparison_correction(p_values, 'holm')
        
        assert len(corrected) == len(p_values)
        assert all(c >= p for c, p in zip(corrected, p_values))
    
    def test_multiple_comparison_correction_fdr(self):
        """Test FDR correction."""
        analysis = StatisticalAnalysis()
        
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        corrected = analysis.multiple_comparison_correction(p_values, 'fdr_bh')
        
        assert len(corrected) == len(p_values)
        assert all(c >= p for c, p in zip(corrected, p_values))
    
    def test_multiple_comparison_invalid_method(self):
        """Test invalid correction method."""
        analysis = StatisticalAnalysis()
        
        p_values = [0.01, 0.02, 0.03]
        
        with pytest.raises(ValueError, match="Unknown correction method"):
            analysis.multiple_comparison_correction(p_values, 'invalid_method')
    
    def test_generate_summary_statistics(self, sample_results):
        """Test summary statistics generation."""
        analysis = StatisticalAnalysis()
        
        summary = analysis.generate_summary_statistics(sample_results)
        
        assert isinstance(summary, dict)
        assert 'method1' in summary
        assert 'method2' in summary
        
        # Check that statistics are computed for each method
        method1_stats = summary['method1']
        assert 'parameter_accuracy' in method1_stats
        
        accuracy_stats = method1_stats['parameter_accuracy']
        assert 'mean' in accuracy_stats
        assert 'std' in accuracy_stats
        assert 'median' in accuracy_stats
        assert 'min' in accuracy_stats
        assert 'max' in accuracy_stats
        assert 'n_samples' in accuracy_stats
    
    def test_summary_statistics_empty_results(self):
        """Test summary statistics with empty results."""
        analysis = StatisticalAnalysis()
        
        empty_results = {'method1': []}
        summary = analysis.generate_summary_statistics(empty_results)
        
        assert 'method1' in summary
        assert summary['method1'] == {}  # Should be empty for no data
    
    def test_summary_statistics_invalid_values(self):
        """Test summary statistics with invalid values."""
        analysis = StatisticalAnalysis()
        
        invalid_results = {
            'method1': [
                {'metric1': float('nan'), 'metric2': 1.0},
                {'metric1': float('inf'), 'metric2': 2.0},
                {'metric1': 3.0, 'metric2': 3.0}
            ]
        }
        
        summary = analysis.generate_summary_statistics(invalid_results)
        
        # Should only include valid values
        assert 'method1' in summary
        method_stats = summary['method1']
        
        # metric1 should only have one valid value
        if 'metric1' in method_stats:
            assert method_stats['metric1']['n_samples'] == 1
        
        # metric2 should have all three values
        assert method_stats['metric2']['n_samples'] == 3


if __name__ == "__main__":
    pytest.main([__file__])
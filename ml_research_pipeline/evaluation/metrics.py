"""
Comprehensive evaluation metrics and statistical analysis for PINN methods.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
import warnings
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for a single metric result."""
    name: str
    value: float
    std: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StatisticalTest:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for PINN methods.
    
    Computes parameter accuracy, adaptation speed, computational efficiency,
    and physics consistency metrics.
    """
    
    def __init__(self):
        """Initialize evaluation metrics calculator."""
        self.metric_registry = {
            'parameter_accuracy': self._compute_parameter_accuracy,
            'adaptation_speed': self._compute_adaptation_speed,
            'computational_efficiency': self._compute_computational_efficiency,
            'physics_consistency': self._compute_physics_consistency,
            'convergence_rate': self._compute_convergence_rate,
            'generalization_error': self._compute_generalization_error,
            'uncertainty_quality': self._compute_uncertainty_quality
        }
        
        logger.info(f"Initialized EvaluationMetrics with {len(self.metric_registry)} metrics")
    
    def compute_all_metrics(self, 
                           results: Dict[str, List[Dict[str, Any]]],
                           ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, MetricResult]:
        """
        Compute all available metrics for given results.
        
        Args:
            results: Dictionary mapping method names to lists of task results
            ground_truth: Optional ground truth data for validation
        
        Returns:
            Dictionary mapping metric names to MetricResult objects
        """
        all_metrics = {}
        
        for metric_name, metric_func in self.metric_registry.items():
            try:
                metric_result = metric_func(results, ground_truth)
                all_metrics[metric_name] = metric_result
                logger.debug(f"Computed {metric_name}: {metric_result.value:.4f}")
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                all_metrics[metric_name] = MetricResult(
                    name=metric_name,
                    value=float('nan'),
                    metadata={'error': str(e)}
                )
        
        return all_metrics
    
    def _compute_parameter_accuracy(self, 
                                  results: Dict[str, List[Dict[str, Any]]],
                                  ground_truth: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Compute parameter inference accuracy metrics."""
        accuracies = []
        
        for method_name, method_results in results.items():
            method_accuracies = []
            
            for task_result in method_results:
                if 'parameter_accuracy' in task_result:
                    accuracy = task_result['parameter_accuracy']
                    if not np.isnan(accuracy) and not np.isinf(accuracy):
                        method_accuracies.append(accuracy)
            
            if method_accuracies:
                accuracies.extend(method_accuracies)
        
        if not accuracies:
            return MetricResult(
                name='parameter_accuracy',
                value=0.0,
                metadata={'error': 'No valid accuracy measurements found'}
            )
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        # Compute confidence interval
        if len(accuracies) > 1:
            ci = stats.t.interval(0.95, len(accuracies)-1, 
                                 loc=mean_accuracy, 
                                 scale=stats.sem(accuracies))
        else:
            ci = None
        
        return MetricResult(
            name='parameter_accuracy',
            value=mean_accuracy,
            std=std_accuracy,
            confidence_interval=ci,
            metadata={
                'n_samples': len(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'median_accuracy': np.median(accuracies)
            }
        )
    
    def _compute_adaptation_speed(self, 
                                results: Dict[str, List[Dict[str, Any]]],
                                ground_truth: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Compute adaptation speed metrics (steps to convergence)."""
        adaptation_steps = []
        
        for method_name, method_results in results.items():
            for task_result in method_results:
                if 'adaptation_steps' in task_result:
                    steps = task_result['adaptation_steps']
                    if not np.isnan(steps) and not np.isinf(steps) and steps > 0:
                        adaptation_steps.append(steps)
        
        if not adaptation_steps:
            return MetricResult(
                name='adaptation_speed',
                value=float('inf'),
                metadata={'error': 'No valid adaptation step measurements found'}
            )
        
        mean_steps = np.mean(adaptation_steps)
        std_steps = np.std(adaptation_steps)
        
        # Compute confidence interval
        if len(adaptation_steps) > 1:
            ci = stats.t.interval(0.95, len(adaptation_steps)-1,
                                 loc=mean_steps,
                                 scale=stats.sem(adaptation_steps))
        else:
            ci = None
        
        return MetricResult(
            name='adaptation_speed',
            value=mean_steps,
            std=std_steps,
            confidence_interval=ci,
            metadata={
                'n_samples': len(adaptation_steps),
                'min_steps': np.min(adaptation_steps),
                'max_steps': np.max(adaptation_steps),
                'median_steps': np.median(adaptation_steps)
            }
        )
    
    def _compute_computational_efficiency(self, 
                                        results: Dict[str, List[Dict[str, Any]]],
                                        ground_truth: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Compute computational efficiency metrics."""
        computation_times = []
        
        for method_name, method_results in results.items():
            for task_result in method_results:
                if 'computation_time' in task_result:
                    time_val = task_result['computation_time']
                    if not np.isnan(time_val) and not np.isinf(time_val) and time_val > 0:
                        computation_times.append(time_val)
        
        if not computation_times:
            return MetricResult(
                name='computational_efficiency',
                value=0.0,
                metadata={'error': 'No valid computation time measurements found'}
            )
        
        mean_time = np.mean(computation_times)
        std_time = np.std(computation_times)
        
        # Efficiency is inverse of time (higher is better)
        efficiency = 1.0 / mean_time if mean_time > 0 else 0.0
        
        return MetricResult(
            name='computational_efficiency',
            value=efficiency,
            std=std_time / (mean_time ** 2) if mean_time > 0 else 0.0,
            metadata={
                'mean_time': mean_time,
                'std_time': std_time,
                'n_samples': len(computation_times),
                'min_time': np.min(computation_times),
                'max_time': np.max(computation_times)
            }
        )
    
    def _compute_physics_consistency(self, 
                                   results: Dict[str, List[Dict[str, Any]]],
                                   ground_truth: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Compute physics consistency metrics (PDE residual magnitude)."""
        physics_residuals = []
        
        for method_name, method_results in results.items():
            for task_result in method_results:
                if 'physics_residual' in task_result:
                    residual = task_result['physics_residual']
                    if not np.isnan(residual) and not np.isinf(residual):
                        physics_residuals.append(residual)
        
        if not physics_residuals:
            return MetricResult(
                name='physics_consistency',
                value=float('inf'),
                metadata={'error': 'No valid physics residual measurements found'}
            )
        
        mean_residual = np.mean(physics_residuals)
        std_residual = np.std(physics_residuals)
        
        # Consistency is inverse of residual (higher is better)
        consistency = 1.0 / (mean_residual + 1e-8)
        
        return MetricResult(
            name='physics_consistency',
            value=consistency,
            metadata={
                'mean_residual': mean_residual,
                'std_residual': std_residual,
                'n_samples': len(physics_residuals),
                'min_residual': np.min(physics_residuals),
                'max_residual': np.max(physics_residuals)
            }
        )
    
    def _compute_convergence_rate(self, 
                                results: Dict[str, List[Dict[str, Any]]],
                                ground_truth: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Compute convergence rate metrics."""
        convergence_rates = []
        
        for method_name, method_results in results.items():
            for task_result in method_results:
                # Estimate convergence rate from adaptation steps and final accuracy
                if ('adaptation_steps' in task_result and 
                    'parameter_accuracy' in task_result):
                    
                    steps = task_result['adaptation_steps']
                    accuracy = task_result['parameter_accuracy']
                    
                    if (not np.isnan(steps) and not np.isnan(accuracy) and 
                        steps > 0 and accuracy > 0):
                        # Rate = accuracy / steps (higher is better)
                        rate = accuracy / steps
                        convergence_rates.append(rate)
        
        if not convergence_rates:
            return MetricResult(
                name='convergence_rate',
                value=0.0,
                metadata={'error': 'No valid convergence rate measurements found'}
            )
        
        mean_rate = np.mean(convergence_rates)
        std_rate = np.std(convergence_rates)
        
        return MetricResult(
            name='convergence_rate',
            value=mean_rate,
            std=std_rate,
            metadata={
                'n_samples': len(convergence_rates),
                'min_rate': np.min(convergence_rates),
                'max_rate': np.max(convergence_rates)
            }
        )
    
    def _compute_generalization_error(self, 
                                    results: Dict[str, List[Dict[str, Any]]],
                                    ground_truth: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Compute generalization error metrics."""
        errors = []
        
        for method_name, method_results in results.items():
            for task_result in method_results:
                if 'total_error' in task_result:
                    error = task_result['total_error']
                    if not np.isnan(error) and not np.isinf(error):
                        errors.append(error)
        
        if not errors:
            return MetricResult(
                name='generalization_error',
                value=float('inf'),
                metadata={'error': 'No valid error measurements found'}
            )
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        return MetricResult(
            name='generalization_error',
            value=mean_error,
            std=std_error,
            metadata={
                'n_samples': len(errors),
                'min_error': np.min(errors),
                'max_error': np.max(errors),
                'median_error': np.median(errors)
            }
        )
    
    def _compute_uncertainty_quality(self, 
                                   results: Dict[str, List[Dict[str, Any]]],
                                   ground_truth: Optional[Dict[str, Any]] = None) -> MetricResult:
        """Compute uncertainty quantification quality metrics."""
        # This would require uncertainty-specific measurements
        # For now, return a placeholder
        return MetricResult(
            name='uncertainty_quality',
            value=0.5,  # Placeholder
            metadata={'note': 'Uncertainty quality metric not yet implemented'}
        )


class StatisticalAnalysis:
    """
    Statistical analysis tools for method comparison and significance testing.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analysis.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
        logger.info(f"Initialized StatisticalAnalysis with alpha={alpha}")
    
    def compare_methods(self, 
                       results: Dict[str, List[Dict[str, Any]]],
                       metric_name: str = 'parameter_accuracy') -> Dict[str, Dict[str, StatisticalTest]]:
        """
        Compare multiple methods using statistical tests.
        
        Args:
            results: Dictionary mapping method names to lists of task results
            metric_name: Name of metric to compare
        
        Returns:
            Dictionary of pairwise comparison results
        """
        method_names = list(results.keys())
        comparisons = {}
        
        # Extract metric values for each method
        method_values = {}
        for method_name in method_names:
            values = []
            for task_result in results[method_name]:
                if metric_name in task_result:
                    value = task_result[metric_name]
                    if not np.isnan(value) and not np.isinf(value):
                        values.append(value)
            method_values[method_name] = values
        
        # Perform pairwise comparisons
        for i, method1 in enumerate(method_names):
            comparisons[method1] = {}
            for j, method2 in enumerate(method_names):
                if i != j:
                    values1 = method_values[method1]
                    values2 = method_values[method2]
                    
                    if len(values1) > 0 and len(values2) > 0:
                        test_result = self._perform_statistical_test(
                            values1, values2, method1, method2, metric_name
                        )
                        comparisons[method1][method2] = test_result
        
        return comparisons
    
    def _perform_statistical_test(self, 
                                values1: List[float], 
                                values2: List[float],
                                method1: str,
                                method2: str,
                                metric_name: str) -> StatisticalTest:
        """Perform statistical test between two sets of values."""
        
        # Choose appropriate test based on data characteristics
        if len(values1) < 30 or len(values2) < 30:
            # Use non-parametric test for small samples
            try:
                statistic, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
                test_name = "Mann-Whitney U"
            except Exception:
                # Fallback to t-test
                statistic, p_value = ttest_ind(values1, values2, equal_var=False)
                test_name = "Welch's t-test"
        else:
            # Use t-test for larger samples
            statistic, p_value = ttest_ind(values1, values2, equal_var=False)
            test_name = "Welch's t-test"
        
        # Compute effect size (Cohen's d)
        effect_size = self._compute_cohens_d(values1, values2)
        
        # Interpret results
        interpretation = self._interpret_test_result(p_value, effect_size, method1, method2)
        
        # Compute confidence interval for difference in means
        ci = self._compute_difference_ci(values1, values2)
        
        return StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
            metadata={
                'method1': method1,
                'method2': method2,
                'metric': metric_name,
                'n1': len(values1),
                'n2': len(values2),
                'mean1': np.mean(values1),
                'mean2': np.mean(values2),
                'std1': np.std(values1),
                'std2': np.std(values2)
            }
        )
    
    def _compute_cohens_d(self, values1: List[float], values2: List[float]) -> float:
        """Compute Cohen's d effect size."""
        mean1, mean2 = np.mean(values1), np.mean(values2)
        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
        n1, n2 = len(values1), len(values2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d
    
    def _interpret_test_result(self, p_value: float, effect_size: float, 
                             method1: str, method2: str) -> str:
        """Interpret statistical test results."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        # Effect size interpretation
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            effect_magnitude = "negligible"
        elif abs_effect < 0.5:
            effect_magnitude = "small"
        elif abs_effect < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
        
        direction = method1 if effect_size > 0 else method2
        
        interpretation = (
            f"Difference between {method1} and {method2} is {significance} "
            f"(p={p_value:.4f}) with {effect_magnitude} effect size "
            f"(Cohen's d={effect_size:.3f}). {direction} performs better."
        )
        
        return interpretation
    
    def _compute_difference_ci(self, values1: List[float], values2: List[float], 
                             confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for difference in means."""
        mean1, mean2 = np.mean(values1), np.mean(values2)
        std1, std2 = np.std(values1, ddof=1), np.std(values2, ddof=1)
        n1, n2 = len(values1), len(values2)
        
        # Standard error of difference
        se_diff = np.sqrt(std1**2/n1 + std2**2/n2)
        
        # Degrees of freedom (Welch's formula)
        df = (std1**2/n1 + std2**2/n2)**2 / (
            (std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1)
        )
        
        # Critical value
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Confidence interval
        diff = mean1 - mean2
        margin = t_critical * se_diff
        
        return (diff - margin, diff + margin)
    
    def perform_power_analysis(self, 
                             effect_size: float,
                             alpha: float = None,
                             power: float = 0.8) -> Dict[str, float]:
        """
        Perform power analysis for experimental design validation.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level (uses instance alpha if None)
            power: Desired statistical power
        
        Returns:
            Dictionary with power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        # Approximate sample size calculation for two-sample t-test
        # Using formula: n ≈ 2 * (z_α/2 + z_β)² / δ²
        # where δ is effect size, z_α/2 is critical value, z_β is power
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        if effect_size == 0:
            required_n = float('inf')
        else:
            required_n = 2 * (z_alpha + z_beta)**2 / (effect_size**2)
        
        # Compute achieved power for different sample sizes
        sample_sizes = [10, 20, 50, 100, 200, 500]
        achieved_powers = []
        
        for n in sample_sizes:
            if effect_size == 0:
                achieved_power = alpha  # Type I error rate
            else:
                # Non-centrality parameter
                ncp = effect_size * np.sqrt(n/2)
                # Achieved power (approximate)
                achieved_power = 1 - stats.norm.cdf(z_alpha - ncp)
            achieved_powers.append(achieved_power)
        
        return {
            'required_sample_size': required_n,
            'effect_size': effect_size,
            'alpha': alpha,
            'desired_power': power,
            'sample_sizes': sample_sizes,
            'achieved_powers': achieved_powers
        }
    
    def multiple_comparison_correction(self, 
                                     p_values: List[float],
                                     method: str = 'bonferroni') -> List[float]:
        """
        Apply multiple comparison correction to p-values.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'holm', 'fdr_bh')
        
        Returns:
            List of corrected p-values
        """
        p_array = np.array(p_values)
        n_tests = len(p_array)
        
        if method == 'bonferroni':
            # Bonferroni correction
            corrected = np.minimum(p_array * n_tests, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_array)
            corrected = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = n_tests - i
                corrected[idx] = min(p_array[idx] * correction_factor, 1.0)
                
                # Ensure monotonicity
                if i > 0:
                    prev_idx = sorted_indices[i-1]
                    corrected[idx] = max(corrected[idx], corrected[prev_idx])
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_array)
            corrected = np.zeros_like(p_array)
            
            for i in range(n_tests-1, -1, -1):
                idx = sorted_indices[i]
                correction_factor = n_tests / (i + 1)
                corrected[idx] = min(p_array[idx] * correction_factor, 1.0)
                
                # Ensure monotonicity
                if i < n_tests - 1:
                    next_idx = sorted_indices[i+1]
                    corrected[idx] = min(corrected[idx], corrected[next_idx])
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return corrected.tolist()
    
    def generate_summary_statistics(self, 
                                  results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Generate comprehensive summary statistics for all methods."""
        summary = {}
        
        for method_name, method_results in results.items():
            method_summary = {}
            
            # Collect all metrics
            all_metrics = defaultdict(list)
            for task_result in method_results:
                for metric_name, value in task_result.items():
                    if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                        all_metrics[metric_name].append(value)
            
            # Compute statistics for each metric
            for metric_name, values in all_metrics.items():
                if values:
                    method_summary[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'n_samples': len(values)
                    }
                    
                    # Add confidence interval
                    if len(values) > 1:
                        ci = stats.t.interval(0.95, len(values)-1,
                                            loc=np.mean(values),
                                            scale=stats.sem(values))
                        method_summary[metric_name]['ci_95'] = ci
            
            summary[method_name] = method_summary
        
        return summary
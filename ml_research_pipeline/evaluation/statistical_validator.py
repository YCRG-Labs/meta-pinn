"""
Statistical Validation Framework - StatisticalValidator Class

This module implements comprehensive statistical validation methods including
bootstrap confidence intervals, permutation tests, and multiple hypothesis correction.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from scipy.stats import bootstrap


@dataclass
class StatisticalResult:
    """Container for statistical test results"""

    statistic: float
    p_value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    test_name: str = ""
    significant: bool = False
    corrected_p_value: Optional[float] = None


@dataclass
class BootstrapResult:
    """Container for bootstrap results"""

    statistic: float
    confidence_interval: Tuple[float, float]
    bootstrap_distribution: np.ndarray
    bias: float
    standard_error: float


class StatisticalTest(ABC):
    """Abstract base class for statistical tests"""

    @abstractmethod
    def test(
        self, data1: np.ndarray, data2: Optional[np.ndarray] = None, **kwargs
    ) -> StatisticalResult:
        """Perform the statistical test"""
        pass


class PermutationTest(StatisticalTest):
    """Permutation test implementation"""

    def __init__(self, n_permutations: int = 10000, random_state: Optional[int] = None):
        self.n_permutations = n_permutations
        self.random_state = random_state

    def test(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        statistic_func: Callable = None,
        **kwargs,
    ) -> StatisticalResult:
        """
        Perform permutation test

        Args:
            data1: First dataset
            data2: Second dataset
            statistic_func: Function to compute test statistic

        Returns:
            StatisticalResult with test results
        """
        if statistic_func is None:
            statistic_func = lambda x, y: np.mean(x) - np.mean(y)

        # Observed test statistic
        observed_stat = statistic_func(data1, data2)

        # Combined data for permutation
        combined = np.concatenate([data1, data2])
        n1, n2 = len(data1), len(data2)

        # Generate permutation distribution
        rng = np.random.RandomState(self.random_state)
        perm_stats = []

        for _ in range(self.n_permutations):
            # Randomly permute and split
            perm_indices = rng.permutation(len(combined))
            perm_data1 = combined[perm_indices[:n1]]
            perm_data2 = combined[perm_indices[n1:]]

            perm_stat = statistic_func(perm_data1, perm_data2)
            perm_stats.append(perm_stat)

        perm_stats = np.array(perm_stats)

        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))

        # Effect size (Cohen's d for mean difference)
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1))
            / (n1 + n2 - 2)
        )
        effect_size = observed_stat / pooled_std if pooled_std > 0 else 0

        return StatisticalResult(
            statistic=observed_stat,
            p_value=p_value,
            effect_size=effect_size,
            test_name="Permutation Test",
            significant=p_value < 0.05,
        )


class StatisticalValidator:
    """
    Comprehensive statistical validation framework with bootstrap confidence intervals,
    permutation tests, and multiple hypothesis correction.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
        n_permutations: int = 10000,
        random_state: Optional[int] = None,
    ):
        """
        Initialize StatisticalValidator

        Args:
            alpha: Significance level for tests
            n_bootstrap: Number of bootstrap samples
            n_permutations: Number of permutations for permutation tests
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.random_state = random_state

        # Initialize test objects
        self.permutation_test = PermutationTest(n_permutations, random_state)

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: Callable = np.mean,
        confidence_level: float = 0.95,
        method: str = "percentile",
    ) -> BootstrapResult:
        """
        Calculate bootstrap confidence interval for a statistic

        Args:
            data: Input data array
            statistic_func: Function to compute statistic
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Bootstrap method ('percentile', 'bias_corrected', 'bca')

        Returns:
            BootstrapResult with confidence interval and statistics
        """
        # Validate input data
        if len(data) == 0:
            raise ValueError(
                "Cannot compute bootstrap confidence interval for empty data"
            )

        # Original statistic
        original_stat = statistic_func(data)

        # Bootstrap sampling
        rng = np.random.RandomState(self.random_state)
        bootstrap_stats = []

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            bootstrap_sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # Calculate confidence interval
        alpha = 1 - confidence_level

        if method == "percentile":
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        elif method == "bias_corrected":
            # Bias-corrected method
            bias = np.mean(bootstrap_stats) - original_stat
            corrected_stats = bootstrap_stats - bias
            lower = np.percentile(corrected_stats, 100 * alpha / 2)
            upper = np.percentile(corrected_stats, 100 * (1 - alpha / 2))
        else:  # Default to percentile
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        # Calculate bias and standard error
        bias = np.mean(bootstrap_stats) - original_stat
        standard_error = np.std(bootstrap_stats, ddof=1)

        return BootstrapResult(
            statistic=original_stat,
            confidence_interval=(lower, upper),
            bootstrap_distribution=bootstrap_stats,
            bias=bias,
            standard_error=standard_error,
        )

    def permutation_test_difference(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        statistic_func: Optional[Callable] = None,
    ) -> StatisticalResult:
        """
        Perform permutation test for difference between two groups

        Args:
            data1: First group data
            data2: Second group data
            statistic_func: Custom statistic function (default: mean difference)

        Returns:
            StatisticalResult with test results
        """
        return self.permutation_test.test(data1, data2, statistic_func)

    def multiple_hypothesis_correction(
        self, p_values: List[float], method: str = "bonferroni"
    ) -> Tuple[List[float], List[bool]]:
        """
        Apply multiple hypothesis correction to p-values

        Args:
            p_values: List of uncorrected p-values
            method: Correction method ('bonferroni', 'fdr_bh', 'fdr_by')

        Returns:
            Tuple of (corrected_p_values, rejected_hypotheses)
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)

        if method == "bonferroni":
            # Bonferroni correction
            corrected_p = np.minimum(p_values * n_tests, 1.0)
            rejected = corrected_p < self.alpha

        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            # Find largest k such that P(k) <= (k/m) * alpha
            rejected_sorted = np.zeros(n_tests, dtype=bool)
            corrected_p_sorted = np.ones(n_tests)

            for i in range(n_tests - 1, -1, -1):
                threshold = (i + 1) / n_tests * self.alpha
                if sorted_p[i] <= threshold:
                    rejected_sorted[i:] = True
                    break

            # Calculate corrected p-values
            for i in range(n_tests):
                corrected_p_sorted[i] = min(1.0, sorted_p[i] * n_tests / (i + 1))

            # Restore original order
            rejected = np.zeros(n_tests, dtype=bool)
            corrected_p = np.ones(n_tests)
            rejected[sorted_indices] = rejected_sorted
            corrected_p[sorted_indices] = corrected_p_sorted

        elif method == "fdr_by":
            # Benjamini-Yekutieli FDR correction (more conservative)
            c_n = np.sum(1.0 / np.arange(1, n_tests + 1))  # Harmonic series
            adjusted_alpha = self.alpha / c_n

            # Apply BH procedure with adjusted alpha
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            rejected_sorted = np.zeros(n_tests, dtype=bool)
            corrected_p_sorted = np.ones(n_tests)

            for i in range(n_tests - 1, -1, -1):
                threshold = (i + 1) / n_tests * adjusted_alpha
                if sorted_p[i] <= threshold:
                    rejected_sorted[i:] = True
                    break

            # Calculate corrected p-values
            for i in range(n_tests):
                corrected_p_sorted[i] = min(1.0, sorted_p[i] * n_tests * c_n / (i + 1))

            # Restore original order
            rejected = np.zeros(n_tests, dtype=bool)
            corrected_p = np.ones(n_tests)
            rejected[sorted_indices] = rejected_sorted
            corrected_p[sorted_indices] = corrected_p_sorted

        else:
            raise ValueError(f"Unknown correction method: {method}")

        return corrected_p.tolist(), rejected.tolist()

    def validate_distribution_assumptions(
        self, data: np.ndarray, distribution: str = "normal"
    ) -> Dict[str, StatisticalResult]:
        """
        Test distributional assumptions using various statistical tests

        Args:
            data: Data to test
            distribution: Distribution to test against ('normal', 'uniform', 'exponential')

        Returns:
            Dictionary of test results
        """
        results = {}

        if distribution == "normal":
            # Shapiro-Wilk test for normality
            if len(data) <= 5000:  # Shapiro-Wilk has sample size limitations
                stat, p_val = stats.shapiro(data)
                results["shapiro_wilk"] = StatisticalResult(
                    statistic=stat,
                    p_value=p_val,
                    test_name="Shapiro-Wilk Normality Test",
                    significant=p_val < self.alpha,
                )

            # Kolmogorov-Smirnov test against normal distribution
            # Standardize data first
            standardized = (data - np.mean(data)) / np.std(data, ddof=1)
            stat, p_val = stats.kstest(standardized, "norm")
            results["ks_normal"] = StatisticalResult(
                statistic=stat,
                p_value=p_val,
                test_name="Kolmogorov-Smirnov Normality Test",
                significant=p_val < self.alpha,
            )

            # Anderson-Darling test for normality
            stat, critical_vals, significance_levels = stats.anderson(data, dist="norm")
            # Find appropriate significance level
            p_val = None
            for i, level in enumerate(significance_levels):
                if stat < critical_vals[i]:
                    p_val = 1 - level / 100
                    break
            if p_val is None:
                p_val = 0.001  # Very small p-value if all critical values exceeded

            results["anderson_darling"] = StatisticalResult(
                statistic=stat,
                p_value=p_val,
                test_name="Anderson-Darling Normality Test",
                significant=p_val < self.alpha if p_val is not None else True,
            )

        elif distribution == "uniform":
            # Kolmogorov-Smirnov test against uniform distribution
            # Normalize to [0,1]
            normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            stat, p_val = stats.kstest(normalized, "uniform")
            results["ks_uniform"] = StatisticalResult(
                statistic=stat,
                p_value=p_val,
                test_name="Kolmogorov-Smirnov Uniformity Test",
                significant=p_val < self.alpha,
            )

        return results

    def effect_size_analysis(
        self, data1: np.ndarray, data2: np.ndarray, effect_type: str = "cohens_d"
    ) -> Dict[str, float]:
        """
        Calculate various effect size measures

        Args:
            data1: First group data
            data2: Second group data
            effect_type: Type of effect size ('cohens_d', 'glass_delta', 'hedges_g')

        Returns:
            Dictionary of effect size measures
        """
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)

        results = {}

        if effect_type == "cohens_d":
            # Cohen's d - pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            results["cohens_d"] = cohens_d

        elif effect_type == "glass_delta":
            # Glass's delta - uses control group standard deviation
            glass_delta = (mean1 - mean2) / np.sqrt(var2) if var2 > 0 else 0
            results["glass_delta"] = glass_delta

        elif effect_type == "hedges_g":
            # Hedges' g - bias-corrected Cohen's d
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            # Bias correction factor
            correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
            hedges_g = cohens_d * correction
            results["hedges_g"] = hedges_g

        return results

    def comprehensive_validation(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        validation_type: str = "single_sample",
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation

        Args:
            data: Data for validation (single array or list of arrays)
            validation_type: Type of validation ('single_sample', 'two_sample', 'multiple_sample')

        Returns:
            Dictionary with comprehensive validation results
        """
        results = {
            "summary": {},
            "tests": {},
            "assumptions": {},
            "effect_sizes": {},
            "confidence_intervals": {},
        }

        if validation_type == "single_sample":
            if not isinstance(data, np.ndarray):
                raise ValueError("Single sample validation requires numpy array")

            # Basic statistics
            results["summary"] = {
                "n": len(data),
                "mean": np.mean(data),
                "std": np.std(data, ddof=1),
                "median": np.median(data),
                "skewness": stats.skew(data),
                "kurtosis": stats.kurtosis(data),
            }

            # Distribution tests
            results["assumptions"] = self.validate_distribution_assumptions(data)

            # Bootstrap confidence intervals
            results["confidence_intervals"]["mean"] = (
                self.bootstrap_confidence_interval(data, np.mean)
            )
            results["confidence_intervals"]["median"] = (
                self.bootstrap_confidence_interval(data, np.median)
            )
            results["confidence_intervals"]["std"] = self.bootstrap_confidence_interval(
                data, lambda x: np.std(x, ddof=1)
            )

        elif validation_type == "two_sample":
            if not isinstance(data, list) or len(data) != 2:
                raise ValueError("Two sample validation requires list of two arrays")

            data1, data2 = data[0], data[1]

            # Basic statistics for both groups
            results["summary"] = {
                "group1": {
                    "n": len(data1),
                    "mean": np.mean(data1),
                    "std": np.std(data1, ddof=1),
                },
                "group2": {
                    "n": len(data2),
                    "mean": np.mean(data2),
                    "std": np.std(data2, ddof=1),
                },
            }

            # Statistical tests
            results["tests"]["permutation"] = self.permutation_test_difference(
                data1, data2
            )

            # t-test for comparison
            stat, p_val = stats.ttest_ind(data1, data2)
            results["tests"]["t_test"] = StatisticalResult(
                statistic=stat,
                p_value=p_val,
                test_name="Independent t-test",
                significant=p_val < self.alpha,
            )

            # Effect sizes
            results["effect_sizes"] = self.effect_size_analysis(data1, data2)

        return results

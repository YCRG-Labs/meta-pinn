"""
Performance regression testing system.

This module provides automated performance regression detection and validation
to ensure optimizations don't degrade system performance over time.
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .performance_profiler import PerformanceMetrics, benchmark_function


@dataclass
class PerformanceBaseline:
    """Container for performance baseline data."""

    test_name: str
    timestamp: float
    git_commit: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    metrics: Optional[PerformanceMetrics] = None
    config_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.metrics:
            data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceBaseline":
        """Create from dictionary."""
        if "metrics" in data and data["metrics"]:
            data["metrics"] = PerformanceMetrics(**data["metrics"])
        return cls(**data)


@dataclass
class RegressionResult:
    """Container for regression test results."""

    test_name: str
    baseline_metrics: PerformanceMetrics
    current_metrics: PerformanceMetrics
    regression_detected: bool
    regression_percentage: float
    threshold_exceeded: Dict[str, bool]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "current_metrics": self.current_metrics.to_dict(),
            "regression_detected": self.regression_detected,
            "regression_percentage": self.regression_percentage,
            "threshold_exceeded": self.threshold_exceeded,
            "details": self.details,
        }


class PerformanceRegressionTester:
    """
    Automated performance regression testing system.

    Maintains performance baselines and detects regressions across
    different system configurations and code changes.
    """

    def __init__(
        self,
        baseline_dir: Union[str, Path],
        regression_thresholds: Optional[Dict[str, float]] = None,
        auto_update_baselines: bool = False,
    ):
        """
        Initialize regression tester.

        Args:
            baseline_dir: Directory to store performance baselines
            regression_thresholds: Thresholds for regression detection (metric -> percentage)
            auto_update_baselines: Whether to automatically update baselines
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        # Default regression thresholds (percentage increase that triggers regression)
        self.regression_thresholds = regression_thresholds or {
            "cpu_time": 10.0,  # 10% increase in CPU time
            "gpu_time": 10.0,  # 10% increase in GPU time
            "memory_peak": 15.0,  # 15% increase in peak memory
            "latency": 10.0,  # 10% increase in latency
            "throughput": -10.0,  # 10% decrease in throughput (negative because lower is worse)
        }

        self.auto_update_baselines = auto_update_baselines

        # Load existing baselines
        self.baselines = self._load_baselines()

        # Test results storage
        self.test_results = []

        logging.info(f"PerformanceRegressionTester initialized: {self.baseline_dir}")
        logging.info(f"Regression thresholds: {self.regression_thresholds}")

    def _load_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Load existing performance baselines."""
        baselines = {}

        baseline_file = self.baseline_dir / "baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, "r") as f:
                    data = json.load(f)

                for test_name, baseline_data in data.items():
                    baselines[test_name] = PerformanceBaseline.from_dict(baseline_data)

                logging.info(f"Loaded {len(baselines)} performance baselines")

            except Exception as e:
                logging.error(f"Failed to load baselines: {e}")

        return baselines

    def _save_baselines(self):
        """Save performance baselines to disk."""
        baseline_file = self.baseline_dir / "baselines.json"

        try:
            data = {}
            for test_name, baseline in self.baselines.items():
                data[test_name] = baseline.to_dict()

            with open(baseline_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logging.info(f"Saved {len(self.baselines)} performance baselines")

        except Exception as e:
            logging.error(f"Failed to save baselines: {e}")

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration for baseline identification."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        import psutil

        info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024**3,
            "cuda_available": torch.cuda.is_available(),
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "torch_version": torch.__version__,
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_device_name": torch.cuda.get_device_name(),
                    "cuda_version": torch.version.cuda,
                }
            )

        return info

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    def create_baseline(
        self,
        test_name: str,
        test_function: callable,
        test_config: Dict[str, Any],
        *args,
        **kwargs,
    ) -> PerformanceBaseline:
        """
        Create a new performance baseline.

        Args:
            test_name: Name of the test
            test_function: Function to benchmark
            test_config: Test configuration
            *args: Arguments for test function
            **kwargs: Keyword arguments for test function

        Returns:
            Created performance baseline
        """
        logging.info(f"Creating baseline for test: {test_name}")

        # Benchmark the function
        metrics = benchmark_function(
            test_function,
            *args,
            num_runs=kwargs.pop("num_runs", 10),
            warmup_runs=kwargs.pop("warmup_runs", 3),
            **kwargs,
        )

        # Create baseline
        baseline = PerformanceBaseline(
            test_name=test_name,
            timestamp=time.time(),
            git_commit=self._get_git_commit(),
            system_info=self._get_system_info(),
            metrics=metrics,
            config_hash=self._compute_config_hash(test_config),
        )

        # Store baseline
        self.baselines[test_name] = baseline
        self._save_baselines()

        logging.info(f"Baseline created for {test_name}: {metrics.to_dict()}")
        return baseline

    def run_regression_test(
        self,
        test_name: str,
        test_function: callable,
        test_config: Dict[str, Any],
        *args,
        **kwargs,
    ) -> RegressionResult:
        """
        Run regression test against existing baseline.

        Args:
            test_name: Name of the test
            test_function: Function to benchmark
            test_config: Test configuration
            *args: Arguments for test function
            **kwargs: Keyword arguments for test function

        Returns:
            Regression test result
        """
        logging.info(f"Running regression test: {test_name}")

        # Check if baseline exists
        if test_name not in self.baselines:
            logging.warning(f"No baseline found for {test_name}, creating new baseline")
            baseline = self.create_baseline(
                test_name, test_function, test_config, *args, **kwargs
            )

            return RegressionResult(
                test_name=test_name,
                baseline_metrics=baseline.metrics,
                current_metrics=baseline.metrics,
                regression_detected=False,
                regression_percentage=0.0,
                threshold_exceeded={},
                details={"status": "baseline_created"},
            )

        baseline = self.baselines[test_name]

        # Check configuration compatibility
        current_config_hash = self._compute_config_hash(test_config)
        if baseline.config_hash != current_config_hash:
            logging.warning(
                f"Configuration changed for {test_name}, results may not be comparable"
            )

        # Benchmark current performance
        current_metrics = benchmark_function(
            test_function,
            *args,
            num_runs=kwargs.pop("num_runs", 10),
            warmup_runs=kwargs.pop("warmup_runs", 3),
            **kwargs,
        )

        # Compare with baseline
        regression_result = self._compare_with_baseline(
            test_name, baseline.metrics, current_metrics
        )

        # Update baseline if auto-update is enabled and no regression detected
        if self.auto_update_baselines and not regression_result.regression_detected:
            self._update_baseline(test_name, current_metrics, test_config)

        # Store test result
        self.test_results.append(regression_result)

        return regression_result

    def _compare_with_baseline(
        self,
        test_name: str,
        baseline_metrics: PerformanceMetrics,
        current_metrics: PerformanceMetrics,
    ) -> RegressionResult:
        """Compare current metrics with baseline."""
        threshold_exceeded = {}
        regression_percentages = {}

        # Compare each metric
        for metric_name, threshold in self.regression_thresholds.items():
            baseline_value = getattr(baseline_metrics, metric_name, 0)
            current_value = getattr(current_metrics, metric_name, 0)

            if baseline_value == 0:
                regression_percentage = 0.0
            else:
                if metric_name == "throughput":
                    # For throughput, lower is worse (negative regression)
                    regression_percentage = (
                        (baseline_value - current_value) / baseline_value
                    ) * 100
                else:
                    # For other metrics, higher is worse (positive regression)
                    regression_percentage = (
                        (current_value - baseline_value) / baseline_value
                    ) * 100

            regression_percentages[metric_name] = regression_percentage

            # Check if threshold is exceeded
            if metric_name == "throughput":
                threshold_exceeded[metric_name] = regression_percentage > abs(threshold)
            else:
                threshold_exceeded[metric_name] = regression_percentage > threshold

        # Overall regression detection
        regression_detected = any(threshold_exceeded.values())
        overall_regression = (
            max(regression_percentages.values()) if regression_percentages else 0.0
        )

        details = {
            "regression_percentages": regression_percentages,
            "baseline_timestamp": (
                self.baselines[test_name].timestamp
                if test_name in self.baselines
                else None
            ),
            "system_info_changed": self._system_info_changed(test_name),
            "git_commit": self._get_git_commit(),
        }

        return RegressionResult(
            test_name=test_name,
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            regression_detected=regression_detected,
            regression_percentage=overall_regression,
            threshold_exceeded=threshold_exceeded,
            details=details,
        )

    def _system_info_changed(self, test_name: str) -> bool:
        """Check if system info has changed since baseline."""
        if test_name not in self.baselines or not self.baselines[test_name].system_info:
            return False

        baseline_info = self.baselines[test_name].system_info
        current_info = self._get_system_info()

        # Check key system parameters
        key_params = ["cuda_device_count", "cuda_device_name", "memory_gb", "cpu_count"]

        for param in key_params:
            if baseline_info.get(param) != current_info.get(param):
                return True

        return False

    def _update_baseline(
        self,
        test_name: str,
        new_metrics: PerformanceMetrics,
        test_config: Dict[str, Any],
    ):
        """Update existing baseline with new metrics."""
        if test_name in self.baselines:
            self.baselines[test_name].metrics = new_metrics
            self.baselines[test_name].timestamp = time.time()
            self.baselines[test_name].git_commit = self._get_git_commit()
            self.baselines[test_name].system_info = self._get_system_info()
            self.baselines[test_name].config_hash = self._compute_config_hash(
                test_config
            )

            self._save_baselines()
            logging.info(f"Updated baseline for {test_name}")

    def run_test_suite(
        self, test_suite: Dict[str, Dict[str, Any]]
    ) -> Dict[str, RegressionResult]:
        """
        Run a suite of regression tests.

        Args:
            test_suite: Dictionary mapping test names to test configurations
                       Each config should have 'function', 'args', 'kwargs', 'config'

        Returns:
            Dictionary mapping test names to regression results
        """
        results = {}

        logging.info(f"Running regression test suite with {len(test_suite)} tests")

        for test_name, test_spec in test_suite.items():
            try:
                result = self.run_regression_test(
                    test_name,
                    test_spec["function"],
                    test_spec.get("config", {}),
                    *test_spec.get("args", []),
                    **test_spec.get("kwargs", {}),
                )
                results[test_name] = result

                if result.regression_detected:
                    logging.warning(
                        f"Regression detected in {test_name}: "
                        f"{result.regression_percentage:.2f}% degradation"
                    )
                else:
                    logging.info(f"No regression in {test_name}")

            except Exception as e:
                logging.error(f"Error running test {test_name}: {e}")
                results[test_name] = None

        return results

    def generate_regression_report(
        self, results: Optional[Dict[str, RegressionResult]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive regression test report.

        Args:
            results: Optional specific results to report on

        Returns:
            Comprehensive regression report
        """
        if results is None:
            results = {r.test_name: r for r in self.test_results}

        # Filter out None results
        valid_results = {k: v for k, v in results.items() if v is not None}

        # Summary statistics
        total_tests = len(valid_results)
        regressions_detected = sum(
            1 for r in valid_results.values() if r.regression_detected
        )

        # Categorize regressions by severity
        severe_regressions = []
        moderate_regressions = []
        minor_regressions = []

        for test_name, result in valid_results.items():
            if result.regression_detected:
                if result.regression_percentage > 25:
                    severe_regressions.append((test_name, result))
                elif result.regression_percentage > 10:
                    moderate_regressions.append((test_name, result))
                else:
                    minor_regressions.append((test_name, result))

        # Generate recommendations
        recommendations = self._generate_regression_recommendations(valid_results)

        report = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "regressions_detected": regressions_detected,
                "regression_rate": (
                    (regressions_detected / total_tests * 100) if total_tests > 0 else 0
                ),
                "severe_regressions": len(severe_regressions),
                "moderate_regressions": len(moderate_regressions),
                "minor_regressions": len(minor_regressions),
            },
            "detailed_results": {
                name: result.to_dict() for name, result in valid_results.items()
            },
            "regression_categories": {
                "severe": [
                    {"test": name, "regression_pct": result.regression_percentage}
                    for name, result in severe_regressions
                ],
                "moderate": [
                    {"test": name, "regression_pct": result.regression_percentage}
                    for name, result in moderate_regressions
                ],
                "minor": [
                    {"test": name, "regression_pct": result.regression_percentage}
                    for name, result in minor_regressions
                ],
            },
            "recommendations": recommendations,
            "system_info": self._get_system_info(),
        }

        # Save report
        report_file = self.baseline_dir / f"regression_report_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logging.info(f"Regression report generated: {report_file}")
        return report

    def _generate_regression_recommendations(
        self, results: Dict[str, RegressionResult]
    ) -> List[str]:
        """Generate recommendations based on regression results."""
        recommendations = []

        # Count regression types
        memory_regressions = sum(
            1
            for r in results.values()
            if r.threshold_exceeded.get("memory_peak", False)
        )
        cpu_regressions = sum(
            1 for r in results.values() if r.threshold_exceeded.get("cpu_time", False)
        )
        gpu_regressions = sum(
            1 for r in results.values() if r.threshold_exceeded.get("gpu_time", False)
        )

        if memory_regressions > 0:
            recommendations.append(
                f"Memory regressions detected in {memory_regressions} tests. "
                "Consider memory profiling and optimization."
            )

        if cpu_regressions > 0:
            recommendations.append(
                f"CPU performance regressions detected in {cpu_regressions} tests. "
                "Review algorithmic changes and consider CPU profiling."
            )

        if gpu_regressions > 0:
            recommendations.append(
                f"GPU performance regressions detected in {gpu_regressions} tests. "
                "Check CUDA kernel efficiency and memory access patterns."
            )

        # General recommendations
        if (
            len([r for r in results.values() if r.regression_detected])
            > len(results) * 0.3
        ):
            recommendations.append(
                "High regression rate detected. Consider reverting recent changes "
                "and investigating systematic performance issues."
            )

        return recommendations

    def cleanup_old_baselines(self, days_old: int = 30):
        """Remove baselines older than specified days."""
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 3600)

        old_baselines = []
        for test_name, baseline in self.baselines.items():
            if baseline.timestamp < cutoff_time:
                old_baselines.append(test_name)

        for test_name in old_baselines:
            del self.baselines[test_name]
            logging.info(f"Removed old baseline for {test_name}")

        if old_baselines:
            self._save_baselines()
            logging.info(f"Cleaned up {len(old_baselines)} old baselines")

    def export_baselines(self, export_path: Union[str, Path]):
        """Export baselines to external file."""
        export_path = Path(export_path)

        data = {}
        for test_name, baseline in self.baselines.items():
            data[test_name] = baseline.to_dict()

        with open(export_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logging.info(f"Exported {len(self.baselines)} baselines to {export_path}")

    def import_baselines(self, import_path: Union[str, Path]):
        """Import baselines from external file."""
        import_path = Path(import_path)

        if not import_path.exists():
            logging.error(f"Import file not found: {import_path}")
            return

        try:
            with open(import_path, "r") as f:
                data = json.load(f)

            imported_count = 0
            for test_name, baseline_data in data.items():
                self.baselines[test_name] = PerformanceBaseline.from_dict(baseline_data)
                imported_count += 1

            self._save_baselines()
            logging.info(f"Imported {imported_count} baselines from {import_path}")

        except Exception as e:
            logging.error(f"Failed to import baselines: {e}")


def create_performance_regression_tester(
    baseline_dir: Union[str, Path], config: Optional[Dict[str, Any]] = None
) -> PerformanceRegressionTester:
    """
    Factory function to create a PerformanceRegressionTester with sensible defaults.

    Args:
        baseline_dir: Directory to store baselines
        config: Configuration dictionary

    Returns:
        PerformanceRegressionTester instance
    """
    defaults = {
        "regression_thresholds": {
            "cpu_time": 15.0,  # 15% increase threshold
            "gpu_time": 15.0,  # 15% increase threshold
            "memory_peak": 20.0,  # 20% increase threshold
            "latency": 15.0,  # 15% increase threshold
            "throughput": -15.0,  # 15% decrease threshold
        },
        "auto_update_baselines": False,
    }

    if config:
        defaults.update(config)

    return PerformanceRegressionTester(baseline_dir=baseline_dir, **defaults)

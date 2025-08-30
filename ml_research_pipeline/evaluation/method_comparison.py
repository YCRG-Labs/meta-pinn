"""
Automated method comparison system for evaluating PINN methods.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .benchmark_suite import BenchmarkResult, PINNBenchmarkSuite
from .metrics import (
    EvaluationMetrics,
    MetricResult,
    StatisticalAnalysis,
    StatisticalTest,
)

logger = logging.getLogger(__name__)


@dataclass
class MethodComparisonConfig:
    """Configuration for method comparison."""

    benchmarks: List[str]  # List of benchmark names to run
    metrics: List[str]  # List of metrics to evaluate
    statistical_tests: List[str]  # List of statistical tests to perform
    significance_level: float = 0.05
    multiple_comparison_correction: str = "bonferroni"
    save_results: bool = True
    save_dir: str = "results/method_comparison"
    generate_plots: bool = True
    generate_tables: bool = True


@dataclass
class ComparisonResult:
    """Results from method comparison."""

    config: MethodComparisonConfig
    benchmark_results: Dict[str, Dict[str, BenchmarkResult]]
    metric_results: Dict[str, Dict[str, MetricResult]]
    statistical_results: Dict[str, Dict[str, Dict[str, StatisticalTest]]]
    rankings: Dict[str, List[Tuple[str, float]]]
    summary: Dict[str, Any]
    timestamp: float


class MethodComparison:
    """
    Automated method comparison system.

    Evaluates multiple PINN methods across benchmarks and provides
    comprehensive statistical analysis and ranking.
    """

    def __init__(self, config: MethodComparisonConfig):
        """
        Initialize method comparison system.

        Args:
            config: Configuration for method comparison
        """
        self.config = config
        self.benchmark_suite = PINNBenchmarkSuite()
        self.metrics_calculator = EvaluationMetrics()
        self.statistical_analyzer = StatisticalAnalysis(alpha=config.significance_level)

        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized MethodComparison with {len(config.benchmarks)} benchmarks"
        )

    def run_comparison(self, methods: Dict[str, Any]) -> ComparisonResult:
        """
        Run comprehensive method comparison.

        Args:
            methods: Dictionary mapping method names to method objects

        Returns:
            ComparisonResult containing all comparison results
        """
        logger.info(f"Starting method comparison with {len(methods)} methods")
        start_time = time.time()

        # Step 1: Run benchmarks
        benchmark_results = self._run_benchmarks(methods)

        # Step 2: Compute metrics
        metric_results = self._compute_metrics(benchmark_results)

        # Step 3: Perform statistical analysis
        statistical_results = self._perform_statistical_analysis(benchmark_results)

        # Step 4: Generate rankings
        rankings = self._generate_rankings(metric_results)

        # Step 5: Generate summary
        summary = self._generate_summary(
            benchmark_results, metric_results, statistical_results, rankings
        )

        # Create comparison result
        comparison_result = ComparisonResult(
            config=self.config,
            benchmark_results=benchmark_results,
            metric_results=metric_results,
            statistical_results=statistical_results,
            rankings=rankings,
            summary=summary,
            timestamp=time.time(),
        )

        # Save results
        if self.config.save_results:
            self._save_results(comparison_result)

        # Generate outputs
        if self.config.generate_tables:
            self._generate_tables(comparison_result)

        if self.config.generate_plots:
            self._generate_plots(comparison_result)

        total_time = time.time() - start_time
        logger.info(f"Method comparison completed in {total_time:.2f}s")

        return comparison_result

    def _run_benchmarks(
        self, methods: Dict[str, Any]
    ) -> Dict[str, Dict[str, BenchmarkResult]]:
        """Run all specified benchmarks with all methods."""
        logger.info("Running benchmarks...")

        benchmark_results = {}

        for benchmark_name in self.config.benchmarks:
            logger.info(f"Running benchmark: {benchmark_name}")
            benchmark_results[benchmark_name] = {}

            for method_name, method in methods.items():
                try:
                    logger.info(f"  Evaluating {method_name}")
                    result = self.benchmark_suite.run_benchmark(
                        benchmark_name, method, method_name
                    )
                    benchmark_results[benchmark_name][method_name] = result

                except Exception as e:
                    logger.error(
                        f"Error running {method_name} on {benchmark_name}: {e}"
                    )
                    # Create dummy result for failed runs
                    benchmark_results[benchmark_name][method_name] = BenchmarkResult(
                        benchmark_name=benchmark_name,
                        method_name=method_name,
                        metrics={"error": float("inf")},
                        task_results=[],
                        runtime_info={"total_time": float("inf")},
                        metadata={"error": str(e)},
                    )

        return benchmark_results

    def _compute_metrics(
        self, benchmark_results: Dict[str, Dict[str, BenchmarkResult]]
    ) -> Dict[str, Dict[str, MetricResult]]:
        """Compute evaluation metrics for all methods and benchmarks."""
        logger.info("Computing evaluation metrics...")

        metric_results = {}

        for benchmark_name, benchmark_data in benchmark_results.items():
            logger.info(f"Computing metrics for {benchmark_name}")

            # Convert benchmark results to format expected by metrics calculator
            results_for_metrics = {}
            for method_name, benchmark_result in benchmark_data.items():
                results_for_metrics[method_name] = benchmark_result.task_results

            # Compute all metrics
            benchmark_metrics = self.metrics_calculator.compute_all_metrics(
                results_for_metrics
            )
            metric_results[benchmark_name] = benchmark_metrics

        return metric_results

    def _perform_statistical_analysis(
        self, benchmark_results: Dict[str, Dict[str, BenchmarkResult]]
    ) -> Dict[str, Dict[str, Dict[str, StatisticalTest]]]:
        """Perform statistical analysis comparing methods."""
        logger.info("Performing statistical analysis...")

        statistical_results = {}

        for benchmark_name, benchmark_data in benchmark_results.items():
            logger.info(f"Statistical analysis for {benchmark_name}")

            # Convert to format for statistical analysis
            results_for_stats = {}
            for method_name, benchmark_result in benchmark_data.items():
                results_for_stats[method_name] = benchmark_result.task_results

            benchmark_stats = {}

            # Perform comparisons for each metric
            for metric_name in self.config.metrics:
                try:
                    comparisons = self.statistical_analyzer.compare_methods(
                        results_for_stats, metric_name
                    )
                    benchmark_stats[metric_name] = comparisons
                except Exception as e:
                    logger.warning(
                        f"Failed statistical analysis for {metric_name}: {e}"
                    )
                    benchmark_stats[metric_name] = {}

            statistical_results[benchmark_name] = benchmark_stats

        return statistical_results

    def _generate_rankings(
        self, metric_results: Dict[str, Dict[str, MetricResult]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Generate method rankings for each metric."""
        logger.info("Generating method rankings...")

        rankings = {}

        # Aggregate metrics across benchmarks
        aggregated_metrics = defaultdict(lambda: defaultdict(list))

        for benchmark_name, benchmark_metrics in metric_results.items():
            for metric_name, metric_result in benchmark_metrics.items():
                if not np.isnan(metric_result.value) and not np.isinf(
                    metric_result.value
                ):
                    # Extract method name from metric result or use benchmark data
                    # For now, we'll need to track methods differently
                    pass

        # Alternative approach: rank methods within each benchmark
        for benchmark_name, benchmark_metrics in metric_results.items():
            benchmark_rankings = {}

            for metric_name in self.config.metrics:
                if metric_name in benchmark_metrics:
                    # For now, create dummy rankings
                    # In practice, this would rank methods based on their performance
                    method_scores = []

                    # This is a simplified version - in practice you'd extract
                    # method-specific scores from the benchmark results
                    metric_result = benchmark_metrics[metric_name]
                    if not np.isnan(metric_result.value):
                        method_scores.append(("method_aggregate", metric_result.value))

                    # Sort by score (higher is better for most metrics)
                    if metric_name in ["adaptation_speed", "generalization_error"]:
                        # Lower is better for these metrics
                        method_scores.sort(key=lambda x: x[1])
                    else:
                        # Higher is better
                        method_scores.sort(key=lambda x: x[1], reverse=True)

                    benchmark_rankings[metric_name] = method_scores

            rankings[benchmark_name] = benchmark_rankings

        return rankings

    def _generate_summary(
        self,
        benchmark_results: Dict[str, Dict[str, BenchmarkResult]],
        metric_results: Dict[str, Dict[str, MetricResult]],
        statistical_results: Dict[str, Dict[str, Dict[str, StatisticalTest]]],
        rankings: Dict[str, List[Tuple[str, float]]],
    ) -> Dict[str, Any]:
        """Generate comprehensive summary of comparison results."""
        logger.info("Generating summary...")

        summary = {
            "overview": {
                "n_benchmarks": len(self.config.benchmarks),
                "n_methods": len(
                    set(
                        method_name
                        for benchmark_data in benchmark_results.values()
                        for method_name in benchmark_data.keys()
                    )
                ),
                "n_metrics": len(self.config.metrics),
                "total_comparisons": 0,
            },
            "benchmark_summary": {},
            "method_performance": {},
            "statistical_significance": {},
            "recommendations": [],
        }

        # Benchmark summary
        for benchmark_name, benchmark_data in benchmark_results.items():
            n_methods = len(benchmark_data)
            n_tasks = sum(
                len(result.task_results) for result in benchmark_data.values()
            )
            total_time = sum(
                result.runtime_info.get("total_time", 0)
                for result in benchmark_data.values()
            )

            summary["benchmark_summary"][benchmark_name] = {
                "n_methods": n_methods,
                "n_tasks": n_tasks,
                "total_runtime": total_time,
                "avg_runtime_per_method": (
                    total_time / n_methods if n_methods > 0 else 0
                ),
            }

        # Method performance summary
        method_names = set()
        for benchmark_data in benchmark_results.values():
            method_names.update(benchmark_data.keys())

        for method_name in method_names:
            method_summary = {
                "benchmarks_completed": 0,
                "avg_metrics": {},
                "best_performance": [],
                "worst_performance": [],
            }

            # Count successful completions
            for benchmark_name, benchmark_data in benchmark_results.items():
                if (
                    method_name in benchmark_data
                    and "error" not in benchmark_data[method_name].metadata
                ):
                    method_summary["benchmarks_completed"] += 1

            summary["method_performance"][method_name] = method_summary

        # Statistical significance summary
        significant_differences = 0
        total_comparisons = 0

        for benchmark_name, benchmark_stats in statistical_results.items():
            for metric_name, metric_comparisons in benchmark_stats.items():
                for method1, method1_comparisons in metric_comparisons.items():
                    for method2, test_result in method1_comparisons.items():
                        total_comparisons += 1
                        if test_result.p_value < self.config.significance_level:
                            significant_differences += 1

        summary["overview"]["total_comparisons"] = total_comparisons
        summary["statistical_significance"] = {
            "total_tests": total_comparisons,
            "significant_differences": significant_differences,
            "significance_rate": (
                significant_differences / total_comparisons
                if total_comparisons > 0
                else 0
            ),
        }

        # Generate recommendations
        recommendations = []

        if significant_differences > 0:
            recommendations.append(
                f"Found {significant_differences} statistically significant differences "
                f"out of {total_comparisons} comparisons."
            )

        if len(method_names) > 1:
            recommendations.append(
                "Consider the effect sizes in addition to p-values when interpreting results."
            )

        if total_comparisons > 10:
            recommendations.append(
                f"With {total_comparisons} comparisons, consider using multiple comparison "
                f"correction (currently using {self.config.multiple_comparison_correction})."
            )

        summary["recommendations"] = recommendations

        return summary

    def _save_results(self, comparison_result: ComparisonResult):
        """Save comparison results to files."""
        logger.info("Saving comparison results...")

        timestamp_str = str(int(comparison_result.timestamp))

        # Save main results
        results_file = self.save_dir / f"comparison_results_{timestamp_str}.json"

        # Convert results to serializable format
        serializable_results = {
            "config": {
                "benchmarks": comparison_result.config.benchmarks,
                "metrics": comparison_result.config.metrics,
                "statistical_tests": comparison_result.config.statistical_tests,
                "significance_level": comparison_result.config.significance_level,
                "multiple_comparison_correction": comparison_result.config.multiple_comparison_correction,
            },
            "summary": comparison_result.summary,
            "timestamp": comparison_result.timestamp,
        }

        # Add benchmark results summary
        serializable_results["benchmark_results"] = {}
        for (
            benchmark_name,
            benchmark_data,
        ) in comparison_result.benchmark_results.items():
            serializable_results["benchmark_results"][benchmark_name] = {}
            for method_name, result in benchmark_data.items():
                serializable_results["benchmark_results"][benchmark_name][
                    method_name
                ] = {
                    "metrics": result.metrics,
                    "runtime_info": result.runtime_info,
                    "n_tasks": len(result.task_results),
                }

        # Add metric results
        serializable_results["metric_results"] = {}
        for (
            benchmark_name,
            benchmark_metrics,
        ) in comparison_result.metric_results.items():
            serializable_results["metric_results"][benchmark_name] = {}
            for metric_name, metric_result in benchmark_metrics.items():
                serializable_results["metric_results"][benchmark_name][metric_name] = {
                    "value": metric_result.value,
                    "std": metric_result.std,
                    "confidence_interval": metric_result.confidence_interval,
                    "metadata": metric_result.metadata,
                }

        # Add statistical results summary
        serializable_results["statistical_results"] = {}
        for (
            benchmark_name,
            benchmark_stats,
        ) in comparison_result.statistical_results.items():
            serializable_results["statistical_results"][benchmark_name] = {}
            for metric_name, metric_comparisons in benchmark_stats.items():
                serializable_results["statistical_results"][benchmark_name][
                    metric_name
                ] = {}
                for method1, method1_comparisons in metric_comparisons.items():
                    serializable_results["statistical_results"][benchmark_name][
                        metric_name
                    ][method1] = {}
                    for method2, test_result in method1_comparisons.items():
                        serializable_results["statistical_results"][benchmark_name][
                            metric_name
                        ][method1][method2] = {
                            "test_name": test_result.test_name,
                            "statistic": test_result.statistic,
                            "p_value": test_result.p_value,
                            "effect_size": test_result.effect_size,
                            "interpretation": test_result.interpretation,
                        }

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_results = convert_numpy_types(serializable_results)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved comparison results to {results_file}")

    def _generate_tables(self, comparison_result: ComparisonResult):
        """Generate LaTeX tables for comparison results."""
        logger.info("Generating LaTeX tables...")

        timestamp_str = str(int(comparison_result.timestamp))

        # Generate summary table
        summary_table = self._create_summary_table(comparison_result)
        summary_file = self.save_dir / f"summary_table_{timestamp_str}.tex"

        with open(summary_file, "w") as f:
            f.write(summary_table)

        # Generate detailed comparison tables for each benchmark
        for benchmark_name in comparison_result.config.benchmarks:
            if benchmark_name in comparison_result.benchmark_results:
                comparison_table = self._create_comparison_table(
                    comparison_result, benchmark_name
                )
                table_file = (
                    self.save_dir / f"{benchmark_name}_comparison_{timestamp_str}.tex"
                )

                with open(table_file, "w") as f:
                    f.write(comparison_table)

        logger.info("Generated LaTeX tables")

    def _create_summary_table(self, comparison_result: ComparisonResult) -> str:
        """Create LaTeX summary table."""

        table_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Method Comparison Summary}",
            "\\label{tab:method_comparison_summary}",
            "\\begin{tabular}{|l|c|c|c|}",
            "\\hline",
            "Benchmark & Methods & Tasks & Runtime (s) \\\\",
            "\\hline",
        ]

        for benchmark_name, benchmark_summary in comparison_result.summary[
            "benchmark_summary"
        ].items():
            n_methods = benchmark_summary["n_methods"]
            n_tasks = benchmark_summary["n_tasks"]
            total_runtime = benchmark_summary["total_runtime"]

            escaped_name = benchmark_name.replace("_", "\\_")
            table_lines.append(
                f"{escaped_name} & {n_methods} & {n_tasks} & {total_runtime:.2f} \\\\"
            )

        table_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

        return "\n".join(table_lines)

    def _create_comparison_table(
        self, comparison_result: ComparisonResult, benchmark_name: str
    ) -> str:
        """Create LaTeX comparison table for a specific benchmark."""

        if benchmark_name not in comparison_result.benchmark_results:
            return ""

        benchmark_data = comparison_result.benchmark_results[benchmark_name]
        method_names = list(benchmark_data.keys())

        escaped_benchmark_name = benchmark_name.replace("_", " ").title()
        escaped_metrics = [m.replace("_", "\\_") for m in self.config.metrics]

        table_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{escaped_benchmark_name} Benchmark Results}}",
            f"\\label{{tab:{benchmark_name}_results}}",
            "\\begin{tabular}{|l|" + "c|" * len(self.config.metrics) + "}",
            "\\hline",
            "Method & " + " & ".join(escaped_metrics) + " \\\\",
            "\\hline",
        ]

        for method_name in method_names:
            escaped_method_name = method_name.replace("_", "\\_")
            row_data = [escaped_method_name]

            for metric_name in self.config.metrics:
                if (
                    benchmark_name in comparison_result.metric_results
                    and metric_name in comparison_result.metric_results[benchmark_name]
                ):

                    metric_result = comparison_result.metric_results[benchmark_name][
                        metric_name
                    ]
                    value = metric_result.value

                    if np.isnan(value) or np.isinf(value):
                        row_data.append("N/A")
                    else:
                        row_data.append(f"{value:.3f}")
                else:
                    row_data.append("N/A")

            table_lines.append(" & ".join(row_data) + " \\\\")

        table_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

        return "\n".join(table_lines)

    def _generate_plots(self, comparison_result: ComparisonResult):
        """Generate plots for comparison results."""
        logger.info("Generating plots...")

        # For now, just log that plots would be generated
        # In a full implementation, this would create matplotlib/seaborn plots

        timestamp_str = str(int(comparison_result.timestamp))

        plot_types = [
            "method_performance_comparison",
            "statistical_significance_heatmap",
            "runtime_comparison",
            "convergence_analysis",
        ]

        for plot_type in plot_types:
            plot_file = self.save_dir / f"{plot_type}_{timestamp_str}.png"
            logger.info(f"Would generate plot: {plot_file}")

        logger.info("Plot generation completed (placeholder)")

    def load_comparison_results(self, results_file: str) -> ComparisonResult:
        """Load previously saved comparison results."""

        with open(results_file, "r") as f:
            data = json.load(f)

        # Reconstruct comparison result from saved data
        # This is a simplified version - full implementation would
        # properly reconstruct all objects

        config = MethodComparisonConfig(
            benchmarks=data["config"]["benchmarks"],
            metrics=data["config"]["metrics"],
            statistical_tests=data["config"]["statistical_tests"],
            significance_level=data["config"]["significance_level"],
            multiple_comparison_correction=data["config"][
                "multiple_comparison_correction"
            ],
        )

        # Create placeholder objects for the loaded data
        comparison_result = ComparisonResult(
            config=config,
            benchmark_results={},  # Would reconstruct from data
            metric_results={},  # Would reconstruct from data
            statistical_results={},  # Would reconstruct from data
            rankings={},  # Would reconstruct from data
            summary=data["summary"],
            timestamp=data["timestamp"],
        )

        return comparison_result

    def get_method_ranking(
        self, comparison_result: ComparisonResult, metric_name: str
    ) -> List[Tuple[str, float]]:
        """Get method ranking for a specific metric across all benchmarks."""

        method_scores = defaultdict(list)

        # Aggregate scores across benchmarks
        for (
            benchmark_name,
            benchmark_metrics,
        ) in comparison_result.metric_results.items():
            if metric_name in benchmark_metrics:
                metric_result = benchmark_metrics[metric_name]
                if not np.isnan(metric_result.value) and not np.isinf(
                    metric_result.value
                ):
                    # This is simplified - would need to track method-specific scores
                    method_scores["aggregate_method"].append(metric_result.value)

        # Compute average scores
        method_averages = []
        for method_name, scores in method_scores.items():
            if scores:
                avg_score = np.mean(scores)
                method_averages.append((method_name, avg_score))

        # Sort by score
        if metric_name in ["adaptation_speed", "generalization_error"]:
            method_averages.sort(key=lambda x: x[1])  # Lower is better
        else:
            method_averages.sort(key=lambda x: x[1], reverse=True)  # Higher is better

        return method_averages

"""
Advanced result management and analysis system for experiments.
"""

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .logging_utils import LoggerMixin


@dataclass
class ExperimentResult:
    """Structured experiment result."""

    experiment_id: str
    name: str
    timestamp: str
    status: str  # completed, failed, running

    # Configuration
    config_hash: str
    hyperparameters: Dict[str, Any]

    # Results
    final_metrics: Dict[str, float]
    best_metrics: Dict[str, float]
    training_history: List[Dict[str, Any]]

    # Metadata
    duration_seconds: Optional[float]
    hardware_info: Dict[str, Any]
    reproducibility_info: Dict[str, Any]

    # Analysis
    convergence_analysis: Optional[Dict[str, Any]] = None
    statistical_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ResultComparison:
    """Comparison between experiment results."""

    experiment_ids: List[str]
    comparison_type: str  # pairwise, group, baseline

    # Metric comparisons
    metric_comparisons: Dict[str, Dict[str, Any]]
    statistical_tests: Dict[str, Dict[str, Any]]

    # Performance analysis
    performance_ranking: List[Tuple[str, float]]

    # Summary
    summary: Dict[str, Any]
    timestamp: str

    # Optional fields
    significance_matrix: Optional[np.ndarray] = None
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.significance_matrix is not None:
            data["significance_matrix"] = self.significance_matrix.tolist()
        return data


class ResultManager(LoggerMixin):
    """Advanced result management system."""

    def __init__(self, results_dir: Path):
        """Initialize result manager.

        Args:
            results_dir: Directory for storing results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Database files
        self.results_db_file = self.results_dir / "results_database.json"
        self.comparisons_file = self.results_dir / "comparisons.json"
        self.analysis_cache_file = self.results_dir / "analysis_cache.json"

        # Load existing data
        self.results_db = self._load_results_db()
        self.comparisons = self._load_comparisons()
        self.analysis_cache = self._load_analysis_cache()

    def _load_results_db(self) -> Dict[str, Any]:
        """Load results database."""
        if self.results_db_file.exists():
            with open(self.results_db_file, "r") as f:
                return json.load(f)
        return {"results": {}, "metadata": {"created_at": datetime.now().isoformat()}}

    def _save_results_db(self):
        """Save results database."""
        self.results_db["metadata"]["updated_at"] = datetime.now().isoformat()
        with open(self.results_db_file, "w") as f:
            json.dump(self.results_db, f, indent=2, default=str)

    def _load_comparisons(self) -> Dict[str, Any]:
        """Load comparisons."""
        if self.comparisons_file.exists():
            with open(self.comparisons_file, "r") as f:
                return json.load(f)
        return {}

    def _save_comparisons(self):
        """Save comparisons."""
        with open(self.comparisons_file, "w") as f:
            json.dump(self.comparisons, f, indent=2, default=str)

    def _load_analysis_cache(self) -> Dict[str, Any]:
        """Load analysis cache."""
        if self.analysis_cache_file.exists():
            with open(self.analysis_cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_analysis_cache(self):
        """Save analysis cache."""
        with open(self.analysis_cache_file, "w") as f:
            json.dump(self.analysis_cache, f, indent=2, default=str)

    def store_result(self, result: ExperimentResult):
        """Store experiment result.

        Args:
            result: Experiment result to store
        """
        # Perform analysis
        if result.training_history:
            result.convergence_analysis = self._analyze_convergence(
                result.training_history
            )
            result.statistical_summary = self._compute_statistical_summary(
                result.final_metrics
            )

        # Store in database
        self.results_db["results"][result.experiment_id] = result.to_dict()
        self._save_results_db()

        # Save detailed result file
        result_file = self.results_dir / f"{result.experiment_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Save training history separately if large
        if result.training_history and len(result.training_history) > 100:
            history_file = self.results_dir / f"{result.experiment_id}_history.pkl"
            with open(history_file, "wb") as f:
                pickle.dump(result.training_history, f)

        self.log_info(f"Stored result for experiment {result.experiment_id}")

    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment result.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment result or None if not found
        """
        if experiment_id not in self.results_db["results"]:
            return None

        result_data = self.results_db["results"][experiment_id]

        # Load training history from separate file if needed
        history_file = self.results_dir / f"{experiment_id}_history.pkl"
        if history_file.exists() and not result_data.get("training_history"):
            with open(history_file, "rb") as f:
                result_data["training_history"] = pickle.load(f)

        return ExperimentResult.from_dict(result_data)

    def list_results(
        self,
        status: Optional[str] = None,
        name_pattern: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> List[str]:
        """List experiment results with filtering.

        Args:
            status: Filter by status
            name_pattern: Filter by name pattern
            date_range: Filter by date range (start, end)

        Returns:
            List of experiment IDs
        """
        results = []

        for exp_id, result_data in self.results_db["results"].items():
            # Status filter
            if status and result_data.get("status") != status:
                continue

            # Name pattern filter
            if name_pattern and name_pattern not in result_data.get("name", ""):
                continue

            # Date range filter
            if date_range:
                result_date = result_data.get("timestamp")
                if result_date:
                    if result_date < date_range[0] or result_date > date_range[1]:
                        continue

            results.append(exp_id)

        return results

    def compare_results(
        self,
        experiment_ids: List[str],
        comparison_type: str = "group",
        metrics: Optional[List[str]] = None,
    ) -> ResultComparison:
        """Compare experiment results.

        Args:
            experiment_ids: List of experiment IDs to compare
            comparison_type: Type of comparison (pairwise, group, baseline)
            metrics: Specific metrics to compare (None for all)

        Returns:
            Result comparison
        """
        # Check cache first
        comparison_id = hashlib.md5(
            json.dumps(experiment_ids + [comparison_type], sort_keys=True).encode()
        ).hexdigest()

        if comparison_id in self.comparisons:
            # Return cached comparison
            cached_data = self.comparisons[comparison_id]
            return ResultComparison(
                experiment_ids=cached_data["experiment_ids"],
                comparison_type=cached_data["comparison_type"],
                metric_comparisons=cached_data["metric_comparisons"],
                statistical_tests=cached_data["statistical_tests"],
                performance_ranking=cached_data["performance_ranking"],
                significance_matrix=(
                    np.array(cached_data["significance_matrix"])
                    if cached_data.get("significance_matrix")
                    else None
                ),
                similarity_score=cached_data.get("similarity_score", 0.0),
                summary=cached_data["summary"],
                timestamp=cached_data["timestamp"],
            )

        # Load results
        results = []
        for exp_id in experiment_ids:
            result = self.get_result(exp_id)
            if result is None:
                raise ValueError(f"Result not found for experiment {exp_id}")
            results.append(result)

        # Determine metrics to compare
        if metrics is None:
            all_metrics = set()
            for result in results:
                all_metrics.update(result.final_metrics.keys())
            metrics = list(all_metrics)

        # Perform metric comparisons
        metric_comparisons = {}
        for metric in metrics:
            metric_values = []
            for result in results:
                if metric in result.final_metrics:
                    metric_values.append(result.final_metrics[metric])
                else:
                    metric_values.append(None)

            metric_comparisons[metric] = self._compare_metric_values(
                experiment_ids, metric_values, metric
            )

        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(results, metrics)

        # Create performance ranking
        performance_ranking = self._create_performance_ranking(results, metrics)

        # Create significance matrix
        significance_matrix = self._create_significance_matrix(
            statistical_tests, experiment_ids
        )

        # Generate summary
        summary = self._generate_comparison_summary(
            results, metric_comparisons, statistical_tests, performance_ranking
        )

        # Calculate similarity score based on metric differences
        similarity_score = 0.0
        if len(experiment_ids) == 2:
            # For pairwise comparison, calculate similarity based on metric differences
            total_diff = 0.0
            metric_count = 0
            for metric, comparison_data in metric_comparisons.items():
                if "difference" in comparison_data:
                    total_diff += abs(comparison_data["difference"])
                    metric_count += 1
            if metric_count > 0:
                similarity_score = max(0.0, 1.0 - (total_diff / metric_count))

        comparison = ResultComparison(
            experiment_ids=experiment_ids,
            comparison_type=comparison_type,
            metric_comparisons=metric_comparisons,
            statistical_tests=statistical_tests,
            performance_ranking=performance_ranking,
            significance_matrix=significance_matrix,
            similarity_score=similarity_score,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )

        # Cache comparison (comparison_id already calculated above)
        self.comparisons[comparison_id] = comparison.to_dict()
        self._save_comparisons()

        return comparison

    def _compare_metric_values(
        self, experiment_ids: List[str], values: List[Optional[float]], metric_name: str
    ) -> Dict[str, Any]:
        """Compare metric values across experiments."""
        valid_values = [v for v in values if v is not None]
        valid_ids = [
            exp_id for exp_id, v in zip(experiment_ids, values) if v is not None
        ]

        if not valid_values:
            return {"error": "No valid values for comparison"}

        comparison = {
            "metric_name": metric_name,
            "experiment_values": dict(zip(experiment_ids, values)),
            "statistics": {
                "mean": np.mean(valid_values),
                "std": np.std(valid_values),
                "min": np.min(valid_values),
                "max": np.max(valid_values),
                "median": np.median(valid_values),
            },
            "best_experiment": valid_ids[
                np.argmin(valid_values)
            ],  # Assuming lower is better
            "worst_experiment": valid_ids[np.argmax(valid_values)],
            "relative_differences": {},
        }

        # Calculate relative differences from best
        best_value = np.min(valid_values)
        for exp_id, value in zip(experiment_ids, values):
            if value is not None:
                rel_diff = (
                    (value - best_value) / abs(best_value) if best_value != 0 else 0
                )
                comparison["relative_differences"][exp_id] = rel_diff

        return comparison

    def _perform_statistical_tests(
        self, results: List[ExperimentResult], metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform statistical tests on results."""
        try:
            from scipy import stats
        except ImportError:
            self.log_warning("SciPy not available for statistical tests")
            return {}

        statistical_tests = {}

        for metric in metrics:
            # Collect values for this metric
            metric_data = []
            experiment_ids = []

            for result in results:
                if metric in result.final_metrics:
                    metric_data.append(result.final_metrics[metric])
                    experiment_ids.append(result.experiment_id)

            if len(metric_data) < 2:
                continue

            # Perform tests
            tests = {}

            # Normality test (Shapiro-Wilk)
            if len(metric_data) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(metric_data)
                tests["normality"] = {
                    "test": "shapiro_wilk",
                    "statistic": shapiro_stat,
                    "p_value": shapiro_p,
                    "is_normal": shapiro_p > 0.05,
                }

            # Pairwise comparisons (if more than 2 experiments)
            if len(metric_data) > 2:
                # ANOVA (if normal) or Kruskal-Wallis (if not normal)
                is_normal = tests.get("normality", {}).get("is_normal", True)

                if is_normal:
                    # One-way ANOVA
                    f_stat, anova_p = stats.f_oneway(*[[val] for val in metric_data])
                    tests["group_comparison"] = {
                        "test": "anova",
                        "statistic": f_stat,
                        "p_value": anova_p,
                        "significant": anova_p < 0.05,
                    }
                else:
                    # Kruskal-Wallis test
                    h_stat, kw_p = stats.kruskal(*[[val] for val in metric_data])
                    tests["group_comparison"] = {
                        "test": "kruskal_wallis",
                        "statistic": h_stat,
                        "p_value": kw_p,
                        "significant": kw_p < 0.05,
                    }

            # Pairwise t-tests or Mann-Whitney U tests
            pairwise_tests = {}
            for i in range(len(metric_data)):
                for j in range(i + 1, len(metric_data)):
                    exp_id_1 = experiment_ids[i]
                    exp_id_2 = experiment_ids[j]
                    val_1 = metric_data[i]
                    val_2 = metric_data[j]

                    # Since we only have single values, we can't perform proper statistical tests
                    # Instead, we'll compute effect size and relative difference
                    effect_size = abs(val_1 - val_2) / max(
                        abs(val_1), abs(val_2), 1e-10
                    )

                    pairwise_tests[f"{exp_id_1}_vs_{exp_id_2}"] = {
                        "values": [val_1, val_2],
                        "difference": val_1 - val_2,
                        "relative_difference": (
                            (val_1 - val_2) / abs(val_2) if val_2 != 0 else 0
                        ),
                        "effect_size": effect_size,
                        "better_experiment": (
                            exp_id_1 if val_1 < val_2 else exp_id_2
                        ),  # Assuming lower is better
                    }

            tests["pairwise"] = pairwise_tests
            statistical_tests[metric] = tests

        return statistical_tests

    def _create_performance_ranking(
        self, results: List[ExperimentResult], metrics: List[str]
    ) -> List[Tuple[str, float]]:
        """Create performance ranking of experiments."""
        # Simple ranking based on average normalized performance
        experiment_scores = {}

        for result in results:
            scores = []
            for metric in metrics:
                if metric in result.final_metrics:
                    # Normalize metric (assuming lower is better)
                    all_values = [
                        r.final_metrics.get(metric)
                        for r in results
                        if metric in r.final_metrics
                    ]
                    all_values = [v for v in all_values if v is not None]

                    if all_values:
                        min_val = min(all_values)
                        max_val = max(all_values)

                        if max_val != min_val:
                            normalized = (result.final_metrics[metric] - min_val) / (
                                max_val - min_val
                            )
                        else:
                            normalized = 0.0

                        scores.append(normalized)

            if scores:
                experiment_scores[result.experiment_id] = np.mean(scores)
            else:
                experiment_scores[result.experiment_id] = float("inf")

        # Sort by score (lower is better)
        ranking = sorted(experiment_scores.items(), key=lambda x: x[1])

        return ranking

    def _create_significance_matrix(
        self, statistical_tests: Dict[str, Dict[str, Any]], experiment_ids: List[str]
    ) -> Optional[np.ndarray]:
        """Create significance matrix for pairwise comparisons."""
        n_experiments = len(experiment_ids)
        if n_experiments < 2:
            return None

        # Initialize matrix with zeros
        significance_matrix = np.zeros((n_experiments, n_experiments))

        # Fill matrix based on pairwise tests
        for metric, tests in statistical_tests.items():
            if "pairwise" in tests:
                for comparison, test_result in tests["pairwise"].items():
                    # Parse experiment IDs from comparison key
                    exp_ids = comparison.replace("_vs_", " ").split()
                    if len(exp_ids) == 2:
                        try:
                            idx1 = experiment_ids.index(exp_ids[0])
                            idx2 = experiment_ids.index(exp_ids[1])

                            # Use effect size as significance measure
                            effect_size = test_result.get("effect_size", 0)
                            significance_matrix[idx1, idx2] = effect_size
                            significance_matrix[idx2, idx1] = effect_size
                        except ValueError:
                            continue

        return significance_matrix

    def _generate_comparison_summary(
        self,
        results: List[ExperimentResult],
        metric_comparisons: Dict[str, Dict[str, Any]],
        statistical_tests: Dict[str, Dict[str, Any]],
        performance_ranking: List[Tuple[str, float]],
    ) -> Dict[str, Any]:
        """Generate comparison summary."""
        summary = {
            "total_experiments": len(results),
            "compared_metrics": list(metric_comparisons.keys()),
            "best_overall_experiment": (
                performance_ranking[0][0] if performance_ranking else None
            ),
            "metric_winners": {},
            "significant_differences": [],
            "recommendations": [],
        }

        # Find best experiment for each metric
        for metric, comparison in metric_comparisons.items():
            if "best_experiment" in comparison:
                summary["metric_winners"][metric] = comparison["best_experiment"]

        # Find significant differences
        for metric, tests in statistical_tests.items():
            if "pairwise" in tests:
                for comparison, test_result in tests["pairwise"].items():
                    effect_size = test_result.get("effect_size", 0)
                    if effect_size > 0.2:  # Medium effect size threshold
                        summary["significant_differences"].append(
                            {
                                "metric": metric,
                                "comparison": comparison,
                                "effect_size": effect_size,
                                "better_experiment": test_result.get(
                                    "better_experiment"
                                ),
                            }
                        )

        # Generate recommendations
        if performance_ranking:
            best_exp = performance_ranking[0][0]
            summary["recommendations"].append(
                f"Experiment {best_exp} shows the best overall performance"
            )

        if len(summary["significant_differences"]) > 0:
            summary["recommendations"].append(
                f"Found {len(summary['significant_differences'])} significant differences between experiments"
            )
        else:
            summary["recommendations"].append(
                "No significant differences found between experiments"
            )

        return summary

    def _analyze_convergence(
        self, training_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze convergence from training history."""
        if not training_history:
            return {}

        # Extract loss values
        loss_values = []
        steps = []

        for entry in training_history:
            if "loss" in entry:
                loss_values.append(entry["loss"])
                steps.append(entry.get("step", len(steps)))

        if not loss_values:
            return {}

        analysis = {
            "total_steps": len(loss_values),
            "initial_loss": loss_values[0],
            "final_loss": loss_values[-1],
            "best_loss": min(loss_values),
            "loss_reduction": loss_values[0] - loss_values[-1],
            "relative_improvement": (
                (loss_values[0] - loss_values[-1]) / abs(loss_values[0])
                if loss_values[0] != 0
                else 0
            ),
        }

        # Detect convergence point (where loss stops improving significantly)
        convergence_step = None
        window_size = min(10, len(loss_values) // 4)

        if window_size > 1:
            for i in range(window_size, len(loss_values)):
                recent_window = loss_values[i - window_size : i]
                current_loss = loss_values[i]

                # Check if improvement is less than 1% over the window
                min_recent = min(recent_window)
                if (
                    abs(min_recent) > 1e-10
                    and abs(current_loss - min_recent) / abs(min_recent) < 0.01
                ):
                    convergence_step = steps[i]
                    break

        analysis["convergence_step"] = convergence_step

        # Stability analysis (variance in final 20% of training)
        final_portion = loss_values[int(0.8 * len(loss_values)) :]
        if len(final_portion) > 1:
            analysis["final_stability"] = {
                "mean": np.mean(final_portion),
                "std": np.std(final_portion),
                "coefficient_of_variation": (
                    np.std(final_portion) / abs(np.mean(final_portion))
                    if np.mean(final_portion) != 0
                    else 0
                ),
            }

        return analysis

    def _compute_statistical_summary(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compute statistical summary of metrics."""
        if not metrics:
            return {}

        values = list(metrics.values())

        summary = {
            "num_metrics": len(metrics),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "metrics": list(metrics.keys()),
        }

        return summary

    def generate_results_report(
        self,
        experiment_ids: Optional[List[str]] = None,
        include_comparisons: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive results report.

        Args:
            experiment_ids: Specific experiments to include (None for all)
            include_comparisons: Whether to include comparison analysis

        Returns:
            Comprehensive results report
        """
        if experiment_ids is None:
            experiment_ids = list(self.results_db["results"].keys())

        # Load results
        results = []
        for exp_id in experiment_ids:
            result = self.get_result(exp_id)
            if result:
                results.append(result)

        # Basic statistics
        completed_results = [r for r in results if r.status == "completed"]
        failed_results = [r for r in results if r.status == "failed"]

        report = {
            "summary": {
                "total_experiments": len(results),
                "completed_experiments": len(completed_results),
                "failed_experiments": len(failed_results),
                "success_rate": len(completed_results) / len(results) if results else 0,
                "generated_at": datetime.now().isoformat(),
            },
            "experiment_details": [r.to_dict() for r in results],
            "performance_analysis": {},
            "reproducibility_analysis": {},
        }

        # Performance analysis
        if completed_results:
            all_metrics = set()
            for result in completed_results:
                all_metrics.update(result.final_metrics.keys())

            performance_stats = {}
            for metric in all_metrics:
                values = [
                    r.final_metrics[metric]
                    for r in completed_results
                    if metric in r.final_metrics
                ]
                if values:
                    performance_stats[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "best_experiment": [
                            r.experiment_id
                            for r in completed_results
                            if r.final_metrics.get(metric) == np.min(values)
                        ][0],
                    }

            report["performance_analysis"] = performance_stats

        # Reproducibility analysis
        reproducible_count = sum(
            1
            for r in results
            if r.reproducibility_info.get("validation_performed", False)
            and r.reproducibility_info.get("overall_status") == "PASSED"
        )

        report["reproducibility_analysis"] = {
            "reproducible_experiments": reproducible_count,
            "reproducibility_rate": reproducible_count / len(results) if results else 0,
            "validation_performed": sum(
                1
                for r in results
                if r.reproducibility_info.get("validation_performed", False)
            ),
        }

        # Include comparisons if requested
        if include_comparisons and len(completed_results) > 1:
            comparison = self.compare_results(
                [r.experiment_id for r in completed_results]
            )
            report["comparison_analysis"] = comparison.to_dict()

        return report

    def export_results(self, output_file: Path, format: str = "json"):
        """Export all results to file.

        Args:
            output_file: Output file path
            format: Export format (json, csv, excel)
        """
        if format == "json":
            export_data = {
                "results_db": self.results_db,
                "comparisons": self.comparisons,
                "exported_at": datetime.now().isoformat(),
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

        elif format == "csv":
            # Create DataFrame from results
            rows = []
            for exp_id, result_data in self.results_db["results"].items():
                row = {
                    "experiment_id": exp_id,
                    "name": result_data.get("name"),
                    "status": result_data.get("status"),
                    "timestamp": result_data.get("timestamp"),
                    "duration_seconds": result_data.get("duration_seconds"),
                }

                # Add final metrics
                final_metrics = result_data.get("final_metrics", {})
                for metric, value in final_metrics.items():
                    row[f"final_{metric}"] = value

                # Add hyperparameters
                hyperparams = result_data.get("hyperparameters", {})
                for param, value in hyperparams.items():
                    row[f"param_{param}"] = value

                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)

        elif format == "excel":
            # Create multiple sheets
            try:
                import openpyxl
            except ImportError:
                raise ImportError(
                    "openpyxl is required for Excel export. Install with: pip install openpyxl"
                )

            with pd.ExcelWriter(output_file) as writer:
                # Results summary
                rows = []
                for exp_id, result_data in self.results_db["results"].items():
                    row = {
                        "experiment_id": exp_id,
                        "name": result_data.get("name"),
                        "status": result_data.get("status"),
                        "timestamp": result_data.get("timestamp"),
                        "duration_seconds": result_data.get("duration_seconds"),
                    }

                    # Add final metrics
                    final_metrics = result_data.get("final_metrics", {})
                    for metric, value in final_metrics.items():
                        row[f"final_{metric}"] = value

                    rows.append(row)

                df_results = pd.DataFrame(rows)
                df_results.to_excel(writer, sheet_name="Results", index=False)

                # Hyperparameters sheet
                hyperparam_rows = []
                for exp_id, result_data in self.results_db["results"].items():
                    hyperparams = result_data.get("hyperparameters", {})
                    for param, value in hyperparams.items():
                        hyperparam_rows.append(
                            {
                                "experiment_id": exp_id,
                                "parameter": param,
                                "value": value,
                            }
                        )

                if hyperparam_rows:
                    df_hyperparams = pd.DataFrame(hyperparam_rows)
                    df_hyperparams.to_excel(
                        writer, sheet_name="Hyperparameters", index=False
                    )

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.log_info(f"Exported results to {output_file} in {format} format")

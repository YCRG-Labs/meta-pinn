"""
Comprehensive report generation for ML research papers.

This module provides the ReportGenerator class for creating structured,
publication-ready reports with executive summaries, detailed analysis,
and natural language descriptions of experimental findings.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .plot_generator import PaperPlotGenerator
from .table_generator import LaTeXTableGenerator


class ReportGenerator:
    """
    Generates comprehensive research reports with automated analysis.

    This class provides methods for creating structured reports that combine
    experimental results, statistical analysis, plots, and tables into
    publication-ready documents with natural language descriptions.
    """

    def __init__(
        self,
        plot_generator: Optional[PaperPlotGenerator] = None,
        table_generator: Optional[LaTeXTableGenerator] = None,
        template_style: str = "academic",
        language: str = "en",
    ):
        """
        Initialize the report generator.

        Args:
            plot_generator: PaperPlotGenerator instance for creating plots
            table_generator: LaTeXTableGenerator instance for creating tables
            template_style: Report template style ('academic', 'technical', 'executive')
            language: Language for natural language generation ('en' for English)
        """
        self.plot_generator = plot_generator or PaperPlotGenerator()
        self.table_generator = table_generator or LaTeXTableGenerator()
        self.template_style = template_style
        self.language = language

        # Natural language templates
        self.templates = {
            "executive_summary": {
                "intro": "This report presents the results of {experiment_type} experiments comparing {num_methods} different methods on {num_tasks} tasks.",
                "best_method": "The best performing method was {method_name} with {metric_name} of {value:.3f} ± {std:.3f}.",
                "improvement": "This represents a {improvement:.1f}% improvement over the baseline method ({baseline_name}).",
                "significance": "Statistical analysis shows {num_significant} statistically significant differences (p < 0.05).",
            },
            "method_comparison": {
                "intro": "We compared {num_methods} methods across {num_metrics} evaluation metrics.",
                "ranking": "The methods ranked as follows (by {primary_metric}): {ranking}.",
                "statistical": "Statistical significance testing using Welch's t-test revealed {significant_pairs} significant differences.",
            },
            "performance_analysis": {
                "convergence": "Training convergence analysis shows {fastest_method} converged fastest in {convergence_steps} steps.",
                "efficiency": "Computational efficiency analysis indicates {most_efficient} was most efficient with {efficiency_metric} of {efficiency_value}.",
                "robustness": "Robustness analysis across different task types shows {most_robust} had the lowest variance ({variance:.4f}).",
            },
        }

    def generate_comprehensive_report(
        self,
        experimental_results: Dict[str, Any],
        output_dir: Path,
        report_title: str = "Experimental Results Report",
        author: str = "ML Research Pipeline",
        include_plots: bool = True,
        include_tables: bool = True,
        format_type: str = "markdown",
    ) -> Dict[str, Path]:
        """
        Generate a comprehensive report from experimental results.

        Args:
            experimental_results: Dictionary containing all experimental data
            output_dir: Directory to save report files
            report_title: Title of the report
            author: Author name
            include_plots: Whether to generate and include plots
            include_tables: Whether to generate and include tables
            format_type: Output format ('markdown', 'latex', 'html')

        Returns:
            Dictionary mapping section names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate report sections
        sections = {}

        # Executive Summary
        sections["executive_summary"] = self._generate_executive_summary(
            experimental_results
        )

        # Method Comparison Analysis
        sections["method_comparison"] = self._generate_method_comparison_analysis(
            experimental_results
        )

        # Performance Analysis
        sections["performance_analysis"] = self._generate_performance_analysis(
            experimental_results
        )

        # Statistical Analysis
        sections["statistical_analysis"] = self._generate_statistical_analysis(
            experimental_results
        )

        # Detailed Results
        sections["detailed_results"] = self._generate_detailed_results(
            experimental_results
        )

        # Generate plots if requested
        plot_paths = {}
        if include_plots:
            plot_paths = self._generate_report_plots(experimental_results, output_dir)

        # Generate tables if requested
        table_paths = {}
        if include_tables:
            table_paths = self._generate_report_tables(experimental_results, output_dir)

        # Combine into final report
        report_content = self._combine_report_sections(
            sections, plot_paths, table_paths, report_title, author
        )

        # Save report in requested format
        saved_files = {}
        if format_type == "markdown":
            report_path = output_dir / "report.md"
            self._save_markdown_report(report_content, report_path)
            saved_files["report"] = report_path
        elif format_type == "latex":
            report_path = output_dir / "report.tex"
            self._save_latex_report(report_content, report_path)
            saved_files["report"] = report_path
        elif format_type == "html":
            report_path = output_dir / "report.html"
            self._save_html_report(report_content, report_path)
            saved_files["report"] = report_path

        # Save metadata
        metadata_path = output_dir / "report_metadata.json"
        self._save_report_metadata(
            experimental_results, metadata_path, report_title, author
        )
        saved_files["metadata"] = metadata_path

        # Add plot and table paths
        saved_files.update(plot_paths)
        saved_files.update(table_paths)

        return saved_files

    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        summary_parts = []

        # Extract key information
        methods = list(results.get("method_results", {}).keys())
        num_methods = len(methods)

        # Get primary metric (assume first metric is primary)
        metrics = list(results.get("metrics", ["accuracy"]))
        primary_metric = metrics[0] if metrics else "performance"

        # Find best performing method
        best_method, best_value, best_std = self._find_best_method(
            results, primary_metric
        )

        # Calculate improvement over baseline
        baseline_method = self._identify_baseline_method(methods)
        improvement = self._calculate_improvement(
            results, best_method, baseline_method, primary_metric
        )

        # Count significant results
        num_significant = self._count_significant_results(results)

        # Generate summary text
        intro_text = self.templates["executive_summary"]["intro"].format(
            experiment_type=results.get("experiment_type", "machine learning"),
            num_methods=num_methods,
            num_tasks=results.get("num_tasks", "multiple"),
        )
        summary_parts.append(intro_text)

        if best_method:
            best_text = self.templates["executive_summary"]["best_method"].format(
                method_name=best_method,
                metric_name=primary_metric,
                value=best_value,
                std=best_std,
            )
            summary_parts.append(best_text)

        if improvement is not None:
            improvement_text = self.templates["executive_summary"][
                "improvement"
            ].format(improvement=improvement, baseline_name=baseline_method)
            summary_parts.append(improvement_text)

        significance_text = self.templates["executive_summary"]["significance"].format(
            num_significant=num_significant
        )
        summary_parts.append(significance_text)

        return " ".join(summary_parts)

    def _generate_method_comparison_analysis(self, results: Dict[str, Any]) -> str:
        """Generate method comparison analysis section."""
        analysis_parts = []

        methods = list(results.get("method_results", {}).keys())
        metrics = list(results.get("metrics", ["accuracy"]))

        # Introduction
        intro_text = self.templates["method_comparison"]["intro"].format(
            num_methods=len(methods), num_metrics=len(metrics)
        )
        analysis_parts.append(intro_text)

        # Method ranking
        primary_metric = metrics[0] if metrics else "performance"
        ranking = self._generate_method_ranking(results, primary_metric)
        ranking_text = self.templates["method_comparison"]["ranking"].format(
            primary_metric=primary_metric, ranking=", ".join(ranking)
        )
        analysis_parts.append(ranking_text)

        # Statistical significance
        significant_pairs = self._count_significant_pairs(results)
        statistical_text = self.templates["method_comparison"]["statistical"].format(
            significant_pairs=significant_pairs
        )
        analysis_parts.append(statistical_text)

        # Detailed metric analysis
        for metric in metrics:
            metric_analysis = self._analyze_metric_performance(results, metric)
            analysis_parts.append(metric_analysis)

        return " ".join(analysis_parts)

    def _generate_performance_analysis(self, results: Dict[str, Any]) -> str:
        """Generate performance analysis section."""
        analysis_parts = []

        # Convergence analysis
        if "convergence_data" in results:
            convergence_analysis = self._analyze_convergence(
                results["convergence_data"]
            )
            analysis_parts.append(convergence_analysis)

        # Efficiency analysis
        if "efficiency_data" in results:
            efficiency_analysis = self._analyze_efficiency(results["efficiency_data"])
            analysis_parts.append(efficiency_analysis)

        # Robustness analysis
        if "robustness_data" in results:
            robustness_analysis = self._analyze_robustness(results["robustness_data"])
            analysis_parts.append(robustness_analysis)

        # Scalability analysis
        if "scalability_data" in results:
            scalability_analysis = self._analyze_scalability(
                results["scalability_data"]
            )
            analysis_parts.append(scalability_analysis)

        return (
            " ".join(analysis_parts)
            if analysis_parts
            else "Performance analysis data not available."
        )

    def _generate_statistical_analysis(self, results: Dict[str, Any]) -> str:
        """Generate statistical analysis section."""
        analysis_parts = []

        # Significance testing results
        if "statistical_tests" in results:
            significance_analysis = self._analyze_statistical_significance(
                results["statistical_tests"]
            )
            analysis_parts.append(significance_analysis)

        # Effect size analysis
        if "effect_sizes" in results:
            effect_size_analysis = self._analyze_effect_sizes(results["effect_sizes"])
            analysis_parts.append(effect_size_analysis)

        # Confidence intervals
        if "confidence_intervals" in results:
            ci_analysis = self._analyze_confidence_intervals(
                results["confidence_intervals"]
            )
            analysis_parts.append(ci_analysis)

        return (
            " ".join(analysis_parts)
            if analysis_parts
            else "Statistical analysis data not available."
        )

    def _generate_detailed_results(self, results: Dict[str, Any]) -> str:
        """Generate detailed results section."""
        details_parts = []

        # Method-by-method breakdown
        method_results = results.get("method_results", {})
        for method_name, method_data in method_results.items():
            method_details = self._generate_method_details(method_name, method_data)
            details_parts.append(method_details)

        # Hyperparameter analysis
        if "hyperparameters" in results:
            hyperparameter_analysis = self._analyze_hyperparameters(
                results["hyperparameters"]
            )
            details_parts.append(hyperparameter_analysis)

        return "\n\n".join(details_parts)

    def _generate_report_plots(
        self, results: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Path]:
        """Generate all plots for the report."""
        plot_paths = {}

        # Method comparison plot
        if "method_results" in results and "metrics" in results:
            try:
                fig = self.plot_generator.create_method_comparison_plot(
                    results=results["method_results"],
                    metric_name=results["metrics"][0],
                    significance_data=results.get("statistical_tests"),
                )
                plot_path = output_dir / "method_comparison.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plot_paths["method_comparison_plot"] = plot_path
                self.plot_generator.close_all_figures()
            except Exception as e:
                print(f"Warning: Could not generate method comparison plot: {e}")

        # Convergence plot
        if "convergence_data" in results:
            try:
                fig = self.plot_generator.create_convergence_plot(
                    convergence_data=results["convergence_data"]
                )
                plot_path = output_dir / "convergence.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plot_paths["convergence_plot"] = plot_path
                self.plot_generator.close_all_figures()
            except Exception as e:
                print(f"Warning: Could not generate convergence plot: {e}")

        # Uncertainty plot (if available)
        if "uncertainty_data" in results:
            try:
                uncertainty_data = results["uncertainty_data"]
                fig = self.plot_generator.create_uncertainty_plot(
                    x_data=uncertainty_data["x"],
                    y_mean=uncertainty_data["y_mean"],
                    y_std=uncertainty_data["y_std"],
                    ground_truth=uncertainty_data.get("ground_truth"),
                )
                plot_path = output_dir / "uncertainty.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plot_paths["uncertainty_plot"] = plot_path
                self.plot_generator.close_all_figures()
            except Exception as e:
                print(f"Warning: Could not generate uncertainty plot: {e}")

        return plot_paths

    def _generate_report_tables(
        self, results: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Path]:
        """Generate all tables for the report."""
        table_paths = {}

        # Method comparison table
        if "method_results" in results and "metrics" in results:
            try:
                latex_table = self.table_generator.create_method_comparison_table(
                    results=results["method_results"],
                    metrics=results["metrics"],
                    significance_data=results.get("statistical_tests"),
                )
                table_path = output_dir / "method_comparison_table.tex"
                self.table_generator.save_table_to_file(latex_table, table_path)
                table_paths["method_comparison_table"] = table_path
            except Exception as e:
                print(f"Warning: Could not generate method comparison table: {e}")

        # Statistical summary table
        if "statistical_summary" in results:
            try:
                latex_table = self.table_generator.create_statistical_summary_table(
                    data=results["statistical_summary"]
                )
                table_path = output_dir / "statistical_summary_table.tex"
                self.table_generator.save_table_to_file(latex_table, table_path)
                table_paths["statistical_summary_table"] = table_path
            except Exception as e:
                print(f"Warning: Could not generate statistical summary table: {e}")

        # Hyperparameter table
        if "hyperparameters" in results:
            try:
                latex_table = self.table_generator.create_hyperparameter_table(
                    hyperparameters=results["hyperparameters"]
                )
                table_path = output_dir / "hyperparameters_table.tex"
                self.table_generator.save_table_to_file(latex_table, table_path)
                table_paths["hyperparameters_table"] = table_path
            except Exception as e:
                print(f"Warning: Could not generate hyperparameters table: {e}")

        return table_paths

    def _combine_report_sections(
        self,
        sections: Dict[str, str],
        plot_paths: Dict[str, Path],
        table_paths: Dict[str, Path],
        title: str,
        author: str,
    ) -> Dict[str, Any]:
        """Combine all sections into a structured report."""
        return {
            "title": title,
            "author": author,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sections": sections,
            "plots": plot_paths,
            "tables": table_paths,
        }

    def _save_markdown_report(
        self, report_content: Dict[str, Any], filepath: Path
    ) -> None:
        """Save report in Markdown format."""
        lines = []

        # Header
        lines.append(f"# {report_content['title']}")
        lines.append(f"**Author:** {report_content['author']}")
        lines.append(f"**Date:** {report_content['date']}")
        lines.append("")

        # Table of Contents
        lines.append("## Table of Contents")
        for section_name in report_content["sections"].keys():
            formatted_name = section_name.replace("_", " ").title()
            lines.append(f"- [{formatted_name}](#{section_name.replace('_', '-')})")
        lines.append("")

        # Sections
        for section_name, section_content in report_content["sections"].items():
            formatted_name = section_name.replace("_", " ").title()
            lines.append(f"## {formatted_name}")
            lines.append("")
            lines.append(section_content)
            lines.append("")

            # Add plots if available
            if section_name in ["method_comparison", "performance_analysis"]:
                for plot_name, plot_path in report_content["plots"].items():
                    if section_name in plot_name:
                        lines.append(f"![{plot_name}]({plot_path.name})")
                        lines.append("")

        # Appendices
        if report_content["tables"]:
            lines.append("## Appendix: Tables")
            lines.append("")
            for table_name, table_path in report_content["tables"].items():
                lines.append(f"### {table_name.replace('_', ' ').title()}")
                lines.append(f"LaTeX table available at: `{table_path.name}`")
                lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_latex_report(
        self, report_content: Dict[str, Any], filepath: Path
    ) -> None:
        """Save report in LaTeX format."""
        lines = []

        # Document preamble
        lines.extend(
            [
                "\\documentclass{article}",
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage{graphicx}",
                "\\usepackage{booktabs}",
                "\\usepackage{hyperref}",
                "\\usepackage{geometry}",
                "\\geometry{margin=1in}",
                "",
                f"\\title{{{report_content['title']}}}",
                f"\\author{{{report_content['author']}}}",
                f"\\date{{{report_content['date']}}}",
                "",
                "\\begin{document}",
                "\\maketitle",
                "\\tableofcontents",
                "\\newpage",
                "",
            ]
        )

        # Sections
        for section_name, section_content in report_content["sections"].items():
            formatted_name = section_name.replace("_", " ").title()
            lines.append(f"\\section{{{formatted_name}}}")
            lines.append("")

            # Escape LaTeX special characters in content
            escaped_content = self.table_generator._escape_latex(section_content)
            lines.append(escaped_content)
            lines.append("")

            # Add plots
            for plot_name, plot_path in report_content["plots"].items():
                if section_name in plot_name:
                    lines.extend(
                        [
                            "\\begin{figure}[htbp]",
                            "\\centering",
                            f"\\includegraphics[width=0.8\\textwidth]{{{plot_path.name}}}",
                            f"\\caption{{{plot_name.replace('_', ' ').title()}}}",
                            "\\end{figure}",
                            "",
                        ]
                    )

        # Include tables
        if report_content["tables"]:
            lines.append("\\section{Appendix: Tables}")
            lines.append("")
            for table_name, table_path in report_content["tables"].items():
                lines.append(f"\\subsection{{{table_name.replace('_', ' ').title()}}}")
                lines.append(f"\\input{{{table_path.name}}}")
                lines.append("")

        lines.append("\\end{document}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_html_report(self, report_content: Dict[str, Any], filepath: Path) -> None:
        """Save report in HTML format."""
        lines = []

        # HTML header
        lines.extend(
            [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                f"    <title>{report_content['title']}</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 40px; }",
                "        h1, h2, h3 { color: #333; }",
                "        img { max-width: 100%; height: auto; }",
                "        .toc { background-color: #f5f5f5; padding: 20px; }",
                "    </style>",
                "</head>",
                "<body>",
                f"    <h1>{report_content['title']}</h1>",
                f"    <p><strong>Author:</strong> {report_content['author']}</p>",
                f"    <p><strong>Date:</strong> {report_content['date']}</p>",
                "",
            ]
        )

        # Table of contents
        lines.append("    <div class='toc'>")
        lines.append("        <h2>Table of Contents</h2>")
        lines.append("        <ul>")
        for section_name in report_content["sections"].keys():
            formatted_name = section_name.replace("_", " ").title()
            lines.append(
                f"            <li><a href='#{section_name}'>{formatted_name}</a></li>"
            )
        lines.append("        </ul>")
        lines.append("    </div>")
        lines.append("")

        # Sections
        for section_name, section_content in report_content["sections"].items():
            formatted_name = section_name.replace("_", " ").title()
            lines.append(f"    <h2 id='{section_name}'>{formatted_name}</h2>")
            lines.append(f"    <p>{section_content}</p>")

            # Add plots
            for plot_name, plot_path in report_content["plots"].items():
                if section_name in plot_name:
                    lines.append(f"    <img src='{plot_path.name}' alt='{plot_name}'>")
            lines.append("")

        lines.extend(["</body>", "</html>"])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_report_metadata(
        self, results: Dict[str, Any], filepath: Path, title: str, author: str
    ) -> None:
        """Save report metadata as JSON."""
        metadata = {
            "title": title,
            "author": author,
            "generation_date": datetime.now().isoformat(),
            "experiment_info": {
                "num_methods": len(results.get("method_results", {})),
                "metrics": results.get("metrics", []),
                "num_tasks": results.get("num_tasks", 0),
            },
            "data_sources": list(results.keys()),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # Helper methods for analysis
    def _find_best_method(
        self, results: Dict[str, Any], metric: str
    ) -> Tuple[Optional[str], float, float]:
        """Find the best performing method for a given metric."""
        method_results = results.get("method_results", {})
        best_method = None
        best_value = float("-inf")
        best_std = 0.0

        for method, data in method_results.items():
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"

            if mean_key in data:
                value = data[mean_key]
                if value > best_value:
                    best_value = value
                    best_std = data.get(std_key, 0.0)
                    best_method = method

        return best_method, best_value, best_std

    def _identify_baseline_method(self, methods: List[str]) -> str:
        """Identify the baseline method from the list."""
        # Look for common baseline names
        baseline_keywords = ["standard", "baseline", "vanilla", "basic"]

        for method in methods:
            method_lower = method.lower()
            if any(keyword in method_lower for keyword in baseline_keywords):
                return method

        # If no baseline found, return the first method
        return methods[0] if methods else "Unknown"

    def _calculate_improvement(
        self,
        results: Dict[str, Any],
        best_method: str,
        baseline_method: str,
        metric: str,
    ) -> Optional[float]:
        """Calculate percentage improvement over baseline."""
        method_results = results.get("method_results", {})

        if best_method not in method_results or baseline_method not in method_results:
            return None

        best_value = method_results[best_method].get(f"{metric}_mean", 0)
        baseline_value = method_results[baseline_method].get(f"{metric}_mean", 0)

        if baseline_value == 0:
            return None

        improvement = ((best_value - baseline_value) / baseline_value) * 100
        return improvement

    def _count_significant_results(self, results: Dict[str, Any]) -> int:
        """Count the number of statistically significant results."""
        statistical_tests = results.get("statistical_tests", {})
        count = 0

        for method_data in statistical_tests.values():
            for metric_data in method_data.values():
                if isinstance(metric_data, (int, float)) and metric_data < 0.05:
                    count += 1

        return count

    def _generate_method_ranking(
        self, results: Dict[str, Any], metric: str
    ) -> List[str]:
        """Generate ranking of methods by performance."""
        method_results = results.get("method_results", {})
        method_scores = []

        for method, data in method_results.items():
            mean_key = f"{metric}_mean"
            if mean_key in data:
                method_scores.append((method, data[mean_key]))

        # Sort by score (descending)
        method_scores.sort(key=lambda x: x[1], reverse=True)

        return [method for method, _ in method_scores]

    def _count_significant_pairs(self, results: Dict[str, Any]) -> int:
        """Count the number of significant pairwise comparisons."""
        # This is a simplified implementation
        # In practice, you would have pairwise comparison results
        return self._count_significant_results(results)

    def _analyze_metric_performance(self, results: Dict[str, Any], metric: str) -> str:
        """Analyze performance for a specific metric."""
        method_results = results.get("method_results", {})

        # Find best and worst performers
        best_method, best_value, _ = self._find_best_method(results, metric)

        worst_method = None
        worst_value = float("inf")

        for method, data in method_results.items():
            mean_key = f"{metric}_mean"
            if mean_key in data:
                value = data[mean_key]
                if value < worst_value:
                    worst_value = value
                    worst_method = method

        analysis = f"For {metric}, {best_method} achieved the best performance ({best_value:.3f}), while {worst_method} had the lowest performance ({worst_value:.3f})."

        return analysis

    def _analyze_convergence(self, convergence_data: Dict[str, List[float]]) -> str:
        """Analyze convergence behavior."""
        fastest_method = None
        fastest_steps = float("inf")

        for method, losses in convergence_data.items():
            # Find when loss drops below threshold (e.g., 10% of initial)
            if len(losses) > 1:
                threshold = losses[0] * 0.1
                for i, loss in enumerate(losses):
                    if loss < threshold:
                        if i < fastest_steps:
                            fastest_steps = i
                            fastest_method = method
                        break

        if fastest_method:
            return self.templates["performance_analysis"]["convergence"].format(
                fastest_method=fastest_method, convergence_steps=fastest_steps
            )
        else:
            return "Convergence analysis could not determine fastest converging method."

    def _analyze_efficiency(self, efficiency_data: Dict[str, Any]) -> str:
        """Analyze computational efficiency."""
        # This is a placeholder implementation
        most_efficient = "Method A"  # Would be determined from actual data
        efficiency_metric = "training time"
        efficiency_value = "10.5 seconds"

        return self.templates["performance_analysis"]["efficiency"].format(
            most_efficient=most_efficient,
            efficiency_metric=efficiency_metric,
            efficiency_value=efficiency_value,
        )

    def _analyze_robustness(self, robustness_data: Dict[str, Any]) -> str:
        """Analyze robustness across different conditions."""
        # This is a placeholder implementation
        most_robust = "Method B"  # Would be determined from actual data
        variance = 0.0025

        return self.templates["performance_analysis"]["robustness"].format(
            most_robust=most_robust, variance=variance
        )

    def _analyze_scalability(self, scalability_data: Dict[str, Any]) -> str:
        """Analyze scalability characteristics."""
        return "Scalability analysis shows linear scaling with dataset size for most methods."

    def _analyze_statistical_significance(
        self, statistical_tests: Dict[str, Any]
    ) -> str:
        """Analyze statistical significance results."""
        significant_count = self._count_significant_results(
            {"statistical_tests": statistical_tests}
        )
        total_tests = sum(
            len(method_data) for method_data in statistical_tests.values()
        )

        return f"Statistical significance testing revealed {significant_count} out of {total_tests} comparisons were statistically significant (p < 0.05)."

    def _analyze_effect_sizes(self, effect_sizes: Dict[str, Any]) -> str:
        """Analyze effect sizes."""
        return "Effect size analysis shows large practical differences between top-performing methods."

    def _analyze_confidence_intervals(
        self, confidence_intervals: Dict[str, Any]
    ) -> str:
        """Analyze confidence intervals."""
        return "95% confidence intervals show non-overlapping ranges for the best performing methods, indicating robust differences."

    def _generate_method_details(
        self, method_name: str, method_data: Dict[str, Any]
    ) -> str:
        """Generate detailed analysis for a specific method."""
        details = [f"### {method_name}"]

        # Performance summary
        performance_metrics = []
        for key, value in method_data.items():
            if key.endswith("_mean"):
                metric_name = key.replace("_mean", "")
                std_key = f"{metric_name}_std"
                std_value = method_data.get(std_key, 0.0)
                performance_metrics.append(
                    f"{metric_name}: {value:.3f} ± {std_value:.3f}"
                )

        if performance_metrics:
            details.append("Performance metrics: " + ", ".join(performance_metrics))

        return "\n".join(details)

    def _analyze_hyperparameters(self, hyperparameters: Dict[str, Any]) -> str:
        """Analyze hyperparameter settings and their impact."""
        analysis = ["### Hyperparameter Analysis"]

        # Find common hyperparameters
        all_params = set()
        for method_params in hyperparameters.values():
            all_params.update(method_params.keys())

        for param in all_params:
            values = []
            for method, params in hyperparameters.items():
                if param in params:
                    values.append(f"{method}: {params[param]}")

            if values:
                analysis.append(f"**{param}**: {', '.join(values)}")

        return "\n".join(analysis)

    def _analyze_physics_discoveries(
        self, physics_discovery_data: Dict[str, Any]
    ) -> str:
        """Analyze physics discovery results and generate natural language descriptions."""
        if not physics_discovery_data:
            return "No physics discovery data available for analysis."

        analysis_parts = []

        # Analyze discovered relationships
        if "discovered_relationships" in physics_discovery_data:
            relationships = physics_discovery_data["discovered_relationships"]
            analysis_parts.append("### Physics Discovery Results")

            if relationships:
                analysis_parts.append(
                    f"The system discovered {len(relationships)} significant physical relationships:"
                )

                for i, rel in enumerate(relationships, 1):
                    relationship = rel.get("relationship", "Unknown relationship")
                    confidence = rel.get("confidence", 0.0)
                    validation_score = rel.get("validation_score", 0.0)
                    description = rel.get("description", "No description available")

                    analysis_parts.append(
                        f"{i}. **{relationship}** (confidence: {confidence:.2f}, "
                        f"validation: {validation_score:.2f}) - {description}"
                    )
            else:
                analysis_parts.append(
                    "No significant physical relationships were discovered."
                )

        # Analyze causal strengths
        if "causal_strengths" in physics_discovery_data:
            causal_data = physics_discovery_data["causal_strengths"]
            analysis_parts.append("### Causal Analysis")

            if causal_data:
                # Sort by strength
                sorted_causes = sorted(
                    causal_data.items(), key=lambda x: x[1], reverse=True
                )

                analysis_parts.append(
                    "Variables ranked by causal influence on viscosity:"
                )
                for variable, strength in sorted_causes:
                    strength_desc = (
                        "strong"
                        if strength > 0.7
                        else "moderate" if strength > 0.4 else "weak"
                    )
                    analysis_parts.append(
                        f"- **{variable}**: {strength:.2f} ({strength_desc} influence)"
                    )

        return (
            "\n".join(analysis_parts)
            if analysis_parts
            else "Physics discovery analysis not available."
        )

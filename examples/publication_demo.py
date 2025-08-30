"""
Demonstration of publication-ready output generation capabilities.

This script showcases the PaperPlotGenerator, LaTeXTableGenerator, and ReportGenerator
classes for creating publication-quality research outputs.
"""

import tempfile
from pathlib import Path

import numpy as np

from ml_research_pipeline.papers import (
    LaTeXTableGenerator,
    PaperPlotGenerator,
    ReportGenerator,
)


def main():
    """Demonstrate publication output generation."""
    print("Publication-Ready Output Generation Demo")
    print("=" * 50)

    # Sample experimental results
    experimental_results = {
        "experiment_type": "Meta-Learning PINN Comparison",
        "num_tasks": 150,
        "metrics": ["accuracy", "convergence_speed", "computational_efficiency"],
        "method_results": {
            "Meta-PINN": {
                "accuracy_mean": 0.952,
                "accuracy_std": 0.018,
                "convergence_speed_mean": 8.3,
                "convergence_speed_std": 1.2,
                "computational_efficiency_mean": 0.847,
                "computational_efficiency_std": 0.065,
            },
            "Standard PINN": {
                "accuracy_mean": 0.874,
                "accuracy_std": 0.042,
                "convergence_speed_mean": 23.7,
                "convergence_speed_std": 4.1,
                "computational_efficiency_mean": 0.623,
                "computational_efficiency_std": 0.089,
            },
            "Transfer Learning PINN": {
                "accuracy_mean": 0.913,
                "accuracy_std": 0.028,
                "convergence_speed_mean": 15.2,
                "convergence_speed_std": 2.3,
                "computational_efficiency_mean": 0.734,
                "computational_efficiency_std": 0.071,
            },
            "Fourier Neural Operator": {
                "accuracy_mean": 0.889,
                "accuracy_std": 0.035,
                "convergence_speed_mean": 12.8,
                "convergence_speed_std": 1.9,
                "computational_efficiency_mean": 0.792,
                "computational_efficiency_std": 0.058,
            },
        },
        "statistical_tests": {
            "Meta-PINN": {
                "accuracy": 0.0001,
                "convergence_speed": 0.001,
                "computational_efficiency": 0.002,
            },
            "Standard PINN": {
                "accuracy": 0.15,
                "convergence_speed": 0.08,
                "computational_efficiency": 0.12,
            },
            "Transfer Learning PINN": {
                "accuracy": 0.03,
                "convergence_speed": 0.02,
                "computational_efficiency": 0.04,
            },
            "Fourier Neural Operator": {
                "accuracy": 0.06,
                "convergence_speed": 0.01,
                "computational_efficiency": 0.03,
            },
        },
        "convergence_data": {
            "Meta-PINN": [1.0, 0.45, 0.18, 0.08, 0.04, 0.02, 0.01],
            "Standard PINN": [1.0, 0.82, 0.67, 0.54, 0.43, 0.35, 0.29],
            "Transfer Learning PINN": [1.0, 0.63, 0.38, 0.22, 0.13, 0.08, 0.05],
            "Fourier Neural Operator": [1.0, 0.71, 0.48, 0.31, 0.19, 0.12, 0.07],
        },
        "hyperparameters": {
            "Meta-PINN": {
                "learning_rate": 0.001,
                "meta_learning_rate": 0.01,
                "adaptation_steps": 5,
                "batch_size": 32,
                "hidden_layers": [128, 256, 128],
            },
            "Standard PINN": {
                "learning_rate": 0.01,
                "batch_size": 64,
                "hidden_layers": [64, 128, 64],
                "optimizer": "Adam",
            },
            "Transfer Learning PINN": {
                "learning_rate": 0.005,
                "fine_tune_rate": 0.001,
                "batch_size": 48,
                "hidden_layers": [96, 192, 96],
            },
            "Fourier Neural Operator": {
                "learning_rate": 0.002,
                "modes": 16,
                "width": 64,
                "batch_size": 40,
            },
        },
    }

    # Create output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "publication_outputs"
        output_dir.mkdir(exist_ok=True)

        print(f"Generating outputs in: {output_dir}")

        # 1. Demonstrate plot generation
        print("\n1. Generating Publication-Quality Plots...")
        plot_generator = PaperPlotGenerator(dpi=300, font_size=12)

        # Method comparison plot
        fig1 = plot_generator.create_method_comparison_plot(
            results=experimental_results["method_results"],
            metric_name="accuracy",
            title="Method Comparison: Accuracy",
            significance_data=experimental_results["statistical_tests"],
            save_path=output_dir / "method_comparison_accuracy.png",
        )
        print("   ✓ Method comparison plot (accuracy)")

        # Convergence plot
        fig2 = plot_generator.create_convergence_plot(
            convergence_data=experimental_results["convergence_data"],
            title="Training Convergence Comparison",
            ylabel="Normalized Loss",
            save_path=output_dir / "convergence_comparison.png",
        )
        print("   ✓ Convergence comparison plot")

        # Uncertainty visualization (synthetic data)
        x_data = np.linspace(0, 10, 100)
        y_mean = np.sin(x_data) * np.exp(-x_data / 10)
        y_std = 0.1 * np.ones_like(x_data) + 0.05 * np.abs(y_mean)
        ground_truth = y_mean + 0.02 * np.random.randn(len(x_data))

        fig3 = plot_generator.create_uncertainty_plot(
            x_data=x_data,
            y_mean=y_mean,
            y_std=y_std,
            ground_truth=ground_truth,
            title="Uncertainty Quantification Example",
            xlabel="Spatial Coordinate",
            ylabel="Predicted Value",
            save_path=output_dir / "uncertainty_example.png",
        )
        print("   ✓ Uncertainty quantification plot")

        plot_generator.close_all_figures()

        # 2. Demonstrate table generation
        print("\n2. Generating LaTeX Tables...")
        table_generator = LaTeXTableGenerator(precision=3, use_booktabs=True)

        # Method comparison table
        method_table = table_generator.create_method_comparison_table(
            results=experimental_results["method_results"],
            metrics=["accuracy", "convergence_speed", "computational_efficiency"],
            significance_data=experimental_results["statistical_tests"],
            caption="Comprehensive method comparison across all evaluation metrics",
            label="tab:method_comparison",
        )
        table_generator.save_table_to_file(
            method_table, output_dir / "method_comparison_table.tex"
        )
        print("   ✓ Method comparison table")

        # Hyperparameter table
        hyperparameter_table = table_generator.create_hyperparameter_table(
            hyperparameters=experimental_results["hyperparameters"],
            caption="Hyperparameter settings for all evaluated methods",
            label="tab:hyperparameters",
        )
        table_generator.save_table_to_file(
            hyperparameter_table, output_dir / "hyperparameters_table.tex"
        )
        print("   ✓ Hyperparameter settings table")

        # Statistical summary
        statistical_summary = {
            "Accuracy": {"mean": 0.907, "std": 0.034, "min": 0.874, "max": 0.952},
            "Convergence Speed": {"mean": 15.0, "std": 6.4, "min": 8.3, "max": 23.7},
            "Computational Efficiency": {
                "mean": 0.749,
                "std": 0.093,
                "min": 0.623,
                "max": 0.847,
            },
        }

        summary_table = table_generator.create_statistical_summary_table(
            data=statistical_summary,
            caption="Statistical summary of performance metrics across all methods",
            label="tab:statistical_summary",
        )
        table_generator.save_table_to_file(
            summary_table, output_dir / "statistical_summary_table.tex"
        )
        print("   ✓ Statistical summary table")

        # 3. Demonstrate comprehensive report generation
        print("\n3. Generating Comprehensive Reports...")
        report_generator = ReportGenerator(
            plot_generator=plot_generator, table_generator=table_generator
        )

        # Generate reports in multiple formats
        for format_type in ["markdown", "latex", "html"]:
            saved_files = report_generator.generate_comprehensive_report(
                experimental_results=experimental_results,
                output_dir=output_dir / f"report_{format_type}",
                report_title="Meta-Learning PINN: Comprehensive Experimental Analysis",
                author="ML Research Pipeline",
                include_plots=True,
                include_tables=True,
                format_type=format_type,
            )
            print(f"   ✓ {format_type.upper()} report generated")

        # 4. Display sample outputs
        print("\n4. Sample Outputs:")
        print("-" * 30)

        # Show executive summary
        executive_summary = report_generator._generate_executive_summary(
            experimental_results
        )
        print("Executive Summary:")
        print(
            executive_summary[:200] + "..."
            if len(executive_summary) > 200
            else executive_summary
        )

        print("\nMethod Ranking (by accuracy):")
        ranking = report_generator._generate_method_ranking(
            experimental_results, "accuracy"
        )
        for i, method in enumerate(ranking, 1):
            accuracy = experimental_results["method_results"][method]["accuracy_mean"]
            print(f"   {i}. {method}: {accuracy:.3f}")

        # Show sample LaTeX table snippet
        print(f"\nSample LaTeX Table (first 5 lines):")
        table_lines = method_table.split("\n")[:5]
        for line in table_lines:
            print(f"   {line}")

        print(f"\nAll outputs saved to: {output_dir}")
        print("\nFiles generated:")
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                print(f"   - {file_path.relative_to(output_dir)}")

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("The generated outputs are ready for inclusion in research papers.")


if __name__ == "__main__":
    main()

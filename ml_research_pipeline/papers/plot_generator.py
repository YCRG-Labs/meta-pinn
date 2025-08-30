"""
Publication-quality plot generation for ML research papers.

This module provides the PaperPlotGenerator class for creating standardized,
publication-ready plots with proper formatting for academic publications.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class PaperPlotGenerator:
    """
    Generates publication-quality plots with standardized formatting.

    This class provides methods for creating various types of plots commonly
    used in ML research papers, with consistent styling and formatting suitable
    for academic publications.
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figure_size: Tuple[float, float] = (8, 6),
        dpi: int = 300,
        font_family: str = "serif",
        font_size: int = 12,
    ):
        """
        Initialize the plot generator with publication settings.

        Args:
            style: Matplotlib/seaborn style to use
            figure_size: Default figure size in inches (width, height)
            dpi: Resolution for saved figures
            font_family: Font family for text
            font_size: Base font size
        """
        self.style = style
        self.figure_size = figure_size
        self.dpi = dpi
        self.font_family = font_family
        self.font_size = font_size

        # Set up publication-quality defaults
        self._setup_publication_style()

        # Color palette for consistent plots
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#C73E1D",
            "neutral": "#6C757D",
            "light": "#F8F9FA",
            "dark": "#212529",
        }

        # Method colors for consistent comparison plots
        self.method_colors = {
            "Meta-PINN": self.colors["primary"],
            "Standard PINN": self.colors["secondary"],
            "Transfer Learning": self.colors["accent"],
            "FNO": self.colors["success"],
            "DeepONet": self.colors["neutral"],
            "Bayesian Meta-PINN": "#8E44AD",
        }

    def _setup_publication_style(self):
        """Set up matplotlib and seaborn for publication-quality plots."""
        # Set style
        plt.style.use(self.style)

        # Configure matplotlib parameters
        plt.rcParams.update(
            {
                "figure.figsize": self.figure_size,
                "figure.dpi": self.dpi,
                "savefig.dpi": self.dpi,
                "font.family": self.font_family,
                "font.size": self.font_size,
                "axes.titlesize": self.font_size + 2,
                "axes.labelsize": self.font_size,
                "xtick.labelsize": self.font_size - 1,
                "ytick.labelsize": self.font_size - 1,
                "legend.fontsize": self.font_size - 1,
                "lines.linewidth": 2,
                "lines.markersize": 6,
                "axes.linewidth": 1.2,
                "grid.linewidth": 0.8,
                "grid.alpha": 0.3,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.1,
                "text.usetex": False,  # Set to True if LaTeX is available
            }
        )

        # Set seaborn palette
        sns.set_palette("husl")

    def create_method_comparison_plot(
        self,
        results: Dict[str, Dict[str, float]],
        metric_name: str,
        title: Optional[str] = None,
        ylabel: Optional[str] = None,
        save_path: Optional[Path] = None,
        show_significance: bool = True,
        significance_data: Optional[Dict] = None,
    ) -> plt.Figure:
        """
        Create a bar plot comparing different methods on a specific metric.

        Args:
            results: Dictionary mapping method names to metric dictionaries
            metric_name: Name of the metric to plot
            title: Plot title (auto-generated if None)
            ylabel: Y-axis label (uses metric_name if None)
            save_path: Path to save the figure
            show_significance: Whether to show significance indicators
            significance_data: Statistical significance test results

        Returns:
            matplotlib Figure object
        """
        if not results:
            raise ValueError("Results dictionary cannot be empty")

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Extract data
        methods = list(results.keys())
        try:
            means = [results[method][metric_name + "_mean"] for method in methods]
            stds = [results[method][metric_name + "_std"] for method in methods]
        except KeyError as e:
            raise KeyError(f"Missing metric '{metric_name}' in results data: {e}")

        # Create bar plot
        x_pos = np.arange(len(methods))
        colors = [
            self.method_colors.get(method, self.colors["neutral"]) for method in methods
        ]

        bars = ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Customize plot
        ax.set_xlabel("Method", fontweight="bold")
        ax.set_ylabel(
            ylabel or metric_name.replace("_", " ").title(), fontweight="bold"
        )
        ax.set_title(
            title or f'{metric_name.replace("_", " ").title()} Comparison',
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right")

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01 * max(means),
                f"{mean:.3f}±{std:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Add significance indicators if provided
        if show_significance and significance_data:
            self._add_significance_indicators(ax, x_pos, means, stds, significance_data)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_convergence_plot(
        self,
        convergence_data: Dict[str, List[float]],
        title: Optional[str] = None,
        xlabel: str = "Iteration",
        ylabel: str = "Loss",
        save_path: Optional[Path] = None,
        log_scale: bool = False,
    ) -> plt.Figure:
        """
        Create a convergence plot showing training progress for different methods.

        Args:
            convergence_data: Dictionary mapping method names to loss sequences
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the figure
            log_scale: Whether to use log scale for y-axis

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        for method, losses in convergence_data.items():
            color = self.method_colors.get(method, self.colors["neutral"])
            ax.plot(losses, label=method, color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(
            title or "Training Convergence Comparison", fontweight="bold", pad=20
        )

        if log_scale:
            ax.set_yscale("log")

        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_uncertainty_plot(
        self,
        x_data: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        xlabel: str = "Input",
        ylabel: str = "Output",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create an uncertainty visualization plot with confidence intervals.

        Args:
            x_data: Input data points
            y_mean: Predicted mean values
            y_std: Predicted standard deviations
            ground_truth: True values (optional)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Validate input dimensions
        if len(x_data) != len(y_mean) or len(y_mean) != len(y_std):
            raise ValueError("x_data, y_mean, and y_std must have the same length")

        if ground_truth is not None and len(ground_truth) != len(x_data):
            raise ValueError("ground_truth must have the same length as x_data")

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot mean prediction
        ax.plot(
            x_data,
            y_mean,
            color=self.colors["primary"],
            linewidth=2,
            label="Prediction",
            alpha=0.8,
        )

        # Plot confidence intervals
        ax.fill_between(
            x_data,
            y_mean - 2 * y_std,
            y_mean + 2 * y_std,
            color=self.colors["primary"],
            alpha=0.2,
            label="95% CI",
        )
        ax.fill_between(
            x_data,
            y_mean - y_std,
            y_mean + y_std,
            color=self.colors["primary"],
            alpha=0.3,
            label="68% CI",
        )

        # Plot ground truth if available
        if ground_truth is not None:
            ax.plot(
                x_data,
                ground_truth,
                color=self.colors["success"],
                linewidth=2,
                linestyle="--",
                label="Ground Truth",
                alpha=0.8,
            )

        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title or "Uncertainty Quantification", fontweight="bold", pad=20)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_heatmap(
        self,
        data: np.ndarray,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        cmap: str = "viridis",
        save_path: Optional[Path] = None,
        annotate: bool = True,
    ) -> plt.Figure:
        """
        Create a heatmap visualization.

        Args:
            data: 2D array of values to plot
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            title: Plot title
            cmap: Colormap to use
            save_path: Path to save the figure
            annotate: Whether to annotate cells with values

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Create heatmap
        im = ax.imshow(data, cmap=cmap, aspect="auto")

        # Set ticks and labels
        if x_labels:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
        if y_labels:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=self.font_size - 1)

        # Annotate cells if requested
        if annotate:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if data[i, j] < np.mean(data) else "black",
                    )

        ax.set_title(title or "Heatmap", fontweight="bold", pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_scatter_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title: Optional[str] = None,
        xlabel: str = "X",
        ylabel: str = "Y",
        save_path: Optional[Path] = None,
        add_regression_line: bool = False,
    ) -> plt.Figure:
        """
        Create a scatter plot with optional regression line.

        Args:
            x_data: X coordinates
            y_data: Y coordinates
            labels: Point labels for legend
            colors: Colors for different groups
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the figure
            add_regression_line: Whether to add regression line

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        if labels is None:
            # Single group scatter
            color = colors[0] if colors else self.colors["primary"]
            ax.scatter(
                x_data,
                y_data,
                color=color,
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
        else:
            # Multiple groups
            unique_labels = list(set(labels))
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                color = (
                    colors[i]
                    if colors and i < len(colors)
                    else list(self.method_colors.values())[i % len(self.method_colors)]
                )
                ax.scatter(
                    x_data[mask],
                    y_data[mask],
                    label=label,
                    color=color,
                    alpha=0.7,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )
            ax.legend(frameon=True, fancybox=True, shadow=True)

        # Add regression line if requested
        if add_regression_line:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(
                x_data,
                p(x_data),
                color=self.colors["success"],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label="Regression",
            )
            if labels is not None:
                ax.legend(frameon=True, fancybox=True, shadow=True)

        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title or "Scatter Plot", fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def _add_significance_indicators(
        self,
        ax: plt.Axes,
        x_pos: np.ndarray,
        means: List[float],
        stds: List[float],
        significance_data: Dict,
    ) -> None:
        """Add statistical significance indicators to bar plots."""
        max_height = max([m + s for m, s in zip(means, stds)])

        # Add significance stars based on p-values
        for i, (method, stats) in enumerate(significance_data.items()):
            if "p_value" in stats:
                p_val = stats["p_value"]
                if p_val < 0.001:
                    sig_text = "***"
                elif p_val < 0.01:
                    sig_text = "**"
                elif p_val < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"

                # Place significance indicator above error bar
                y_pos = means[i] + stds[i] + 0.05 * max_height
                ax.text(
                    x_pos[i],
                    y_pos,
                    sig_text,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

    def save_all_figures(
        self,
        figures: Dict[str, plt.Figure],
        output_dir: Path,
        formats: List[str] = ["png", "pdf"],
    ) -> None:
        """
        Save multiple figures to specified directory in multiple formats.

        Args:
            figures: Dictionary mapping figure names to Figure objects
            output_dir: Directory to save figures
            formats: List of file formats to save
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, fig in figures.items():
            for fmt in formats:
                save_path = output_dir / f"{name}.{fmt}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", format=fmt)

        print(f"Saved {len(figures)} figures in {len(formats)} formats to {output_dir}")

    def close_all_figures(self) -> None:
        """Close all matplotlib figures to free memory."""
        plt.close("all")

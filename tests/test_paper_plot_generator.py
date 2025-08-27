"""
Unit tests for PaperPlotGenerator class.

Tests plot generation functionality, formatting accuracy, and data visualization
correctness for publication-quality plots.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from ml_research_pipeline.papers.plot_generator import PaperPlotGenerator


class TestPaperPlotGenerator:
    """Test suite for PaperPlotGenerator class."""
    
    @pytest.fixture
    def plot_generator(self):
        """Create a PaperPlotGenerator instance for testing."""
        return PaperPlotGenerator()
    
    @pytest.fixture
    def sample_results_data(self):
        """Sample experimental results data for testing."""
        return {
            'Meta-PINN': {
                'accuracy_mean': 0.95,
                'accuracy_std': 0.02,
                'speed_mean': 10.5,
                'speed_std': 1.2
            },
            'Standard PINN': {
                'accuracy_mean': 0.87,
                'accuracy_std': 0.04,
                'speed_mean': 25.3,
                'speed_std': 3.1
            },
            'Transfer Learning': {
                'accuracy_mean': 0.91,
                'accuracy_std': 0.03,
                'speed_mean': 15.8,
                'speed_std': 2.0
            }
        }
    
    @pytest.fixture
    def sample_convergence_data(self):
        """Sample convergence data for testing."""
        return {
            'Meta-PINN': [1.0, 0.5, 0.25, 0.12, 0.06, 0.03],
            'Standard PINN': [1.0, 0.8, 0.6, 0.45, 0.35, 0.28],
            'Transfer Learning': [1.0, 0.7, 0.4, 0.22, 0.15, 0.10]
        }
    
    def test_initialization(self):
        """Test PaperPlotGenerator initialization."""
        generator = PaperPlotGenerator(
            style='seaborn-v0_8-whitegrid',
            figure_size=(10, 8),
            dpi=150,
            font_family='sans-serif',
            font_size=14
        )
        
        assert generator.style == 'seaborn-v0_8-whitegrid'
        assert generator.figure_size == (10, 8)
        assert generator.dpi == 150
        assert generator.font_family == 'sans-serif'
        assert generator.font_size == 14
        
        # Check color palettes are set
        assert 'primary' in generator.colors
        assert 'Meta-PINN' in generator.method_colors
    
    def test_method_comparison_plot_creation(self, plot_generator, sample_results_data):
        """Test creation of method comparison plots."""
        fig = plot_generator.create_method_comparison_plot(
            results=sample_results_data,
            metric_name='accuracy',
            title='Test Accuracy Comparison',
            ylabel='Accuracy'
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot has the correct number of bars
        ax = fig.get_axes()[0]
        # Count the number of bar containers (each method gets one bar)
        bar_containers = [child for child in ax.containers if hasattr(child, '__iter__')]
        if bar_containers:
            # If bar containers exist, count bars in the first container
            num_bars = len(bar_containers[0])
        else:
            # Fallback: count Rectangle patches with reasonable width
            rectangles = [child for child in ax.patches 
                         if hasattr(child, 'get_width') and child.get_width() > 0.1]
            num_bars = len(rectangles)
        
        assert num_bars == len(sample_results_data)
        
        # Check axis labels
        assert ax.get_xlabel() == 'Method'
        assert ax.get_ylabel() == 'Accuracy'
        assert 'Test Accuracy Comparison' in ax.get_title()
        
        plt.close(fig)
    
    def test_method_comparison_plot_with_significance(self, plot_generator, sample_results_data):
        """Test method comparison plot with significance indicators."""
        significance_data = {
            'Meta-PINN': {'p_value': 0.001},
            'Standard PINN': {'p_value': 0.05},
            'Transfer Learning': {'p_value': 0.1}
        }
        
        fig = plot_generator.create_method_comparison_plot(
            results=sample_results_data,
            metric_name='accuracy',
            show_significance=True,
            significance_data=significance_data
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check that significance indicators are added
        ax = fig.get_axes()[0]
        text_objects = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        significance_texts = [t for t in text_objects if t.get_text() in ['***', '**', '*', 'ns']]
        assert len(significance_texts) > 0
        
        plt.close(fig)
    
    def test_convergence_plot_creation(self, plot_generator, sample_convergence_data):
        """Test creation of convergence plots."""
        fig = plot_generator.create_convergence_plot(
            convergence_data=sample_convergence_data,
            title='Training Convergence',
            xlabel='Epoch',
            ylabel='Loss'
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot has the correct number of lines
        ax = fig.get_axes()[0]
        lines = ax.get_lines()
        assert len(lines) == len(sample_convergence_data)
        
        # Check axis labels
        assert ax.get_xlabel() == 'Epoch'
        assert ax.get_ylabel() == 'Loss'
        assert 'Training Convergence' in ax.get_title()
        
        # Check legend
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == len(sample_convergence_data)
        
        plt.close(fig)
    
    def test_convergence_plot_log_scale(self, plot_generator, sample_convergence_data):
        """Test convergence plot with log scale."""
        fig = plot_generator.create_convergence_plot(
            convergence_data=sample_convergence_data,
            log_scale=True
        )
        
        ax = fig.get_axes()[0]
        assert ax.get_yscale() == 'log'
        
        plt.close(fig)
    
    def test_uncertainty_plot_creation(self, plot_generator):
        """Test creation of uncertainty plots."""
        x_data = np.linspace(0, 10, 50)
        y_mean = np.sin(x_data)
        y_std = 0.1 * np.ones_like(x_data)
        ground_truth = np.sin(x_data) + 0.05 * np.random.randn(len(x_data))
        
        fig = plot_generator.create_uncertainty_plot(
            x_data=x_data,
            y_mean=y_mean,
            y_std=y_std,
            ground_truth=ground_truth,
            title='Uncertainty Visualization',
            xlabel='X',
            ylabel='Y'
        )
        
        assert isinstance(fig, plt.Figure)
        
        ax = fig.get_axes()[0]
        
        # Check axis labels
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        assert 'Uncertainty Visualization' in ax.get_title()
        
        # Check legend
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert 'Prediction' in legend_texts
        assert 'Ground Truth' in legend_texts
        
        plt.close(fig)
    
    def test_uncertainty_plot_without_ground_truth(self, plot_generator):
        """Test uncertainty plot without ground truth data."""
        x_data = np.linspace(0, 10, 50)
        y_mean = np.sin(x_data)
        y_std = 0.1 * np.ones_like(x_data)
        
        fig = plot_generator.create_uncertainty_plot(
            x_data=x_data,
            y_mean=y_mean,
            y_std=y_std
        )
        
        assert isinstance(fig, plt.Figure)
        
        ax = fig.get_axes()[0]
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert 'Ground Truth' not in legend_texts
        
        plt.close(fig)
    
    def test_heatmap_creation(self, plot_generator):
        """Test creation of heatmap plots."""
        data = np.random.rand(5, 4)
        x_labels = ['A', 'B', 'C', 'D']
        y_labels = ['1', '2', '3', '4', '5']
        
        fig = plot_generator.create_heatmap(
            data=data,
            x_labels=x_labels,
            y_labels=y_labels,
            title='Test Heatmap',
            annotate=True
        )
        
        assert isinstance(fig, plt.Figure)
        
        ax = fig.get_axes()[0]
        
        # Check title
        assert 'Test Heatmap' in ax.get_title()
        
        # Check labels
        assert len(ax.get_xticklabels()) == len(x_labels)
        assert len(ax.get_yticklabels()) == len(y_labels)
        
        plt.close(fig)
    
    def test_heatmap_without_labels(self, plot_generator):
        """Test heatmap creation without labels."""
        data = np.random.rand(3, 3)
        
        fig = plot_generator.create_heatmap(
            data=data,
            annotate=False
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_scatter_plot_creation(self, plot_generator):
        """Test creation of scatter plots."""
        x_data = np.random.randn(100)
        y_data = 2 * x_data + np.random.randn(100) * 0.5
        
        fig = plot_generator.create_scatter_plot(
            x_data=x_data,
            y_data=y_data,
            title='Test Scatter Plot',
            xlabel='X Variable',
            ylabel='Y Variable',
            add_regression_line=True
        )
        
        assert isinstance(fig, plt.Figure)
        
        ax = fig.get_axes()[0]
        
        # Check axis labels
        assert ax.get_xlabel() == 'X Variable'
        assert ax.get_ylabel() == 'Y Variable'
        assert 'Test Scatter Plot' in ax.get_title()
        
        # Check that regression line is added
        lines = ax.get_lines()
        assert len(lines) > 0  # Should have regression line
        
        plt.close(fig)
    
    def test_scatter_plot_with_groups(self, plot_generator):
        """Test scatter plot with multiple groups."""
        x_data = np.random.randn(100)
        y_data = np.random.randn(100)
        labels = ['Group A'] * 50 + ['Group B'] * 50
        colors = ['red', 'blue']
        
        fig = plot_generator.create_scatter_plot(
            x_data=x_data,
            y_data=y_data,
            labels=labels,
            colors=colors
        )
        
        assert isinstance(fig, plt.Figure)
        
        ax = fig.get_axes()[0]
        legend = ax.get_legend()
        assert legend is not None
        
        plt.close(fig)
    
    def test_save_figures_functionality(self, plot_generator, sample_results_data):
        """Test saving figures to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test figure
            fig = plot_generator.create_method_comparison_plot(
                results=sample_results_data,
                metric_name='accuracy'
            )
            
            figures = {'test_plot': fig}
            
            plot_generator.save_all_figures(
                figures=figures,
                output_dir=temp_path,
                formats=['png', 'pdf']
            )
            
            # Check that files were created
            assert (temp_path / 'test_plot.png').exists()
            assert (temp_path / 'test_plot.pdf').exists()
            
            plt.close(fig)
    
    def test_figure_saving_with_path(self, plot_generator, sample_results_data):
        """Test saving individual figures with save_path parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / 'test_figure.png'
            
            fig = plot_generator.create_method_comparison_plot(
                results=sample_results_data,
                metric_name='accuracy',
                save_path=save_path
            )
            
            assert save_path.exists()
            plt.close(fig)
    
    def test_close_all_figures(self, plot_generator):
        """Test closing all figures."""
        # Create some figures
        fig1 = plt.figure()
        fig2 = plt.figure()
        
        # Check figures exist
        assert len(plt.get_fignums()) >= 2
        
        # Close all figures
        plot_generator.close_all_figures()
        
        # Check all figures are closed
        assert len(plt.get_fignums()) == 0
    
    def test_color_consistency(self, plot_generator):
        """Test that colors are consistent across plots."""
        # Test method colors
        assert 'Meta-PINN' in plot_generator.method_colors
        assert 'Standard PINN' in plot_generator.method_colors
        
        # Test general colors
        assert 'primary' in plot_generator.colors
        assert 'secondary' in plot_generator.colors
        
        # Colors should be valid hex codes or color names
        for color in plot_generator.colors.values():
            assert isinstance(color, str)
            assert len(color) > 0
    
    def test_plot_formatting_consistency(self, plot_generator, sample_results_data):
        """Test that plot formatting is consistent across different plot types."""
        # Create different types of plots
        fig1 = plot_generator.create_method_comparison_plot(
            results=sample_results_data,
            metric_name='accuracy'
        )
        
        convergence_data = {
            'Method A': [1.0, 0.5, 0.25],
            'Method B': [1.0, 0.7, 0.4]
        }
        fig2 = plot_generator.create_convergence_plot(convergence_data)
        
        # Check that both figures have consistent DPI
        assert fig1.dpi == plot_generator.dpi
        assert fig2.dpi == plot_generator.dpi
        
        # Check font sizes are consistent
        ax1 = fig1.get_axes()[0]
        ax2 = fig2.get_axes()[0]
        
        # Both should have titles
        assert ax1.get_title() != ''
        assert ax2.get_title() != ''
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_error_handling_invalid_data(self, plot_generator):
        """Test error handling with invalid data."""
        # Test with empty results
        with pytest.raises(ValueError, match="Results dictionary cannot be empty"):
            plot_generator.create_method_comparison_plot(
                results={},
                metric_name='accuracy'
            )
        
        # Test with missing metric in results
        invalid_results = {
            'Method1': {'other_metric_mean': 0.5, 'other_metric_std': 0.1}
        }
        with pytest.raises(KeyError, match="Missing metric 'accuracy'"):
            plot_generator.create_method_comparison_plot(
                results=invalid_results,
                metric_name='accuracy'
            )
        
        # Test with mismatched data lengths
        with pytest.raises(ValueError, match="must have the same length"):
            plot_generator.create_uncertainty_plot(
                x_data=np.array([1, 2, 3]),
                y_mean=np.array([1, 2]),  # Different length
                y_std=np.array([0.1, 0.1, 0.1])
            )
    
    @pytest.mark.parametrize("metric_name,expected_ylabel", [
        ("accuracy", "Accuracy"),
        ("loss_value", "Loss Value"),
        ("training_time", "Training Time"),
        ("parameter_error", "Parameter Error")
    ])
    def test_automatic_ylabel_generation(self, plot_generator, sample_results_data, metric_name, expected_ylabel):
        """Test automatic y-axis label generation from metric names."""
        # Add the metric to sample data
        for method_data in sample_results_data.values():
            method_data[f'{metric_name}_mean'] = 0.5
            method_data[f'{metric_name}_std'] = 0.1
        
        fig = plot_generator.create_method_comparison_plot(
            results=sample_results_data,
            metric_name=metric_name
        )
        
        ax = fig.get_axes()[0]
        assert ax.get_ylabel() == expected_ylabel
        
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__])
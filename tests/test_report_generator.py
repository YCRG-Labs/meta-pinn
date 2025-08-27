"""
Unit tests for ReportGenerator class.

Tests comprehensive report generation functionality, natural language generation,
and integration with plot and table generators.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

from ml_research_pipeline.papers.report_generator import ReportGenerator
from ml_research_pipeline.papers.plot_generator import PaperPlotGenerator
from ml_research_pipeline.papers.table_generator import LaTeXTableGenerator


class TestReportGenerator:
    """Test suite for ReportGenerator class."""
    
    @pytest.fixture
    def report_generator(self):
        """Create a ReportGenerator instance for testing."""
        return ReportGenerator()
    
    @pytest.fixture
    def mock_plot_generator(self):
        """Create a mock plot generator."""
        mock_gen = Mock(spec=PaperPlotGenerator)
        mock_gen.create_method_comparison_plot.return_value = Mock()
        mock_gen.create_convergence_plot.return_value = Mock()
        mock_gen.create_uncertainty_plot.return_value = Mock()
        mock_gen.close_all_figures.return_value = None
        return mock_gen
    
    @pytest.fixture
    def mock_table_generator(self):
        """Create a mock table generator."""
        mock_gen = Mock(spec=LaTeXTableGenerator)
        mock_gen.create_method_comparison_table.return_value = "\\begin{table}...\\end{table}"
        mock_gen.create_statistical_summary_table.return_value = "\\begin{table}...\\end{table}"
        mock_gen.create_hyperparameter_table.return_value = "\\begin{table}...\\end{table}"
        mock_gen.save_table_to_file.return_value = None
        mock_gen._escape_latex.side_effect = lambda x: x.replace('&', '\\&')
        return mock_gen
    
    @pytest.fixture
    def sample_experimental_results(self):
        """Sample experimental results data for testing."""
        return {
            'experiment_type': 'meta-learning PINN',
            'num_tasks': 100,
            'metrics': ['accuracy', 'speed', 'loss'],
            'method_results': {
                'Meta-PINN': {
                    'accuracy_mean': 0.95,
                    'accuracy_std': 0.02,
                    'speed_mean': 10.5,
                    'speed_std': 1.2,
                    'loss_mean': 0.001,
                    'loss_std': 0.0002
                },
                'Standard PINN': {
                    'accuracy_mean': 0.87,
                    'accuracy_std': 0.04,
                    'speed_mean': 25.3,
                    'speed_std': 3.1,
                    'loss_mean': 0.005,
                    'loss_std': 0.001
                },
                'Transfer Learning': {
                    'accuracy_mean': 0.91,
                    'accuracy_std': 0.03,
                    'speed_mean': 15.8,
                    'speed_std': 2.0,
                    'loss_mean': 0.003,
                    'loss_std': 0.0005
                }
            },
            'statistical_tests': {
                'Meta-PINN': {
                    'accuracy': 0.001,
                    'speed': 0.01,
                    'loss': 0.001
                },
                'Standard PINN': {
                    'accuracy': 0.1,
                    'speed': 0.05,
                    'loss': 0.2
                },
                'Transfer Learning': {
                    'accuracy': 0.02,
                    'speed': 0.03,
                    'loss': 0.01
                }
            },
            'convergence_data': {
                'Meta-PINN': [1.0, 0.5, 0.25, 0.12, 0.06, 0.03],
                'Standard PINN': [1.0, 0.8, 0.6, 0.45, 0.35, 0.28],
                'Transfer Learning': [1.0, 0.7, 0.4, 0.22, 0.15, 0.10]
            },
            'hyperparameters': {
                'Meta-PINN': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'hidden_layers': [64, 128, 64]
                },
                'Standard PINN': {
                    'learning_rate': 0.01,
                    'batch_size': 64,
                    'hidden_layers': [32, 64, 32]
                }
            }
        }
    
    def test_initialization(self):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator(
            template_style='technical',
            language='en'
        )
        
        assert generator.template_style == 'technical'
        assert generator.language == 'en'
        assert generator.plot_generator is not None
        assert generator.table_generator is not None
        
        # Check templates are loaded
        assert 'executive_summary' in generator.templates
        assert 'method_comparison' in generator.templates
    
    def test_initialization_with_custom_generators(self, mock_plot_generator, mock_table_generator):
        """Test initialization with custom generators."""
        generator = ReportGenerator(
            plot_generator=mock_plot_generator,
            table_generator=mock_table_generator
        )
        
        assert generator.plot_generator is mock_plot_generator
        assert generator.table_generator is mock_table_generator
    
    def test_generate_comprehensive_report_markdown(self, report_generator, sample_experimental_results):
        """Test comprehensive report generation in Markdown format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = report_generator.generate_comprehensive_report(
                experimental_results=sample_experimental_results,
                output_dir=output_dir,
                report_title="Test Report",
                author="Test Author",
                format_type='markdown'
            )
            
            # Check that report file was created
            assert 'report' in saved_files
            report_path = saved_files['report']
            assert report_path.exists()
            assert report_path.suffix == '.md'
            
            # Check report content
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert 'Test Report' in content
            assert 'Test Author' in content
            assert 'Executive Summary' in content
            assert 'Method Comparison' in content
            
            # Check metadata file
            assert 'metadata' in saved_files
            metadata_path = saved_files['metadata']
            assert metadata_path.exists()
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            assert metadata['title'] == 'Test Report'
            assert metadata['author'] == 'Test Author'
    
    def test_generate_comprehensive_report_latex(self, report_generator, sample_experimental_results):
        """Test comprehensive report generation in LaTeX format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = report_generator.generate_comprehensive_report(
                experimental_results=sample_experimental_results,
                output_dir=output_dir,
                format_type='latex'
            )
            
            # Check that report file was created
            assert 'report' in saved_files
            report_path = saved_files['report']
            assert report_path.exists()
            assert report_path.suffix == '.tex'
            
            # Check LaTeX structure
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert '\\documentclass{article}' in content
            assert '\\begin{document}' in content
            assert '\\end{document}' in content
            assert '\\section{' in content
    
    def test_generate_comprehensive_report_html(self, report_generator, sample_experimental_results):
        """Test comprehensive report generation in HTML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = report_generator.generate_comprehensive_report(
                experimental_results=sample_experimental_results,
                output_dir=output_dir,
                format_type='html'
            )
            
            # Check that report file was created
            assert 'report' in saved_files
            report_path = saved_files['report']
            assert report_path.exists()
            assert report_path.suffix == '.html'
            
            # Check HTML structure
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert '<!DOCTYPE html>' in content
            assert '<html>' in content
            assert '</html>' in content
            assert '<h1>' in content
    
    def test_generate_executive_summary(self, report_generator, sample_experimental_results):
        """Test executive summary generation."""
        summary = report_generator._generate_executive_summary(sample_experimental_results)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # Check key information is included
        assert 'meta-learning PINN' in summary
        assert '3 different methods' in summary
        assert 'Meta-PINN' in summary  # Best performing method
        assert 'accuracy' in summary
    
    def test_generate_method_comparison_analysis(self, report_generator, sample_experimental_results):
        """Test method comparison analysis generation."""
        analysis = report_generator._generate_method_comparison_analysis(sample_experimental_results)
        
        assert isinstance(analysis, str)
        assert len(analysis) > 0
        
        # Check method ranking is included
        assert 'Meta-PINN' in analysis
        assert 'Standard PINN' in analysis
        assert 'Transfer Learning' in analysis
    
    def test_generate_performance_analysis(self, report_generator, sample_experimental_results):
        """Test performance analysis generation."""
        analysis = report_generator._generate_performance_analysis(sample_experimental_results)
        
        assert isinstance(analysis, str)
        assert len(analysis) > 0
        
        # Should include convergence analysis since convergence_data is provided
        assert 'convergence' in analysis.lower() or 'converged' in analysis.lower()
    
    def test_generate_statistical_analysis(self, report_generator, sample_experimental_results):
        """Test statistical analysis generation."""
        analysis = report_generator._generate_statistical_analysis(sample_experimental_results)
        
        assert isinstance(analysis, str)
        assert len(analysis) > 0
        
        # Should mention statistical significance
        assert 'statistical' in analysis.lower() or 'significance' in analysis.lower()
    
    def test_generate_detailed_results(self, report_generator, sample_experimental_results):
        """Test detailed results generation."""
        details = report_generator._generate_detailed_results(sample_experimental_results)
        
        assert isinstance(details, str)
        assert len(details) > 0
        
        # Should include method details
        assert 'Meta-PINN' in details
        assert 'Standard PINN' in details
        assert 'Transfer Learning' in details
    
    def test_generate_report_plots_with_mocks(self, mock_plot_generator, sample_experimental_results):
        """Test report plot generation with mocked plot generator."""
        generator = ReportGenerator(plot_generator=mock_plot_generator)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            plot_paths = generator._generate_report_plots(sample_experimental_results, output_dir)
            
            # Check that plot generation methods were called
            mock_plot_generator.create_method_comparison_plot.assert_called_once()
            mock_plot_generator.create_convergence_plot.assert_called_once()
            mock_plot_generator.close_all_figures.assert_called()
            
            # Check that plot paths are returned
            assert isinstance(plot_paths, dict)
    
    def test_generate_report_tables_with_mocks(self, mock_table_generator, sample_experimental_results):
        """Test report table generation with mocked table generator."""
        generator = ReportGenerator(table_generator=mock_table_generator)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            table_paths = generator._generate_report_tables(sample_experimental_results, output_dir)
            
            # Check that table generation methods were called
            mock_table_generator.create_method_comparison_table.assert_called_once()
            mock_table_generator.save_table_to_file.assert_called()
            
            # Check that table paths are returned
            assert isinstance(table_paths, dict)
    
    def test_find_best_method(self, report_generator, sample_experimental_results):
        """Test finding the best performing method."""
        best_method, best_value, best_std = report_generator._find_best_method(
            sample_experimental_results, 'accuracy'
        )
        
        assert best_method == 'Meta-PINN'
        assert best_value == 0.95
        assert best_std == 0.02
    
    def test_identify_baseline_method(self, report_generator):
        """Test baseline method identification."""
        methods = ['Meta-PINN', 'Standard PINN', 'Transfer Learning']
        baseline = report_generator._identify_baseline_method(methods)
        
        # Should identify 'Standard PINN' as baseline due to 'Standard' keyword
        assert baseline == 'Standard PINN'
        
        # Test with no obvious baseline
        methods_no_baseline = ['Method A', 'Method B', 'Method C']
        baseline = report_generator._identify_baseline_method(methods_no_baseline)
        assert baseline == 'Method A'  # Should return first method
    
    def test_calculate_improvement(self, report_generator, sample_experimental_results):
        """Test improvement calculation."""
        improvement = report_generator._calculate_improvement(
            sample_experimental_results, 'Meta-PINN', 'Standard PINN', 'accuracy'
        )
        
        # Meta-PINN: 0.95, Standard PINN: 0.87
        # Improvement = (0.95 - 0.87) / 0.87 * 100 ≈ 9.2%
        assert improvement is not None
        assert abs(improvement - 9.195) < 0.01  # Allow small floating point error
    
    def test_count_significant_results(self, report_generator, sample_experimental_results):
        """Test counting significant results."""
        count = report_generator._count_significant_results(sample_experimental_results)
        
        # Should count p-values < 0.05
        # Meta-PINN: 3 significant (0.001, 0.01, 0.001)
        # Standard PINN: 1 significant (0.05 is not < 0.05, but others might be)
        # Transfer Learning: 3 significant (0.02, 0.03, 0.01)
        assert count > 0
        assert isinstance(count, int)
    
    def test_generate_method_ranking(self, report_generator, sample_experimental_results):
        """Test method ranking generation."""
        ranking = report_generator._generate_method_ranking(sample_experimental_results, 'accuracy')
        
        # Should be ranked by accuracy: Meta-PINN (0.95), Transfer Learning (0.91), Standard PINN (0.87)
        assert ranking == ['Meta-PINN', 'Transfer Learning', 'Standard PINN']
    
    def test_analyze_convergence(self, report_generator):
        """Test convergence analysis."""
        convergence_data = {
            'Method A': [1.0, 0.5, 0.08, 0.05, 0.03],  # Converges at step 2
            'Method B': [1.0, 0.8, 0.6, 0.4, 0.2, 0.08, 0.05]  # Converges at step 5
        }
        
        analysis = report_generator._analyze_convergence(convergence_data)
        
        assert isinstance(analysis, str)
        assert 'Method A' in analysis  # Should identify Method A as fastest
    
    def test_analyze_metric_performance(self, report_generator, sample_experimental_results):
        """Test metric performance analysis."""
        analysis = report_generator._analyze_metric_performance(sample_experimental_results, 'accuracy')
        
        assert isinstance(analysis, str)
        assert 'Meta-PINN' in analysis  # Best performer
        assert 'Standard PINN' in analysis  # Worst performer
        assert 'accuracy' in analysis
    
    def test_generate_method_details(self, report_generator):
        """Test method details generation."""
        method_data = {
            'accuracy_mean': 0.95,
            'accuracy_std': 0.02,
            'speed_mean': 10.5,
            'speed_std': 1.2
        }
        
        details = report_generator._generate_method_details('Meta-PINN', method_data)
        
        assert isinstance(details, str)
        assert 'Meta-PINN' in details
        assert '0.95' in details
        assert 'accuracy' in details
        assert 'speed' in details
    
    def test_analyze_hyperparameters(self, report_generator, sample_experimental_results):
        """Test hyperparameter analysis."""
        analysis = report_generator._analyze_hyperparameters(sample_experimental_results['hyperparameters'])
        
        assert isinstance(analysis, str)
        assert 'learning_rate' in analysis
        assert 'batch_size' in analysis
        assert 'Meta-PINN' in analysis
        assert 'Standard PINN' in analysis
    
    def test_report_generation_without_plots_and_tables(self, report_generator, sample_experimental_results):
        """Test report generation without plots and tables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = report_generator.generate_comprehensive_report(
                experimental_results=sample_experimental_results,
                output_dir=output_dir,
                include_plots=False,
                include_tables=False
            )
            
            # Should still generate report and metadata
            assert 'report' in saved_files
            assert 'metadata' in saved_files
            
            # Should not have plot or table files
            plot_files = [k for k in saved_files.keys() if 'plot' in k]
            table_files = [k for k in saved_files.keys() if 'table' in k]
            
            assert len(plot_files) == 0
            assert len(table_files) == 0
    
    def test_error_handling_missing_data(self, report_generator):
        """Test error handling with missing or incomplete data."""
        incomplete_results = {
            'method_results': {
                'Method A': {'accuracy_mean': 0.8}
                # Missing std, other metrics
            }
        }
        
        # Should not crash, should handle gracefully
        summary = report_generator._generate_executive_summary(incomplete_results)
        assert isinstance(summary, str)
        
        analysis = report_generator._generate_method_comparison_analysis(incomplete_results)
        assert isinstance(analysis, str)
    
    def test_report_content_structure(self, report_generator, sample_experimental_results):
        """Test that report content has proper structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = report_generator.generate_comprehensive_report(
                experimental_results=sample_experimental_results,
                output_dir=output_dir,
                format_type='markdown'
            )
            
            report_path = saved_files['report']
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = [
                'Executive Summary',
                'Method Comparison',
                'Performance Analysis',
                'Statistical Analysis',
                'Detailed Results'
            ]
            
            for section in required_sections:
                assert section in content, f"Missing section: {section}"
            
            # Check for table of contents
            assert 'Table of Contents' in content
    
    def test_natural_language_templates(self, report_generator):
        """Test that natural language templates are properly formatted."""
        templates = report_generator.templates
        
        # Check that templates exist
        assert 'executive_summary' in templates
        assert 'method_comparison' in templates
        assert 'performance_analysis' in templates
        
        # Check that templates have proper format strings
        exec_templates = templates['executive_summary']
        assert '{experiment_type}' in exec_templates['intro']
        assert '{method_name}' in exec_templates['best_method']
        assert '{improvement' in exec_templates['improvement']  # Check for improvement format string (may have format spec)
    
    def test_metadata_generation(self, report_generator, sample_experimental_results):
        """Test metadata generation and content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / 'metadata.json'
            
            report_generator._save_report_metadata(
                sample_experimental_results,
                metadata_path,
                'Test Report',
                'Test Author'
            )
            
            assert metadata_path.exists()
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            assert metadata['title'] == 'Test Report'
            assert metadata['author'] == 'Test Author'
            assert 'generation_date' in metadata
            assert 'experiment_info' in metadata
            assert metadata['experiment_info']['num_methods'] == 3
            assert 'accuracy' in metadata['experiment_info']['metrics']
    
    @pytest.mark.parametrize("format_type,expected_extension", [
        ('markdown', '.md'),
        ('latex', '.tex'),
        ('html', '.html')
    ])
    def test_different_output_formats(self, report_generator, sample_experimental_results, format_type, expected_extension):
        """Test report generation in different formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = report_generator.generate_comprehensive_report(
                experimental_results=sample_experimental_results,
                output_dir=output_dir,
                format_type=format_type
            )
            
            report_path = saved_files['report']
            assert report_path.suffix == expected_extension
            assert report_path.exists()
            
            # Check that file has content
            assert report_path.stat().st_size > 0


if __name__ == '__main__':
    pytest.main([__file__])
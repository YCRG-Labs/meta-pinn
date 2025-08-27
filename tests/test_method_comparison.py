"""
Unit tests for automated method comparison system.
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ml_research_pipeline.evaluation.method_comparison import (
    MethodComparison,
    MethodComparisonConfig,
    ComparisonResult
)
from ml_research_pipeline.evaluation.benchmark_suite import BenchmarkResult
from ml_research_pipeline.evaluation.metrics import MetricResult, StatisticalTest


class TestMethodComparisonConfig:
    """Test method comparison configuration."""
    
    def test_config_creation(self):
        """Test creating method comparison configuration."""
        config = MethodComparisonConfig(
            benchmarks=['cavity_flow', 'channel_flow'],
            metrics=['parameter_accuracy', 'adaptation_speed'],
            statistical_tests=['t_test', 'mann_whitney'],
            significance_level=0.05,
            save_dir="test_results"
        )
        
        assert config.benchmarks == ['cavity_flow', 'channel_flow']
        assert config.metrics == ['parameter_accuracy', 'adaptation_speed']
        assert config.significance_level == 0.05
        assert config.save_dir == "test_results"


class TestComparisonResult:
    """Test comparison result data structure."""
    
    def test_comparison_result_creation(self):
        """Test creating comparison result."""
        config = MethodComparisonConfig(
            benchmarks=['test_benchmark'],
            metrics=['test_metric'],
            statistical_tests=['t_test']
        )
        
        result = ComparisonResult(
            config=config,
            benchmark_results={},
            metric_results={},
            statistical_results={},
            rankings={},
            summary={'test': 'data'},
            timestamp=1234567890.0
        )
        
        assert result.config == config
        assert result.summary == {'test': 'data'}
        assert result.timestamp == 1234567890.0


class TestMethodComparison:
    """Test method comparison system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create sample configuration."""
        return MethodComparisonConfig(
            benchmarks=['cavity_flow', 'channel_flow'],
            metrics=['parameter_accuracy', 'adaptation_speed', 'computational_efficiency'],
            statistical_tests=['t_test'],
            significance_level=0.05,
            save_dir=temp_dir,
            save_results=True,
            generate_plots=False,  # Disable for testing
            generate_tables=True
        )
    
    @pytest.fixture
    def sample_methods(self):
        """Create sample methods for testing."""
        method1 = Mock()
        method1.adapt_to_task = Mock(return_value=[])
        method1.forward = Mock(return_value=np.random.randn(100, 3))
        
        method2 = Mock()
        method2.adapt_to_task = Mock(return_value=[])
        method2.forward = Mock(return_value=np.random.randn(100, 3))
        
        return {'method1': method1, 'method2': method2}
    
    def test_method_comparison_initialization(self, sample_config):
        """Test method comparison initialization."""
        comparison = MethodComparison(sample_config)
        
        assert comparison.config == sample_config
        assert comparison.benchmark_suite is not None
        assert comparison.metrics_calculator is not None
        assert comparison.statistical_analyzer is not None
        assert comparison.save_dir.exists()
    
    @patch('ml_research_pipeline.evaluation.method_comparison.PINNBenchmarkSuite')
    def test_run_benchmarks(self, mock_benchmark_suite, sample_config, sample_methods):
        """Test running benchmarks."""
        # Mock benchmark suite
        mock_suite_instance = Mock()
        mock_benchmark_suite.return_value = mock_suite_instance
        
        # Create mock benchmark results
        mock_result = BenchmarkResult(
            benchmark_name="cavity_flow",
            method_name="method1",
            metrics={'parameter_accuracy': 0.85},
            task_results=[{'parameter_accuracy': 0.85, 'adaptation_steps': 10}],
            runtime_info={'total_time': 5.0},
            metadata={}
        )
        mock_suite_instance.run_benchmark.return_value = mock_result
        
        comparison = MethodComparison(sample_config)
        benchmark_results = comparison._run_benchmarks(sample_methods)
        
        assert isinstance(benchmark_results, dict)
        assert len(benchmark_results) == len(sample_config.benchmarks)
        
        for benchmark_name in sample_config.benchmarks:
            assert benchmark_name in benchmark_results
            assert len(benchmark_results[benchmark_name]) == len(sample_methods)
    
    def test_compute_metrics(self, sample_config):
        """Test computing metrics."""
        comparison = MethodComparison(sample_config)
        
        # Create sample benchmark results
        benchmark_results = {
            'cavity_flow': {
                'method1': BenchmarkResult(
                    benchmark_name="cavity_flow",
                    method_name="method1",
                    metrics={'parameter_accuracy': 0.85},
                    task_results=[
                        {'parameter_accuracy': 0.85, 'adaptation_steps': 10, 'computation_time': 5.0},
                        {'parameter_accuracy': 0.90, 'adaptation_steps': 8, 'computation_time': 4.5}
                    ],
                    runtime_info={'total_time': 10.0},
                    metadata={}
                )
            }
        }
        
        metric_results = comparison._compute_metrics(benchmark_results)
        
        assert isinstance(metric_results, dict)
        assert 'cavity_flow' in metric_results
        
        cavity_metrics = metric_results['cavity_flow']
        assert isinstance(cavity_metrics, dict)
        
        # Check that metrics were computed
        for metric_result in cavity_metrics.values():
            assert isinstance(metric_result, MetricResult)
    
    def test_perform_statistical_analysis(self, sample_config):
        """Test performing statistical analysis."""
        comparison = MethodComparison(sample_config)
        
        # Create sample benchmark results with multiple methods
        benchmark_results = {
            'cavity_flow': {
                'method1': BenchmarkResult(
                    benchmark_name="cavity_flow",
                    method_name="method1",
                    metrics={},
                    task_results=[
                        {'parameter_accuracy': 0.85, 'adaptation_steps': 10},
                        {'parameter_accuracy': 0.90, 'adaptation_steps': 8}
                    ],
                    runtime_info={'total_time': 10.0},
                    metadata={}
                ),
                'method2': BenchmarkResult(
                    benchmark_name="cavity_flow",
                    method_name="method2",
                    metrics={},
                    task_results=[
                        {'parameter_accuracy': 0.75, 'adaptation_steps': 15},
                        {'parameter_accuracy': 0.80, 'adaptation_steps': 12}
                    ],
                    runtime_info={'total_time': 15.0},
                    metadata={}
                )
            }
        }
        
        statistical_results = comparison._perform_statistical_analysis(benchmark_results)
        
        assert isinstance(statistical_results, dict)
        assert 'cavity_flow' in statistical_results
        
        cavity_stats = statistical_results['cavity_flow']
        assert isinstance(cavity_stats, dict)
    
    def test_generate_rankings(self, sample_config):
        """Test generating method rankings."""
        comparison = MethodComparison(sample_config)
        
        # Create sample metric results
        metric_results = {
            'cavity_flow': {
                'parameter_accuracy': MetricResult(
                    name='parameter_accuracy',
                    value=0.85,
                    std=0.05
                ),
                'adaptation_speed': MetricResult(
                    name='adaptation_speed',
                    value=10.0,
                    std=2.0
                )
            }
        }
        
        rankings = comparison._generate_rankings(metric_results)
        
        assert isinstance(rankings, dict)
        # Rankings structure may vary based on implementation
    
    def test_generate_summary(self, sample_config):
        """Test generating summary."""
        comparison = MethodComparison(sample_config)
        
        # Create sample data
        benchmark_results = {
            'cavity_flow': {
                'method1': BenchmarkResult(
                    benchmark_name="cavity_flow",
                    method_name="method1",
                    metrics={},
                    task_results=[{'parameter_accuracy': 0.85}],
                    runtime_info={'total_time': 10.0},
                    metadata={}
                )
            }
        }
        
        metric_results = {
            'cavity_flow': {
                'parameter_accuracy': MetricResult(
                    name='parameter_accuracy',
                    value=0.85
                )
            }
        }
        
        statistical_results = {
            'cavity_flow': {
                'parameter_accuracy': {
                    'method1': {
                        'method2': StatisticalTest(
                            test_name='t-test',
                            statistic=2.5,
                            p_value=0.01,
                            effect_size=0.8
                        )
                    }
                }
            }
        }
        
        rankings = {}
        
        summary = comparison._generate_summary(
            benchmark_results, metric_results, statistical_results, rankings
        )
        
        assert isinstance(summary, dict)
        assert 'overview' in summary
        assert 'benchmark_summary' in summary
        assert 'method_performance' in summary
        assert 'statistical_significance' in summary
        assert 'recommendations' in summary
        
        # Check overview
        overview = summary['overview']
        assert 'n_benchmarks' in overview
        assert 'n_methods' in overview
        assert 'n_metrics' in overview
    
    def test_save_results(self, sample_config):
        """Test saving results."""
        comparison = MethodComparison(sample_config)
        
        # Create sample comparison result
        comparison_result = ComparisonResult(
            config=sample_config,
            benchmark_results={},
            metric_results={},
            statistical_results={},
            rankings={},
            summary={'test': 'data'},
            timestamp=1234567890.0
        )
        
        comparison._save_results(comparison_result)
        
        # Check that file was created
        save_files = list(comparison.save_dir.glob("comparison_results_*.json"))
        assert len(save_files) > 0
        
        # Check file content
        with open(save_files[0], 'r') as f:
            saved_data = json.load(f)
        
        assert 'config' in saved_data
        assert 'summary' in saved_data
        assert 'timestamp' in saved_data
        assert saved_data['summary'] == {'test': 'data'}
    
    def test_generate_tables(self, sample_config):
        """Test generating LaTeX tables."""
        comparison = MethodComparison(sample_config)
        
        # Create sample comparison result
        comparison_result = ComparisonResult(
            config=sample_config,
            benchmark_results={
                'cavity_flow': {
                    'method1': BenchmarkResult(
                        benchmark_name="cavity_flow",
                        method_name="method1",
                        metrics={},
                        task_results=[],
                        runtime_info={'total_time': 10.0},
                        metadata={}
                    )
                }
            },
            metric_results={
                'cavity_flow': {
                    'parameter_accuracy': MetricResult(
                        name='parameter_accuracy',
                        value=0.85
                    )
                }
            },
            statistical_results={},
            rankings={},
            summary={
                'benchmark_summary': {
                    'cavity_flow': {
                        'n_methods': 1,
                        'n_tasks': 10,
                        'total_runtime': 10.0
                    }
                }
            },
            timestamp=1234567890.0
        )
        
        comparison._generate_tables(comparison_result)
        
        # Check that LaTeX files were created
        tex_files = list(comparison.save_dir.glob("*.tex"))
        assert len(tex_files) > 0
        
        # Check that summary table exists
        summary_files = list(comparison.save_dir.glob("summary_table_*.tex"))
        assert len(summary_files) > 0
        
        # Check content of summary table
        with open(summary_files[0], 'r') as f:
            content = f.read()
        
        assert "\\begin{table}" in content
        assert "\\end{table}" in content
        assert "Method Comparison Summary" in content
    
    def test_create_summary_table(self, sample_config):
        """Test creating summary table."""
        comparison = MethodComparison(sample_config)
        
        comparison_result = ComparisonResult(
            config=sample_config,
            benchmark_results={},
            metric_results={},
            statistical_results={},
            rankings={},
            summary={
                'benchmark_summary': {
                    'cavity_flow': {
                        'n_methods': 2,
                        'n_tasks': 20,
                        'total_runtime': 25.5
                    },
                    'channel_flow': {
                        'n_methods': 2,
                        'n_tasks': 15,
                        'total_runtime': 18.2
                    }
                }
            },
            timestamp=1234567890.0
        )
        
        table_latex = comparison._create_summary_table(comparison_result)
        
        assert isinstance(table_latex, str)
        assert "\\begin{table}" in table_latex
        assert "\\end{table}" in table_latex
        assert "cavity\\_flow" in table_latex  # Escaped underscore
        assert "channel\\_flow" in table_latex
        assert "25.50" in table_latex  # Runtime formatting
    
    def test_create_comparison_table(self, sample_config):
        """Test creating comparison table for specific benchmark."""
        comparison = MethodComparison(sample_config)
        
        comparison_result = ComparisonResult(
            config=sample_config,
            benchmark_results={
                'cavity_flow': {
                    'method1': BenchmarkResult(
                        benchmark_name="cavity_flow",
                        method_name="method1",
                        metrics={},
                        task_results=[],
                        runtime_info={},
                        metadata={}
                    ),
                    'method2': BenchmarkResult(
                        benchmark_name="cavity_flow",
                        method_name="method2",
                        metrics={},
                        task_results=[],
                        runtime_info={},
                        metadata={}
                    )
                }
            },
            metric_results={
                'cavity_flow': {
                    'parameter_accuracy': MetricResult(
                        name='parameter_accuracy',
                        value=0.85
                    ),
                    'adaptation_speed': MetricResult(
                        name='adaptation_speed',
                        value=10.0
                    )
                }
            },
            statistical_results={},
            rankings={},
            summary={},
            timestamp=1234567890.0
        )
        
        table_latex = comparison._create_comparison_table(comparison_result, 'cavity_flow')
        
        assert isinstance(table_latex, str)
        assert "\\begin{table}" in table_latex
        assert "\\end{table}" in table_latex
        assert "Cavity Flow Benchmark Results" in table_latex
        assert "method1" in table_latex
        assert "method2" in table_latex
    
    def test_create_comparison_table_invalid_benchmark(self, sample_config):
        """Test creating comparison table for invalid benchmark."""
        comparison = MethodComparison(sample_config)
        
        comparison_result = ComparisonResult(
            config=sample_config,
            benchmark_results={},
            metric_results={},
            statistical_results={},
            rankings={},
            summary={},
            timestamp=1234567890.0
        )
        
        table_latex = comparison._create_comparison_table(comparison_result, 'invalid_benchmark')
        
        assert table_latex == ""
    
    def test_load_comparison_results(self, sample_config, temp_dir):
        """Test loading comparison results."""
        comparison = MethodComparison(sample_config)
        
        # Create sample saved results
        saved_data = {
            'config': {
                'benchmarks': ['cavity_flow'],
                'metrics': ['parameter_accuracy'],
                'statistical_tests': ['t_test'],
                'significance_level': 0.05,
                'multiple_comparison_correction': 'bonferroni'
            },
            'summary': {'test': 'loaded_data'},
            'timestamp': 1234567890.0
        }
        
        results_file = Path(temp_dir) / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(saved_data, f)
        
        loaded_result = comparison.load_comparison_results(str(results_file))
        
        assert isinstance(loaded_result, ComparisonResult)
        assert loaded_result.summary == {'test': 'loaded_data'}
        assert loaded_result.timestamp == 1234567890.0
        assert loaded_result.config.benchmarks == ['cavity_flow']
    
    def test_get_method_ranking(self, sample_config):
        """Test getting method ranking."""
        comparison = MethodComparison(sample_config)
        
        comparison_result = ComparisonResult(
            config=sample_config,
            benchmark_results={},
            metric_results={
                'cavity_flow': {
                    'parameter_accuracy': MetricResult(
                        name='parameter_accuracy',
                        value=0.85
                    )
                },
                'channel_flow': {
                    'parameter_accuracy': MetricResult(
                        name='parameter_accuracy',
                        value=0.90
                    )
                }
            },
            statistical_results={},
            rankings={},
            summary={},
            timestamp=1234567890.0
        )
        
        ranking = comparison.get_method_ranking(comparison_result, 'parameter_accuracy')
        
        assert isinstance(ranking, list)
        # Ranking structure may vary based on implementation
    
    @patch('ml_research_pipeline.evaluation.method_comparison.PINNBenchmarkSuite')
    @patch('ml_research_pipeline.evaluation.method_comparison.EvaluationMetrics')
    @patch('ml_research_pipeline.evaluation.method_comparison.StatisticalAnalysis')
    def test_run_comparison_integration(self, mock_stats, mock_metrics, mock_benchmark, 
                                      sample_config, sample_methods):
        """Test full comparison integration."""
        # Mock all components
        mock_suite_instance = Mock()
        mock_benchmark.return_value = mock_suite_instance
        
        mock_result = BenchmarkResult(
            benchmark_name="cavity_flow",
            method_name="method1",
            metrics={'parameter_accuracy': 0.85},
            task_results=[{'parameter_accuracy': 0.85}],
            runtime_info={'total_time': 5.0},
            metadata={}
        )
        mock_suite_instance.run_benchmark.return_value = mock_result
        
        mock_metrics_instance = Mock()
        mock_metrics.return_value = mock_metrics_instance
        mock_metrics_instance.compute_all_metrics.return_value = {
            'parameter_accuracy': MetricResult(name='parameter_accuracy', value=0.85)
        }
        
        mock_stats_instance = Mock()
        mock_stats.return_value = mock_stats_instance
        mock_stats_instance.compare_methods.return_value = {}
        
        # Disable plot generation for testing
        sample_config.generate_plots = False
        
        comparison = MethodComparison(sample_config)
        result = comparison.run_comparison(sample_methods)
        
        assert isinstance(result, ComparisonResult)
        assert result.config == sample_config
        assert isinstance(result.summary, dict)
        assert result.timestamp > 0


if __name__ == "__main__":
    pytest.main([__file__])
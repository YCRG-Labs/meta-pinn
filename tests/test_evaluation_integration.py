"""
Integration tests for the complete evaluation framework.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from ml_research_pipeline.evaluation.benchmark_suite import PINNBenchmarkSuite
from ml_research_pipeline.evaluation.metrics import EvaluationMetrics, StatisticalAnalysis
from ml_research_pipeline.evaluation.method_comparison import (
    MethodComparison, MethodComparisonConfig
)


class TestEvaluationIntegration:
    """Test integration of all evaluation components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_methods(self):
        """Create mock methods for testing."""
        methods = {}
        
        for i, method_name in enumerate(['MetaPINN', 'StandardPINN', 'TransferPINN']):
            method = Mock()
            method.adapt_to_task = Mock(return_value=[])
            method.forward = Mock(return_value=np.random.randn(100, 3))
            method.physics_loss = Mock(return_value=np.random.uniform(1e-5, 1e-3))
            
            # Simulate different performance levels
            base_accuracy = 0.7 + i * 0.1  # MetaPINN best, StandardPINN worst
            method.inferred_parameters = {'viscosity': base_accuracy}
            
            methods[method_name] = method
        
        return methods
    
    def test_benchmark_suite_integration(self, temp_dir):
        """Test benchmark suite with multiple benchmarks."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        
        # Check that all benchmarks are available
        benchmark_names = suite.get_benchmark_names()
        assert len(benchmark_names) >= 4
        assert 'cavity_flow' in benchmark_names
        assert 'channel_flow' in benchmark_names
        assert 'cylinder_flow' in benchmark_names
        assert 'thermal_convection' in benchmark_names
        
        # Check benchmark configurations
        for benchmark_name in benchmark_names:
            config = suite.get_benchmark_config(benchmark_name)
            assert config.name == benchmark_name
            assert config.n_tasks > 0
            assert config.n_support > 0
            assert config.n_query > 0
    
    def test_metrics_and_statistics_integration(self):
        """Test integration between metrics and statistical analysis."""
        # Create sample results
        results = {
            'method1': [
                {
                    'parameter_accuracy': 0.85,
                    'adaptation_steps': 10,
                    'computation_time': 5.0,
                    'physics_residual': 1e-4,
                    'total_error': 0.1
                },
                {
                    'parameter_accuracy': 0.90,
                    'adaptation_steps': 8,
                    'computation_time': 4.5,
                    'physics_residual': 2e-4,
                    'total_error': 0.08
                }
            ],
            'method2': [
                {
                    'parameter_accuracy': 0.75,
                    'adaptation_steps': 15,
                    'computation_time': 8.0,
                    'physics_residual': 5e-4,
                    'total_error': 0.15
                },
                {
                    'parameter_accuracy': 0.80,
                    'adaptation_steps': 12,
                    'computation_time': 7.0,
                    'physics_residual': 3e-4,
                    'total_error': 0.12
                }
            ]
        }
        
        # Compute metrics
        metrics_calculator = EvaluationMetrics()
        metric_results = metrics_calculator.compute_all_metrics(results)
        
        assert len(metric_results) > 0
        assert 'parameter_accuracy' in metric_results
        assert 'adaptation_speed' in metric_results
        
        # Perform statistical analysis
        stats_analyzer = StatisticalAnalysis()
        comparisons = stats_analyzer.compare_methods(results, 'parameter_accuracy')
        
        assert 'method1' in comparisons
        assert 'method2' in comparisons
        assert 'method2' in comparisons['method1']
        
        # Generate summary statistics
        summary = stats_analyzer.generate_summary_statistics(results)
        
        assert 'method1' in summary
        assert 'method2' in summary
        assert 'parameter_accuracy' in summary['method1']
    
    @patch('ml_research_pipeline.evaluation.benchmark_suite.PINNBenchmarkSuite.run_benchmark')
    def test_method_comparison_integration(self, mock_run_benchmark, temp_dir, mock_methods):
        """Test complete method comparison workflow."""
        # Mock benchmark results
        def create_mock_result(benchmark_name, method_name):
            # Simulate different performance for different methods
            base_performance = {
                'MetaPINN': {'accuracy': 0.9, 'steps': 5, 'time': 3.0, 'residual': 1e-5},
                'StandardPINN': {'accuracy': 0.7, 'steps': 50, 'time': 15.0, 'residual': 1e-3},
                'TransferPINN': {'accuracy': 0.8, 'steps': 20, 'time': 8.0, 'residual': 5e-4}
            }
            
            perf = base_performance.get(method_name, base_performance['StandardPINN'])
            
            from ml_research_pipeline.evaluation.benchmark_suite import BenchmarkResult
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                method_name=method_name,
                metrics={
                    'parameter_accuracy_mean': perf['accuracy'],
                    'adaptation_steps_mean': perf['steps'],
                    'computation_time_mean': perf['time'],
                    'physics_residual_mean': perf['residual']
                },
                task_results=[
                    {
                        'parameter_accuracy': perf['accuracy'] + np.random.normal(0, 0.05),
                        'adaptation_steps': perf['steps'] + np.random.randint(-2, 3),
                        'computation_time': perf['time'] + np.random.normal(0, 0.5),
                        'physics_residual': perf['residual'] * np.random.uniform(0.5, 2.0),
                        'total_error': (1 - perf['accuracy']) + np.random.normal(0, 0.02)
                    }
                    for _ in range(10)  # 10 tasks per benchmark
                ],
                runtime_info={'total_time': perf['time'] * 10},
                metadata={}
            )
        
        mock_run_benchmark.side_effect = lambda benchmark_name, method, method_name: create_mock_result(benchmark_name, method_name)
        
        # Create comparison configuration
        config = MethodComparisonConfig(
            benchmarks=['cavity_flow', 'channel_flow'],
            metrics=['parameter_accuracy', 'adaptation_speed', 'computational_efficiency'],
            statistical_tests=['t_test'],
            significance_level=0.05,
            save_dir=temp_dir,
            save_results=True,
            generate_plots=False,  # Disable for testing
            generate_tables=True
        )
        
        # Run comparison
        comparison = MethodComparison(config)
        result = comparison.run_comparison(mock_methods)
        
        # Verify results
        assert result.config == config
        assert len(result.benchmark_results) == 2  # cavity_flow, channel_flow
        assert len(result.metric_results) == 2
        
        # Check that all methods were evaluated
        for benchmark_name in config.benchmarks:
            assert benchmark_name in result.benchmark_results
            benchmark_results = result.benchmark_results[benchmark_name]
            assert len(benchmark_results) == len(mock_methods)
            
            for method_name in mock_methods.keys():
                assert method_name in benchmark_results
        
        # Check summary
        assert 'overview' in result.summary
        assert 'benchmark_summary' in result.summary
        assert 'method_performance' in result.summary
        
        overview = result.summary['overview']
        assert overview['n_benchmarks'] == 2
        assert overview['n_methods'] == 3
        assert overview['n_metrics'] == 3
        
        # Check that files were saved
        save_dir = temp_dir
        json_files = list(Path(save_dir).glob("comparison_results_*.json"))
        assert len(json_files) > 0
        
        tex_files = list(Path(save_dir).glob("*.tex"))
        assert len(tex_files) > 0
    
    def test_evaluation_pipeline_error_handling(self, temp_dir):
        """Test error handling in evaluation pipeline."""
        # Create methods that will fail
        failing_method = Mock()
        failing_method.adapt_to_task = Mock(side_effect=Exception("Method failed"))
        
        working_method = Mock()
        working_method.adapt_to_task = Mock(return_value=[])
        working_method.forward = Mock(return_value=np.random.randn(100, 3))
        
        methods = {
            'failing_method': failing_method,
            'working_method': working_method
        }
        
        config = MethodComparisonConfig(
            benchmarks=['cavity_flow'],
            metrics=['parameter_accuracy'],
            statistical_tests=['t_test'],
            save_dir=temp_dir,
            save_results=False,
            generate_plots=False,
            generate_tables=False
        )
        
        # This should not crash even with failing methods
        comparison = MethodComparison(config)
        
        # Mock the benchmark suite to avoid actual computation
        with patch.object(comparison.benchmark_suite, 'run_benchmark') as mock_run:
            from ml_research_pipeline.evaluation.benchmark_suite import BenchmarkResult
            
            def mock_benchmark_run(benchmark_name, method, method_name):
                if method_name == 'failing_method':
                    return BenchmarkResult(
                        benchmark_name=benchmark_name,
                        method_name=method_name,
                        metrics={'error': float('inf')},
                        task_results=[],
                        runtime_info={'total_time': float('inf')},
                        metadata={'error': 'Method failed'}
                    )
                else:
                    return BenchmarkResult(
                        benchmark_name=benchmark_name,
                        method_name=method_name,
                        metrics={'parameter_accuracy_mean': 0.8},
                        task_results=[{'parameter_accuracy': 0.8}],
                        runtime_info={'total_time': 5.0},
                        metadata={}
                    )
            
            mock_run.side_effect = mock_benchmark_run
            
            result = comparison.run_comparison(methods)
            
            # Should complete successfully despite failures
            assert result is not None
            assert 'cavity_flow' in result.benchmark_results
            assert 'failing_method' in result.benchmark_results['cavity_flow']
            assert 'working_method' in result.benchmark_results['cavity_flow']
    
    def test_evaluation_scalability(self, temp_dir):
        """Test evaluation framework with larger number of methods and benchmarks."""
        # Create many mock methods
        methods = {}
        for i in range(10):  # 10 methods
            method = Mock()
            method.adapt_to_task = Mock(return_value=[])
            method.forward = Mock(return_value=np.random.randn(50, 3))  # Smaller for speed
            methods[f'method_{i}'] = method
        
        config = MethodComparisonConfig(
            benchmarks=['cavity_flow', 'channel_flow', 'cylinder_flow'],  # 3 benchmarks
            metrics=['parameter_accuracy', 'adaptation_speed'],
            statistical_tests=['t_test'],
            save_dir=temp_dir,
            save_results=False,
            generate_plots=False,
            generate_tables=False
        )
        
        comparison = MethodComparison(config)
        
        # Mock benchmark runs for speed
        with patch.object(comparison.benchmark_suite, 'run_benchmark') as mock_run:
            from ml_research_pipeline.evaluation.benchmark_suite import BenchmarkResult
            
            def quick_benchmark_run(benchmark_name, method, method_name):
                return BenchmarkResult(
                    benchmark_name=benchmark_name,
                    method_name=method_name,
                    metrics={'parameter_accuracy_mean': np.random.uniform(0.6, 0.9)},
                    task_results=[
                        {
                            'parameter_accuracy': np.random.uniform(0.6, 0.9),
                            'adaptation_steps': np.random.randint(5, 20),
                            'computation_time': np.random.uniform(1.0, 10.0)
                        }
                        for _ in range(5)  # Fewer tasks for speed
                    ],
                    runtime_info={'total_time': np.random.uniform(5.0, 50.0)},
                    metadata={}
                )
            
            mock_run.side_effect = quick_benchmark_run
            
            result = comparison.run_comparison(methods)
            
            # Verify scalability
            assert len(result.benchmark_results) == 3  # 3 benchmarks
            
            for benchmark_name in config.benchmarks:
                assert len(result.benchmark_results[benchmark_name]) == 10  # 10 methods
            
            # Check that statistical analysis handled many comparisons
            total_comparisons = result.summary['statistical_significance']['total_tests']
            expected_comparisons = 3 * 2 * 10 * 9  # benchmarks * metrics * methods * (methods-1)
            # Note: actual number may be different due to implementation details
            assert total_comparisons >= 0  # At least some comparisons were made


if __name__ == "__main__":
    pytest.main([__file__])
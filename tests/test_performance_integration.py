"""
Integration tests for the complete performance benchmarking and optimization system.

These tests demonstrate the full workflow of performance profiling, scalability testing,
and regression detection for meta-learning PINNs.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch

from ml_research_pipeline.evaluation.performance_profiler import (
    PerformanceBenchmarkSuite,
    MemoryMonitor,
    ComputationProfiler,
    ScalabilityTester
)
from ml_research_pipeline.evaluation.performance_regression import (
    PerformanceRegressionTester,
    create_performance_regression_tester
)
from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.config.model_config import MetaPINNConfig


class MockMetaPINN(nn.Module):
    """Mock MetaPINN for testing without full dependencies."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = config.layers if hasattr(config, 'layers') else [2, 64, 64, 1]
        
        # Build network
        network_layers = []
        for i in range(len(layers) - 1):
            network_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # No activation on output layer
                network_layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*network_layers)
    
    def forward(self, x):
        return self.network(x)
    
    def physics_loss(self, coords, task_config, params=None):
        """Mock physics loss computation."""
        batch_size = coords.shape[0]
        return {
            'pde_residual': torch.randn(batch_size).mean(),
            'boundary_loss': torch.randn(batch_size).mean(),
            'total': torch.randn(1).abs()
        }
    
    def adapt_to_task(self, task, adaptation_steps=5):
        """Mock task adaptation."""
        # Simulate adaptation by running forward passes
        coords = task['support_set']['coords']
        for _ in range(adaptation_steps):
            _ = self.forward(coords)
        
        # Return current parameters
        return {name: param.clone() for name, param in self.named_parameters()}


class TestPerformanceIntegration:
    """Integration tests for the complete performance system."""
    
    def test_complete_performance_workflow(self):
        """Test the complete performance benchmarking workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize performance suite
            suite = PerformanceBenchmarkSuite(temp_dir)
            
            # Create model factory
            def model_fn():
                config = type('Config', (), {
                    'layers': [2, 64, 64, 1],
                    'activation': 'tanh',
                    'physics_loss_weight': 1.0
                })()
                return MockMetaPINN(config)
            
            # Define test configurations
            test_configs = {
                'batch_sizes': [1, 4, 8],
                'model_configs': [
                    {'layers': [2, 32, 1]},
                    {'layers': [2, 64, 1]},
                    {'layers': [2, 128, 1]}
                ],
                'world_sizes': [1, 2],
                'input_shape': (2,),
                'batch_size': 4,
                'num_iterations': 3
            }
            
            # Run full benchmark
            results = suite.run_full_benchmark(model_fn, test_configs)
            
            # Verify results structure
            assert isinstance(results, dict)
            assert 'timestamp' in results
            assert 'system_info' in results
            assert 'memory_optimization' in results
            assert 'batch_size_scaling' in results
            assert 'model_size_scaling' in results
            assert 'distributed_scaling' in results
            
            # Verify batch size scaling results
            batch_results = results['batch_size_scaling']['results']
            assert len(batch_results) == 3
            for result in batch_results:
                if 'error' not in result:
                    assert 'batch_size' in result
                    assert 'throughput_samples_per_s' in result
                    assert 'memory_stats' in result
            
            # Verify model size scaling results
            model_results = results['model_size_scaling']['results']
            assert len(model_results) == 3
            for result in model_results:
                if 'error' not in result:
                    assert 'total_params' in result
                    assert 'metrics' in result
            
            # Verify files were created
            results_file = Path(temp_dir) / "full_benchmark_results.json"
            assert results_file.exists()
    
    def test_memory_monitoring_integration(self):
        """Test memory monitoring during model operations."""
        monitor = MemoryMonitor(monitor_interval=0.01)
        
        # Start monitoring
        monitor.start_monitoring()
        
        try:
            # Create and use models of different sizes
            models = []
            for hidden_size in [32, 64, 128]:
                config = type('Config', (), {'layers': [2, hidden_size, 1]})()
                model = MockMetaPINN(config)
                models.append(model)
                
                # Run forward passes
                x = torch.randn(10, 2)
                _ = model(x)
            
            # Let monitoring collect data
            time.sleep(0.05)
            
            # Get memory statistics
            stats = monitor.get_memory_stats()
            
            assert 'cpu_memory' in stats
            assert stats['cpu_memory']['current_gb'] > 0
            assert stats['cpu_memory']['peak_gb'] >= stats['cpu_memory']['current_gb']
            
            # Test memory optimization
            optimization_result = monitor.optimize_memory()
            assert 'actions_taken' in optimization_result
            assert 'recommendations' in optimization_result
            
        finally:
            monitor.stop_monitoring()
    
    def test_computation_profiling_integration(self):
        """Test computation profiling with meta-learning operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            profiler = ComputationProfiler(temp_dir)
            
            def meta_learning_step():
                """Simulate a meta-learning step."""
                config = type('Config', (), {'layers': [2, 64, 1]})()
                model = MockMetaPINN(config)
                
                # Create mock task
                task = {
                    'support_set': {
                        'coords': torch.randn(10, 2),
                        'data': torch.randn(10, 1)
                    },
                    'query_set': {
                        'coords': torch.randn(5, 2),
                        'data': torch.randn(5, 1)
                    },
                    'config': {'viscosity_type': 'linear'}
                }
                
                # Simulate adaptation
                adapted_params = model.adapt_to_task(task, adaptation_steps=3)
                
                # Simulate meta-gradient computation
                query_coords = task['query_set']['coords']
                predictions = model(query_coords)
                loss = torch.nn.functional.mse_loss(predictions, task['query_set']['data'])
                loss.backward()
                
                return loss.item()
            
            # Profile the meta-learning step
            result, metrics = profiler.profile_function(meta_learning_step)
            
            assert isinstance(result, float)
            assert metrics.cpu_time > 0
            assert metrics.latency > 0
            
            # Generate optimization report
            report = profiler.generate_optimization_report()
            assert 'bottlenecks' in report
            assert 'recommendations' in report
    
    def test_scalability_testing_integration(self):
        """Test scalability testing with different configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = ScalabilityTester(temp_dir)
            
            def model_fn():
                config = type('Config', (), {'layers': [2, 64, 1]})()
                return MockMetaPINN(config)
            
            # Test batch size scaling
            batch_results = tester.test_batch_size_scaling(
                model_fn=model_fn,
                batch_sizes=[1, 2, 4],
                input_shape=(2,),
                num_iterations=3
            )
            
            assert 'results' in batch_results
            assert 'optimal_batch_size' in batch_results
            assert len(batch_results['results']) == 3
            
            # Test model size scaling
            model_configs = [
                {'layers': [2, 32, 1]},
                {'layers': [2, 64, 1]},
                {'layers': [2, 128, 1]}
            ]
            
            model_results = tester.test_model_size_scaling(
                model_configs=model_configs,
                input_shape=(2,),
                batch_size=4,
                num_iterations=3
            )
            
            assert 'results' in model_results
            assert len(model_results['results']) == 3
            
            # Verify parameter counts increase
            valid_results = [r for r in model_results['results'] if 'error' not in r]
            if len(valid_results) > 1:
                params = [r['total_params'] for r in valid_results]
                assert params == sorted(params)  # Should be increasing
    
    def test_regression_testing_integration(self):
        """Test regression testing with model performance changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            def fast_inference(batch_size):
                """Fast model inference."""
                config = type('Config', (), {'layers': [2, 32, 1]})()
                model = MockMetaPINN(config)
                x = torch.randn(batch_size, 2)
                with torch.no_grad():
                    return model(x)
            
            def slow_inference(batch_size):
                """Slower model inference."""
                config = type('Config', (), {'layers': [2, 128, 128, 1]})()  # Larger model
                model = MockMetaPINN(config)
                x = torch.randn(batch_size, 2)
                with torch.no_grad():
                    return model(x)
            
            # Create baseline with fast model
            baseline = tester.create_baseline(
                "model_inference",
                fast_inference,
                {'model_type': 'fast'},
                8,  # batch_size
                num_runs=5,
                warmup_runs=2
            )
            
            assert baseline.test_name == "model_inference"
            assert baseline.metrics.cpu_time > 0
            
            # Test with slower model (should detect regression)
            result = tester.run_regression_test(
                "model_inference",
                slow_inference,
                {'model_type': 'slow'},
                8,  # batch_size
                num_runs=5,
                warmup_runs=2
            )
            
            assert result.test_name == "model_inference"
            assert result.current_metrics.cpu_time >= result.baseline_metrics.cpu_time
            
            # Generate regression report
            report = tester.generate_regression_report({'model_inference': result})
            
            assert 'summary' in report
            assert 'detailed_results' in report
            assert report['summary']['total_tests'] == 1
    
    def test_end_to_end_performance_pipeline(self):
        """Test complete end-to-end performance analysis pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize all components
            benchmark_suite = PerformanceBenchmarkSuite(temp_dir / "benchmarks")
            regression_tester = PerformanceRegressionTester(temp_dir / "regression")
            
            def create_model(config):
                """Create model from configuration."""
                return MockMetaPINN(config)
            
            # Define test scenarios
            scenarios = {
                'small_model': {
                    'config': type('Config', (), {'layers': [2, 32, 1]})(),
                    'batch_size': 4
                },
                'large_model': {
                    'config': type('Config', (), {'layers': [2, 128, 128, 1]})(),
                    'batch_size': 4
                }
            }
            
            # Step 1: Run initial benchmarks and create baselines
            for scenario_name, scenario in scenarios.items():
                def model_inference(batch_size, config=scenario['config']):
                    model = create_model(config)
                    x = torch.randn(batch_size, 2)
                    with torch.no_grad():
                        return model(x)
                
                # Create regression baseline
                regression_tester.create_baseline(
                    f"{scenario_name}_inference",
                    model_inference,
                    {'scenario': scenario_name},
                    scenario['batch_size'],
                    num_runs=3,
                    warmup_runs=1
                )
            
            # Step 2: Run comprehensive benchmarks
            def model_fn():
                return create_model(scenarios['small_model']['config'])
            
            benchmark_configs = {
                'batch_sizes': [1, 4, 8],
                'input_shape': (2,),
                'num_iterations': 3
            }
            
            benchmark_results = benchmark_suite.run_full_benchmark(model_fn, benchmark_configs)
            
            # Step 3: Simulate performance regression
            def slower_inference(batch_size):
                """Simulate slower inference (e.g., after code changes)."""
                model = create_model(scenarios['large_model']['config'])  # Use larger model
                x = torch.randn(batch_size, 2)
                with torch.no_grad():
                    time.sleep(0.001)  # Add artificial delay
                    return model(x)
            
            # Test for regression
            regression_result = regression_tester.run_regression_test(
                "small_model_inference",
                slower_inference,
                {'scenario': 'small_model_modified'},
                scenarios['small_model']['batch_size'],
                num_runs=3,
                warmup_runs=1
            )
            
            # Step 4: Generate comprehensive reports
            regression_report = regression_tester.generate_regression_report(
                {'small_model_inference': regression_result}
            )
            
            # Verify all components worked together
            assert isinstance(benchmark_results, dict)
            assert 'batch_size_scaling' in benchmark_results
            
            assert isinstance(regression_result.regression_detected, bool)
            assert regression_result.current_metrics.cpu_time > 0
            
            assert isinstance(regression_report, dict)
            assert 'summary' in regression_report
            assert 'recommendations' in regression_report
            
            # Verify files were created
            benchmark_file = Path(temp_dir) / "benchmarks" / "full_benchmark_results.json"
            assert benchmark_file.exists()
            
            regression_files = list(Path(temp_dir / "regression").glob("regression_report_*.json"))
            assert len(regression_files) > 0
    
    def test_performance_optimization_recommendations(self):
        """Test generation of performance optimization recommendations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create profiler and run some operations
            profiler = ComputationProfiler(temp_dir)
            
            def memory_intensive_operation():
                """Operation that uses significant memory."""
                tensors = []
                for i in range(10):
                    tensor = torch.randn(100, 100)
                    tensors.append(tensor)
                
                # Simulate computation
                result = torch.stack(tensors).sum()
                return result
            
            def cpu_intensive_operation():
                """Operation that uses significant CPU."""
                x = torch.randn(200, 200)
                for _ in range(5):
                    x = torch.matmul(x, x.T)
                return x.sum()
            
            # Profile operations
            with profiler.profile_context("memory_test"):
                memory_intensive_operation()
            
            with profiler.profile_context("cpu_test"):
                cpu_intensive_operation()
            
            # Generate optimization report
            report = profiler.generate_optimization_report()
            
            assert 'bottlenecks' in report
            assert 'recommendations' in report
            assert isinstance(report['recommendations'], list)
            
            # Should have some recommendations
            if report['bottlenecks']:
                assert len(report['recommendations']) > 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_performance_integration(self):
        """Test GPU-specific performance monitoring and optimization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            suite = PerformanceBenchmarkSuite(temp_dir)
            
            def gpu_model_fn():
                config = type('Config', (), {'layers': [2, 64, 1]})()
                model = MockMetaPINN(config)
                return model.cuda()
            
            # Test GPU memory monitoring
            monitor = MemoryMonitor()
            monitor.start_monitoring()
            
            try:
                # Create GPU tensors
                gpu_tensors = []
                for i in range(5):
                    tensor = torch.randn(100, 100).cuda()
                    gpu_tensors.append(tensor)
                
                # Get GPU memory stats
                stats = monitor.get_memory_stats()
                
                assert 'gpu_memory' in stats
                assert stats['gpu_memory']['allocated_gb'] > 0
                
                # Test memory optimization
                optimization_result = monitor.optimize_memory()
                assert 'actions_taken' in optimization_result
                
            finally:
                monitor.stop_monitoring()
            
            # Test GPU scalability
            test_configs = {
                'batch_sizes': [1, 4],
                'input_shape': (2,),
                'num_iterations': 2
            }
            
            results = suite.run_full_benchmark(gpu_model_fn, test_configs)
            
            # Should have GPU-specific information
            assert 'system_info' in results
            assert results['system_info']['cuda_available']
            assert results['system_info']['cuda_device_count'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
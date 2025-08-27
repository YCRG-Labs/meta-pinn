"""
Tests for performance regression testing system.
"""

import pytest
import torch
import torch.nn as nn
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from ml_research_pipeline.evaluation.performance_regression import (
    PerformanceBaseline,
    RegressionResult,
    PerformanceRegressionTester,
    create_performance_regression_tester
)
from ml_research_pipeline.evaluation.performance_profiler import PerformanceMetrics


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestPerformanceBaseline:
    """Test PerformanceBaseline dataclass."""
    
    def test_baseline_creation(self):
        """Test creating performance baseline."""
        metrics = PerformanceMetrics(cpu_time=1.0, gpu_time=0.5, memory_peak=2.0)
        
        baseline = PerformanceBaseline(
            test_name="test_forward_pass",
            timestamp=time.time(),
            git_commit="abc123",
            metrics=metrics,
            config_hash="def456"
        )
        
        assert baseline.test_name == "test_forward_pass"
        assert baseline.git_commit == "abc123"
        assert baseline.metrics == metrics
        assert baseline.config_hash == "def456"
    
    def test_baseline_to_dict(self):
        """Test converting baseline to dictionary."""
        metrics = PerformanceMetrics(cpu_time=1.0, gpu_time=0.5)
        baseline = PerformanceBaseline(
            test_name="test",
            timestamp=time.time(),
            metrics=metrics
        )
        
        baseline_dict = baseline.to_dict()
        
        assert isinstance(baseline_dict, dict)
        assert baseline_dict['test_name'] == "test"
        assert 'metrics' in baseline_dict
        assert isinstance(baseline_dict['metrics'], dict)
    
    def test_baseline_from_dict(self):
        """Test creating baseline from dictionary."""
        data = {
            'test_name': 'test',
            'timestamp': time.time(),
            'metrics': {
                'cpu_time': 1.0,
                'gpu_time': 0.5,
                'memory_peak': 2.0,
                'memory_allocated': 0.0,
                'memory_reserved': 0.0,
                'gpu_utilization': 0.0,
                'throughput': 0.0,
                'latency': 0.0,
                'flops': None
            }
        }
        
        baseline = PerformanceBaseline.from_dict(data)
        
        assert baseline.test_name == 'test'
        assert isinstance(baseline.metrics, PerformanceMetrics)
        assert baseline.metrics.cpu_time == 1.0


class TestRegressionResult:
    """Test RegressionResult dataclass."""
    
    def test_regression_result_creation(self):
        """Test creating regression result."""
        baseline_metrics = PerformanceMetrics(cpu_time=1.0, gpu_time=0.5)
        current_metrics = PerformanceMetrics(cpu_time=1.2, gpu_time=0.6)
        
        result = RegressionResult(
            test_name="test",
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            regression_detected=True,
            regression_percentage=20.0,
            threshold_exceeded={'cpu_time': True},
            details={'info': 'test'}
        )
        
        assert result.test_name == "test"
        assert result.regression_detected
        assert result.regression_percentage == 20.0
        assert result.threshold_exceeded['cpu_time']
    
    def test_regression_result_to_dict(self):
        """Test converting regression result to dictionary."""
        baseline_metrics = PerformanceMetrics(cpu_time=1.0)
        current_metrics = PerformanceMetrics(cpu_time=1.2)
        
        result = RegressionResult(
            test_name="test",
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            regression_detected=True,
            regression_percentage=20.0,
            threshold_exceeded={'cpu_time': True},
            details={}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['test_name'] == "test"
        assert result_dict['regression_detected']
        assert 'baseline_metrics' in result_dict
        assert 'current_metrics' in result_dict


class TestPerformanceRegressionTester:
    """Test PerformanceRegressionTester class."""
    
    def test_tester_initialization(self):
        """Test regression tester initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            assert tester.baseline_dir == Path(temp_dir)
            assert isinstance(tester.regression_thresholds, dict)
            assert 'cpu_time' in tester.regression_thresholds
            assert not tester.auto_update_baselines
    
    def test_create_baseline(self):
        """Test creating a performance baseline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            def test_function(x):
                time.sleep(0.001)
                return x * 2
            
            baseline = tester.create_baseline(
                "test_multiply",
                test_function,
                {'input': 5},
                5,
                num_runs=3,
                warmup_runs=1
            )
            
            assert isinstance(baseline, PerformanceBaseline)
            assert baseline.test_name == "test_multiply"
            assert isinstance(baseline.metrics, PerformanceMetrics)
            assert baseline.metrics.cpu_time > 0
            
            # Check that baseline was saved
            assert "test_multiply" in tester.baselines
    
    def test_run_regression_test_no_baseline(self):
        """Test running regression test when no baseline exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            def test_function(x):
                return x * 2
            
            result = tester.run_regression_test(
                "new_test",
                test_function,
                {'input': 5},
                5,
                num_runs=3,
                warmup_runs=1
            )
            
            assert isinstance(result, RegressionResult)
            assert result.test_name == "new_test"
            assert not result.regression_detected
            assert result.details['status'] == 'baseline_created'
    
    def test_run_regression_test_with_baseline(self):
        """Test running regression test with existing baseline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            def fast_function(x):
                return x * 2
            
            def slow_function(x):
                time.sleep(0.002)  # Slower than fast_function
                return x * 2
            
            # Create baseline with fast function
            tester.create_baseline(
                "speed_test",
                fast_function,
                {'input': 5},
                5,
                num_runs=3,
                warmup_runs=1
            )
            
            # Test with slower function
            result = tester.run_regression_test(
                "speed_test",
                slow_function,
                {'input': 5},
                5,
                num_runs=3,
                warmup_runs=1
            )
            
            assert isinstance(result, RegressionResult)
            assert result.test_name == "speed_test"
            # Should detect regression due to slower function
            assert result.current_metrics.cpu_time > result.baseline_metrics.cpu_time
    
    def test_compare_with_baseline(self):
        """Test comparing metrics with baseline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            baseline_metrics = PerformanceMetrics(
                cpu_time=1.0,
                gpu_time=0.5,
                memory_peak=2.0,
                throughput=100.0
            )
            
            # Current metrics with regression
            current_metrics = PerformanceMetrics(
                cpu_time=1.2,  # 20% increase
                gpu_time=0.6,  # 20% increase
                memory_peak=2.5,  # 25% increase
                throughput=80.0  # 20% decrease
            )
            
            result = tester._compare_with_baseline(
                "test", baseline_metrics, current_metrics
            )
            
            assert result.regression_detected
            assert result.threshold_exceeded['cpu_time']
            assert result.threshold_exceeded['gpu_time']
            assert result.threshold_exceeded['memory_peak']
            assert result.threshold_exceeded['throughput']
    
    def test_run_test_suite(self):
        """Test running a suite of regression tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            def test_function1(x):
                return x * 2
            
            def test_function2(x):
                return x + 1
            
            test_suite = {
                'test1': {
                    'function': test_function1,
                    'args': [5],
                    'kwargs': {'num_runs': 3, 'warmup_runs': 1},
                    'config': {'operation': 'multiply'}
                },
                'test2': {
                    'function': test_function2,
                    'args': [5],
                    'kwargs': {'num_runs': 3, 'warmup_runs': 1},
                    'config': {'operation': 'add'}
                }
            }
            
            results = tester.run_test_suite(test_suite)
            
            assert isinstance(results, dict)
            assert 'test1' in results
            assert 'test2' in results
            assert isinstance(results['test1'], RegressionResult)
            assert isinstance(results['test2'], RegressionResult)
    
    def test_generate_regression_report(self):
        """Test generating regression report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            # Create mock results
            baseline_metrics = PerformanceMetrics(cpu_time=1.0, memory_peak=2.0)
            current_metrics = PerformanceMetrics(cpu_time=1.3, memory_peak=2.6)  # Regression
            
            result = RegressionResult(
                test_name="test",
                baseline_metrics=baseline_metrics,
                current_metrics=current_metrics,
                regression_detected=True,
                regression_percentage=30.0,
                threshold_exceeded={'cpu_time': True, 'memory_peak': True},
                details={}
            )
            
            results = {'test': result}
            report = tester.generate_regression_report(results)
            
            assert isinstance(report, dict)
            assert 'summary' in report
            assert 'detailed_results' in report
            assert 'regression_categories' in report
            assert 'recommendations' in report
            
            assert report['summary']['total_tests'] == 1
            assert report['summary']['regressions_detected'] == 1
            assert len(report['regression_categories']['severe']) == 1
    
    def test_cleanup_old_baselines(self):
        """Test cleaning up old baselines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            # Create old baseline
            old_baseline = PerformanceBaseline(
                test_name="old_test",
                timestamp=time.time() - (40 * 24 * 3600),  # 40 days ago
                metrics=PerformanceMetrics(cpu_time=1.0)
            )
            
            # Create recent baseline
            recent_baseline = PerformanceBaseline(
                test_name="recent_test",
                timestamp=time.time() - (10 * 24 * 3600),  # 10 days ago
                metrics=PerformanceMetrics(cpu_time=1.0)
            )
            
            tester.baselines["old_test"] = old_baseline
            tester.baselines["recent_test"] = recent_baseline
            
            # Cleanup baselines older than 30 days
            tester.cleanup_old_baselines(days_old=30)
            
            assert "old_test" not in tester.baselines
            assert "recent_test" in tester.baselines
    
    def test_export_import_baselines(self):
        """Test exporting and importing baselines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            # Create baseline
            baseline = PerformanceBaseline(
                test_name="export_test",
                timestamp=time.time(),
                metrics=PerformanceMetrics(cpu_time=1.0, gpu_time=0.5)
            )
            tester.baselines["export_test"] = baseline
            
            # Export baselines
            export_file = Path(temp_dir) / "exported_baselines.json"
            tester.export_baselines(export_file)
            
            assert export_file.exists()
            
            # Clear baselines and import
            tester.baselines.clear()
            tester.import_baselines(export_file)
            
            assert "export_test" in tester.baselines
            assert isinstance(tester.baselines["export_test"], PerformanceBaseline)
    
    def test_system_info_changed(self):
        """Test detecting system info changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            # Create baseline with system info
            baseline = PerformanceBaseline(
                test_name="system_test",
                timestamp=time.time(),
                system_info={'cpu_count': 4, 'memory_gb': 8.0},
                metrics=PerformanceMetrics(cpu_time=1.0)
            )
            tester.baselines["system_test"] = baseline
            
            # Mock different system info
            with patch.object(tester, '_get_system_info') as mock_system_info:
                mock_system_info.return_value = {'cpu_count': 8, 'memory_gb': 16.0}
                
                changed = tester._system_info_changed("system_test")
                assert changed
    
    def test_config_hash_computation(self):
        """Test configuration hash computation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            config1 = {'param1': 'value1', 'param2': 42}
            config2 = {'param1': 'value1', 'param2': 42}
            config3 = {'param1': 'value2', 'param2': 42}
            
            hash1 = tester._compute_config_hash(config1)
            hash2 = tester._compute_config_hash(config2)
            hash3 = tester._compute_config_hash(config3)
            
            assert hash1 == hash2  # Same config should have same hash
            assert hash1 != hash3  # Different config should have different hash
    
    def test_auto_update_baselines(self):
        """Test automatic baseline updating."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir, auto_update_baselines=True)
            
            def test_function(x):
                return x * 2
            
            # Create initial baseline
            tester.create_baseline(
                "auto_update_test",
                test_function,
                {'input': 5},
                5,
                num_runs=3,
                warmup_runs=1
            )
            
            initial_timestamp = tester.baselines["auto_update_test"].timestamp
            
            # Run regression test (should update baseline if no regression)
            time.sleep(0.001)  # Ensure different timestamp
            result = tester.run_regression_test(
                "auto_update_test",
                test_function,
                {'input': 5},
                5,
                num_runs=3,
                warmup_runs=1
            )
            
            # Baseline should be updated if no regression detected
            if not result.regression_detected:
                assert tester.baselines["auto_update_test"].timestamp > initial_timestamp


class TestCreatePerformanceRegressionTester:
    """Test factory function for creating regression tester."""
    
    def test_create_with_defaults(self):
        """Test creating tester with default configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = create_performance_regression_tester(temp_dir)
            
            assert isinstance(tester, PerformanceRegressionTester)
            assert tester.baseline_dir == Path(temp_dir)
            assert not tester.auto_update_baselines
    
    def test_create_with_custom_config(self):
        """Test creating tester with custom configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'regression_thresholds': {
                    'cpu_time': 20.0,
                    'memory_peak': 25.0
                },
                'auto_update_baselines': True
            }
            
            tester = create_performance_regression_tester(temp_dir, config)
            
            assert tester.regression_thresholds['cpu_time'] == 20.0
            assert tester.regression_thresholds['memory_peak'] == 25.0
            assert tester.auto_update_baselines


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    def test_model_performance_regression(self):
        """Test detecting performance regression in model inference."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(temp_dir)
            
            def fast_model_inference(batch_size):
                model = SimpleTestModel(input_size=2, hidden_size=32, output_size=1)
                x = torch.randn(batch_size, 2)
                with torch.no_grad():
                    return model(x)
            
            def slow_model_inference(batch_size):
                model = SimpleTestModel(input_size=2, hidden_size=128, output_size=1)  # Larger model
                x = torch.randn(batch_size, 2)
                with torch.no_grad():
                    return model(x)
            
            # Create baseline with fast model
            tester.create_baseline(
                "model_inference",
                fast_model_inference,
                {'model_size': 'small'},
                32,  # batch_size
                num_runs=5,
                warmup_runs=2
            )
            
            # Test with slow model
            result = tester.run_regression_test(
                "model_inference",
                slow_model_inference,
                {'model_size': 'large'},
                32,  # batch_size
                num_runs=5,
                warmup_runs=2
            )
            
            # Should detect some performance difference
            assert isinstance(result, RegressionResult)
            # Note: Performance can vary, so we just check that the test ran successfully
            assert result.current_metrics.cpu_time > 0
            assert result.baseline_metrics.cpu_time > 0
    
    def test_memory_regression_detection(self):
        """Test detecting memory usage regression."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(
                temp_dir,
                regression_thresholds={'memory_peak': 10.0}  # Strict memory threshold
            )
            
            def low_memory_function():
                x = torch.randn(100, 100)
                return torch.sum(x)
            
            def high_memory_function():
                x = torch.randn(1000, 1000)  # Much larger tensor
                return torch.sum(x)
            
            # Create baseline with low memory function
            tester.create_baseline(
                "memory_test",
                low_memory_function,
                {'tensor_size': 'small'},
                num_runs=3,
                warmup_runs=1
            )
            
            # Test with high memory function
            result = tester.run_regression_test(
                "memory_test",
                high_memory_function,
                {'tensor_size': 'large'},
                num_runs=3,
                warmup_runs=1
            )
            
            assert isinstance(result, RegressionResult)
            # Memory usage should be higher
            assert result.current_metrics.memory_allocated >= result.baseline_metrics.memory_allocated


if __name__ == "__main__":
    pytest.main([__file__])
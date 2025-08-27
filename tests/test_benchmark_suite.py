"""
Unit tests for the PINN benchmark suite.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from ml_research_pipeline.evaluation.benchmark_suite import (
    PINNBenchmarkSuite,
    BenchmarkConfig,
    BenchmarkResult,
    CavityFlowBenchmark,
    ChannelFlowBenchmark,
    CylinderFlowBenchmark,
    ThermalConvectionBenchmark
)


class TestBenchmarkConfig:
    """Test benchmark configuration."""
    
    def test_benchmark_config_creation(self):
        """Test creating benchmark configuration."""
        config = BenchmarkConfig(
            name="test_benchmark",
            problem_type="cavity",
            domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            reynolds_range=(100.0, 1000.0),
            viscosity_types=["constant", "linear"],
            n_tasks=10,
            n_support=50,
            n_query=200,
            geometry_params={"lid_velocity": 1.0},
            boundary_conditions={"type": "lid_driven"}
        )
        
        assert config.name == "test_benchmark"
        assert config.problem_type == "cavity"
        assert config.n_tasks == 10
        assert config.n_support == 50
        assert config.n_query == 200


class TestCavityFlowBenchmark:
    """Test cavity flow benchmark."""
    
    @pytest.fixture
    def cavity_config(self):
        """Create cavity flow configuration."""
        return BenchmarkConfig(
            name="cavity_test",
            problem_type="cavity",
            domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            reynolds_range=(100.0, 1000.0),
            viscosity_types=["constant"],
            n_tasks=5,
            n_support=25,
            n_query=100,
            geometry_params={"lid_velocity": 1.0},
            boundary_conditions={"type": "lid_driven"}
        )
    
    def test_cavity_benchmark_creation(self, cavity_config):
        """Test creating cavity flow benchmark."""
        benchmark = CavityFlowBenchmark(cavity_config)
        assert benchmark.config == cavity_config
        assert benchmark.analytical_solver is not None
    
    def test_get_evaluation_points(self, cavity_config):
        """Test getting evaluation points for cavity flow."""
        benchmark = CavityFlowBenchmark(cavity_config)
        eval_points = benchmark.get_evaluation_points()
        
        assert eval_points.shape[1] == 2  # 2D coordinates
        assert eval_points.shape[0] > 0
        
        # Check domain bounds
        x_coords = eval_points[:, 0]
        y_coords = eval_points[:, 1]
        assert torch.all(x_coords >= 0.0) and torch.all(x_coords <= 1.0)
        assert torch.all(y_coords >= 0.0) and torch.all(y_coords <= 1.0)
    
    def test_validate_solution(self, cavity_config):
        """Test solution validation for cavity flow."""
        benchmark = CavityFlowBenchmark(cavity_config)
        
        # Create dummy predictions and ground truth
        n_points = 100
        prediction = torch.randn(n_points, 3)  # [u, v, p]
        ground_truth = torch.randn(n_points, 3)
        
        metrics = benchmark.validate_solution(prediction, ground_truth)
        
        # Check that all expected metrics are present
        expected_metrics = ['u_mse', 'v_mse', 'p_mse', 'boundary_u_error', 
                          'boundary_v_error', 'total_velocity_error']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert metrics[metric] >= 0.0


class TestChannelFlowBenchmark:
    """Test channel flow benchmark."""
    
    @pytest.fixture
    def channel_config(self):
        """Create channel flow configuration."""
        return BenchmarkConfig(
            name="channel_test",
            problem_type="channel",
            domain_bounds={"x": (0.0, 2.0), "y": (0.0, 1.0)},
            reynolds_range=(50.0, 500.0),
            viscosity_types=["constant"],
            n_tasks=5,
            n_support=25,
            n_query=100,
            geometry_params={"inlet_velocity": 1.0},
            boundary_conditions={"type": "poiseuille"}
        )
    
    def test_channel_benchmark_creation(self, channel_config):
        """Test creating channel flow benchmark."""
        benchmark = ChannelFlowBenchmark(channel_config)
        assert benchmark.config == channel_config
    
    def test_get_evaluation_points(self, channel_config):
        """Test getting evaluation points for channel flow."""
        benchmark = ChannelFlowBenchmark(channel_config)
        eval_points = benchmark.get_evaluation_points()
        
        assert eval_points.shape[1] == 2  # 2D coordinates
        
        # Check domain bounds
        x_coords = eval_points[:, 0]
        y_coords = eval_points[:, 1]
        assert torch.all(x_coords >= 0.0) and torch.all(x_coords <= 2.0)
        assert torch.all(y_coords >= 0.0) and torch.all(y_coords <= 1.0)
    
    def test_validate_solution(self, channel_config):
        """Test solution validation for channel flow."""
        benchmark = ChannelFlowBenchmark(channel_config)
        
        # Create dummy predictions and ground truth
        n_points = 100
        prediction = torch.randn(n_points, 3)
        ground_truth = torch.randn(n_points, 3)
        
        metrics = benchmark.validate_solution(prediction, ground_truth)
        
        expected_metrics = ['u_mse', 'v_mse', 'p_mse', 'profile_error', 
                          'mass_conservation_error', 'total_error']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)


class TestCylinderFlowBenchmark:
    """Test cylinder flow benchmark."""
    
    @pytest.fixture
    def cylinder_config(self):
        """Create cylinder flow configuration."""
        return BenchmarkConfig(
            name="cylinder_test",
            problem_type="cylinder",
            domain_bounds={"x": (-2.0, 8.0), "y": (-3.0, 3.0)},
            reynolds_range=(20.0, 200.0),
            viscosity_types=["constant"],
            n_tasks=3,
            n_support=30,
            n_query=120,
            geometry_params={"cylinder_radius": 0.5, "cylinder_center": (0.0, 0.0)},
            boundary_conditions={"type": "cylinder_flow"}
        )
    
    def test_cylinder_benchmark_creation(self, cylinder_config):
        """Test creating cylinder flow benchmark."""
        benchmark = CylinderFlowBenchmark(cylinder_config)
        assert benchmark.config == cylinder_config
    
    def test_get_evaluation_points(self, cylinder_config):
        """Test getting evaluation points for cylinder flow."""
        benchmark = CylinderFlowBenchmark(cylinder_config)
        eval_points = benchmark.get_evaluation_points()
        
        assert eval_points.shape[1] == 2  # 2D coordinates
        
        # Check that points are outside cylinder (radius > 0.5)
        distances = torch.sqrt(eval_points[:, 0] ** 2 + eval_points[:, 1] ** 2)
        assert torch.all(distances > 0.5)
    
    def test_validate_solution(self, cylinder_config):
        """Test solution validation for cylinder flow."""
        benchmark = CylinderFlowBenchmark(cylinder_config)
        
        # Create dummy predictions and ground truth
        n_points = 100
        prediction = torch.randn(n_points, 3)
        ground_truth = torch.randn(n_points, 3)
        
        metrics = benchmark.validate_solution(prediction, ground_truth)
        
        expected_metrics = ['u_mse', 'v_mse', 'p_mse', 'cylinder_boundary_error', 
                          'wake_structure_error', 'total_error']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)


class TestThermalConvectionBenchmark:
    """Test thermal convection benchmark."""
    
    @pytest.fixture
    def thermal_config(self):
        """Create thermal convection configuration."""
        return BenchmarkConfig(
            name="thermal_test",
            problem_type="thermal",
            domain_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            reynolds_range=(100.0, 1000.0),
            viscosity_types=["temperature_dependent"],
            n_tasks=3,
            n_support=30,
            n_query=120,
            geometry_params={"rayleigh_number": 1e6},
            boundary_conditions={"type": "natural_convection"}
        )
    
    def test_thermal_benchmark_creation(self, thermal_config):
        """Test creating thermal convection benchmark."""
        benchmark = ThermalConvectionBenchmark(thermal_config)
        assert benchmark.config == thermal_config
    
    def test_validate_solution(self, thermal_config):
        """Test solution validation for thermal convection."""
        benchmark = ThermalConvectionBenchmark(thermal_config)
        
        # Create dummy predictions and ground truth with temperature
        n_points = 100
        prediction = torch.randn(n_points, 4)  # [u, v, p, T]
        ground_truth = torch.randn(n_points, 4)
        
        metrics = benchmark.validate_solution(prediction, ground_truth)
        
        expected_metrics = ['u_mse', 'v_mse', 'p_mse', 'T_mse', 
                          'thermal_bc_error', 'buoyancy_error', 'total_error']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)


class TestPINNBenchmarkSuite:
    """Test the main benchmark suite."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_benchmark_suite_creation(self, temp_dir):
        """Test creating benchmark suite."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        
        assert len(suite.benchmarks) == 4  # cavity, channel, cylinder, thermal
        assert "cavity_flow" in suite.benchmarks
        assert "channel_flow" in suite.benchmarks
        assert "cylinder_flow" in suite.benchmarks
        assert "thermal_convection" in suite.benchmarks
    
    def test_get_benchmark_names(self, temp_dir):
        """Test getting benchmark names."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        names = suite.get_benchmark_names()
        
        expected_names = ["cavity_flow", "channel_flow", "cylinder_flow", "thermal_convection"]
        assert set(names) == set(expected_names)
    
    def test_get_benchmark_config(self, temp_dir):
        """Test getting benchmark configuration."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        config = suite.get_benchmark_config("cavity_flow")
        
        assert isinstance(config, BenchmarkConfig)
        assert config.name == "cavity_flow"
        assert config.problem_type == "cavity"
    
    def test_get_benchmark_config_invalid(self, temp_dir):
        """Test getting configuration for invalid benchmark."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        
        with pytest.raises(ValueError, match="Unknown benchmark"):
            suite.get_benchmark_config("invalid_benchmark")
    
    @patch('ml_research_pipeline.evaluation.benchmark_suite.PINNBenchmarkSuite._evaluate_method_on_task')
    def test_run_benchmark(self, mock_evaluate, temp_dir):
        """Test running a single benchmark."""
        # Mock the evaluation method
        mock_evaluate.return_value = {
            'parameter_accuracy': 0.85,
            'adaptation_steps': 5,
            'physics_residual': 1e-4,
            'total_error': 0.1
        }
        
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        
        # Create mock method
        mock_method = Mock()
        mock_method.adapt_to_task = Mock(return_value=[])
        mock_method.forward = Mock(return_value=torch.randn(100, 3))
        
        # Mock task generator
        with patch.object(suite.benchmarks["cavity_flow"], 'setup_task_generator') as mock_setup:
            mock_task_generator = Mock()
            mock_task_generator.generate_task_batch = Mock(return_value=[Mock() for _ in range(3)])
            mock_setup.return_value = mock_task_generator
            
            result = suite.run_benchmark("cavity_flow", mock_method, "test_method")
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "cavity_flow"
        assert result.method_name == "test_method"
        assert "parameter_accuracy_mean" in result.metrics
        assert len(result.task_results) == 3
    
    def test_run_benchmark_invalid(self, temp_dir):
        """Test running benchmark with invalid name."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        mock_method = Mock()
        
        with pytest.raises(ValueError, match="Unknown benchmark"):
            suite.run_benchmark("invalid_benchmark", mock_method, "test_method")
    
    def test_compute_parameter_accuracy(self, temp_dir):
        """Test parameter accuracy computation."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        
        # Test with matching parameters
        mock_method = Mock()
        mock_method.inferred_parameters = {"viscosity": 0.1, "reynolds": 100.0}
        
        mock_task = Mock()
        mock_task.true_parameters = {"viscosity": 0.11, "reynolds": 95.0}
        
        accuracy = suite._compute_parameter_accuracy(mock_method, mock_task)
        assert 0.0 <= accuracy <= 1.0
    
    def test_compute_parameter_accuracy_no_params(self, temp_dir):
        """Test parameter accuracy when parameters not available."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        
        mock_method = Mock()
        mock_task = Mock()
        
        accuracy = suite._compute_parameter_accuracy(mock_method, mock_task)
        assert accuracy == 0.5  # Default value
    
    def test_compute_physics_residual(self, temp_dir):
        """Test physics residual computation."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        
        # Test with method that has physics_loss
        mock_method = Mock()
        mock_method.physics_loss = Mock(return_value=torch.tensor(1e-4))
        mock_method.forward = Mock(return_value=torch.randn(10, 3))
        
        mock_task = Mock()
        eval_points = torch.randn(10, 2)
        
        residual = suite._compute_physics_residual(mock_method, mock_task, eval_points)
        assert isinstance(residual, float)
        assert residual >= 0.0
    
    def test_compute_physics_residual_no_method(self, temp_dir):
        """Test physics residual when method doesn't have physics_loss."""
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        
        mock_method = Mock()
        # Remove physics_loss attribute to simulate method without it
        if hasattr(mock_method, 'physics_loss'):
            delattr(mock_method, 'physics_loss')
        
        mock_task = Mock()
        eval_points = torch.randn(10, 2)
        
        residual = suite._compute_physics_residual(mock_method, mock_task, eval_points)
        assert residual == 0.0
    
    @patch('ml_research_pipeline.evaluation.benchmark_suite.PINNBenchmarkSuite.run_benchmark')
    def test_run_full_benchmark_suite(self, mock_run_benchmark, temp_dir):
        """Test running full benchmark suite."""
        # Mock individual benchmark runs
        mock_result = BenchmarkResult(
            benchmark_name="test_benchmark",
            method_name="test_method",
            metrics={"accuracy": 0.85},
            task_results=[],
            runtime_info={"total_time": 10.0},
            metadata={}
        )
        mock_run_benchmark.return_value = mock_result
        
        suite = PINNBenchmarkSuite(save_dir=temp_dir)
        methods = {"method1": Mock(), "method2": Mock()}
        
        results = suite.run_full_benchmark_suite(methods)
        
        # Check structure
        assert len(results) == 4  # 4 benchmarks
        for benchmark_name in suite.get_benchmark_names():
            assert benchmark_name in results
            assert len(results[benchmark_name]) == 2  # 2 methods
            for method_name in methods.keys():
                assert method_name in results[benchmark_name]
        
        # Check that summary file was created
        summary_file = Path(temp_dir) / "benchmark_summary.json"
        assert summary_file.exists()


class TestBenchmarkResult:
    """Test benchmark result data structure."""
    
    def test_benchmark_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            method_name="test_method",
            metrics={"accuracy": 0.85, "speed": 10.0},
            task_results=[{"task1": "result1"}],
            runtime_info={"total_time": 100.0},
            metadata={"config": "test_config"}
        )
        
        assert result.benchmark_name == "test_benchmark"
        assert result.method_name == "test_method"
        assert result.metrics["accuracy"] == 0.85
        assert len(result.task_results) == 1
        assert result.runtime_info["total_time"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__])
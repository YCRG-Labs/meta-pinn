"""
Unit tests for the FluidTaskGenerator class.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

from ml_research_pipeline.core.task_generator import FluidTaskGenerator, FluidTask
from ml_research_pipeline.config.data_config import DataConfig, TaskConfig, ViscosityConfig, GeometryConfig


class TestFluidTask:
    """Test FluidTask class functionality."""
    
    def test_task_creation(self):
        """Test basic task creation and validation."""
        config = TaskConfig(
            task_id="test_001",
            task_type="linear_viscosity",
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 0.01, 'gradient_x': 0.1}
        )
        
        # Create sample data
        coords = torch.randn(50, 2)
        velocity = torch.randn(50, 2)
        pressure = torch.randn(50, 1)
        
        support_set = {'coords': coords, 'velocity': velocity, 'pressure': pressure}
        query_set = {'coords': coords, 'velocity': velocity, 'pressure': pressure}
        
        task = FluidTask(
            config=config,
            support_set=support_set,
            query_set=query_set
        )
        
        assert task.config.task_id == "test_001"
        assert task.config.task_type == "linear_viscosity"
        assert task.support_set['coords'].shape == (50, 2)
    
    def test_task_validation_missing_keys(self):
        """Test task validation with missing required keys."""
        config = TaskConfig(task_id="test_002")
        
        # Missing required keys
        support_set = {'coords': torch.randn(10, 2)}
        query_set = {'coords': torch.randn(10, 2)}
        
        with pytest.raises(ValueError, match="Missing required key"):
            FluidTask(config=config, support_set=support_set, query_set=query_set)
    
    def test_task_validation_inconsistent_shapes(self):
        """Test task validation with inconsistent tensor shapes."""
        config = TaskConfig(task_id="test_003")
        
        # Inconsistent shapes
        support_set = {
            'coords': torch.randn(10, 2),
            'velocity': torch.randn(5, 2),  # Wrong number of points
            'pressure': torch.randn(10, 1)
        }
        query_set = {
            'coords': torch.randn(10, 2),
            'velocity': torch.randn(10, 2),
            'pressure': torch.randn(10, 1)
        }
        
        with pytest.raises(ValueError, match="Inconsistent number of points"):
            FluidTask(config=config, support_set=support_set, query_set=query_set)
    
    def test_task_serialization(self):
        """Test task serialization and deserialization."""
        config = TaskConfig(
            task_id="test_004",
            task_type="exponential_viscosity",
            reynolds_number=200.0
        )
        
        coords = torch.randn(20, 2)
        velocity = torch.randn(20, 2)
        pressure = torch.randn(20, 1)
        
        support_set = {'coords': coords, 'velocity': velocity, 'pressure': pressure}
        query_set = {'coords': coords, 'velocity': velocity, 'pressure': pressure}
        
        original_task = FluidTask(
            config=config,
            support_set=support_set,
            query_set=query_set,
            metadata={'test': 'value'}
        )
        
        # Serialize and deserialize
        task_dict = original_task.to_dict()
        restored_task = FluidTask.from_dict(task_dict)
        
        assert restored_task.config.task_id == original_task.config.task_id
        assert restored_task.config.task_type == original_task.config.task_type
        assert torch.allclose(restored_task.support_set['coords'], original_task.support_set['coords'])
        assert restored_task.metadata == original_task.metadata


class TestFluidTaskGenerator:
    """Test FluidTaskGenerator class functionality."""
    
    @pytest.fixture
    def data_config(self):
        """Create test data configuration."""
        return DataConfig(
            n_train_tasks=100,
            n_val_tasks=20,
            n_test_tasks=20,
            task_types=['linear_viscosity', 'bilinear_viscosity', 'exponential_viscosity'],
            domain_bounds={'x': (0.0, 1.0), 'y': (0.0, 1.0)},
            spatial_resolution=[32, 32],
            n_boundary_points=50,
            n_interior_points=200
        )
    
    @pytest.fixture
    def viscosity_config(self):
        """Create test viscosity configuration."""
        return ViscosityConfig(
            base_viscosity_range=(0.01, 0.1),
            gradient_range=(-1.0, 1.0)
        )
    
    @pytest.fixture
    def geometry_config(self):
        """Create test geometry configuration."""
        return GeometryConfig(
            geometry_type="channel",
            length=1.0,
            width=1.0
        )
    
    @pytest.fixture
    def task_generator(self, data_config, viscosity_config, geometry_config):
        """Create test task generator."""
        return FluidTaskGenerator(
            data_config=data_config,
            viscosity_config=viscosity_config,
            geometry_config=geometry_config,
            seed=42
        )
    
    def test_generator_initialization(self, task_generator):
        """Test task generator initialization."""
        assert len(task_generator.data_config.task_types) == 3
        assert np.allclose(task_generator.task_weights, [1/3, 1/3, 1/3])
        assert task_generator.task_cache == {}
    
    def test_task_config_generation(self, task_generator):
        """Test task configuration generation."""
        config = task_generator._generate_task_config('linear_viscosity')
        
        assert config.task_type == 'linear_viscosity'
        assert 10.0 <= config.reynolds_number <= 5000.0
        assert 'base_viscosity' in config.viscosity_params
        assert 'gradient_x' in config.viscosity_params
        assert 'gradient_y' in config.viscosity_params
        assert config.geometry_type in ['channel', 'cavity', 'cylinder']
    
    def test_viscosity_params_generation(self, task_generator):
        """Test viscosity parameter generation for different task types."""
        # Linear viscosity
        params = task_generator._generate_viscosity_params('linear_viscosity')
        assert 'base_viscosity' in params
        assert 'gradient_x' in params
        assert 'gradient_y' in params
        
        # Bilinear viscosity
        params = task_generator._generate_viscosity_params('bilinear_viscosity')
        assert 'base_viscosity' in params
        assert 'gradient_x' in params
        assert 'gradient_y' in params
        assert 'cross_term' in params
        
        # Exponential viscosity
        params = task_generator._generate_viscosity_params('exponential_viscosity')
        assert 'base_viscosity' in params
        assert 'decay_rate_x' in params
        assert 'decay_rate_y' in params
        assert 'amplitude' in params
        
        # Temperature dependent
        params = task_generator._generate_viscosity_params('temperature_dependent')
        assert 'base_viscosity' in params
        assert 'reference_temperature' in params
        assert 'activation_energy' in params
        
        # Non-Newtonian
        params = task_generator._generate_viscosity_params('non_newtonian')
        assert 'base_viscosity' in params
        assert 'consistency_index' in params
        assert 'flow_behavior_index' in params
    
    def test_geometry_params_generation(self, task_generator):
        """Test geometry parameter generation."""
        # Channel geometry
        params = task_generator._generate_geometry_params('channel')
        assert 'length' in params
        assert 'width' in params
        assert 'inlet_profile' in params
        
        # Cavity geometry
        params = task_generator._generate_geometry_params('cavity')
        assert 'length' in params
        assert 'width' in params
        assert 'lid_velocity' in params
        
        # Cylinder geometry
        params = task_generator._generate_geometry_params('cylinder')
        assert 'domain_length' in params
        assert 'cylinder_radius' in params
        assert 'cylinder_position' in params
    
    def test_boundary_conditions_generation(self, task_generator):
        """Test boundary condition generation."""
        # Channel boundary conditions
        bc = task_generator._generate_boundary_conditions('channel')
        assert 'inlet' in bc
        assert 'outlet' in bc
        assert 'walls' in bc
        assert bc['inlet']['type'] == 'dirichlet'
        assert bc['outlet']['type'] == 'neumann'
        
        # Cavity boundary conditions
        bc = task_generator._generate_boundary_conditions('cavity')
        assert 'lid' in bc
        assert 'walls' in bc
        
        # Cylinder boundary conditions
        bc = task_generator._generate_boundary_conditions('cylinder')
        assert 'inlet' in bc
        assert 'outlet' in bc
        assert 'walls' in bc
        assert 'cylinder' in bc
    
    def test_coordinate_sampling(self, task_generator):
        """Test coordinate sampling methods."""
        config = TaskConfig(task_id="test_coords")
        
        # Test uniform sampling
        coords = task_generator._sample_coordinates(100, config)
        assert coords.shape == (100, 2)
        assert torch.all(coords >= 0.0)
        assert torch.all(coords <= 1.0)
        
        # Test that coordinates are within domain bounds
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        x_bounds = task_generator.data_config.domain_bounds['x']
        y_bounds = task_generator.data_config.domain_bounds['y']
        
        assert torch.all(x_coords >= x_bounds[0])
        assert torch.all(x_coords <= x_bounds[1])
        assert torch.all(y_coords >= y_bounds[0])
        assert torch.all(y_coords <= y_bounds[1])
    
    def test_adaptive_sampling(self, task_generator):
        """Test adaptive coordinate sampling."""
        bounds = (0.0, 1.0)
        samples = task_generator._adaptive_sample(bounds, 1000)
        
        assert len(samples) == 1000
        assert np.all(samples >= bounds[0])
        assert np.all(samples <= bounds[1])
        
        # Check that there's higher density near boundaries
        near_lower = np.sum(samples < 0.1)
        near_upper = np.sum(samples > 0.9)
        middle = np.sum((samples >= 0.4) & (samples <= 0.6))
        
        # Should have more points near boundaries than in middle
        assert (near_lower + near_upper) > middle * 0.5
    
    def test_task_batch_generation(self, task_generator):
        """Test batch task generation."""
        batch_size = 5
        n_support = 50
        n_query = 100
        
        tasks = task_generator.generate_task_batch(batch_size, n_support, n_query)
        
        assert len(tasks) == batch_size
        
        for task in tasks:
            assert isinstance(task, FluidTask)
            assert task.support_set['coords'].shape == (n_support, 2)
            assert task.query_set['coords'].shape == (n_query, 2)
            assert task.config.task_type in task_generator.data_config.task_types
    
    def test_task_batch_generation_specific_types(self, task_generator):
        """Test batch generation with specific task types."""
        batch_size = 3
        n_support = 30
        n_query = 50
        task_types = ['linear_viscosity', 'exponential_viscosity']
        
        tasks = task_generator.generate_task_batch(
            batch_size, n_support, n_query, task_types=task_types
        )
        
        assert len(tasks) == batch_size
        
        for task in tasks:
            assert task.config.task_type in task_types
    
    def test_task_config_validation(self, task_generator):
        """Test task configuration validation."""
        # Valid configuration
        valid_config = TaskConfig(
            task_id="valid_001",
            task_type="linear_viscosity",
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 0.01},
            geometry_type="channel",
            geometry_params={'length': 1.0, 'width': 0.5},
            boundary_conditions={'inlet': {'type': 'dirichlet', 'value': [1.0, 0.0]}}
        )
        
        assert task_generator.validate_task_config(valid_config) == True
        
        # Invalid Reynolds number
        invalid_config = TaskConfig(
            task_id="invalid_001",
            reynolds_number=10000.0,  # Too high
            viscosity_params={'base_viscosity': 0.01},
            boundary_conditions={'inlet': {'type': 'dirichlet'}}
        )
        
        assert task_generator.validate_task_config(invalid_config) == False
        
        # Invalid viscosity
        invalid_config = TaskConfig(
            task_id="invalid_002",
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 1e10},  # Too high
            boundary_conditions={'inlet': {'type': 'dirichlet'}}
        )
        
        assert task_generator.validate_task_config(invalid_config) == False
        
        # Missing boundary conditions
        invalid_config = TaskConfig(
            task_id="invalid_003",
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 0.01},
            boundary_conditions={}  # Empty
        )
        
        assert task_generator.validate_task_config(invalid_config) == False
    
    def test_task_statistics(self, task_generator):
        """Test task statistics computation."""
        # Generate some tasks
        tasks = task_generator.generate_task_batch(10, 50, 100)
        
        stats = task_generator.get_task_statistics(tasks)
        
        assert stats['n_tasks'] == 10
        assert 'task_type_distribution' in stats
        assert 'reynolds_statistics' in stats
        assert 'geometry_distribution' in stats
        
        # Check Reynolds statistics
        reynolds_stats = stats['reynolds_statistics']
        assert 'mean' in reynolds_stats
        assert 'std' in reynolds_stats
        assert 'min' in reynolds_stats
        assert 'max' in reynolds_stats
        
        # Check that all values are reasonable
        assert 10.0 <= reynolds_stats['min'] <= 5000.0
        assert 10.0 <= reynolds_stats['max'] <= 5000.0
        assert reynolds_stats['min'] <= reynolds_stats['mean'] <= reynolds_stats['max']
    
    def test_task_id_generation(self, task_generator):
        """Test unique task ID generation."""
        # Generate multiple task IDs
        task_ids = set()
        for _ in range(100):
            task_id = task_generator._generate_task_id('linear_viscosity')
            assert len(task_id) == 12  # MD5 hash truncated to 12 characters
            task_ids.add(task_id)
        
        # All IDs should be unique
        assert len(task_ids) == 100
    
    def test_reproducibility_with_seed(self):
        """Test that generator produces reproducible results with seed."""
        data_config = DataConfig(task_types=['linear_viscosity'])
        
        # Test that same generator with reset seed produces same results
        gen = FluidTaskGenerator(data_config, seed=42)
        tasks1 = gen.generate_task_batch(3, 10, 20)
        
        # Reset the generator with same seed
        gen = FluidTaskGenerator(data_config, seed=42)
        tasks2 = gen.generate_task_batch(3, 10, 20)
        
        # Should produce identical results
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.config.reynolds_number == t2.config.reynolds_number
            assert t1.config.viscosity_params == t2.config.viscosity_params
            assert torch.allclose(t1.support_set['coords'], t2.support_set['coords'])
        
        # Test that different seeds produce different results
        gen_diff = FluidTaskGenerator(data_config, seed=123)
        tasks_diff = gen_diff.generate_task_batch(3, 10, 20)
        
        # Should produce different results
        different_found = False
        for t1, t_diff in zip(tasks1, tasks_diff):
            if t1.config.reynolds_number != t_diff.config.reynolds_number:
                different_found = True
                break
        assert different_found, "Different seeds should produce different results"
    
    def test_empty_task_statistics(self, task_generator):
        """Test statistics computation with empty task list."""
        stats = task_generator.get_task_statistics([])
        assert stats == {}


if __name__ == "__main__":
    pytest.main([__file__])
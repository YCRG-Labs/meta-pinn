"""
Unit tests for analytical solution generators.
"""

import pytest
import numpy as np
import torch
from typing import Dict

from ml_research_pipeline.core.analytical_solutions import (
    AnalyticalSolutionGenerator, AnalyticalSolution
)
from ml_research_pipeline.config.data_config import TaskConfig


class TestAnalyticalSolution:
    """Test AnalyticalSolution class functionality."""
    
    def test_solution_creation(self):
        """Test basic analytical solution creation."""
        coords = torch.randn(50, 2)
        velocity = torch.randn(50, 2)
        pressure = torch.randn(50, 1)
        viscosity = torch.randn(50, 1)
        
        solution = AnalyticalSolution(
            velocity=velocity,
            pressure=pressure,
            coordinates=coords,
            viscosity_field=viscosity,
            metadata={'test': 'value'}
        )
        
        assert solution.velocity.shape == (50, 2)
        assert solution.pressure.shape == (50, 1)
        assert solution.coordinates.shape == (50, 2)
        assert solution.viscosity_field.shape == (50, 1)
        assert solution.metadata['test'] == 'value'
    
    def test_solution_without_viscosity(self):
        """Test solution creation without viscosity field."""
        coords = torch.randn(30, 2)
        velocity = torch.randn(30, 2)
        pressure = torch.randn(30, 1)
        
        solution = AnalyticalSolution(
            velocity=velocity,
            pressure=pressure,
            coordinates=coords
        )
        
        assert solution.viscosity_field is None
        assert solution.metadata == {}


class TestAnalyticalSolutionGenerator:
    """Test AnalyticalSolutionGenerator class functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create analytical solution generator."""
        return AnalyticalSolutionGenerator()
    
    @pytest.fixture
    def sample_coords(self):
        """Create sample coordinate grid."""
        x = torch.linspace(0, 1, 21)
        y = torch.linspace(0, 1, 11)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        return coords
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert len(generator.solution_registry) == 5
        assert 'poiseuille_flow' in generator.solution_registry
        assert 'couette_flow' in generator.solution_registry
        assert 'stokes_flow_cylinder' in generator.solution_registry
    
    def test_solution_type_inference(self, generator):
        """Test automatic solution type inference."""
        # Channel geometry should infer Poiseuille flow
        config = TaskConfig(
            geometry_type='channel',
            task_type='linear_viscosity'
        )
        solution_type = generator._infer_solution_type(config)
        assert solution_type == 'linear_viscosity_channel'
        
        # Cavity geometry should infer Couette flow
        config = TaskConfig(
            geometry_type='cavity',
            task_type='linear_viscosity'
        )
        solution_type = generator._infer_solution_type(config)
        assert solution_type == 'couette_flow'
        
        # Cylinder geometry should infer Stokes flow
        config = TaskConfig(
            geometry_type='cylinder',
            task_type='exponential_viscosity'
        )
        solution_type = generator._infer_solution_type(config)
        assert solution_type == 'stokes_flow_cylinder'
    
    def test_poiseuille_flow_solution(self, generator, sample_coords):
        """Test Poiseuille flow analytical solution."""
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='channel',
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 0.01},
            geometry_params={'width': 1.0, 'length': 2.0},
            density=1.0
        )
        
        solution = generator._poiseuille_flow(config, sample_coords)
        
        assert isinstance(solution, AnalyticalSolution)
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.pressure.shape == (len(sample_coords), 1)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        
        # Check that velocity is parabolic in y-direction
        y_coords = sample_coords[:, 1]
        u_velocity = solution.velocity[:, 0]
        
        # At y=0 and y=1 (walls), velocity should be zero
        wall_points = (torch.abs(y_coords) < 1e-6) | (torch.abs(y_coords - 1.0) < 1e-6)
        if torch.any(wall_points):
            assert torch.allclose(u_velocity[wall_points], torch.zeros_like(u_velocity[wall_points]), atol=1e-6)
        
        # v-velocity should be zero everywhere
        v_velocity = solution.velocity[:, 1]
        assert torch.allclose(v_velocity, torch.zeros_like(v_velocity))
        
        # Viscosity should be constant
        viscosity = solution.viscosity_field[:, 0]
        assert torch.allclose(viscosity, torch.full_like(viscosity, 0.01))
    
    def test_couette_flow_solution(self, generator, sample_coords):
        """Test Couette flow analytical solution."""
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='cavity',
            viscosity_params={'base_viscosity': 0.02},
            geometry_params={'width': 1.0, 'lid_velocity': 2.0}
        )
        
        solution = generator._couette_flow(config, sample_coords)
        
        assert isinstance(solution, AnalyticalSolution)
        assert solution.velocity.shape == (len(sample_coords), 2)
        
        # Check linear velocity profile
        y_coords = sample_coords[:, 1]
        u_velocity = solution.velocity[:, 0]
        
        # Velocity should be linear in y: u = U_lid * y / H
        expected_u = 2.0 * y_coords / 1.0
        assert torch.allclose(u_velocity, expected_u, atol=1e-6)
        
        # v-velocity should be zero
        v_velocity = solution.velocity[:, 1]
        assert torch.allclose(v_velocity, torch.zeros_like(v_velocity))
        
        # Pressure should be zero (no pressure gradient)
        pressure = solution.pressure[:, 0]
        assert torch.allclose(pressure, torch.zeros_like(pressure))
    
    def test_stokes_flow_cylinder_solution(self, generator):
        """Test Stokes flow around cylinder solution."""
        # Create coordinates around cylinder
        theta = torch.linspace(0, 2*np.pi, 50)
        r = torch.full_like(theta, 0.2)  # Outside cylinder
        x = 0.5 + r * torch.cos(theta)
        y = 0.5 + r * torch.sin(theta)
        coords = torch.stack([x, y], dim=1)
        
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='cylinder',
            viscosity_params={'base_viscosity': 0.01},
            geometry_params={
                'cylinder_radius': 0.1,
                'cylinder_position': [0.5, 0.5]
            }
        )
        
        solution = generator._stokes_flow_cylinder(config, coords)
        
        assert isinstance(solution, AnalyticalSolution)
        assert solution.velocity.shape == (len(coords), 2)
        
        # Check that velocity field is reasonable
        velocity_magnitude = torch.norm(solution.velocity, dim=1)
        assert torch.all(velocity_magnitude >= 0)
        assert torch.max(velocity_magnitude) > 0  # Should have some flow
    
    def test_linear_viscosity_channel_solution(self, generator, sample_coords):
        """Test channel flow with linear viscosity variation."""
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='channel',
            reynolds_number=50.0,
            viscosity_params={
                'base_viscosity': 0.01,
                'gradient_y': 0.05
            },
            geometry_params={'width': 1.0},
            density=1.0
        )
        
        solution = generator._linear_viscosity_channel(config, sample_coords)
        
        assert isinstance(solution, AnalyticalSolution)
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        
        # Check viscosity variation
        y_coords = sample_coords[:, 1]
        viscosity = solution.viscosity_field[:, 0]
        expected_viscosity = 0.01 + 0.05 * y_coords
        assert torch.allclose(viscosity, expected_viscosity, atol=1e-6)
        
        # Check that velocity varies with viscosity
        u_velocity = solution.velocity[:, 0]
        assert torch.max(u_velocity) > 0
        assert torch.min(u_velocity) >= 0  # Should be non-negative in channel
    
    def test_exponential_viscosity_channel_solution(self, generator, sample_coords):
        """Test channel flow with exponential viscosity variation."""
        config = TaskConfig(
            task_type='exponential_viscosity',
            geometry_type='channel',
            reynolds_number=75.0,
            viscosity_params={
                'base_viscosity': 0.01,
                'decay_rate_y': 1.0
            },
            geometry_params={'width': 1.0},
            density=1.0
        )
        
        solution = generator._exponential_viscosity_channel(config, sample_coords)
        
        assert isinstance(solution, AnalyticalSolution)
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        
        # Check exponential viscosity variation
        y_coords = sample_coords[:, 1]
        viscosity = solution.viscosity_field[:, 0]
        expected_viscosity = 0.01 * torch.exp(1.0 * y_coords)
        assert torch.allclose(viscosity, expected_viscosity, atol=1e-6)
        
        # Check that velocity field is reasonable
        u_velocity = solution.velocity[:, 0]
        assert torch.max(u_velocity) > 0
    
    def test_generate_solution_with_inference(self, generator, sample_coords):
        """Test solution generation with automatic type inference."""
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='channel',
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 0.01, 'gradient_y': 0.1},
            geometry_params={'width': 1.0},
            density=1.0
        )
        
        solution = generator.generate_solution(config, sample_coords)
        
        assert isinstance(solution, AnalyticalSolution)
        assert solution.metadata['solution_type'] == 'linear_viscosity_channel'
    
    def test_generate_solution_with_explicit_type(self, generator, sample_coords):
        """Test solution generation with explicit solution type."""
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='channel',
            viscosity_params={'base_viscosity': 0.01}
        )
        
        solution = generator.generate_solution(config, sample_coords, solution_type='poiseuille_flow')
        
        assert isinstance(solution, AnalyticalSolution)
        assert solution.metadata['solution_type'] == 'poiseuille_flow'
    
    def test_invalid_solution_type(self, generator, sample_coords):
        """Test error handling for invalid solution type."""
        config = TaskConfig(task_type='linear_viscosity')
        
        with pytest.raises(ValueError, match="Unknown solution type"):
            generator.generate_solution(config, sample_coords, solution_type='invalid_solution')
    
    def test_solution_validation(self, generator, sample_coords):
        """Test analytical solution validation."""
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='channel',
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 0.01},
            geometry_params={'width': 1.0},
            density=1.0
        )
        
        solution = generator._poiseuille_flow(config, sample_coords)
        validation_metrics = generator.validate_solution(solution, config)
        
        assert isinstance(validation_metrics, dict)
        assert 'max_velocity' in validation_metrics
        assert 'min_velocity' in validation_metrics
        assert 'max_pressure' in validation_metrics
        assert 'min_pressure' in validation_metrics
        assert 'max_viscosity' in validation_metrics
        assert 'min_viscosity' in validation_metrics
        assert 'negative_viscosity_points' in validation_metrics
        
        # Check that validation metrics are reasonable
        assert validation_metrics['max_velocity'] >= validation_metrics['min_velocity']
        assert validation_metrics['max_pressure'] >= validation_metrics['min_pressure']
        assert validation_metrics['max_viscosity'] >= validation_metrics['min_viscosity']
        assert validation_metrics['negative_viscosity_points'] == 0  # No negative viscosity
        assert validation_metrics['min_viscosity'] > 0  # Positive viscosity
    
    def test_reynolds_number_validation(self, generator):
        """Test Reynolds number consistency validation."""
        # Create simple coordinate grid
        coords = torch.tensor([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]])
        
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='channel',
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 0.01},
            geometry_params={'width': 1.0},
            density=1.0
        )
        
        solution = generator._poiseuille_flow(config, coords)
        validation_metrics = generator.validate_solution(solution, config)
        
        assert 'reynolds_error' in validation_metrics
        # Reynolds error should be reasonable (analytical solution should be consistent)
        assert validation_metrics['reynolds_error'] < 1.0  # Less than 100% error
    
    def test_get_available_solutions(self, generator):
        """Test getting available solution descriptions."""
        solutions = generator.get_available_solutions()
        
        assert isinstance(solutions, dict)
        assert len(solutions) == 5
        assert 'poiseuille_flow' in solutions
        assert 'couette_flow' in solutions
        assert 'stokes_flow_cylinder' in solutions
        assert 'linear_viscosity_channel' in solutions
        assert 'exponential_viscosity_channel' in solutions
        
        # Check that descriptions are strings
        for solution_type, description in solutions.items():
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_boundary_conditions_poiseuille(self, generator):
        """Test that Poiseuille flow satisfies no-slip boundary conditions."""
        # Create coordinates including boundary points
        x = torch.linspace(0, 1, 11)
        y = torch.tensor([0.0, 0.5, 1.0])  # Include walls at y=0 and y=1
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        config = TaskConfig(
            task_type='linear_viscosity',
            geometry_type='channel',
            reynolds_number=100.0,
            viscosity_params={'base_viscosity': 0.01},
            geometry_params={'width': 1.0},
            density=1.0
        )
        
        solution = generator._poiseuille_flow(config, coords)
        
        # Check no-slip at walls
        y_coords = coords[:, 1]
        u_velocity = solution.velocity[:, 0]
        
        # At y=0 (bottom wall)
        bottom_wall = torch.abs(y_coords) < 1e-6
        if torch.any(bottom_wall):
            assert torch.allclose(u_velocity[bottom_wall], torch.zeros_like(u_velocity[bottom_wall]), atol=1e-6)
        
        # At y=1 (top wall)
        top_wall = torch.abs(y_coords - 1.0) < 1e-6
        if torch.any(top_wall):
            assert torch.allclose(u_velocity[top_wall], torch.zeros_like(u_velocity[top_wall]), atol=1e-6)
    
    def test_viscosity_positivity(self, generator, sample_coords):
        """Test that all analytical solutions produce positive viscosity."""
        configs = [
            TaskConfig(
                task_type='linear_viscosity',
                geometry_type='channel',
                viscosity_params={'base_viscosity': 0.01, 'gradient_y': 0.05},
                geometry_params={'width': 1.0}
            ),
            TaskConfig(
                task_type='exponential_viscosity',
                geometry_type='channel',
                viscosity_params={'base_viscosity': 0.01, 'decay_rate_y': 0.5},
                geometry_params={'width': 1.0}
            )
        ]
        
        for config in configs:
            solution = generator.generate_solution(config, sample_coords)
            if solution.viscosity_field is not None:
                assert torch.all(solution.viscosity_field > 0), f"Negative viscosity in {config.task_type}"


if __name__ == "__main__":
    pytest.main([__file__])
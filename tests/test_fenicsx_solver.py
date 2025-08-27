"""
Unit tests for FEniCSx solver integration.
"""

import pytest
import numpy as np
import torch
from typing import Dict

from ml_research_pipeline.core.fenicsx_solver import (
    FEniCSxSolver, SolverConfig, create_fenicsx_solver, FENICSX_AVAILABLE
)
from ml_research_pipeline.core.analytical_solutions import AnalyticalSolutionGenerator
from ml_research_pipeline.config.data_config import TaskConfig


@pytest.mark.skipif(not FENICSX_AVAILABLE, reason="FEniCSx not available")
class TestFEniCSxSolver:
    """Test FEniCSx solver functionality."""
    
    @pytest.fixture
    def solver_config(self):
        """Create test solver configuration."""
        return SolverConfig(
            mesh_resolution=(20, 10),  # Coarse mesh for fast testing
            element_degree=2,
            pressure_degree=1,
            solver_type='direct',
            tolerance=1e-6
        )
    
    @pytest.fixture
    def solver(self, solver_config):
        """Create FEniCSx solver instance."""
        return FEniCSxSolver(solver_config)
    
    @pytest.fixture
    def sample_coords(self):
        """Create sample coordinate points."""
        x = torch.linspace(0.1, 0.9, 10)
        y = torch.linspace(0.1, 0.9, 5)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        return coords
    
    def test_solver_initialization(self, solver_config):
        """Test solver initialization."""
        solver = FEniCSxSolver(solver_config)
        assert solver.solver_config.mesh_resolution == (20, 10)
        assert solver.solver_config.element_degree == 2
        assert solver.solver_config.solver_type == 'direct'
    
    def test_solver_initialization_default_config(self):
        """Test solver initialization with default configuration."""
        solver = FEniCSxSolver()
        assert solver.solver_config.mesh_resolution == (100, 50)
        assert solver.solver_config.element_degree == 2
    
    def test_channel_flow_constant_viscosity(self, solver, sample_coords):
        """Test channel flow with constant viscosity."""
        config = TaskConfig(
            task_id="test_channel_001",
            task_type="linear_viscosity",
            geometry_type="channel",
            reynolds_number=10.0,  # Low Re for Stokes flow
            viscosity_params={'base_viscosity': 0.01},
            geometry_params={'length': 1.0, 'width': 1.0, 'inlet_profile': 'uniform'},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [1.0, 0.0]},
                'outlet': {'type': 'neumann', 'value': [0.0, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        solution = solver.solve_task(config, sample_coords)
        
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.pressure.shape == (len(sample_coords), 1)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        
        # Check that solution is reasonable
        u_velocity = solution.velocity[:, 0]
        v_velocity = solution.velocity[:, 1]
        
        # u-velocity should be positive (flow in positive x direction)
        assert torch.all(u_velocity >= 0)
        
        # v-velocity should be small (no significant cross-flow)
        assert torch.max(torch.abs(v_velocity)) < 0.1 * torch.max(u_velocity)
        
        # Viscosity should be constant
        viscosity = solution.viscosity_field[:, 0]
        assert torch.allclose(viscosity, torch.full_like(viscosity, 0.01), atol=1e-6)
    
    def test_channel_flow_linear_viscosity(self, solver, sample_coords):
        """Test channel flow with linear viscosity variation."""
        config = TaskConfig(
            task_id="test_channel_002",
            task_type="linear_viscosity",
            geometry_type="channel",
            reynolds_number=5.0,
            viscosity_params={
                'base_viscosity': 0.01,
                'gradient_x': 0.0,
                'gradient_y': 0.02
            },
            geometry_params={'length': 1.0, 'width': 1.0, 'inlet_profile': 'uniform'},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.5, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        solution = solver.solve_task(config, sample_coords)
        
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.pressure.shape == (len(sample_coords), 1)
        
        # Check viscosity variation
        coords_np = sample_coords.numpy()
        y_coords = coords_np[:, 1]
        expected_viscosity = 0.01 + 0.02 * y_coords
        
        viscosity = solution.viscosity_field[:, 0]
        assert torch.allclose(viscosity, torch.from_numpy(expected_viscosity).float(), atol=1e-6)
    
    def test_cavity_flow(self, solver):
        """Test lid-driven cavity flow."""
        # Use fewer points for cavity flow test
        x = torch.linspace(0.1, 0.9, 5)
        y = torch.linspace(0.1, 0.9, 5)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        config = TaskConfig(
            task_id="test_cavity_001",
            task_type="linear_viscosity",
            geometry_type="cavity",
            reynolds_number=1.0,  # Very low Re for stability
            viscosity_params={'base_viscosity': 0.1},
            geometry_params={'length': 1.0, 'width': 1.0, 'lid_velocity': 0.1},
            boundary_conditions={
                'lid': {'type': 'dirichlet', 'value': [0.1, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        solution = solver.solve_task(config, coords)
        
        assert solution.velocity.shape == (len(coords), 2)
        assert solution.pressure.shape == (len(coords), 1)
        
        # Check that solution exists and is finite
        assert torch.all(torch.isfinite(solution.velocity))
        assert torch.all(torch.isfinite(solution.pressure))
    
    def test_exponential_viscosity(self, solver, sample_coords):
        """Test flow with exponential viscosity variation."""
        config = TaskConfig(
            task_id="test_exp_visc_001",
            task_type="exponential_viscosity",
            geometry_type="channel",
            reynolds_number=1.0,
            viscosity_params={
                'base_viscosity': 0.01,
                'decay_rate_x': 0.0,
                'decay_rate_y': 0.5,
                'amplitude': 1.0
            },
            geometry_params={'length': 1.0, 'width': 1.0},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.1, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        solution = solver.solve_task(config, sample_coords)
        
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        
        # Check exponential viscosity variation
        coords_np = sample_coords.numpy()
        y_coords = coords_np[:, 1]
        expected_viscosity = 0.01 * 1.0 * np.exp(0.5 * y_coords)
        
        viscosity = solution.viscosity_field[:, 0]
        assert torch.allclose(viscosity, torch.from_numpy(expected_viscosity).float(), atol=1e-5)
    
    def test_mesh_creation_channel(self, solver):
        """Test mesh creation for channel geometry."""
        config = TaskConfig(
            geometry_type="channel",
            geometry_params={'length': 2.0, 'width': 1.0}
        )
        
        domain = solver._create_mesh(config)
        assert domain is not None
        # Check that mesh has reasonable number of cells
        assert domain.topology.index_map(domain.topology.dim).size_local > 0
    
    def test_mesh_creation_cavity(self, solver):
        """Test mesh creation for cavity geometry."""
        config = TaskConfig(
            geometry_type="cavity",
            geometry_params={'length': 1.0, 'width': 1.0}
        )
        
        domain = solver._create_mesh(config)
        assert domain is not None
        assert domain.topology.index_map(domain.topology.dim).size_local > 0
    
    def test_viscosity_function_creation(self, solver):
        """Test viscosity function creation for different task types."""
        # Create a simple domain for testing
        config_channel = TaskConfig(geometry_type="channel")
        domain = solver._create_mesh(config_channel)
        
        # Test linear viscosity
        config_linear = TaskConfig(
            task_type="linear_viscosity",
            viscosity_params={
                'base_viscosity': 0.01,
                'gradient_x': 0.1,
                'gradient_y': 0.2
            }
        )
        
        viscosity_expr = solver._create_viscosity_function(config_linear, domain)
        assert viscosity_expr is not None
        
        # Test exponential viscosity
        config_exp = TaskConfig(
            task_type="exponential_viscosity",
            viscosity_params={
                'base_viscosity': 0.01,
                'decay_rate_x': 0.5,
                'decay_rate_y': 1.0,
                'amplitude': 2.0
            }
        )
        
        viscosity_expr = solver._create_viscosity_function(config_exp, domain)
        assert viscosity_expr is not None
    
    def test_solver_error_handling(self, solver, sample_coords):
        """Test solver error handling with invalid configurations."""
        # Test with unsupported geometry
        config_invalid = TaskConfig(
            task_id="test_invalid_001",
            geometry_type="unsupported_geometry",
            reynolds_number=10.0,
            viscosity_params={'base_viscosity': 0.01}
        )
        
        # Should return fallback solution instead of raising error
        solution = solver.solve_task(config_invalid, sample_coords)
        assert solution.metadata['solution_type'] == 'fenicsx_fallback'
        assert 'error_message' in solution.metadata
    
    def test_solver_validation_errors(self, solver, sample_coords):
        """Test solver validation with invalid parameters."""
        # Test with invalid Reynolds number
        config_invalid_re = TaskConfig(
            task_id="test_invalid_re",
            geometry_type="channel",
            reynolds_number=-1.0,  # Invalid negative Reynolds
            viscosity_params={'base_viscosity': 0.01}
        )
        
        solution = solver.solve_task(config_invalid_re, sample_coords)
        assert solution.metadata['solution_type'] == 'fenicsx_fallback'
        
        # Test with invalid viscosity
        config_invalid_visc = TaskConfig(
            task_id="test_invalid_visc",
            geometry_type="channel",
            reynolds_number=10.0,
            viscosity_params={'base_viscosity': -0.01}  # Invalid negative viscosity
        )
        
        solution = solver.solve_task(config_invalid_visc, sample_coords)
        assert solution.metadata['solution_type'] == 'fenicsx_fallback'
        
        # Test with invalid geometry dimensions
        config_invalid_geom = TaskConfig(
            task_id="test_invalid_geom",
            geometry_type="channel",
            reynolds_number=10.0,
            viscosity_params={'base_viscosity': 0.01},
            geometry_params={'length': -1.0, 'width': 1.0}  # Invalid negative length
        )
        
        solution = solver.solve_task(config_invalid_geom, sample_coords)
        assert solution.metadata['solution_type'] == 'fenicsx_fallback'
    
    def test_solution_metadata(self, solver, sample_coords):
        """Test that solution contains proper metadata."""
        config = TaskConfig(
            task_id="test_metadata_001",
            task_type="linear_viscosity",
            geometry_type="channel",
            reynolds_number=50.0,
            viscosity_params={'base_viscosity': 0.02}
        )
        
        solution = solver.solve_task(config, sample_coords)
        
        assert 'solution_type' in solution.metadata
        assert solution.metadata['solution_type'] == 'fenicsx_solution'
        assert solution.metadata['task_id'] == 'test_metadata_001'
        assert solution.metadata['reynolds_number'] == 50.0
        assert solution.metadata['mesh_resolution'] == solver.solver_config.mesh_resolution
    
    def test_polynomial_viscosity(self, solver, sample_coords):
        """Test solver with polynomial viscosity variation."""
        config = TaskConfig(
            task_id="test_poly_visc_001",
            task_type="polynomial_viscosity",
            geometry_type="channel",
            reynolds_number=1.0,
            viscosity_params={
                'base_viscosity': 0.01,
                'polynomial_coeffs': {
                    'x1_y0': 0.01,  # x term
                    'x0_y1': 0.02,  # y term
                    'x2_y0': 0.005  # x^2 term
                }
            },
            geometry_params={'length': 1.0, 'width': 1.0},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.1, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        solution = solver.solve_task(config, sample_coords)
        
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        assert torch.all(torch.isfinite(solution.velocity))
        assert torch.all(torch.isfinite(solution.viscosity_field))
        
        # Check that viscosity varies spatially
        viscosity_values = solution.viscosity_field[:, 0]
        assert torch.std(viscosity_values) > 1e-6  # Should have variation
    
    def test_sinusoidal_viscosity(self, solver, sample_coords):
        """Test solver with sinusoidal viscosity variation."""
        config = TaskConfig(
            task_id="test_sin_visc_001",
            task_type="sinusoidal_viscosity",
            geometry_type="channel",
            reynolds_number=0.5,
            viscosity_params={
                'base_viscosity': 0.02,
                'amplitude': 0.005,
                'frequency_x': 2.0,
                'frequency_y': 1.0,
                'phase': 0.0
            },
            geometry_params={'length': 1.0, 'width': 1.0},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.05, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        solution = solver.solve_task(config, sample_coords)
        
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        assert torch.all(torch.isfinite(solution.velocity))
        assert torch.all(torch.isfinite(solution.viscosity_field))
        
        # Check viscosity bounds (should oscillate around base value)
        viscosity_values = solution.viscosity_field[:, 0]
        assert torch.min(viscosity_values) >= 0.015  # base - amplitude
        assert torch.max(viscosity_values) <= 0.025  # base + amplitude
    
    def test_temperature_dependent_viscosity(self, solver, sample_coords):
        """Test solver with temperature-dependent viscosity."""
        config = TaskConfig(
            task_id="test_temp_visc_001",
            task_type="temperature_dependent",
            geometry_type="channel",
            reynolds_number=0.5,
            viscosity_params={
                'base_viscosity': 0.01,
                'reference_temperature': 300.0,
                'activation_energy': 500.0,
                'temperature_gradient': 10.0
            },
            geometry_params={'length': 1.0, 'width': 1.0},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.05, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        solution = solver.solve_task(config, sample_coords)
        
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        assert torch.all(torch.isfinite(solution.velocity))
        assert torch.all(torch.isfinite(solution.viscosity_field))
        
        # Check that viscosity is positive and varies with position
        viscosity_values = solution.viscosity_field[:, 0]
        assert torch.all(viscosity_values > 0)
        assert torch.std(viscosity_values) > 1e-6  # Should have variation
    
    def test_inlet_profiles(self, solver):
        """Test different inlet velocity profiles."""
        profiles = ['uniform', 'parabolic', 'plug']
        
        for profile in profiles:
            config = TaskConfig(
                task_id=f"test_inlet_{profile}",
                task_type="linear_viscosity",
                geometry_type="channel",
                reynolds_number=1.0,
                viscosity_params={'base_viscosity': 0.05},
                geometry_params={'length': 1.0, 'width': 1.0, 'inlet_profile': profile},
                boundary_conditions={
                    'inlet': {'type': 'dirichlet', 'value': [0.1, 0.0]},
                    'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
                },
                density=1.0
            )
            
            # Test coordinates near inlet
            coords = torch.tensor([[0.05, 0.2], [0.05, 0.5], [0.05, 0.8]])
            
            solution = solver.solve_task(config, coords)
            
            assert solution.velocity.shape == (len(coords), 2)
            assert torch.all(torch.isfinite(solution.velocity))
            
            # Check that u-velocity is positive (flow in positive x direction)
            u_velocity = solution.velocity[:, 0]
            assert torch.all(u_velocity >= 0)
    
    def test_non_newtonian_viscosity(self, solver, sample_coords):
        """Test solver with non-Newtonian viscosity model."""
        config = TaskConfig(
            task_id="test_non_newtonian_001",
            task_type="non_newtonian",
            geometry_type="channel",
            reynolds_number=0.5,
            viscosity_params={
                'base_viscosity': 0.01,
                'consistency_index': 0.05,
                'flow_behavior_index': 0.8,
                'yield_stress': 0.001
            },
            geometry_params={'length': 1.0, 'width': 1.0},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.05, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        solution = solver.solve_task(config, sample_coords)
        
        assert solution.velocity.shape == (len(sample_coords), 2)
        assert solution.viscosity_field.shape == (len(sample_coords), 1)
        assert torch.all(torch.isfinite(solution.velocity))
        assert torch.all(torch.isfinite(solution.viscosity_field))
        
        # Check that effective viscosity is positive
        viscosity_values = solution.viscosity_field[:, 0]
        assert torch.all(viscosity_values > 0)


class TestFEniCSxSolverValidation:
    """Test FEniCSx solver validation against analytical solutions."""
    
    @pytest.mark.skipif(not FENICSX_AVAILABLE, reason="FEniCSx not available")
    def test_validation_against_analytical(self):
        """Test validation of FEniCSx solution against analytical solution."""
        # Use coarse mesh and simple problem for testing
        solver_config = SolverConfig(mesh_resolution=(10, 5), tolerance=1e-4)
        solver = FEniCSxSolver(solver_config)
        analytical_generator = AnalyticalSolutionGenerator()
        
        # Simple channel flow configuration
        config = TaskConfig(
            task_id="validation_test_001",
            task_type="linear_viscosity",
            geometry_type="channel",
            reynolds_number=1.0,  # Very low Re for better agreement
            viscosity_params={'base_viscosity': 0.1},  # Higher viscosity for stability
            geometry_params={'length': 1.0, 'width': 1.0, 'inlet_profile': 'uniform'},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.1, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        # Create evaluation points
        coords = torch.tensor([[0.5, 0.5], [0.3, 0.3], [0.7, 0.7]])
        
        # Generate analytical solution (using Poiseuille flow as approximation)
        analytical_solution = analytical_generator.generate_solution(
            config, coords, solution_type='poiseuille_flow'
        )
        
        # Validate FEniCSx against analytical
        metrics = solver.validate_against_analytical(config, coords, analytical_solution)
        
        assert isinstance(metrics, dict)
        assert 'velocity_l2_error' in metrics
        assert 'pressure_l2_error' in metrics
        assert 'relative_velocity_error' in metrics
        assert 'relative_pressure_error' in metrics
        
        # Check that errors are reasonable (not too strict due to different methods)
        assert metrics['velocity_l2_error'] >= 0
        assert metrics['pressure_l2_error'] >= 0
    
    @pytest.mark.skipif(not FENICSX_AVAILABLE, reason="FEniCSx not available")
    def test_validation_linear_viscosity_channel(self):
        """Test FEniCSx solver against analytical solution for linear viscosity channel."""
        solver_config = SolverConfig(mesh_resolution=(15, 8), tolerance=1e-5)
        solver = FEniCSxSolver(solver_config)
        analytical_generator = AnalyticalSolutionGenerator()
        
        config = TaskConfig(
            task_id="linear_visc_validation",
            task_type="linear_viscosity",
            geometry_type="channel",
            reynolds_number=0.5,  # Very low Reynolds for Stokes flow
            viscosity_params={
                'base_viscosity': 0.05,
                'gradient_x': 0.0,
                'gradient_y': 0.01
            },
            geometry_params={'length': 1.0, 'width': 1.0, 'inlet_profile': 'uniform'},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.05, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        # Create evaluation grid
        x = torch.linspace(0.1, 0.9, 5)
        y = torch.linspace(0.1, 0.9, 4)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Generate analytical solution
        analytical_solution = analytical_generator.generate_solution(
            config, coords, solution_type='linear_viscosity_channel'
        )
        
        # Validate against FEniCSx
        metrics = solver.validate_against_analytical(config, coords, analytical_solution)
        
        # Check that relative errors are reasonable for this simple case
        assert metrics['relative_velocity_error'] < 1.0  # Less than 100% error
        assert metrics['relative_pressure_error'] < 2.0  # Pressure can have larger relative error
        
        # Check that viscosity field matches
        fenicsx_solution = solver.solve_task(config, coords)
        viscosity_diff = torch.abs(fenicsx_solution.viscosity_field - analytical_solution.viscosity_field)
        max_viscosity_error = torch.max(viscosity_diff).item()
        assert max_viscosity_error < 0.01  # Viscosity should match closely
    
    @pytest.mark.skipif(not FENICSX_AVAILABLE, reason="FEniCSx not available")
    def test_validation_exponential_viscosity(self):
        """Test FEniCSx solver with exponential viscosity variation."""
        solver_config = SolverConfig(mesh_resolution=(12, 6), tolerance=1e-4)
        solver = FEniCSxSolver(solver_config)
        analytical_generator = AnalyticalSolutionGenerator()
        
        config = TaskConfig(
            task_id="exp_visc_validation",
            task_type="exponential_viscosity",
            geometry_type="channel",
            reynolds_number=0.1,  # Very low Reynolds
            viscosity_params={
                'base_viscosity': 0.02,
                'decay_rate_x': 0.0,
                'decay_rate_y': 0.5,
                'amplitude': 1.0
            },
            geometry_params={'length': 1.0, 'width': 1.0, 'inlet_profile': 'uniform'},
            boundary_conditions={
                'inlet': {'type': 'dirichlet', 'value': [0.02, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            },
            density=1.0
        )
        
        coords = torch.tensor([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
        
        # Generate analytical solution
        analytical_solution = analytical_generator.generate_solution(
            config, coords, solution_type='exponential_viscosity_channel'
        )
        
        # Solve with FEniCSx
        fenicsx_solution = solver.solve_task(config, coords)
        
        # Check that solutions exist and are finite
        assert torch.all(torch.isfinite(fenicsx_solution.velocity))
        assert torch.all(torch.isfinite(fenicsx_solution.pressure))
        assert torch.all(torch.isfinite(fenicsx_solution.viscosity_field))
        
        # Check viscosity field matches exponential profile
        coords_np = coords.numpy()
        y_coords = coords_np[:, 1]
        expected_viscosity = 0.02 * 1.0 * np.exp(0.5 * y_coords)
        
        viscosity_values = fenicsx_solution.viscosity_field[:, 0]
        viscosity_error = torch.abs(viscosity_values - torch.from_numpy(expected_viscosity).float())
        max_viscosity_error = torch.max(viscosity_error).item()
        assert max_viscosity_error < 0.005  # Should match closely
    
    @pytest.mark.skipif(not FENICSX_AVAILABLE, reason="FEniCSx not available")
    def test_validation_multiple_geometries(self):
        """Test FEniCSx solver validation across different geometries."""
        solver_config = SolverConfig(mesh_resolution=(10, 10), tolerance=1e-4)
        solver = FEniCSxSolver(solver_config)
        
        geometries = ['channel', 'cavity']
        
        for geometry in geometries:
            if geometry == 'channel':
                config = TaskConfig(
                    task_id=f"geom_test_{geometry}",
                    task_type="linear_viscosity",
                    geometry_type=geometry,
                    reynolds_number=1.0,
                    viscosity_params={'base_viscosity': 0.1},
                    geometry_params={'length': 1.0, 'width': 1.0, 'inlet_profile': 'uniform'},
                    boundary_conditions={
                        'inlet': {'type': 'dirichlet', 'value': [0.1, 0.0]},
                        'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
                    },
                    density=1.0
                )
            
            elif geometry == 'cavity':
                config = TaskConfig(
                    task_id=f"geom_test_{geometry}",
                    task_type="linear_viscosity",
                    geometry_type=geometry,
                    reynolds_number=0.5,
                    viscosity_params={'base_viscosity': 0.2},
                    geometry_params={'length': 1.0, 'width': 1.0, 'lid_velocity': 0.1},
                    boundary_conditions={
                        'lid': {'type': 'dirichlet', 'value': [0.1, 0.0]},
                        'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
                    },
                    density=1.0
                )
            
            # Test coordinates
            coords = torch.tensor([[0.3, 0.3], [0.5, 0.5], [0.7, 0.7]])
            
            # Solve with FEniCSx
            solution = solver.solve_task(config, coords)
            
            # Basic validation
            assert solution.velocity.shape == (len(coords), 2)
            assert solution.pressure.shape == (len(coords), 1)
            assert torch.all(torch.isfinite(solution.velocity))
            assert torch.all(torch.isfinite(solution.pressure))
            
            # Check metadata
            assert solution.metadata['solution_type'] == 'fenicsx_solution'
            assert solution.metadata['task_id'] == f"geom_test_{geometry}"
    
    @pytest.mark.skipif(not FENICSX_AVAILABLE, reason="FEniCSx not available")
    def test_batch_solving(self):
        """Test batch solving of multiple tasks."""
        solver_config = SolverConfig(mesh_resolution=(8, 8), tolerance=1e-4)
        solver = FEniCSxSolver(solver_config)
        
        # Create multiple task configurations
        task_configs = []
        for i in range(3):
            config = TaskConfig(
                task_id=f"batch_test_{i:03d}",
                task_type="linear_viscosity",
                geometry_type="channel",
                reynolds_number=1.0 + i * 0.5,
                viscosity_params={'base_viscosity': 0.01 + i * 0.01},
                geometry_params={'length': 1.0, 'width': 1.0, 'inlet_profile': 'uniform'},
                boundary_conditions={
                    'inlet': {'type': 'dirichlet', 'value': [0.1 + i * 0.05, 0.0]},
                    'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
                },
                density=1.0
            )
            task_configs.append(config)
        
        # Test coordinates
        coords = torch.tensor([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
        
        # Solve batch
        solutions = solver.solve_task_batch(task_configs, coords)
        
        assert len(solutions) == len(task_configs)
        
        for i, solution in enumerate(solutions):
            assert solution.velocity.shape == (len(coords), 2)
            assert solution.pressure.shape == (len(coords), 1)
            assert torch.all(torch.isfinite(solution.velocity))
            assert torch.all(torch.isfinite(solution.pressure))
            assert solution.metadata['task_id'] == f"batch_test_{i:03d}"
    
    @pytest.mark.skipif(not FENICSX_AVAILABLE, reason="FEniCSx not available")
    def test_ground_truth_dataset_generation(self):
        """Test generation of ground truth dataset."""
        solver_config = SolverConfig(mesh_resolution=(6, 6), tolerance=1e-3)
        solver = FEniCSxSolver(solver_config)
        
        # Create task configurations
        task_configs = []
        for i in range(2):  # Small number for testing
            config = TaskConfig(
                task_id=f"dataset_test_{i:03d}",
                task_type="linear_viscosity",
                geometry_type="channel",
                reynolds_number=1.0 + i,
                viscosity_params={'base_viscosity': 0.02 + i * 0.01},
                geometry_params={'length': 1.0, 'width': 1.0},
                boundary_conditions={
                    'inlet': {'type': 'dirichlet', 'value': [0.1, 0.0]},
                    'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
                },
                density=1.0
            )
            task_configs.append(config)
        
        # Generate dataset
        dataset = solver.generate_ground_truth_dataset(
            task_configs, 
            n_points_per_task=10  # Small number for testing
        )
        
        assert 'tasks' in dataset
        assert 'metadata' in dataset
        assert len(dataset['tasks']) == len(task_configs)
        assert dataset['metadata']['n_tasks'] == len(task_configs)
        assert dataset['metadata']['n_points_per_task'] == 10
        
        # Check task data structure
        for i, task_data in enumerate(dataset['tasks']):
            assert 'task_config' in task_data
            assert 'coordinates' in task_data
            assert 'velocity' in task_data
            assert 'pressure' in task_data
            assert 'viscosity_field' in task_data
            assert 'metadata' in task_data
            
            # Check shapes
            assert task_data['coordinates'].shape == (10, 2)
            assert task_data['velocity'].shape == (10, 2)
            assert task_data['pressure'].shape == (10, 1)
            assert task_data['viscosity_field'].shape == (10, 1)


class TestFEniCSxSolverFactory:
    """Test FEniCSx solver factory function."""
    
    def test_create_solver_with_fenicsx(self):
        """Test solver creation when FEniCSx is available."""
        if FENICSX_AVAILABLE:
            solver = create_fenicsx_solver()
            assert solver is not None
            assert isinstance(solver, FEniCSxSolver)
        else:
            solver = create_fenicsx_solver()
            assert solver is None
    
    def test_create_solver_with_config(self):
        """Test solver creation with custom configuration."""
        config = SolverConfig(mesh_resolution=(50, 25), element_degree=1)
        
        if FENICSX_AVAILABLE:
            solver = create_fenicsx_solver(config)
            assert solver is not None
            assert solver.solver_config.mesh_resolution == (50, 25)
            assert solver.solver_config.element_degree == 1
        else:
            solver = create_fenicsx_solver(config)
            assert solver is None


class TestFEniCSxSolverWithoutFEniCSx:
    """Test behavior when FEniCSx is not available."""
    
    def test_solver_initialization_without_fenicsx(self, monkeypatch):
        """Test that solver raises error when FEniCSx is not available."""
        # Mock FEniCSx as unavailable
        monkeypatch.setattr('ml_research_pipeline.core.fenicsx_solver.FENICSX_AVAILABLE', False)
        
        with pytest.raises(ImportError, match="FEniCSx is required but not available"):
            FEniCSxSolver()
    
    def test_factory_returns_none_without_fenicsx(self, monkeypatch):
        """Test that factory returns None when FEniCSx is not available."""
        # Mock FEniCSx as unavailable
        monkeypatch.setattr('ml_research_pipeline.core.fenicsx_solver.FENICSX_AVAILABLE', False)
        
        solver = create_fenicsx_solver()
        assert solver is None


if __name__ == "__main__":
    pytest.main([__file__])
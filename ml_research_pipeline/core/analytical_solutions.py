"""
Analytical solution generators for fluid dynamics problems.

This module provides analytical solutions for simple viscosity profiles
and flow configurations that can be used for validation of numerical methods.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging

from ..config.data_config import TaskConfig


logger = logging.getLogger(__name__)


@dataclass
class AnalyticalSolution:
    """Container for analytical solution data."""
    
    velocity: torch.Tensor  # Shape: (n_points, 2) for [u, v]
    pressure: torch.Tensor  # Shape: (n_points, 1)
    coordinates: torch.Tensor  # Shape: (n_points, 2) for [x, y]
    viscosity_field: Optional[torch.Tensor] = None  # Shape: (n_points, 1)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnalyticalSolutionGenerator:
    """
    Generates analytical solutions for fluid dynamics problems.
    
    Supports various flow configurations with different viscosity profiles
    for validation and testing purposes.
    """
    
    def __init__(self):
        """Initialize the analytical solution generator."""
        self.solution_registry = {
            'poiseuille_flow': self._poiseuille_flow,
            'couette_flow': self._couette_flow,
            'stokes_flow_cylinder': self._stokes_flow_cylinder,
            'linear_viscosity_channel': self._linear_viscosity_channel,
            'exponential_viscosity_channel': self._exponential_viscosity_channel
        }
        
        logger.info(f"Initialized AnalyticalSolutionGenerator with {len(self.solution_registry)} solutions")
    
    def generate_solution(self, 
                         task_config: TaskConfig, 
                         coordinates: torch.Tensor,
                         solution_type: Optional[str] = None) -> AnalyticalSolution:
        """
        Generate analytical solution for given task configuration.
        
        Args:
            task_config: Task configuration specifying the problem
            coordinates: Coordinate points where to evaluate solution
            solution_type: Specific solution type (if None, infer from task_config)
        
        Returns:
            AnalyticalSolution object containing velocity, pressure, and metadata
        """
        if solution_type is None:
            solution_type = self._infer_solution_type(task_config)
        
        if solution_type not in self.solution_registry:
            raise ValueError(f"Unknown solution type: {solution_type}")
        
        logger.debug(f"Generating {solution_type} solution for {len(coordinates)} points")
        
        solution_func = self.solution_registry[solution_type]
        return solution_func(task_config, coordinates)
    
    def _infer_solution_type(self, task_config: TaskConfig) -> str:
        """Infer the appropriate analytical solution type from task configuration."""
        geometry = task_config.geometry_type
        viscosity_type = task_config.task_type
        
        # Simple heuristics for solution type selection
        if geometry == 'channel':
            if viscosity_type == 'linear_viscosity':
                return 'linear_viscosity_channel'
            elif viscosity_type == 'exponential_viscosity':
                return 'exponential_viscosity_channel'
            else:
                return 'poiseuille_flow'  # Default for channel
        
        elif geometry == 'cavity':
            return 'couette_flow'
        
        elif geometry == 'cylinder':
            return 'stokes_flow_cylinder'
        
        else:
            return 'poiseuille_flow'  # Default fallback
    
    def _poiseuille_flow(self, task_config: TaskConfig, coords: torch.Tensor) -> AnalyticalSolution:
        """
        Generate Poiseuille flow solution for channel with constant viscosity.
        
        Assumes 2D channel flow between parallel plates at y=0 and y=H
        with parabolic velocity profile.
        """
        x, y = coords[:, 0], coords[:, 1]
        
        # Get parameters
        viscosity = task_config.viscosity_params.get('base_viscosity', 0.01)
        reynolds = task_config.reynolds_number
        
        # Channel dimensions (assume unit channel if not specified)
        height = task_config.geometry_params.get('width', 1.0)
        length = task_config.geometry_params.get('length', 1.0)
        
        # Pressure gradient (derived from Reynolds number and geometry)
        # Re = rho * U * H / mu, where U is characteristic velocity
        rho = task_config.density
        char_velocity = reynolds * viscosity / (rho * height)
        
        # Pressure gradient for Poiseuille flow: dp/dx = -12*mu*U_avg/H^2
        u_avg = char_velocity
        dp_dx = -12 * viscosity * u_avg / (height**2)
        
        # Velocity profile: u(y) = -(dp/dx) * y * (H - y) / (2 * mu)
        u_velocity = -dp_dx * y * (height - y) / (2 * viscosity)
        v_velocity = torch.zeros_like(u_velocity)
        
        # Pressure field: linear variation in x
        pressure = dp_dx * x
        
        # Viscosity field (constant)
        viscosity_field = torch.full_like(pressure, viscosity)
        
        velocity = torch.stack([u_velocity, v_velocity], dim=1)
        pressure = pressure.unsqueeze(1)
        viscosity_field = viscosity_field.unsqueeze(1)
        
        return AnalyticalSolution(
            velocity=velocity,
            pressure=pressure,
            coordinates=coords,
            viscosity_field=viscosity_field,
            metadata={
                'solution_type': 'poiseuille_flow',
                'reynolds_number': reynolds,
                'viscosity': viscosity,
                'pressure_gradient': dp_dx,
                'channel_height': height
            }
        )
    
    def _couette_flow(self, task_config: TaskConfig, coords: torch.Tensor) -> AnalyticalSolution:
        """
        Generate Couette flow solution for flow between moving plates.
        
        Assumes 2D flow with top plate moving at constant velocity.
        """
        x, y = coords[:, 0], coords[:, 1]
        
        # Get parameters
        viscosity = task_config.viscosity_params.get('base_viscosity', 0.01)
        
        # Plate velocity and separation
        height = task_config.geometry_params.get('width', 1.0)
        lid_velocity = task_config.geometry_params.get('lid_velocity', 1.0)
        
        # Linear velocity profile: u(y) = U_lid * y / H
        u_velocity = lid_velocity * y / height
        v_velocity = torch.zeros_like(u_velocity)
        
        # No pressure gradient in simple Couette flow
        pressure = torch.zeros_like(x)
        
        # Viscosity field (constant)
        viscosity_field = torch.full_like(pressure, viscosity)
        
        velocity = torch.stack([u_velocity, v_velocity], dim=1)
        pressure = pressure.unsqueeze(1)
        viscosity_field = viscosity_field.unsqueeze(1)
        
        return AnalyticalSolution(
            velocity=velocity,
            pressure=pressure,
            coordinates=coords,
            viscosity_field=viscosity_field,
            metadata={
                'solution_type': 'couette_flow',
                'lid_velocity': lid_velocity,
                'viscosity': viscosity,
                'channel_height': height
            }
        )
    
    def _stokes_flow_cylinder(self, task_config: TaskConfig, coords: torch.Tensor) -> AnalyticalSolution:
        """
        Generate Stokes flow solution around a cylinder.
        
        Uses the analytical solution for creeping flow around a circular cylinder.
        """
        x, y = coords[:, 0], coords[:, 1]
        
        # Get parameters
        viscosity = task_config.viscosity_params.get('base_viscosity', 0.01)
        cylinder_radius = task_config.geometry_params.get('cylinder_radius', 0.1)
        cylinder_pos = task_config.geometry_params.get('cylinder_position', [0.5, 0.5])
        
        # Inlet velocity (uniform flow)
        u_inf = 1.0  # Characteristic velocity
        
        # Translate coordinates to cylinder center
        x_rel = x - cylinder_pos[0]
        y_rel = y - cylinder_pos[1]
        
        # Convert to polar coordinates
        r = torch.sqrt(x_rel**2 + y_rel**2)
        theta = torch.atan2(y_rel, x_rel)
        
        # Stokes flow solution in polar coordinates
        # u_r = U_inf * cos(theta) * (1 - R^2/r^2)
        # u_theta = -U_inf * sin(theta) * (1 + R^2/r^2)
        
        # Avoid division by zero at cylinder center
        r_safe = torch.clamp(r, min=cylinder_radius * 1.01)
        
        u_r = u_inf * torch.cos(theta) * (1 - (cylinder_radius**2) / (r_safe**2))
        u_theta = -u_inf * torch.sin(theta) * (1 + (cylinder_radius**2) / (r_safe**2))
        
        # Convert back to Cartesian coordinates
        u_velocity = u_r * torch.cos(theta) - u_theta * torch.sin(theta)
        v_velocity = u_r * torch.sin(theta) + u_theta * torch.cos(theta)
        
        # Apply no-slip boundary condition on cylinder surface
        on_cylinder = r <= cylinder_radius * 1.05
        u_velocity[on_cylinder] = 0.0
        v_velocity[on_cylinder] = 0.0
        
        # Pressure field (simplified)
        pressure = torch.zeros_like(x)
        
        # Viscosity field (constant)
        viscosity_field = torch.full_like(pressure, viscosity)
        
        velocity = torch.stack([u_velocity, v_velocity], dim=1)
        pressure = pressure.unsqueeze(1)
        viscosity_field = viscosity_field.unsqueeze(1)
        
        return AnalyticalSolution(
            velocity=velocity,
            pressure=pressure,
            coordinates=coords,
            viscosity_field=viscosity_field,
            metadata={
                'solution_type': 'stokes_flow_cylinder',
                'cylinder_radius': cylinder_radius,
                'cylinder_position': cylinder_pos,
                'inlet_velocity': u_inf,
                'viscosity': viscosity
            }
        )
    
    def _linear_viscosity_channel(self, task_config: TaskConfig, coords: torch.Tensor) -> AnalyticalSolution:
        """
        Generate solution for channel flow with linear viscosity variation.
        
        Assumes viscosity varies linearly: mu(y) = mu_0 + grad_y * y
        """
        x, y = coords[:, 0], coords[:, 1]
        
        # Get parameters
        mu_0 = task_config.viscosity_params.get('base_viscosity', 0.01)
        grad_y = task_config.viscosity_params.get('gradient_y', 0.1)
        
        # Channel dimensions
        height = task_config.geometry_params.get('width', 1.0)
        
        # Viscosity field: mu(y) = mu_0 + grad_y * y
        viscosity_field = mu_0 + grad_y * y
        
        # For linear viscosity variation, the velocity profile is more complex
        # Approximate solution assuming small viscosity variation
        reynolds = task_config.reynolds_number
        rho = task_config.density
        
        # Characteristic velocity
        char_velocity = reynolds * mu_0 / (rho * height)
        
        # Modified parabolic profile accounting for viscosity variation
        # This is an approximation - exact solution requires solving ODE
        y_norm = y / height
        base_profile = y_norm * (1 - y_norm)  # Normalized parabolic profile
        
        # Correction factor for viscosity variation
        mu_avg = mu_0 + grad_y * height / 2
        correction = mu_avg / viscosity_field
        
        u_velocity = 6 * char_velocity * base_profile * correction
        v_velocity = torch.zeros_like(u_velocity)
        
        # Pressure gradient (approximate)
        dp_dx = -12 * mu_avg * char_velocity / (height**2)
        pressure = dp_dx * x
        
        velocity = torch.stack([u_velocity, v_velocity], dim=1)
        pressure = pressure.unsqueeze(1)
        viscosity_field = viscosity_field.unsqueeze(1)
        
        return AnalyticalSolution(
            velocity=velocity,
            pressure=pressure,
            coordinates=coords,
            viscosity_field=viscosity_field,
            metadata={
                'solution_type': 'linear_viscosity_channel',
                'base_viscosity': mu_0,
                'viscosity_gradient': grad_y,
                'reynolds_number': reynolds,
                'pressure_gradient': dp_dx
            }
        )
    
    def _exponential_viscosity_channel(self, task_config: TaskConfig, coords: torch.Tensor) -> AnalyticalSolution:
        """
        Generate solution for channel flow with exponential viscosity variation.
        
        Assumes viscosity varies exponentially: mu(y) = mu_0 * exp(decay_rate * y)
        """
        x, y = coords[:, 0], coords[:, 1]
        
        # Get parameters
        mu_0 = task_config.viscosity_params.get('base_viscosity', 0.01)
        decay_rate = task_config.viscosity_params.get('decay_rate_y', 1.0)
        
        # Channel dimensions
        height = task_config.geometry_params.get('width', 1.0)
        
        # Viscosity field: mu(y) = mu_0 * exp(decay_rate * y)
        viscosity_field = mu_0 * torch.exp(decay_rate * y)
        
        # For exponential viscosity, approximate the velocity profile
        reynolds = task_config.reynolds_number
        rho = task_config.density
        
        # Characteristic velocity
        char_velocity = reynolds * mu_0 / (rho * height)
        
        # Approximate velocity profile
        # Exact solution requires solving: d/dy(mu(y) * du/dy) = dp/dx
        y_norm = y / height
        base_profile = y_norm * (1 - y_norm)
        
        # Correction for exponential viscosity variation
        mu_avg = mu_0 * (torch.exp(torch.tensor(decay_rate * height)) - 1) / (decay_rate * height)
        correction = mu_avg / viscosity_field
        
        u_velocity = 6 * char_velocity * base_profile * correction
        v_velocity = torch.zeros_like(u_velocity)
        
        # Pressure gradient
        dp_dx = -12 * mu_avg * char_velocity / (height**2)
        pressure = dp_dx * x
        
        velocity = torch.stack([u_velocity, v_velocity], dim=1)
        pressure = pressure.unsqueeze(1)
        viscosity_field = viscosity_field.unsqueeze(1)
        
        return AnalyticalSolution(
            velocity=velocity,
            pressure=pressure,
            coordinates=coords,
            viscosity_field=viscosity_field,
            metadata={
                'solution_type': 'exponential_viscosity_channel',
                'base_viscosity': mu_0,
                'decay_rate': decay_rate,
                'reynolds_number': reynolds,
                'pressure_gradient': dp_dx
            }
        )
    
    def validate_solution(self, solution: AnalyticalSolution, task_config: TaskConfig) -> Dict[str, float]:
        """
        Validate analytical solution against physics constraints.
        
        Args:
            solution: Analytical solution to validate
            task_config: Task configuration
        
        Returns:
            Dictionary of validation metrics
        """
        coords = solution.coordinates
        velocity = solution.velocity
        pressure = solution.pressure
        viscosity = solution.viscosity_field
        
        # Compute derivatives for physics validation
        u, v = velocity[:, 0], velocity[:, 1]
        
        # Approximate derivatives using finite differences
        # Note: This is a simplified validation - full validation would require
        # proper numerical differentiation on the coordinate grid
        
        validation_metrics = {}
        
        # Check continuity equation: du/dx + dv/dy = 0
        if len(coords) > 1:
            # Simple finite difference approximation
            du_dx = torch.gradient(u, spacing=0.01)[0] if len(u) > 1 else torch.zeros_like(u)
            dv_dy = torch.gradient(v, spacing=0.01)[0] if len(v) > 1 else torch.zeros_like(v)
            continuity_residual = torch.abs(du_dx + dv_dy)
            validation_metrics['continuity_error'] = torch.mean(continuity_residual).item()
        
        # Check velocity magnitude bounds
        velocity_magnitude = torch.norm(velocity, dim=1)
        validation_metrics['max_velocity'] = torch.max(velocity_magnitude).item()
        validation_metrics['min_velocity'] = torch.min(velocity_magnitude).item()
        
        # Check pressure bounds
        validation_metrics['max_pressure'] = torch.max(pressure).item()
        validation_metrics['min_pressure'] = torch.min(pressure).item()
        
        # Check viscosity bounds
        if viscosity is not None:
            validation_metrics['max_viscosity'] = torch.max(viscosity).item()
            validation_metrics['min_viscosity'] = torch.min(viscosity).item()
            
            # Check viscosity positivity
            negative_viscosity = torch.sum(viscosity <= 0).item()
            validation_metrics['negative_viscosity_points'] = negative_viscosity
        
        # Reynolds number consistency check
        if viscosity is not None and len(coords) > 0:
            char_length = task_config.geometry_params.get('width', 1.0)
            char_velocity = torch.mean(velocity_magnitude).item()
            mean_viscosity = torch.mean(viscosity).item()
            rho = task_config.density
            
            computed_reynolds = rho * char_velocity * char_length / mean_viscosity
            expected_reynolds = task_config.reynolds_number
            
            validation_metrics['reynolds_error'] = abs(computed_reynolds - expected_reynolds) / expected_reynolds
        
        return validation_metrics
    
    def get_available_solutions(self) -> Dict[str, str]:
        """Get list of available analytical solutions with descriptions."""
        descriptions = {
            'poiseuille_flow': 'Parabolic flow between parallel plates with constant viscosity',
            'couette_flow': 'Linear shear flow between moving plates',
            'stokes_flow_cylinder': 'Creeping flow around a circular cylinder',
            'linear_viscosity_channel': 'Channel flow with linearly varying viscosity',
            'exponential_viscosity_channel': 'Channel flow with exponentially varying viscosity'
        }
        return descriptions
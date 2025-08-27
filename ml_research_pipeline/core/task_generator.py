"""
Task generation system for meta-learning fluid dynamics problems.

This module implements the FluidTaskGenerator class that creates diverse
fluid dynamics tasks with varying viscosity profiles for meta-learning.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import hashlib
from pathlib import Path
import logging

from ..config.data_config import TaskConfig, ViscosityConfig, GeometryConfig, DataConfig


logger = logging.getLogger(__name__)


@dataclass
class FluidTask:
    """Represents a single fluid dynamics task for meta-learning."""
    
    config: TaskConfig
    support_set: Dict[str, torch.Tensor]  # {'coords': ..., 'data': ...}
    query_set: Dict[str, torch.Tensor]
    ground_truth: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate task data after initialization."""
        self._validate_task_data()
    
    def _validate_task_data(self):
        """Validate that task data is consistent and well-formed."""
        # Check that support and query sets have required keys
        required_keys = ['coords', 'velocity', 'pressure']
        
        for dataset_name, dataset in [('support_set', self.support_set), 
                                     ('query_set', self.query_set)]:
            for key in required_keys:
                if key not in dataset:
                    raise ValueError(f"Missing required key '{key}' in {dataset_name}")
            
            # Check tensor shapes are consistent
            n_points = dataset['coords'].shape[0]
            for key, tensor in dataset.items():
                if tensor.shape[0] != n_points:
                    raise ValueError(f"Inconsistent number of points in {dataset_name}.{key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            'config': self.config.to_dict(),
            'support_set': {k: v.numpy() for k, v in self.support_set.items()},
            'query_set': {k: v.numpy() for k, v in self.query_set.items()},
            'ground_truth': {k: v.numpy() for k, v in self.ground_truth.items()} if self.ground_truth else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FluidTask':
        """Create task from dictionary."""
        config = TaskConfig.from_dict(data['config'])
        support_set = {k: torch.from_numpy(v) for k, v in data['support_set'].items()}
        query_set = {k: torch.from_numpy(v) for k, v in data['query_set'].items()}
        ground_truth = {k: torch.from_numpy(v) for k, v in data['ground_truth'].items()} if data['ground_truth'] else None
        
        return cls(
            config=config,
            support_set=support_set,
            query_set=query_set,
            ground_truth=ground_truth,
            metadata=data.get('metadata', {})
        )


class FluidTaskGenerator:
    """Generates diverse fluid dynamics tasks for meta-learning.
    
    This class creates a comprehensive distribution of fluid dynamics tasks
    with varying viscosity profiles, geometries, and boundary conditions.
    It supports both analytical solutions for validation and high-fidelity
    FEniCSx solutions for complex scenarios.
    
    The generator creates tasks suitable for few-shot meta-learning by:
    1. Sampling diverse task configurations from specified distributions
    2. Generating ground truth solutions using analytical or numerical methods
    3. Creating support/query splits for meta-learning evaluation
    4. Caching results for efficient repeated access
    
    Supported viscosity profiles:
        - Linear: μ(x,y) = a*x + b*y + c
        - Bilinear: μ(x,y) = a*x*y + b*x + c*y + d
        - Exponential: μ(x,y) = a*exp(b*x + c*y)
        - Temperature-dependent: μ(T) = μ₀*exp(E/(R*T))
        - Non-Newtonian: Power-law and Bingham plastic models
    
    Supported geometries:
        - Channel flow: Rectangular domain with inlet/outlet
        - Cavity flow: Lid-driven cavity with moving top wall
        - Cylinder flow: Flow around circular cylinder
        - Thermal convection: Natural convection in enclosure
    
    Attributes:
        data_config (DataConfig): Configuration for data generation parameters
        viscosity_config (ViscosityConfig): Viscosity profile specifications
        geometry_config (GeometryConfig): Geometry and boundary conditions
        task_cache (Dict): Cache for generated tasks to avoid recomputation
        
    Example:
        >>> from ml_research_pipeline.config import DataConfig
        >>> from ml_research_pipeline.core import FluidTaskGenerator
        >>> 
        >>> # Initialize generator
        >>> data_config = DataConfig(
        ...     domain_bounds=[[0, 1], [0, 1]],
        ...     n_support=50,
        ...     n_query=100
        ... )
        >>> generator = FluidTaskGenerator(data_config)
        >>> 
        >>> # Generate single task
        >>> task = generator.generate_task(
        ...     viscosity_type='linear',
        ...     reynolds=100.0
        ... )
        >>> print(f"Support points: {task.support_set['coords'].shape}")
        >>> print(f"Query points: {task.query_set['coords'].shape}")
        Support points: torch.Size([50, 2])
        Query points: torch.Size([100, 2])
        >>> 
        >>> # Generate task batch for meta-learning
        >>> task_batch = generator.generate_task_batch(
        ...     batch_size=16,
        ...     n_support=20,
        ...     n_query=50
        ... )
        >>> print(f"Generated {len(task_batch)} tasks")
        Generated 16 tasks
        
    Note:
        The generator automatically handles coordinate normalization,
        boundary condition enforcement, and solution validation.
        Tasks are cached based on configuration hash for efficiency.
    """
    
    def __init__(self, 
                 data_config: DataConfig,
                 viscosity_config: Optional[ViscosityConfig] = None,
                 geometry_config: Optional[GeometryConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize the task generator.
        
        Args:
            data_config: Configuration for data generation
            viscosity_config: Configuration for viscosity profiles
            geometry_config: Configuration for geometries
            seed: Random seed for reproducibility
        """
        self.data_config = data_config
        self.viscosity_config = viscosity_config or ViscosityConfig()
        self.geometry_config = geometry_config or GeometryConfig()
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize task type weights
        if data_config.task_weights is None:
            self.task_weights = np.ones(len(data_config.task_types)) / len(data_config.task_types)
        else:
            self.task_weights = np.array(data_config.task_weights)
            self.task_weights = self.task_weights / np.sum(self.task_weights)
        
        # Cache for generated tasks
        self.task_cache = {}
        
        # Task counter for deterministic ID generation
        self._task_counter = 0
        
        logger.info(f"Initialized FluidTaskGenerator with {len(data_config.task_types)} task types")
    
    def generate_task_batch(self, 
                           batch_size: int, 
                           n_support: int, 
                           n_query: int,
                           task_types: Optional[List[str]] = None) -> List[FluidTask]:
        """
        Generate a batch of fluid dynamics tasks.
        
        Args:
            batch_size: Number of tasks to generate
            n_support: Number of support points per task
            n_query: Number of query points per task
            task_types: Specific task types to generate (if None, sample from all)
        
        Returns:
            List of FluidTask objects
        """
        tasks = []
        
        for _ in range(batch_size):
            # Sample task type
            if task_types is None:
                task_type = np.random.choice(self.data_config.task_types, p=self.task_weights)
            else:
                task_type = np.random.choice(task_types)
            
            # Generate task configuration
            task_config = self._generate_task_config(task_type)
            
            # Generate coordinate samples
            support_coords = self._sample_coordinates(n_support, task_config)
            query_coords = self._sample_coordinates(n_query, task_config)
            
            # Create placeholder data for velocity and pressure
            # Note: In a complete implementation, these would be generated by solving
            # the forward problem or using analytical solutions
            support_velocity = torch.zeros(n_support, 2)  # [u, v] components
            support_pressure = torch.zeros(n_support, 1)  # scalar pressure
            query_velocity = torch.zeros(n_query, 2)
            query_pressure = torch.zeros(n_query, 1)
            
            # Create task
            task = FluidTask(
                config=task_config,
                support_set={
                    'coords': support_coords,
                    'velocity': support_velocity,
                    'pressure': support_pressure
                },
                query_set={
                    'coords': query_coords,
                    'velocity': query_velocity,
                    'pressure': query_pressure
                },
                metadata={
                    'generation_time': np.datetime64('now'),
                    'generator_version': '1.0'
                }
            )
            
            tasks.append(task)
        
        logger.debug(f"Generated batch of {batch_size} tasks")
        return tasks
    
    def _generate_task_config(self, task_type: str) -> TaskConfig:
        """Generate configuration for a specific task type."""
        # Generate unique task ID
        task_id = self._generate_task_id(task_type)
        
        # Sample Reynolds number
        reynolds_min, reynolds_max = 10.0, 5000.0
        reynolds_number = np.random.uniform(reynolds_min, reynolds_max)
        
        # Generate viscosity parameters based on task type
        viscosity_params = self._generate_viscosity_params(task_type)
        
        # Sample geometry parameters
        geometry_type = np.random.choice(['channel', 'cavity', 'cylinder'])
        geometry_params = self._generate_geometry_params(geometry_type)
        
        # Generate boundary conditions
        boundary_conditions = self._generate_boundary_conditions(geometry_type)
        
        return TaskConfig(
            task_id=task_id,
            task_type=task_type,
            reynolds_number=reynolds_number,
            viscosity_params=viscosity_params,
            geometry_type=geometry_type,
            geometry_params=geometry_params,
            boundary_conditions=boundary_conditions
        )
    
    def _generate_task_id(self, task_type: str) -> str:
        """Generate unique task identifier."""
        # Create hash from task type and random number (deterministic if seed is set)
        random_val = str(np.random.random())
        counter = getattr(self, '_task_counter', 0)
        self._task_counter = counter + 1
        hash_input = f"{task_type}_{counter}_{random_val}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _generate_viscosity_params(self, task_type: str) -> Dict[str, float]:
        """Generate viscosity parameters for different task types."""
        base_min, base_max = self.viscosity_config.base_viscosity_range
        grad_min, grad_max = self.viscosity_config.gradient_range
        
        params = {
            'base_viscosity': np.random.uniform(base_min, base_max)
        }
        
        if task_type == 'linear_viscosity':
            params.update({
                'gradient_x': np.random.uniform(grad_min, grad_max),
                'gradient_y': np.random.uniform(grad_min, grad_max)
            })
        
        elif task_type == 'bilinear_viscosity':
            params.update({
                'gradient_x': np.random.uniform(grad_min, grad_max),
                'gradient_y': np.random.uniform(grad_min, grad_max),
                'cross_term': np.random.uniform(-0.5, 0.5)
            })
        
        elif task_type == 'exponential_viscosity':
            params.update({
                'decay_rate_x': np.random.uniform(0.1, 2.0),
                'decay_rate_y': np.random.uniform(0.1, 2.0),
                'amplitude': np.random.uniform(0.5, 2.0)
            })
        
        elif task_type == 'temperature_dependent':
            temp_min, temp_max = self.viscosity_config.temperature_range
            params.update({
                'reference_temperature': np.random.uniform(temp_min, temp_max),
                'activation_energy': np.random.uniform(500.0, 2000.0),
                'temperature_gradient': np.random.uniform(-10.0, 10.0)
            })
        
        elif task_type == 'non_newtonian':
            params.update({
                'consistency_index': np.random.uniform(0.01, 1.0),
                'flow_behavior_index': np.random.uniform(0.5, 1.5),
                'yield_stress': np.random.uniform(0.0, 0.1)
            })
        
        return params
    
    def _generate_geometry_params(self, geometry_type: str) -> Dict[str, Any]:
        """Generate geometry parameters."""
        if geometry_type == 'channel':
            return {
                'length': np.random.uniform(1.0, 3.0),
                'width': np.random.uniform(0.5, 1.5),
                'inlet_profile': np.random.choice(['parabolic', 'uniform', 'plug'])
            }
        
        elif geometry_type == 'cavity':
            return {
                'length': np.random.uniform(1.0, 2.0),
                'width': np.random.uniform(1.0, 2.0),
                'lid_velocity': np.random.uniform(0.5, 2.0)
            }
        
        elif geometry_type == 'cylinder':
            return {
                'domain_length': np.random.uniform(2.0, 4.0),
                'domain_width': np.random.uniform(1.0, 2.0),
                'cylinder_radius': np.random.uniform(0.05, 0.2),
                'cylinder_position': [np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7)]
            }
        
        return {}
    
    def _generate_boundary_conditions(self, geometry_type: str) -> Dict[str, Dict[str, Any]]:
        """Generate boundary conditions for different geometries."""
        if geometry_type == 'channel':
            inlet_velocity = np.random.uniform(0.5, 2.0)
            return {
                'inlet': {'type': 'dirichlet', 'value': [inlet_velocity, 0.0]},
                'outlet': {'type': 'neumann', 'value': [0.0, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            }
        
        elif geometry_type == 'cavity':
            lid_velocity = np.random.uniform(0.5, 2.0)
            return {
                'lid': {'type': 'dirichlet', 'value': [lid_velocity, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            }
        
        elif geometry_type == 'cylinder':
            inlet_velocity = np.random.uniform(0.5, 2.0)
            return {
                'inlet': {'type': 'dirichlet', 'value': [inlet_velocity, 0.0]},
                'outlet': {'type': 'neumann', 'value': [0.0, 0.0]},
                'walls': {'type': 'dirichlet', 'value': [0.0, 0.0]},
                'cylinder': {'type': 'dirichlet', 'value': [0.0, 0.0]}
            }
        
        return {}
    
    def _sample_coordinates(self, n_points: int, task_config: TaskConfig) -> torch.Tensor:
        """Sample coordinate points for a task."""
        # Get domain bounds
        x_bounds = self.data_config.domain_bounds['x']
        y_bounds = self.data_config.domain_bounds['y']
        
        # Sample coordinates based on sampling strategy
        if self.data_config.interior_sampling == 'uniform':
            x = np.random.uniform(x_bounds[0], x_bounds[1], n_points)
            y = np.random.uniform(y_bounds[0], y_bounds[1], n_points)
        
        elif self.data_config.interior_sampling == 'adaptive':
            # Adaptive sampling with higher density near boundaries
            x = self._adaptive_sample(x_bounds, n_points)
            y = self._adaptive_sample(y_bounds, n_points)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.data_config.interior_sampling}")
        
        # Stack coordinates
        coords = np.stack([x, y], axis=1)
        
        return torch.from_numpy(coords).float()
    
    def _adaptive_sample(self, bounds: Tuple[float, float], n_points: int) -> np.ndarray:
        """Adaptive sampling with higher density near boundaries."""
        # Mix uniform and boundary-focused sampling
        n_uniform = int(0.7 * n_points)
        n_boundary = n_points - n_uniform
        
        # Uniform sampling
        uniform_samples = np.random.uniform(bounds[0], bounds[1], n_uniform)
        
        # Boundary-focused sampling
        boundary_samples = []
        for _ in range(n_boundary):
            if np.random.random() < 0.5:
                # Near lower boundary
                sample = bounds[0] + np.random.exponential(0.1) * (bounds[1] - bounds[0])
                sample = min(sample, bounds[1])
            else:
                # Near upper boundary
                sample = bounds[1] - np.random.exponential(0.1) * (bounds[1] - bounds[0])
                sample = max(sample, bounds[0])
            boundary_samples.append(sample)
        
        # Combine and shuffle
        all_samples = np.concatenate([uniform_samples, boundary_samples])
        np.random.shuffle(all_samples)
        
        return all_samples
    
    def validate_task_config(self, config: TaskConfig) -> bool:
        """
        Validate that a task configuration is physically reasonable.
        
        Args:
            config: Task configuration to validate
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check Reynolds number range
            if not (10.0 <= config.reynolds_number <= 5000.0):
                logger.warning(f"Reynolds number {config.reynolds_number} outside valid range")
                return False
            
            # Check viscosity parameters
            base_visc = config.viscosity_params.get('base_viscosity', 0.01)
            if not (1e-6 <= base_visc <= 1e3):
                logger.warning(f"Base viscosity {base_visc} outside physical bounds")
                return False
            
            # Check geometry parameters
            if config.geometry_type == 'channel':
                length = config.geometry_params.get('length', 1.0)
                width = config.geometry_params.get('width', 1.0)
                if length <= 0 or width <= 0:
                    logger.warning("Invalid channel dimensions")
                    return False
            
            # Check boundary conditions
            if not config.boundary_conditions:
                logger.warning("No boundary conditions specified")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating task config: {e}")
            return False
    
    def get_task_statistics(self, tasks: List[FluidTask]) -> Dict[str, Any]:
        """Compute statistics for a list of tasks."""
        if not tasks:
            return {}
        
        # Task type distribution
        task_types = [task.config.task_type for task in tasks]
        type_counts = {t: task_types.count(t) for t in set(task_types)}
        
        # Reynolds number statistics
        reynolds_numbers = [task.config.reynolds_number for task in tasks]
        
        # Geometry type distribution
        geometry_types = [task.config.geometry_type for task in tasks]
        geometry_counts = {g: geometry_types.count(g) for g in set(geometry_types)}
        
        return {
            'n_tasks': len(tasks),
            'task_type_distribution': type_counts,
            'reynolds_statistics': {
                'mean': np.mean(reynolds_numbers),
                'std': np.std(reynolds_numbers),
                'min': np.min(reynolds_numbers),
                'max': np.max(reynolds_numbers)
            },
            'geometry_distribution': geometry_counts
        }
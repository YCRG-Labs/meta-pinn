"""
Data and task configuration for meta-learning experiments.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base_config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """Configuration for data handling and preprocessing."""

    # Dataset settings
    dataset_name: str = "fluid_dynamics_tasks"
    data_dir: str = "data"
    cache_dir: str = "data/cache"

    # Data generation
    n_train_tasks: int = 1000
    n_val_tasks: int = 200
    n_test_tasks: int = 200

    # Task distribution
    task_types: List[str] = field(
        default_factory=lambda: [
            "linear_viscosity",
            "bilinear_viscosity",
            "exponential_viscosity",
            "temperature_dependent",
            "non_newtonian",
        ]
    )
    task_weights: Optional[List[float]] = None

    # Domain settings
    domain_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 1.0)}
    )

    # Discretization
    spatial_resolution: List[int] = field(default_factory=lambda: [64, 64])
    temporal_resolution: int = 100

    # Sampling strategies
    boundary_sampling: str = "uniform"  # uniform, adaptive, importance
    interior_sampling: str = "uniform"
    n_boundary_points: int = 100
    n_interior_points: int = 1000

    # Data preprocessing
    normalize_inputs: bool = True
    normalize_outputs: bool = True
    standardize: bool = True

    # Augmentation
    data_augmentation: bool = False
    augmentation_methods: List[str] = field(
        default_factory=lambda: ["rotation", "scaling", "noise"]
    )
    augmentation_probability: float = 0.5

    # Caching and storage
    use_cache: bool = True
    cache_format: str = "hdf5"  # hdf5, pickle, numpy
    compression: bool = True

    # Parallel data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class TaskConfig(BaseConfig):
    """Configuration for individual fluid dynamics tasks."""

    # Task identification
    task_id: str = ""
    task_type: str = "linear_viscosity"

    # Physics parameters
    reynolds_number: float = 100.0
    viscosity_params: Dict[str, float] = field(default_factory=dict)

    # Geometry settings
    geometry_type: str = "channel"  # channel, cavity, cylinder, custom
    geometry_params: Dict[str, Any] = field(default_factory=dict)

    # Boundary conditions
    boundary_conditions: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "inlet": {"type": "dirichlet", "value": [1.0, 0.0, 0.0]},
            "outlet": {"type": "neumann", "value": [0.0, 0.0, 0.0]},
            "walls": {"type": "dirichlet", "value": [0.0, 0.0, 0.0]},
        }
    )

    # Initial conditions
    initial_conditions: Dict[str, Any] = field(
        default_factory=lambda: {"velocity": [0.0, 0.0], "pressure": 0.0}
    )

    # Source terms
    source_terms: Dict[str, str] = field(default_factory=dict)

    # Material properties
    density: float = 1.0
    viscosity_function: str = "linear"  # linear, bilinear, exponential, custom

    # Numerical settings
    time_stepping: str = "implicit"  # implicit, explicit, semi_implicit
    time_step: float = 0.01
    final_time: float = 1.0

    # Solver settings
    solver_type: str = "fenics"  # fenics, analytical, neural_operator
    solver_params: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    mesh_quality: float = 0.8
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000


@dataclass
class ViscosityConfig(BaseConfig):
    """Configuration for viscosity profile generation."""

    # Viscosity function types
    function_type: str = "linear"  # linear, bilinear, exponential, polynomial, custom

    # Parameter ranges
    base_viscosity_range: Tuple[float, float] = (0.01, 0.1)
    gradient_range: Tuple[float, float] = (-1.0, 1.0)

    # Spatial dependence
    spatial_dependence: List[str] = field(default_factory=lambda: ["x", "y"])
    temporal_dependence: bool = False

    # Temperature dependence (for thermal problems)
    temperature_dependent: bool = False
    temperature_range: Tuple[float, float] = (273.0, 373.0)
    arrhenius_params: Dict[str, float] = field(
        default_factory=lambda: {
            "activation_energy": 1000.0,
            "reference_temperature": 298.0,
        }
    )

    # Non-Newtonian behavior
    non_newtonian: bool = False
    rheology_model: str = "power_law"  # power_law, carreau, cross
    rheology_params: Dict[str, float] = field(default_factory=dict)

    # Smoothness constraints
    smoothness_penalty: float = 0.1
    continuity_enforcement: bool = True

    # Validation
    physical_bounds: Tuple[float, float] = (1e-6, 1e3)
    monotonicity_constraint: Optional[str] = None  # increasing, decreasing, None


@dataclass
class GeometryConfig(BaseConfig):
    """Configuration for computational geometry."""

    # Geometry type
    geometry_type: str = "channel"  # channel, cavity, cylinder, backward_step, custom

    # Dimensional parameters
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0

    # Specific geometry parameters
    cylinder_radius: float = 0.1
    step_height: float = 0.2
    inlet_width: float = 0.1

    # Mesh generation
    mesh_resolution: float = 0.05
    mesh_refinement_levels: int = 2
    adaptive_refinement: bool = True

    # Boundary identification
    boundary_markers: Dict[str, int] = field(
        default_factory=lambda: {"inlet": 1, "outlet": 2, "walls": 3, "cylinder": 4}
    )

    # Mesh quality
    min_angle: float = 20.0
    max_aspect_ratio: float = 10.0

    # Custom geometry
    custom_geometry_file: Optional[str] = None
    geometry_function: Optional[str] = None

"""
Demonstration of FEniCSx solver integration with task generation system.

This script shows how the extended FEniCSx solver integrates with the task
generation system to create ground truth data for meta-learning.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_research_pipeline.core.fenicsx_solver import (
    FEniCSxSolver, SolverConfig, create_fenicsx_solver, FENICSX_AVAILABLE
)
from ml_research_pipeline.core.task_generator import FluidTaskGenerator
from ml_research_pipeline.core.analytical_solutions import AnalyticalSolutionGenerator
from ml_research_pipeline.config.data_config import DataConfig, TaskConfig, ViscosityConfig, GeometryConfig


def demonstrate_fenicsx_integration():
    """Demonstrate FEniCSx solver integration."""
    print("FEniCSx Integration Demonstration")
    print("=" * 50)
    
    # Check FEniCSx availability
    print(f"FEniCSx Available: {FENICSX_AVAILABLE}")
    
    if not FENICSX_AVAILABLE:
        print("\nFEniCSx is not available. This demo will show the interface")
        print("and fallback behavior, but won't run actual FEniCSx simulations.")
    
    print("\n1. Creating Task Generator")
    print("-" * 30)
    
    # Create task generator configuration
    data_config = DataConfig(
        task_types=['linear_viscosity', 'exponential_viscosity', 'bilinear_viscosity'],
        domain_bounds={'x': (0.0, 1.0), 'y': (0.0, 1.0)},
        n_interior_points=100,
        n_boundary_points=50
    )
    
    viscosity_config = ViscosityConfig(
        base_viscosity_range=(0.01, 0.1),
        gradient_range=(-0.5, 0.5)
    )
    
    geometry_config = GeometryConfig(
        geometry_type='channel',
        length=1.0,
        width=1.0
    )
    
    task_generator = FluidTaskGenerator(
        data_config=data_config,
        viscosity_config=viscosity_config,
        geometry_config=geometry_config,
        seed=42
    )
    
    print(f"Task generator created with {len(data_config.task_types)} task types")
    
    print("\n2. Generating Sample Tasks")
    print("-" * 30)
    
    # Generate a batch of tasks
    tasks = task_generator.generate_task_batch(
        batch_size=3,
        n_support=20,
        n_query=10
    )
    
    print(f"Generated {len(tasks)} tasks:")
    for i, task in enumerate(tasks):
        config = task.config
        print(f"  Task {i+1}: {config.task_type}, Re={config.reynolds_number:.2f}")
        print(f"    Geometry: {config.geometry_type}")
        print(f"    Viscosity params: {config.viscosity_params}")
    
    print("\n3. Creating FEniCSx Solver")
    print("-" * 30)
    
    # Create solver configuration
    solver_config = SolverConfig(
        mesh_resolution=(20, 10),  # Coarse mesh for demo
        element_degree=2,
        pressure_degree=1,
        solver_type='direct',
        tolerance=1e-6
    )
    
    # Create solver
    solver = create_fenicsx_solver(solver_config)
    
    if solver is not None:
        print("FEniCSx solver created successfully")
        print(f"  Mesh resolution: {solver_config.mesh_resolution}")
        print(f"  Element degree: {solver_config.element_degree}")
        print(f"  Solver type: {solver_config.solver_type}")
    else:
        print("FEniCSx solver not available - using fallback behavior")
    
    print("\n4. Solving Tasks")
    print("-" * 30)
    
    # Create evaluation coordinates
    coords = torch.tensor([
        [0.2, 0.2], [0.2, 0.5], [0.2, 0.8],
        [0.5, 0.2], [0.5, 0.5], [0.5, 0.8],
        [0.8, 0.2], [0.8, 0.5], [0.8, 0.8]
    ])
    
    print(f"Evaluation coordinates: {coords.shape}")
    
    # Solve each task
    solutions = []
    for i, task in enumerate(tasks):
        print(f"\nSolving Task {i+1}: {task.config.task_id}")
        
        if solver is not None:
            try:
                solution = solver.solve_task(task.config, coords)
                solutions.append(solution)
                
                print(f"  Solution type: {solution.metadata.get('solution_type', 'unknown')}")
                print(f"  Velocity shape: {solution.velocity.shape}")
                print(f"  Pressure shape: {solution.pressure.shape}")
                print(f"  Viscosity field shape: {solution.viscosity_field.shape}")
                
                # Check solution quality
                velocity_magnitude = torch.norm(solution.velocity, dim=1)
                print(f"  Max velocity: {torch.max(velocity_magnitude).item():.4f}")
                print(f"  Pressure range: {torch.max(solution.pressure).item() - torch.min(solution.pressure).item():.4f}")
                
                # Check viscosity field
                viscosity_range = torch.max(solution.viscosity_field) - torch.min(solution.viscosity_field)
                print(f"  Viscosity range: {viscosity_range.item():.6f}")
                
            except Exception as e:
                print(f"  Error solving task: {e}")
                print(f"  This would trigger fallback solution in production")
        else:
            print("  Solver not available - would use fallback solution")
    
    print("\n5. Comparing with Analytical Solutions")
    print("-" * 30)
    
    # Create analytical solution generator
    analytical_generator = AnalyticalSolutionGenerator()
    
    # Compare first task with analytical solution if possible
    if tasks and solver is not None:
        task = tasks[0]
        print(f"Comparing task: {task.config.task_id}")
        
        try:
            # Generate analytical solution
            analytical_solution = analytical_generator.generate_solution(
                task.config, coords
            )
            
            print(f"  Analytical solution type: {analytical_solution.metadata.get('solution_type', 'unknown')}")
            
            if solutions:
                fenicsx_solution = solutions[0]
                
                # Compare solutions
                velocity_diff = torch.norm(fenicsx_solution.velocity - analytical_solution.velocity, dim=1)
                pressure_diff = torch.abs(fenicsx_solution.pressure - analytical_solution.pressure).squeeze()
                
                print(f"  Max velocity difference: {torch.max(velocity_diff).item():.6f}")
                print(f"  Max pressure difference: {torch.max(pressure_diff).item():.6f}")
                
                # Relative errors
                velocity_rel_error = torch.mean(velocity_diff) / torch.mean(torch.norm(analytical_solution.velocity, dim=1))
                pressure_rel_error = torch.mean(pressure_diff) / torch.mean(torch.abs(analytical_solution.pressure))
                
                print(f"  Relative velocity error: {velocity_rel_error.item():.4f}")
                print(f"  Relative pressure error: {pressure_rel_error.item():.4f}")
        
        except Exception as e:
            print(f"  Error in analytical comparison: {e}")
    
    print("\n6. Batch Processing Demonstration")
    print("-" * 30)
    
    if solver is not None:
        try:
            # Create task configurations for batch processing
            batch_configs = [task.config for task in tasks]
            
            print(f"Processing batch of {len(batch_configs)} tasks...")
            
            # Solve batch
            batch_solutions = solver.solve_task_batch(batch_configs, coords)
            
            print(f"Batch processing completed: {len(batch_solutions)} solutions")
            
            # Summary statistics
            for i, solution in enumerate(batch_solutions):
                velocity_mag = torch.norm(solution.velocity, dim=1)
                print(f"  Task {i+1}: max_vel={torch.max(velocity_mag).item():.4f}, "
                      f"solution_type={solution.metadata.get('solution_type', 'unknown')}")
        
        except Exception as e:
            print(f"  Error in batch processing: {e}")
    else:
        print("  Batch processing not available without FEniCSx")
    
    print("\n7. Ground Truth Dataset Generation")
    print("-" * 30)
    
    if solver is not None:
        try:
            # Generate small dataset for demonstration
            task_configs = [task.config for task in tasks[:2]]  # Use first 2 tasks
            
            print(f"Generating ground truth dataset for {len(task_configs)} tasks...")
            
            dataset = solver.generate_ground_truth_dataset(
                task_configs,
                n_points_per_task=25  # Small number for demo
            )
            
            print(f"Dataset generated successfully:")
            print(f"  Number of tasks: {dataset['metadata']['n_tasks']}")
            print(f"  Points per task: {dataset['metadata']['n_points_per_task']}")
            print(f"  Solver config: {dataset['metadata']['solver_config']}")
            
            # Show sample data structure
            if dataset['tasks']:
                sample_task = dataset['tasks'][0]
                print(f"  Sample task data keys: {list(sample_task.keys())}")
                print(f"  Coordinates shape: {sample_task['coordinates'].shape}")
                print(f"  Velocity shape: {sample_task['velocity'].shape}")
                print(f"  Pressure shape: {sample_task['pressure'].shape}")
        
        except Exception as e:
            print(f"  Error in dataset generation: {e}")
    else:
        print("  Dataset generation not available without FEniCSx")
    
    print("\n8. Summary")
    print("-" * 30)
    
    print("FEniCSx Integration Features Demonstrated:")
    print("✓ Task generation with diverse viscosity profiles")
    print("✓ FEniCSx solver configuration and initialization")
    print("✓ Individual task solving with error handling")
    print("✓ Analytical solution comparison")
    print("✓ Batch processing capabilities")
    print("✓ Ground truth dataset generation")
    print("✓ Fallback behavior when FEniCSx is unavailable")
    
    if FENICSX_AVAILABLE:
        print("\nFEniCSx is available - all features are functional")
    else:
        print("\nFEniCSx is not available - install FEniCSx to run actual simulations")
    
    print("\nIntegration complete! The FEniCSx solver is ready for meta-learning.")


if __name__ == "__main__":
    demonstrate_fenicsx_integration()
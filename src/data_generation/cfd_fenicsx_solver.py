"""
Solves the 2D steady Stokes or Navier-Stokes equations with spatially varying viscosity
using FEniCSx. This module provides a more accurate solution compared to the placeholder
data generator, but requires FEniCSx to be installed.

For steady Stokes flow with varying viscosity:
    -∇·(ν(y)∇u) + ∇p = 0
    ∇·u = 0
    
where ν(y) = ν_base + a*y is the spatially varying viscosity.

For Navier-Stokes:
    (u·∇)u - ∇·(ν(y)∇u) + ∇p = 0
    ∇·u = 0
"""

import numpy as np
import os
import sys

# Check if FEniCSx is available
try:
    import dolfinx
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    FENICSX_AVAILABLE = True
except ImportError:
    FENICSX_AVAILABLE = False
    print("Warning: FEniCSx not available. This module requires FEniCSx to be installed.")

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def solve_stokes_varying_viscosity(x_query, y_query, t_query=None, config=None):
    """
    Solve the 2D Stokes or Navier-Stokes equations with spatially varying viscosity
    and evaluate the solution at query points.
    
    Args:
        x_query: x-coordinates of query points
        y_query: y-coordinates of query points
        t_query: time coordinates of query points (optional, for unsteady flow)
        config: Project configuration
        
    Returns:
        Tuple of (u, v, p) arrays at query points
    """
    if not FENICSX_AVAILABLE:
        raise ImportError("FEniCSx is required for this function but is not available.")
    
    if config is None:
        from config import cfg
        config = cfg
    
    # Import here to avoid errors if FEniCSx is not available
    import dolfinx.fem as fem
    import dolfinx.mesh as mesh
    import dolfinx.io as io
    import ufl
    
    # Create mesh
    nx, ny = 100, 50  # Mesh resolution
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [config.X_MIN, config.Y_MIN, config.X_MAX, config.Y_MAX],
        [nx, ny],
        mesh.CellType.triangle
    )
    
    # Function spaces
    P2 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 2)  # Velocity
    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # Pressure
    TH = ufl.MixedElement([P2, P1])  # Taylor-Hood element
    W = fem.FunctionSpace(domain, TH)
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Boundary conditions
    # No-slip on walls (y=Y_MIN and y=Y_MAX)
    def walls(x):
        return np.isclose(x[1], config.Y_MIN) | np.isclose(x[1], config.Y_MAX)
    
    # Parabolic inflow at x=X_MIN
    def inflow(x):
        return np.isclose(x[0], config.X_MIN)
    
    # Natural outflow at x=X_MAX (no explicit BC needed)
    
    # Create boundary conditions
    zero = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
    
    # No-slip on walls
    walls_dofs = fem.locate_dofs_geometrical(W.sub(0), walls)
    bc_walls = fem.dirichletbc(zero, walls_dofs, W.sub(0))
    
    # Parabolic inflow
    inflow_dofs = fem.locate_dofs_geometrical(W.sub(0), inflow)
    
    # Create inflow expression
    h = config.Y_MAX - config.Y_MIN
    u_max = config.U_MAX_INLET
    
    def inflow_expr(x):
        y_rel = (x[1] - config.Y_MIN) / h
        return np.stack((4 * u_max * y_rel * (1 - y_rel), np.zeros_like(y_rel)))
    
    inflow_function = fem.Function(W.sub(0).collapse())
    inflow_function.interpolate(inflow_expr)
    bc_inflow = fem.dirichletbc(inflow_function, inflow_dofs, W.sub(0))
    
    # Collect boundary conditions
    bcs = [bc_walls, bc_inflow]
    
    # Define viscosity function
    nu_base = config.NU_BASE_TRUE
    a_true = config.A_TRUE
    
    # Spatially varying viscosity: nu(y) = nu_base + a*y
    nu_expr = nu_base + a_true * ufl.SpatialCoordinate(domain)[1]
    
    # Define variational problem
    x = ufl.SpatialCoordinate(domain)
    
    # For unsteady flow, add time dependence
    if config.UNSTEADY_FLOW and t_query is not None:
        # Use the first time value for steady-state approximation at that time
        t_val = t_query[0]
        time_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t_val / config.T_MAX)
    else:
        time_factor = 1.0
    
    # Weak form
    if config.REYNOLDS_NUMBER > 0:
        # Navier-Stokes equations
        Re = config.REYNOLDS_NUMBER
        
        # Nonlinear convection term
        F = (1/Re) * ufl.inner(nu_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        F += ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        F -= ufl.inner(p, ufl.div(v)) * ufl.dx
        F += ufl.inner(ufl.div(u), q) * ufl.dx
    else:
        # Stokes equations
        F = ufl.inner(nu_expr * ufl.grad(u), ufl.grad(v)) * ufl.dx
        F -= ufl.inner(p, ufl.div(v)) * ufl.dx
        F += ufl.inner(ufl.div(u), q) * ufl.dx
    
    # Convert to bilinear and linear forms
    a, L = ufl.system(F)
    
    # Create solution function
    w = fem.Function(W)
    
    # Solve problem
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, u=w)
    problem.solve()
    
    # Extract velocity and pressure
    u_sol, p_sol = w.split()
    
    # Evaluate solution at query points
    u_values = np.zeros_like(x_query)
    v_values = np.zeros_like(y_query)
    p_values = np.zeros_like(x_query)
    
    # Create points array for evaluation
    points = np.vstack((x_query.flatten(), y_query.flatten())).T
    
    # Evaluate u and v components
    u_eval = u_sol.eval(points, np.zeros(len(points)))
    p_eval = p_sol.eval(points, np.zeros(len(points)))
    
    # Extract components
    u_values = u_eval[:, 0].reshape(x_query.shape) * time_factor
    v_values = u_eval[:, 1].reshape(y_query.shape) * time_factor
    p_values = p_eval.reshape(x_query.shape) * time_factor
    
    return u_values, v_values, p_values

if __name__ == "__main__":
    if not FENICSX_AVAILABLE:
        print("FEniCSx is not available. Please install it to run this script.")
        sys.exit(1)
    
    from config import cfg
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("Testing FEniCSx solver")
    print("="*70)
    
    # Create a grid for testing
    nx, ny = 50, 30
    x = np.linspace(cfg.X_MIN, cfg.X_MAX, nx)
    y = np.linspace(cfg.Y_MIN, cfg.Y_MAX, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Solve using FEniCSx
    u, v, p = solve_stokes_varying_viscosity(X, Y, config=cfg)
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot u velocity
    im0 = axs[0, 0].contourf(X, Y, u, 50, cmap='viridis')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title('u velocity')
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Plot v velocity
    im1 = axs[0, 1].contourf(X, Y, v, 50, cmap='viridis')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_title('v velocity')
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot pressure
    im2 = axs[1, 0].contourf(X, Y, p, 50, cmap='viridis')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_title('pressure')
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Plot velocity vectors with magnitude as background
    vel_mag = np.sqrt(u**2 + v**2)
    im3 = axs[1, 1].contourf(X, Y, vel_mag, 50, cmap='viridis')
    # Plot vectors (subsample for clarity)
    stride = 5
    axs[1, 1].quiver(X[::stride, ::stride], Y[::stride, ::stride], 
                     u[::stride, ::stride], v[::stride, ::stride],
                     color='white', scale=25)
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_title('velocity magnitude and direction')
    plt.colorbar(im3, ax=axs[1, 1])
    
    plt.tight_layout()
    plt.savefig("fenicsx_solution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("FEniCSx solver test complete")
    print("="*70)

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import cfg

def generate_placeholder_data(x, y, t=None, config=None):
    """
    Generate placeholder data for velocity and pressure fields using analytical approximations.
    This function provides a simple analytical solution that approximates channel flow with
    varying viscosity, without requiring FEniCSx.
    
    Args:
        x: x-coordinates tensor
        y: y-coordinates tensor
        t: time coordinates tensor (optional, for unsteady flow)
        config: Project configuration (optional, uses global cfg if None)
        
    Returns:
        Tuple of (u, v, p) tensors
    """
    if config is None:
        config = cfg
        
    # Ensure inputs are tensors
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=config.DEVICE)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32, device=config.DEVICE)
        
    # Reshape if needed
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    # Domain parameters
    h = config.Y_MAX - config.Y_MIN  # Channel height
    L = config.X_MAX - config.X_MIN  # Channel length
    
    # Physics parameters
    nu_base = config.NU_BASE_TRUE
    a_true = config.A_TRUE
    
    # Calculate viscosity field: nu(y) = nu_base + a * y
    nu = nu_base + a_true * y
    
    # For unsteady flow, add time dependence
    if config.UNSTEADY_FLOW and t is not None:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=config.DEVICE)
        if t.dim() == 1:
            t = t.unsqueeze(1)
            
        # Simple time-dependent factor (oscillating with time)
        time_factor = 1.0 + 0.2 * torch.sin(2 * torch.pi * t / config.T_MAX)
    else:
        time_factor = torch.ones_like(x)
    
    # Analytical approximation for u velocity in channel with varying viscosity
    # This is a simplified model that approximates the effect of varying viscosity
    u_max = config.U_MAX_INLET
    u = 4 * u_max * (y - config.Y_MIN) * (config.Y_MAX - y) / (h**2)
    
    # Add x-dependence (velocity decreases slightly along channel due to viscosity)
    x_factor = 1.0 - 0.1 * (x - config.X_MIN) / L
    u = u * x_factor * time_factor
    
    # v velocity (small vertical component due to varying viscosity)
    v = 0.05 * u_max * torch.sin(torch.pi * y / h) * (x / L) * time_factor
    
    # Pressure field (decreasing along channel)
    p_inlet = 1.0
    p_outlet = 0.0
    p_gradient = (p_outlet - p_inlet) / L
    
    # Base pressure field (linear decrease)
    p = p_inlet + p_gradient * (x - config.X_MIN)
    
    # Add y-dependence (slight variation due to varying viscosity)
    p = p + 0.1 * torch.sin(torch.pi * y / h) * time_factor
    
    # For Navier-Stokes, add nonlinear effects at higher Reynolds numbers
    if config.REYNOLDS_NUMBER > 50:
        # Add recirculation zones near corners at higher Reynolds numbers
        recirculation = 0.2 * (config.REYNOLDS_NUMBER / 100) * torch.exp(-20 * (y - config.Y_MIN) / h) * torch.sin(torch.pi * x / L)
        u = u - recirculation * time_factor
        
        # Add more complex pressure field
        p = p + 0.2 * (config.REYNOLDS_NUMBER / 100) * torch.sin(2 * torch.pi * x / L) * torch.sin(2 * torch.pi * y / h) * time_factor
    
    return u, v, p

def generate_sparse_measurement_data(x, y, t=None, config=None):
    """
    Generate sparse measurement data for training the PINN.
    
    Args:
        x: x-coordinates tensor
        y: y-coordinates tensor
        t: time coordinates tensor (optional, for unsteady flow)
        config: Project configuration (optional, uses global cfg if None)
        
    Returns:
        Tuple of (u, v, p) tensors
    """
    if config is None:
        config = cfg
        
    # Choose data source based on configuration
    if config.DATA_SOURCE == 'placeholder':
        # Use placeholder analytical approximation
        u, v, p = generate_placeholder_data(x, y, t, config)
        
    elif config.DATA_SOURCE == 'load_from_file':
        # Load pre-computed data from file
        if not os.path.exists(config.DATA_FILE):
            print(f"Warning: Data file {config.DATA_FILE} not found. Falling back to placeholder data.")
            u, v, p = generate_placeholder_data(x, y, t, config)
        else:
            # Load data from file
            data = np.load(config.DATA_FILE)
            
            # Extract coordinates and fields
            x_data = torch.tensor(data['x'], dtype=torch.float32, device=config.DEVICE)
            y_data = torch.tensor(data['y'], dtype=torch.float32, device=config.DEVICE)
            u_data = torch.tensor(data['u'], dtype=torch.float32, device=config.DEVICE)
            v_data = torch.tensor(data['v'], dtype=torch.float32, device=config.DEVICE)
            p_data = torch.tensor(data['p'], dtype=torch.float32, device=config.DEVICE)
            
            # If unsteady, also load time
            if config.UNSTEADY_FLOW and 't' in data:
                t_data = torch.tensor(data['t'], dtype=torch.float32, device=config.DEVICE)
                
                # Interpolate to requested points
                u = interpolate_field_3d(x, y, t, x_data, y_data, t_data, u_data)
                v = interpolate_field_3d(x, y, t, x_data, y_data, t_data, v_data)
                p = interpolate_field_3d(x, y, t, x_data, y_data, t_data, p_data)
            else:
                # Interpolate to requested points (2D)
                u = interpolate_field_2d(x, y, x_data, y_data, u_data)
                v = interpolate_field_2d(x, y, x_data, y_data, v_data)
                p = interpolate_field_2d(x, y, x_data, y_data, p_data)
    
    elif config.DATA_SOURCE == 'generate_with_fenicsx':
        # Generate data using FEniCSx solver
        try:
            from src.data_generation.cfd_fenicsx_solver import solve_stokes_varying_viscosity
            
            # Convert to numpy for FEniCSx
            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            
            # Solve using FEniCSx
            if config.UNSTEADY_FLOW and t is not None:
                t_np = t.cpu().numpy()
                u_np, v_np, p_np = solve_stokes_varying_viscosity(x_np, y_np, t_np, config)
            else:
                u_np, v_np, p_np = solve_stokes_varying_viscosity(x_np, y_np, config)
                
            # Convert back to torch tensors
            u = torch.tensor(u_np, dtype=torch.float32, device=config.DEVICE)
            v = torch.tensor(v_np, dtype=torch.float32, device=config.DEVICE)
            p = torch.tensor(p_np, dtype=torch.float32, device=config.DEVICE)
            
        except ImportError:
            print("Warning: FEniCSx not available. Falling back to placeholder data.")
            u, v, p = generate_placeholder_data(x, y, t, config)
    
    else:
        raise ValueError(f"Unknown data source: {config.DATA_SOURCE}")
    
    # Add some noise to simulate measurement errors
    if config.DATA_NOISE_LEVEL > 0:
        u = u + config.DATA_NOISE_LEVEL * torch.randn_like(u)
        v = v + config.DATA_NOISE_LEVEL * torch.randn_like(v)
        p = p + config.DATA_NOISE_LEVEL * torch.randn_like(p)
    
    return u, v, p

def interpolate_field_2d(x_query, y_query, x_data, y_data, field_data):
    """
    Interpolate a 2D field to query points
    
    Args:
        x_query, y_query: Query point coordinates
        x_data, y_data: Data point coordinates
        field_data: Field values at data points
        
    Returns:
        Interpolated field values at query points
    """
    # Simple nearest-neighbor interpolation
    # For a real application, consider more sophisticated interpolation methods
    
    # Reshape query points
    x_q = x_query.reshape(-1)
    y_q = y_query.reshape(-1)
    
    # Initialize output
    field_interp = torch.zeros_like(x_q)
    
    # For each query point, find nearest data point
    for i in range(len(x_q)):
        # Calculate distances to all data points
        dist = (x_data - x_q[i])**2 + (y_data - y_q[i])**2
        
        # Find index of nearest point
        idx = torch.argmin(dist)
        
        # Get field value at nearest point
        field_interp[i] = field_data[idx]
    
    # Reshape to match input
    return field_interp.reshape(x_query.shape)

def interpolate_field_3d(x_query, y_query, t_query, x_data, y_data, t_data, field_data):
    """
    Interpolate a 3D field to query points
    
    Args:
        x_query, y_query, t_query: Query point coordinates
        x_data, y_data, t_data: Data point coordinates
        field_data: Field values at data points
        
    Returns:
        Interpolated field values at query points
    """
    # Simple nearest-neighbor interpolation
    # For a real application, consider more sophisticated interpolation methods
    
    # Reshape query points
    x_q = x_query.reshape(-1)
    y_q = y_query.reshape(-1)
    t_q = t_query.reshape(-1)
    
    # Initialize output
    field_interp = torch.zeros_like(x_q)
    
    # For each query point, find nearest data point
    for i in range(len(x_q)):
        # Calculate distances to all data points
        dist = (x_data - x_q[i])**2 + (y_data - y_q[i])**2 + (t_data - t_q[i])**2
        
        # Find index of nearest point
        idx = torch.argmin(dist)
        
        # Get field value at nearest point
        field_interp[i] = field_data[idx]
    
    # Reshape to match input
    return field_interp.reshape(x_query.shape)

def save_data_to_file(x, y, u, v, p, t=None, filename=None, config=None):
    """
    Save generated data to a file for later use
    
    Args:
        x, y: Coordinate arrays
        u, v, p: Velocity and pressure fields
        t: Time array (optional, for unsteady flow)
        filename: Output filename (optional)
        config: Project configuration (optional)
    """
    if config is None:
        config = cfg
        
    if filename is None:
        filename = os.path.join(config.OUTPUT_DIR, 'generated_data.npz')
    
    # Convert to numpy arrays
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    p_np = p.cpu().numpy()
    
    # Save to file
    if t is not None:
        t_np = t.cpu().numpy()
        np.savez(filename, x=x_np, y=y_np, t=t_np, u=u_np, v=v_np, p=p_np)
    else:
        np.savez(filename, x=x_np, y=y_np, u=u_np, v=v_np, p=p_np)
        
    print(f"Data saved to {filename}")

def visualize_generated_data(x, y, u, v, p, t=None, config=None):
    """
    Visualize generated data
    
    Args:
        x, y: Coordinate arrays
        u, v, p: Velocity and pressure fields
        t: Time value (optional, for unsteady flow)
        config: Project configuration (optional)
    """
    if config is None:
        config = cfg
        
    # Convert to numpy arrays
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    u_np = u.cpu().numpy()
    v_np = v.cpu().numpy()
    p_np = p.cpu().numpy()
    
    # Create meshgrid if x and y are 1D
    if x_np.ndim == 1 and y_np.ndim == 1:
        X, Y = np.meshgrid(x_np, y_np, indexing='ij')
    else:
        X, Y = x_np, y_np
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Title suffix for unsteady flow
    title_suffix = f" (t = {t:.2f})" if t is not None else ""
    
    # Plot u velocity
    im0 = axs[0, 0].contourf(X, Y, u_np, 50, cmap='viridis')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title(f'u velocity{title_suffix}')
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Plot v velocity
    im1 = axs[0, 1].contourf(X, Y, v_np, 50, cmap='viridis')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_title(f'v velocity{title_suffix}')
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot pressure
    im2 = axs[1, 0].contourf(X, Y, p_np, 50, cmap='viridis')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_title(f'pressure{title_suffix}')
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Plot velocity vectors with magnitude as background
    vel_mag = np.sqrt(u_np**2 + v_np**2)
    im3 = axs[1, 1].contourf(X, Y, vel_mag, 50, cmap='viridis')
    # Plot vectors (subsample for clarity)
    stride = 5
    axs[1, 1].quiver(X[::stride, ::stride], Y[::stride, ::stride], 
                     u_np[::stride, ::stride], v_np[::stride, ::stride],
                     color='white', scale=25)
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_title(f'velocity magnitude and direction{title_suffix}')
    plt.colorbar(im3, ax=axs[1, 1])
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if t is not None:
        filename = f"{config.OUTPUT_DIR}/generated_data_t_{t:.2f}.png"
    else:
        filename = f"{config.OUTPUT_DIR}/generated_data.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
        
    print(f"Data visualization saved to {filename}")

if __name__ == "__main__":
    print("="*70)
    print("Testing data generator")
    print("="*70)
    
    # Create a grid for testing
    nx, ny = 50, 30
    x = torch.linspace(cfg.X_MIN, cfg.X_MAX, nx, device=cfg.DEVICE)
    y = torch.linspace(cfg.Y_MIN, cfg.Y_MAX, ny, device=cfg.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Generate data
    u, v, p = generate_placeholder_data(X, Y)
    
    # Visualize data
    visualize_generated_data(X, Y, u, v, p)
    
    # Test with unsteady flow
    if cfg.UNSTEADY_FLOW:
        for t_val in [0.0, 0.5, 1.0]:
            t = torch.full_like(X, t_val)
            u, v, p = generate_placeholder_data(X, Y, t)
            visualize_generated_data(X, Y, u, v, p, t_val)
    
    print("\n" + "="*70)
    print("Data generator test complete")
    print("="*70)

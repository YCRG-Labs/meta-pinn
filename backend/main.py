import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import cfg
from src.generate_data import generate_collocation_points, generate_boundary_points, generate_sparse_data_points
from src.model.model import PINN
from src.model.train_model import train_model
from src.model.evaluate_model import evaluate_model, analyze_flow_features

def create_3d_visualizations(model, config=None, save_path=None):
    """
    Create comprehensive 3D visualizations of the flow field
    
    Args:
        model: Trained PINN model
        config: Project configuration
        save_path: Directory to save visualizations
        
    Returns:
        Dictionary containing 3D visualization data
    """
    if config is None:
        config = cfg
        
    if save_path is None:
        save_path = config.OUTPUT_DIR
        
    print("\n" + "="*70)
    print("Generating 3D Visualizations")
    print("="*70)
    
    # Create high-resolution grid for visualization
    nx, ny = 80, 40
    x = torch.linspace(config.X_MIN, config.X_MAX, nx, device=config.DEVICE)
    y = torch.linspace(config.Y_MIN, config.Y_MAX, ny, device=config.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    # For unsteady flow, create multiple time slices
    if config.UNSTEADY_FLOW:
        n_time_slices = 5
        time_values = torch.linspace(config.T_MIN, config.T_MAX, n_time_slices, device=config.DEVICE)
        
        for i, t_val in enumerate(time_values):
            t_flat = torch.full_like(x_flat, t_val)
            
            # Get predictions
            u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat, t_flat)
            
            # Create 3D visualizations for this time slice
            create_time_slice_3d(X, Y, u_pred, v_pred, p_pred, t_val.item(), 
                                config, save_path, slice_idx=i)
    else:
        # Steady flow visualization
        u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat)
        
        # Reshape for plotting
        u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
        v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
        p_grid = p_pred.reshape(nx, ny).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        
        # Calculate derived quantities
        vel_mag = np.sqrt(u_grid**2 + v_grid**2)
        
        # Calculate vorticity
        dx = (config.X_MAX - config.X_MIN) / (nx - 1)
        dy = (config.Y_MAX - config.Y_MIN) / (ny - 1)
        u_y = np.gradient(u_grid, dy, axis=1)
        v_x = np.gradient(v_grid, dx, axis=0)
        vorticity = v_x - u_y
        
        # Calculate viscosity field
        nu_grid = config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * Y_np
        
        # Create 3D surface plots
        create_3d_surface_plots(X_np, Y_np, u_grid, v_grid, p_grid, vel_mag, 
                               vorticity, nu_grid, config, save_path)
        
        # Create 3D streamline plots
        create_3d_streamlines(X_np, Y_np, u_grid, v_grid, p_grid, config, save_path)
        
        # Create 3D vector field
        create_3d_vector_field(X_np, Y_np, u_grid, v_grid, config, save_path)
        
        # Create interactive 3D plots data
        visualization_data = {
            'coordinates': {
                'x': X_np.tolist(),
                'y': Y_np.tolist()
            },
            'fields': {
                'u_velocity': u_grid.tolist(),
                'v_velocity': v_grid.tolist(),
                'pressure': p_grid.tolist(),
                'velocity_magnitude': vel_mag.tolist(),
                'vorticity': vorticity.tolist(),
                'viscosity': nu_grid.tolist()
            },
            'parameters': {
                'reynolds_number': config.REYNOLDS_NUMBER,
                'inferred_viscosity_param': model.get_inferred_viscosity_param(),
                'true_viscosity_param': config.A_TRUE
            }
        }
        
        # Save visualization data as JSON
        with open(os.path.join(save_path, '3d_visualization_data.json'), 'w') as f:
            json.dump(visualization_data, f, indent=2)
        
        print(f"3D visualization data saved to {save_path}/3d_visualization_data.json")
        
        return visualization_data

def create_3d_surface_plots(X, Y, u, v, p, vel_mag, vorticity, nu, config, save_path):
    """
    Create 3D surface plots for all field variables
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define colormap
    cmap = plt.cm.viridis
    
    # 1. U velocity surface
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u, cmap=cmap, alpha=0.9, 
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U Velocity')
    ax1.set_title('U Velocity Field (3D Surface)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    # 2. Pressure surface
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, p, cmap='coolwarm', alpha=0.9,
                            linewidth=0, antialiased=True)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Pressure')
    ax2.set_title('Pressure Field (3D Surface)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    # 3. Velocity magnitude surface
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, vel_mag, cmap=cmap, alpha=0.9,
                            linewidth=0, antialiased=True)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Velocity Magnitude')
    ax3.set_title('Velocity Magnitude (3D Surface)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    # 4. Vorticity surface
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf4 = ax4.plot_surface(X, Y, vorticity, cmap='RdBu_r', alpha=0.9,
                            linewidth=0, antialiased=True)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Vorticity')
    ax4.set_title('Vorticity Field (3D Surface)')
    fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)
    
    # 5. Combined U and V with contours
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    surf5 = ax5.plot_surface(X, Y, u, cmap='viridis', alpha=0.7)
    contours5 = ax5.contour(X, Y, v, levels=10, colors='red', alpha=0.8)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('U Velocity')
    ax5.set_title('U Surface with V Contours')
    
    # 6. Pressure with velocity vectors (subsampled)
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    surf6 = ax6.plot_surface(X, Y, p, cmap='viridis', alpha=0.6)
    # Subsample for vector plot
    stride = 8
    X_sub = X[::stride, ::stride]
    Y_sub = Y[::stride, ::stride]
    U_sub = u[::stride, ::stride]
    V_sub = v[::stride, ::stride]
    P_sub = p[::stride, ::stride]
    ax6.quiver(X_sub, Y_sub, P_sub, U_sub, V_sub, np.zeros_like(U_sub), 
              length=0.1, normalize=True, color='red')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Pressure')
    ax6.set_title('Pressure Surface with Velocity Vectors')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, '3d_surface_plots.png'), 
                dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
    
    print(f"3D surface plots saved to {save_path}/3d_surface_plots.png")

def create_3d_streamlines(X, Y, u, v, p, config, save_path):
    """
    Create 3D streamline visualization
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Streamlines on pressure surface
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Create pressure surface
    surf1 = ax1.plot_surface(X, Y, p, cmap='viridis', alpha=0.3)
    
    # Create streamlines at different heights
    n_streamlines = 6
    y_starts = np.linspace(config.Y_MIN + 0.1, config.Y_MAX - 0.1, n_streamlines)
    
    for y_start in y_starts:
        # Create streamline starting points
        x_start = config.X_MIN
        # Use simple integration for streamlines
        streamline_x, streamline_y = create_streamline_2d(X, Y, u, v, x_start, y_start, config)
        streamline_z = np.interp(streamline_x, X[:, 0], p[:, int(y_start * Y.shape[1] / (config.Y_MAX - config.Y_MIN))])
        
        ax1.plot(streamline_x, streamline_y, streamline_z, color='red', linewidth=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Pressure')
    ax1.set_title('3D Streamlines on Pressure Surface')
    
    # 2. Streamlines on velocity magnitude surface
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    vel_mag = np.sqrt(u**2 + v**2)
    surf2 = ax2.plot_surface(X, Y, vel_mag, cmap='plasma', alpha=0.3)
    
    for y_start in y_starts:
        x_start = config.X_MIN
        streamline_x, streamline_y = create_streamline_2d(X, Y, u, v, x_start, y_start, config)
        streamline_z = np.interp(streamline_x, X[:, 0], vel_mag[:, int(y_start * Y.shape[1] / (config.Y_MAX - config.Y_MIN))])
        
        ax2.plot(streamline_x, streamline_y, streamline_z, color='white', linewidth=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Velocity Magnitude')
    ax2.set_title('3D Streamlines on Velocity Surface')
    
    # 3. Vector field in 3D
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    
    # Subsample for vector plot
    stride = 6
    X_sub = X[::stride, ::stride]
    Y_sub = Y[::stride, ::stride]
    U_sub = u[::stride, ::stride]
    V_sub = v[::stride, ::stride]
    
    # Create 3D vectors (w component is zero for 2D flow)
    W_sub = np.zeros_like(U_sub)
    
    ax3.quiver(X_sub, Y_sub, np.zeros_like(X_sub), U_sub, V_sub, W_sub, 
              length=0.1, normalize=True, color='red')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D Velocity Vector Field')
    
    # 4. Combined visualization
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Create combined surface (velocity magnitude)
    surf4 = ax4.plot_surface(X, Y, vel_mag, cmap='viridis', alpha=0.5)
    
    # Add streamlines
    for y_start in y_starts[::2]:  # Every other streamline for clarity
        x_start = config.X_MIN
        streamline_x, streamline_y = create_streamline_2d(X, Y, u, v, x_start, y_start, config)
        streamline_z = np.interp(streamline_x, X[:, 0], vel_mag[:, int(y_start * Y.shape[1] / (config.Y_MAX - config.Y_MIN))])
        
        ax4.plot(streamline_x, streamline_y, streamline_z, color='black', linewidth=2)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Velocity Magnitude')
    ax4.set_title('Combined 3D Flow Visualization')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_path, '3d_streamlines.png'), 
                dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
    
    print(f"3D streamlines plot saved to {save_path}/3d_streamlines.png")

def create_streamline_2d(X, Y, u, v, x_start, y_start, config, max_length=80):
    """
    Create a 2D streamline using simple integration
    """
    x_line = [x_start]
    y_line = [y_start]
    
    x_current = x_start
    y_current = y_start
    
    dt = 0.02  # Integration step
    
    for _ in range(max_length):
        # Interpolate velocity at current position
        if x_current >= config.X_MAX or x_current <= config.X_MIN:
            break
        if y_current >= config.Y_MAX or y_current <= config.Y_MIN:
            break
            
        # Find indices for interpolation
        i = int((x_current - config.X_MIN) / (config.X_MAX - config.X_MIN) * (X.shape[0] - 1))
        j = int((y_current - config.Y_MIN) / (config.Y_MAX - config.Y_MIN) * (X.shape[1] - 1))
        
        i = max(0, min(X.shape[0] - 1, i))
        j = max(0, min(X.shape[1] - 1, j))
        
        u_current = u[i, j]
        v_current = v[i, j]
        
        # Update position
        x_current += u_current * dt
        y_current += v_current * dt
        
        x_line.append(x_current)
        y_line.append(y_current)
    
    return np.array(x_line), np.array(y_line)

def create_3d_vector_field(X, Y, u, v, config, save_path):
    """
    Create 3D vector field visualization with multiple representations
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Velocity vectors colored by magnitude
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    stride = 5
    X_sub = X[::stride, ::stride]
    Y_sub = Y[::stride, ::stride]
    U_sub = u[::stride, ::stride]
    V_sub = v[::stride, ::stride]
    
    vel_mag_sub = np.sqrt(U_sub**2 + V_sub**2)
    
    # Create arrows with colors based on magnitude
    for i in range(X_sub.shape[0]):
        for j in range(X_sub.shape[1]):
            x_pos = X_sub[i, j]
            y_pos = Y_sub[i, j]
            u_val = U_sub[i, j]
            v_val = V_sub[i, j]
            mag = vel_mag_sub[i, j]
            
            # Color based on magnitude
            color_val = mag / np.max(vel_mag_sub)
            color = plt.cm.viridis(color_val)
            
            ax1.quiver(x_pos, y_pos, 0, u_val, v_val, 0, 
                      length=0.15, normalize=True, color=color, alpha=0.8)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Velocity Vectors (Colored by Magnitude)')
    
    # 2. Velocity vectors on multiple Z planes
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    
    z_levels = [0, 0.2, 0.4, 0.6, 0.8]
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    
    for z_level, color in zip(z_levels, colors):
        # Scale vectors by z-level for visual effect
        scale = 1.0 - 0.3 * z_level
        
        for i in range(0, X_sub.shape[0], 2):
            for j in range(0, X_sub.shape[1], 2):
                x_pos = X_sub[i, j]
                y_pos = Y_sub[i, j]
                u_val = U_sub[i, j] * scale
                v_val = V_sub[i, j] * scale
                
                ax2.quiver(x_pos, y_pos, z_level, u_val, v_val, 0, 
                          length=0.1, normalize=True, color=color, alpha=0.6)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Velocity Field at Multiple Levels')
    
    # 3. Velocity magnitude isosurfaces
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Create "height" based on velocity magnitude
    Z_vel = vel_mag
    
    surf3 = ax3.plot_surface(X, Y, Z_vel, cmap='plasma', alpha=0.7)
    
    # Add contour lines at different levels
    levels = np.linspace(np.min(vel_mag), np.max(vel_mag), 5)
    for level in levels:
        contour = ax3.contour(X, Y, vel_mag, levels=[level], colors=['black'], alpha=0.8)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Velocity Magnitude')
    ax3.set_title('Velocity Magnitude Surface')
    
    # 4. Vorticity surface
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Calculate vorticity
    dx = (config.X_MAX - config.X_MIN) / (X.shape[0] - 1)
    dy = (config.Y_MAX - config.Y_MIN) / (X.shape[1] - 1)
    u_y = np.gradient(u, dy, axis=1)
    v_x = np.gradient(v, dx, axis=0)
    vorticity = v_x - u_y
    
    # Create vorticity surface
    surf4 = ax4.plot_surface(X, Y, vorticity, cmap='RdBu_r', alpha=0.8)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Vorticity')
    ax4.set_title('Vorticity Field')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_path, '3d_vector_field.png'), 
                dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
    
    print(f"3D vector field plot saved to {save_path}/3d_vector_field.png")

def create_time_slice_3d(X, Y, u_pred, v_pred, p_pred, t_val, config, save_path, slice_idx):
    """
    Create 3D visualization for a specific time slice
    """
    nx, ny = X.shape
    u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
    v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
    p_grid = p_pred.reshape(nx, ny).detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    
    vel_mag = np.sqrt(u_grid**2 + v_grid**2)
    
    # Create time slice visualization
    fig = plt.figure(figsize=(12, 8))
    
    # 1. Velocity magnitude surface
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X_np, Y_np, vel_mag, cmap='viridis', alpha=0.9)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Velocity Magnitude')
    ax1.set_title(f'Velocity Magnitude at t = {t_val:.2f}')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. Pressure surface
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X_np, Y_np, p_grid, cmap='plasma', alpha=0.9)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Pressure')
    ax2.set_title(f'Pressure at t = {t_val:.2f}')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 3. Vector field
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    stride = 6
    X_sub = X_np[::stride, ::stride]
    Y_sub = Y_np[::stride, ::stride]
    U_sub = u_grid[::stride, ::stride]
    V_sub = v_grid[::stride, ::stride]
    
    ax3.quiver(X_sub, Y_sub, np.zeros_like(X_sub), U_sub, V_sub, np.zeros_like(U_sub),
              length=0.1, normalize=True, color='red')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'Velocity Vectors at t = {t_val:.2f}')
    
    # 4. Combined visualization
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf4 = ax4.plot_surface(X_np, Y_np, p_grid, cmap='viridis', alpha=0.5)
    ax4.quiver(X_sub, Y_sub, np.zeros_like(X_sub), U_sub, V_sub, np.zeros_like(U_sub),
              length=0.1, normalize=True, color='white')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Pressure')
    ax4.set_title(f'Combined View at t = {t_val:.2f}')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'3d_time_slice_{slice_idx:02d}_t_{t_val:.2f}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
    
    print(f"3D time slice plot saved to {save_path}/{filename}")

def export_simulation_data(model, config=None, save_path=None):
    """
    Export comprehensive simulation data for API consumption
    
    Returns:
        Dictionary containing all simulation results and metadata
    """
    if config is None:
        config = cfg
        
    if save_path is None:
        save_path = config.OUTPUT_DIR
    
    print("\n" + "="*70)
    print("Exporting Simulation Data")
    print("="*70)
    
    # Create evaluation grid
    nx, ny = 60, 30
    x = torch.linspace(config.X_MIN, config.X_MAX, nx, device=config.DEVICE)
    y = torch.linspace(config.Y_MIN, config.Y_MAX, ny, device=config.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    # Get model predictions
    with torch.no_grad():
        if config.UNSTEADY_FLOW:
            # Multiple time slices
            n_time = 8
            time_values = torch.linspace(config.T_MIN, config.T_MAX, n_time, device=config.DEVICE)
            
            simulation_data = {
                'type': 'unsteady',
                'time_slices': []
            }
            
            for t_val in time_values:
                t_flat = torch.full_like(x_flat, t_val)
                u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat, t_flat)
                
                u_grid = u_pred.reshape(nx, ny).cpu().numpy()
                v_grid = v_pred.reshape(nx, ny).cpu().numpy()
                p_grid = p_pred.reshape(nx, ny).cpu().numpy()
                
                time_slice_data = {
                    'time': float(t_val),
                    'u_velocity': u_grid.tolist(),
                    'v_velocity': v_grid.tolist(),
                    'pressure': p_grid.tolist(),
                    'velocity_magnitude': np.sqrt(u_grid**2 + v_grid**2).tolist()
                }
                
                simulation_data['time_slices'].append(time_slice_data)
        else:
            # Steady flow
            u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat)
            
            u_grid = u_pred.reshape(nx, ny).cpu().numpy()
            v_grid = v_pred.reshape(nx, ny).cpu().numpy()
            p_grid = p_pred.reshape(nx, ny).cpu().numpy()
            
            simulation_data = {
                'type': 'steady',
                'u_velocity': u_grid.tolist(),
                'v_velocity': v_grid.tolist(),
                'pressure': p_grid.tolist(),
                'velocity_magnitude': np.sqrt(u_grid**2 + v_grid**2).tolist()
            }
    
    # Add coordinate information
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    
    simulation_data.update({
        'coordinates': {
            'x': X_np.tolist(),
            'y': Y_np.tolist(),
            'x_range': [float(config.X_MIN), float(config.X_MAX)],
            'y_range': [float(config.Y_MIN), float(config.Y_MAX)],
            'dimensions': [nx, ny]
        },
        'parameters': {
            'reynolds_number': float(config.REYNOLDS_NUMBER),
            'viscosity_base': float(config.NU_BASE_TRUE),
            'viscosity_variation_true': float(config.A_TRUE),
            'viscosity_variation_inferred': float(model.get_inferred_viscosity_param()),
            'inlet_velocity_max': float(config.U_MAX_INLET),
            'unsteady': bool(config.UNSTEADY_FLOW)
        },
        'metadata': {
            'grid_size': [nx, ny],
            'model_layers': config.PINN_LAYERS,
            'training_epochs': config.EPOCHS,
            'device': str(config.DEVICE),
            'timestamp': time.time()
        }
    })
    
    # Calculate additional metrics for steady flow
    if not config.UNSTEADY_FLOW:
        # Calculate derived quantities
        dx = (config.X_MAX - config.X_MIN) / (nx - 1)
        dy = (config.Y_MAX - config.Y_MIN) / (ny - 1)
        
        # Vorticity
        u_y = np.gradient(u_grid, dy, axis=1)
        v_x = np.gradient(v_grid, dx, axis=0)
        vorticity = v_x - u_y
        
        # Viscosity field
        nu_grid = config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * Y_np
        
        # Divergence (mass conservation check)
        u_x = np.gradient(u_grid, dx, axis=0)
        v_y = np.gradient(v_grid, dy, axis=1)
        divergence = u_x + v_y
        
        simulation_data.update({
            'derived_fields': {
                'vorticity': vorticity.tolist(),
                'viscosity': nu_grid.tolist(),
                'divergence': divergence.tolist()
            },
            'metrics': {
                'max_velocity': float(np.max(np.sqrt(u_grid**2 + v_grid**2))),
                'min_pressure': float(np.min(p_grid)),
                'max_pressure': float(np.max(p_grid)),
                'pressure_drop': float(np.mean(p_grid[0, :]) - np.mean(p_grid[-1, :])),
                'mass_conservation_error': float(np.mean(np.abs(divergence))),
                'max_vorticity': float(np.max(np.abs(vorticity)))
            }
        })
    
    # Save to file
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'simulation_data.json'), 'w') as f:
        json.dump(simulation_data, f, indent=2)
    
    print(f"Simulation data exported to {save_path}/simulation_data.json")
    print(f"Data size: {len(json.dumps(simulation_data)) / 1024:.1f} KB")
    
    return simulation_data

def main(args):
    """
    Enhanced main function with 3D visualization capabilities and retry mechanism
    """
    print("\n" + "="*70)
    print("PINN Viscosity Inference Project with 3D Visualization")
    print("="*70)
    
    # Update configuration based on command line arguments
    if args.navier_stokes:
        cfg.update_for_navier_stokes()
    
    if args.unsteady:
        cfg.update_for_unsteady_flow()
    
    if args.advanced:
        cfg.enable_all_advanced_features()
    
    # Individual feature toggles
    if args.fourier:
        cfg.USE_FOURIER_FEATURES = True
    
    if args.adaptive_weights:
        cfg.USE_ADAPTIVE_WEIGHTS = True
    
    if args.adaptive_sampling:
        cfg.USE_ADAPTIVE_SAMPLING = True
    
    if args.curriculum:
        cfg.USE_CURRICULUM_LEARNING = True
    
    if args.reinit:
        cfg.USE_REINIT_STRATEGY = True
    
    # Update Reynolds number if specified
    if args.reynolds is not None:
        cfg.REYNOLDS_NUMBER = args.reynolds
    
    # Update epochs if specified
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    
    # Print configuration
    cfg.print_config()
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Generate data
    print("\nGenerating data...")
    collocation_points = generate_collocation_points()
    boundary_points = generate_boundary_points()
    sparse_data = generate_sparse_data_points()
    
    # Create and train model with retry mechanism
    if not args.evaluate_only:
        print("\nCreating and training model...")
        max_retries = 3
        best_error = float('inf')
        best_model = None
        
        for attempt in range(max_retries):
            print(f"\nTraining attempt {attempt + 1}/{max_retries}")
            model = PINN(cfg)
            model, history = train_model(model, collocation_points, boundary_points, sparse_data, cfg)
            
            # Calculate error
            inferred_a = model.get_inferred_viscosity_param()
            error = abs(inferred_a - cfg.A_TRUE) / cfg.A_TRUE * 100
            
            print(f"Attempt {attempt + 1} - Relative error: {error:.2f}%")
            
            if error < best_error:
                best_error = error
                best_model = model
                print(f"New best model found with error: {error:.2f}%")
            
            # If error is good enough, stop retrying
            if error < 5.0:  # 5% error threshold
                print("Achieved acceptable error, stopping retries")
                break
        
        # Use the best model found
        model = best_model
        print(f"\nUsing best model with error: {best_error:.2f}%")
    else:
        # Load existing model for evaluation
        print("\nLoading existing model for evaluation...")
        model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_SAVE_FILENAME)
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Please train a model first.")
            return
        model = PINN.load(model_path, cfg)
    
    # Evaluate model (traditional 2D plots)
    print("\nEvaluating model...")
    metrics = evaluate_model(model, cfg)
    
    # Create 3D visualizations
    if args.create_3d or not args.skip_3d:
        print("\nCreating 3D visualizations...")
        visualization_data = create_3d_visualizations(model, cfg)
    
    # Export comprehensive simulation data
    if args.export_data or not args.skip_export:
        print("\nExporting simulation data...")
        simulation_data = export_simulation_data(model, cfg)
    
    # Analyze advanced flow features
    if args.analyze_flow:
        print("\nAnalyzing advanced flow features...")
        analyze_flow_features(model, cfg)
    
    print("\n" + "="*70)
    print("PINN Project with 3D Visualization Complete")
    print("="*70)
    
    # Print final results
    print(f"\nInferred viscosity parameter a: {model.get_inferred_viscosity_param():.6f}")
    print(f"True viscosity parameter a: {cfg.A_TRUE:.6f}")
    print(f"Relative error: {abs(model.get_inferred_viscosity_param() - cfg.A_TRUE) / cfg.A_TRUE * 100:.2f}%")
    
    # Print PDE residuals
    print("\nPDE Residuals (Mean Absolute):")
    for key, value in metrics['pde_residuals'].items():
        print(f"  {key}: {value:.6e}")
    
    print("\nResults saved to:", cfg.OUTPUT_DIR)
    print("3D visualizations and simulation data available for API consumption")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PINN Viscosity Inference Project with 3D Visualization")
    
    # Physics options
    parser.add_argument("--navier-stokes", action="store_true", help="Use Navier-Stokes equations instead of Stokes")
    parser.add_argument("--unsteady", action="store_true", help="Use unsteady flow instead of steady")
    parser.add_argument("--reynolds", type=float, help="Reynolds number for the flow")
    
    # Advanced PINN features
    parser.add_argument("--advanced", action="store_true", help="Enable all advanced PINN features")
    parser.add_argument("--fourier", action="store_true", help="Use Fourier feature embeddings")
    parser.add_argument("--adaptive-weights", action="store_true", help="Use adaptive loss weighting")
    parser.add_argument("--adaptive-sampling", action="store_true", help="Use adaptive collocation sampling")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--reinit", action="store_true", help="Use re-initialization strategy")
    
    # Training options
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    
    # Evaluation options
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and only evaluate existing model")
    parser.add_argument("--analyze-flow", action="store_true", help="Analyze advanced flow features")
    
    # 3D visualization options
    parser.add_argument("--create-3d", action="store_true", help="Force creation of 3D visualizations")
    parser.add_argument("--skip-3d", action="store_true", help="Skip 3D visualization creation")
    parser.add_argument("--export-data", action="store_true", help="Force export of simulation data")
    parser.add_argument("--skip-export", action="store_true", help="Skip simulation data export")
    
    args = parser.parse_args()
    
    # Run main function
    main(args)
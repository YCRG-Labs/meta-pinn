import torch
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import cfg
from src.data_generation.data_generator import generate_sparse_measurement_data

def generate_collocation_points(config=None) -> torch.Tensor:
    """
    Generate collocation points for PDE residual evaluation.
    
    Args:
        config: Project configuration (optional, uses global cfg if None)
        
    Returns:
        Tensor of shape [N_COLLOCATION, 2] containing (x,y) coordinates
        or [N_COLLOCATION, 3] containing (x,y,t) if unsteady flow
    """
    if config is None:
        config = cfg
        
    # Random sampling in the domain
    n_points = config.N_COLLOCATION
    x_collocation = torch.rand(n_points, 1, device=config.DEVICE) * (config.X_MAX - config.X_MIN) + config.X_MIN
    y_collocation = torch.rand(n_points, 1, device=config.DEVICE) * (config.Y_MAX - config.Y_MIN) + config.Y_MIN
    
    # Add time dimension for unsteady flow
    if config.UNSTEADY_FLOW:
        t_collocation = torch.rand(n_points, 1, device=config.DEVICE) * (config.T_MAX - config.T_MIN) + config.T_MIN
        collocation_points = torch.cat([x_collocation, y_collocation, t_collocation], dim=1)
        print(f"Generated {n_points} collocation points for PDE residual evaluation (with time dimension)")
    else:
        # Stack into [x, y] coordinates
        collocation_points = torch.cat([x_collocation, y_collocation], dim=1)
        print(f"Generated {n_points} collocation points for PDE residual evaluation")
    
    return collocation_points

def generate_boundary_points(config=None) -> Dict[str, torch.Tensor]:
    """
    Generate boundary points for boundary condition enforcement.
    
    Args:
        config: Project configuration (optional, uses global cfg if None)
        
    Returns:
        Dictionary with keys 'inlet', 'outlet', 'walls' containing boundary points
    """
    if config is None:
        config = cfg
        
    n_points = config.N_BOUNDARY
    
    # Inlet boundary (x=X_MIN)
    x_inlet = torch.full((n_points, 1), config.X_MIN, device=config.DEVICE)
    y_inlet = torch.linspace(config.Y_MIN, config.Y_MAX, n_points, device=config.DEVICE).unsqueeze(1)
    
    # Outlet boundary (x=X_MAX)
    x_outlet = torch.full((n_points, 1), config.X_MAX, device=config.DEVICE)
    y_outlet = torch.linspace(config.Y_MIN, config.Y_MAX, n_points, device=config.DEVICE).unsqueeze(1)
    
    # Wall boundaries (y=Y_MIN and y=Y_MAX)
    # Bottom wall
    x_bottom = torch.linspace(config.X_MIN, config.X_MAX, n_points, device=config.DEVICE).unsqueeze(1)
    y_bottom = torch.full((n_points, 1), config.Y_MIN, device=config.DEVICE)
    
    # Top wall
    x_top = torch.linspace(config.X_MIN, config.X_MAX, n_points, device=config.DEVICE).unsqueeze(1)
    y_top = torch.full((n_points, 1), config.Y_MAX, device=config.DEVICE)
    
    # Add time dimension for unsteady flow
    if config.UNSTEADY_FLOW:
        # For each boundary, create points at different time steps
        n_time = config.N_TIME_BOUNDARY
        t_steps = torch.linspace(config.T_MIN, config.T_MAX, n_time, device=config.DEVICE).unsqueeze(1)
        
        # Repeat spatial coordinates for each time step
        x_inlet_t = x_inlet.repeat(n_time, 1)
        y_inlet_t = y_inlet.repeat(n_time, 1)
        t_inlet = t_steps.repeat_interleave(n_points, dim=0)
        
        x_outlet_t = x_outlet.repeat(n_time, 1)
        y_outlet_t = y_outlet.repeat(n_time, 1)
        t_outlet = t_steps.repeat_interleave(n_points, dim=0)
        
        x_bottom_t = x_bottom.repeat(n_time, 1)
        y_bottom_t = y_bottom.repeat(n_time, 1)
        t_bottom = t_steps.repeat_interleave(n_points, dim=0)
        
        x_top_t = x_top.repeat(n_time, 1)
        y_top_t = y_top.repeat(n_time, 1)
        t_top = t_steps.repeat_interleave(n_points, dim=0)
        
        # Combine coordinates with time
        inlet_points = torch.cat([x_inlet_t, y_inlet_t, t_inlet], dim=1)
        outlet_points = torch.cat([x_outlet_t, y_outlet_t, t_outlet], dim=1)
        bottom_wall = torch.cat([x_bottom_t, y_bottom_t, t_bottom], dim=1)
        top_wall = torch.cat([x_top_t, y_top_t, t_top], dim=1)
    else:
        # Steady flow - just spatial coordinates
        inlet_points = torch.cat([x_inlet, y_inlet], dim=1)
        outlet_points = torch.cat([x_outlet, y_outlet], dim=1)
        bottom_wall = torch.cat([x_bottom, y_bottom], dim=1)
        top_wall = torch.cat([x_top, y_top], dim=1)
    
    # Combine walls
    walls_points = torch.cat([bottom_wall, top_wall], dim=0)
    
    boundary_points = {
        'inlet': inlet_points,
        'outlet': outlet_points,
        'walls': walls_points
    }
    
    if config.UNSTEADY_FLOW:
        print(f"Generated boundary points with time dimension: inlet={inlet_points.shape[0]}, outlet={outlet_points.shape[0]}, walls={walls_points.shape[0]}")
    else:
        print(f"Generated boundary points: inlet={inlet_points.shape[0]}, outlet={outlet_points.shape[0]}, walls={walls_points.shape[0]}")
    
    return boundary_points

def generate_sparse_data_points(config=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate sparse data points for training the PINN.
    
    Args:
        config: Project configuration (optional, uses global cfg if None)
        
    Returns:
        Tuple of (coords, u, v, p) where coords is a tensor of shape [N_DATA_SPARSE, 2]
        or [N_DATA_SPARSE, 3] for unsteady flow, and u, v, p are tensors of shape [N_DATA_SPARSE, 1]
    """
    if config is None:
        config = cfg
        
    n_points = config.N_DATA_SPARSE
    
    # Generate coordinates based on sampling strategy
    if config.SPARSE_DATA_SAMPLING_STRATEGY == 'random':
        # Random sampling in the domain
        x_sparse = torch.rand(n_points, 1, device=config.DEVICE) * (config.X_MAX - config.X_MIN) + config.X_MIN
        y_sparse = torch.rand(n_points, 1, device=config.DEVICE) * (config.Y_MAX - config.Y_MIN) + config.Y_MIN
    
    elif config.SPARSE_DATA_SAMPLING_STRATEGY == 'grid':
        # Grid sampling
        n_x = int(np.sqrt(n_points))
        n_y = n_points // n_x
        x_grid = torch.linspace(config.X_MIN, config.X_MAX, n_x, device=config.DEVICE)
        y_grid = torch.linspace(config.Y_MIN, config.Y_MAX, n_y, device=config.DEVICE)
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        x_sparse = X.reshape(-1, 1)
        y_sparse = Y.reshape(-1, 1)
        
    elif config.SPARSE_DATA_SAMPLING_STRATEGY == 'centerline':
        # Points along the centerline (y = (Y_MIN + Y_MAX)/2)
        x_sparse = torch.linspace(config.X_MIN, config.X_MAX, n_points, device=config.DEVICE).unsqueeze(1)
        y_sparse = torch.full((n_points, 1), (config.Y_MIN + config.Y_MAX)/2, device=config.DEVICE)
        
    elif config.SPARSE_DATA_SAMPLING_STRATEGY == 'wall_proximal':
        # Points near the walls
        x_sparse = torch.linspace(config.X_MIN, config.X_MAX, n_points, device=config.DEVICE).unsqueeze(1)
        # Alternate between points near bottom and top walls
        y_values = []
        wall_distance = 0.05  # Distance from wall
        for i in range(n_points):
            if i % 2 == 0:
                y_values.append(config.Y_MIN + wall_distance)
            else:
                y_values.append(config.Y_MAX - wall_distance)
        y_sparse = torch.tensor(y_values, device=config.DEVICE).unsqueeze(1)
        
    else:
        raise ValueError(f"Unknown sampling strategy: {config.SPARSE_DATA_SAMPLING_STRATEGY}")
    
    # Add time dimension for unsteady flow
    if config.UNSTEADY_FLOW:
        if config.SPARSE_DATA_SAMPLING_STRATEGY == 'time_slices':
            # Sample at specific time slices
            n_time = config.N_TIME_SPARSE
            t_steps = torch.linspace(config.T_MIN, config.T_MAX, n_time, device=config.DEVICE)
            
            # Repeat spatial coordinates for each time step
            x_sparse_t = x_sparse.repeat(n_time, 1)
            y_sparse_t = y_sparse.repeat(n_time, 1)
            
            # Create time coordinates
            t_sparse = torch.cat([t.repeat(n_points, 1) for t in t_steps.unsqueeze(1)])
            
            # Get the corresponding u, v, p values
            u_sparse, v_sparse, p_sparse = generate_sparse_measurement_data(x_sparse_t, y_sparse_t, t_sparse, config)
            
            # Combine x, y, t coordinates
            coords_sparse = torch.cat([x_sparse_t, y_sparse_t, t_sparse], dim=1)
        else:
            # Random time sampling
            t_sparse = torch.rand(n_points, 1, device=config.DEVICE) * (config.T_MAX - config.T_MIN) + config.T_MIN
            
            # Get the corresponding u, v, p values
            u_sparse, v_sparse, p_sparse = generate_sparse_measurement_data(x_sparse, y_sparse, t_sparse, config)
            
            # Combine x, y, t coordinates
            coords_sparse = torch.cat([x_sparse, y_sparse, t_sparse], dim=1)
            
        print(f"Generated {coords_sparse.shape[0]} sparse data points with time dimension using '{config.SPARSE_DATA_SAMPLING_STRATEGY}' strategy")
    else:
        # Get the corresponding u, v, p values for steady flow
        u_sparse, v_sparse, p_sparse = generate_sparse_measurement_data(x_sparse, y_sparse, config)
        
        # Combine x, y coordinates
        coords_sparse = torch.cat([x_sparse, y_sparse], dim=1)
        
        print(f"Generated {coords_sparse.shape[0]} sparse data points using '{config.SPARSE_DATA_SAMPLING_STRATEGY}' strategy")
    
    return coords_sparse, u_sparse, v_sparse, p_sparse

def visualize_data_points(collocation_points, boundary_points, sparse_data, config=None):
    """
    Visualize the generated data points.
    
    Args:
        collocation_points: Tensor of collocation points
        boundary_points: Dictionary of boundary points
        sparse_data: Tuple of (coords, u, v, p)
        config: Project configuration (optional, uses global cfg if None)
    """
    if config is None:
        config = cfg
        
    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot collocation points (subsample for clarity if too many)
    max_plot_points = 1000
    if collocation_points.shape[0] > max_plot_points:
        indices = np.random.choice(collocation_points.shape[0], max_plot_points, replace=False)
        collocation_subset = collocation_points[indices].cpu().numpy()
    else:
        collocation_subset = collocation_points.cpu().numpy()
    
    # For unsteady flow, plot a specific time slice
    if config.UNSTEADY_FLOW and collocation_subset.shape[1] > 2:
        # Choose a specific time slice (e.g., t=T_MIN)
        t_slice = config.T_MIN
        t_tolerance = (config.T_MAX - config.T_MIN) * 0.05  # 5% tolerance
        
        # Filter points close to the chosen time slice
        time_mask = np.abs(collocation_subset[:, 2] - t_slice) < t_tolerance
        collocation_subset = collocation_subset[time_mask]
        
        plt.scatter(collocation_subset[:, 0], collocation_subset[:, 1], s=1, alpha=0.3, color='gray', 
                   label=f'Collocation (t≈{t_slice:.2f})')
        
        # Filter boundary points for this time slice
        inlet_points = boundary_points['inlet'].cpu().numpy()
        outlet_points = boundary_points['outlet'].cpu().numpy()
        walls_points = boundary_points['walls'].cpu().numpy()
        
        inlet_mask = np.abs(inlet_points[:, 2] - t_slice) < t_tolerance
        outlet_mask = np.abs(outlet_points[:, 2] - t_slice) < t_tolerance
        walls_mask = np.abs(walls_points[:, 2] - t_slice) < t_tolerance
        
        inlet_points = inlet_points[inlet_mask]
        outlet_points = outlet_points[outlet_mask]
        walls_points = walls_points[walls_mask]
        
        # Plot boundary points
        if len(inlet_points) > 0:
            plt.scatter(inlet_points[:, 0], inlet_points[:, 1], s=5, color='blue', label=f'Inlet (t≈{t_slice:.2f})')
        if len(outlet_points) > 0:
            plt.scatter(outlet_points[:, 0], outlet_points[:, 1], s=5, color='green', label=f'Outlet (t≈{t_slice:.2f})')
        if len(walls_points) > 0:
            plt.scatter(walls_points[:, 0], walls_points[:, 1], s=5, color='red', label=f'Walls (t≈{t_slice:.2f})')
        
        # Plot sparse data points
        sparse_coords, sparse_u, sparse_v, sparse_p = sparse_data
        sparse_coords = sparse_coords.cpu().numpy()
        sparse_u = sparse_u.cpu().numpy()
        
        # Filter sparse data for this time slice
        sparse_mask = np.abs(sparse_coords[:, 2] - t_slice) < t_tolerance
        sparse_coords_filtered = sparse_coords[sparse_mask]
        sparse_u_filtered = sparse_u[sparse_mask]
        
        if len(sparse_coords_filtered) > 0:
            # Use u velocity for color
            plt.scatter(sparse_coords_filtered[:, 0], sparse_coords_filtered[:, 1], s=30, c=sparse_u_filtered, 
                       cmap='viridis', edgecolors='black', label=f'Sparse Data (t≈{t_slice:.2f})')
            plt.colorbar(label='u velocity')
        
        plt.title(f'Data Points for PINN Training (Time Slice t≈{t_slice:.2f})')
    else:
        # Steady flow visualization
        plt.scatter(collocation_subset[:, 0], collocation_subset[:, 1], s=1, alpha=0.3, color='gray', label='Collocation')
        
        # Plot boundary points
        inlet_points = boundary_points['inlet'].cpu().numpy()
        outlet_points = boundary_points['outlet'].cpu().numpy()
        walls_points = boundary_points['walls'].cpu().numpy()
        
        plt.scatter(inlet_points[:, 0], inlet_points[:, 1], s=5, color='blue', label='Inlet')
        plt.scatter(outlet_points[:, 0], outlet_points[:, 1], s=5, color='green', label='Outlet')
        plt.scatter(walls_points[:, 0], walls_points[:, 1], s=5, color='red', label='Walls')
        
        # Plot sparse data points
        sparse_coords, sparse_u, sparse_v, sparse_p = sparse_data
        sparse_coords = sparse_coords.cpu().numpy()
        sparse_u = sparse_u.cpu().numpy()
        
        # Use u velocity for color
        plt.scatter(sparse_coords[:, 0], sparse_coords[:, 1], s=30, c=sparse_u, cmap='viridis', 
                   edgecolors='black', label='Sparse Data')
        plt.colorbar(label='u velocity')
        
        plt.title('Data Points for PINN Training')
    
    # Set plot properties
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{config.OUTPUT_DIR}/data_points.png", dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
        
    print(f"Data visualization saved to {config.OUTPUT_DIR}/data_points.png")

if __name__ == "__main__":
    print("="*70)
    print("Generating data for PINN training")
    print("="*70)
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Generate collocation points
    collocation_points = generate_collocation_points()
    
    # Generate boundary points
    boundary_points = generate_boundary_points()
    
    # Generate sparse data points
    sparse_data = generate_sparse_data_points()
    
    # Visualize data points
    visualize_data_points(collocation_points, boundary_points, sparse_data)
    
    print("\n" + "="*70)
    print("Data generation complete")
    print("="*70)

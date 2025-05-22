import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import cfg
from src.model.model import PINN

def evaluate_model(model, config=None):
    """
    Evaluate a trained PINN model with advanced visualization and diagnostics
    
    Args:
        model: Trained PINN model
        config: Project configuration (optional, uses global cfg if None)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if config is None:
        config = cfg
        
    print("\n" + "="*70)
    print("Evaluating trained model")
    print("="*70)
    
    # Create a grid for visualization
    nx, ny = 100, 50
    x = torch.linspace(config.X_MIN, config.X_MAX, nx, device=config.DEVICE)
    y = torch.linspace(config.Y_MIN, config.Y_MAX, ny, device=config.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    # For unsteady flow, evaluate at specific time points
    if config.UNSTEADY_FLOW:
        # Evaluate at several time points
        time_points = [config.T_MIN, 
                      (config.T_MIN + config.T_MAX) / 2, 
                      config.T_MAX]
        
        for t_val in time_points:
            t_flat = torch.full_like(x_flat, t_val)
            evaluate_at_time(model, x_flat, y_flat, t_flat, X, Y, t_val, config)
    else:
        # Steady flow evaluation
        # Predict velocity and pressure fields
        u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat)
        
        # Reshape for plotting
        u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
        v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
        p_grid = p_pred.reshape(nx, ny).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        
        # Calculate velocity magnitude
        vel_mag = np.sqrt(u_grid**2 + v_grid**2)
        
        # Calculate viscosity field
        nu_grid = (config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * Y_np)
        
        # Calculate Reynolds number field (local)
        Re_local = vel_mag * config.U_MAX_INLET / nu_grid
        
        # Plot velocity field
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot u velocity
        im0 = axs[0, 0].contourf(X_np, Y_np, u_grid, 50, cmap='viridis')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')
        axs[0, 0].set_title('u velocity')
        plt.colorbar(im0, ax=axs[0, 0])
        
        # Plot v velocity
        im1 = axs[0, 1].contourf(X_np, Y_np, v_grid, 50, cmap='viridis')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')
        axs[0, 1].set_title('v velocity')
        plt.colorbar(im1, ax=axs[0, 1])
        
        # Plot pressure
        im2 = axs[1, 0].contourf(X_np, Y_np, p_grid, 50, cmap='viridis')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('y')
        axs[1, 0].set_title('pressure')
        plt.colorbar(im2, ax=axs[1, 0])
        
        # Plot velocity vectors with magnitude as background
        im3 = axs[1, 1].contourf(X_np, Y_np, vel_mag, 50, cmap='viridis')
        # Plot vectors (subsample for clarity)
        stride = 5
        axs[1, 1].quiver(X_np[::stride, ::stride], Y_np[::stride, ::stride], 
                         u_grid[::stride, ::stride], v_grid[::stride, ::stride],
                         color='white', scale=25)
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('y')
        axs[1, 1].set_title('velocity magnitude and direction')
        plt.colorbar(im3, ax=axs[1, 1])
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'velocity_pressure_fields.png'), dpi=300, bbox_inches='tight')
        
        if config.SHOW_PLOTS_INTERACTIVE:
            plt.show()
        else:
            plt.close()
            
        print(f"Velocity and pressure fields plot saved to {config.OUTPUT_DIR}/velocity_pressure_fields.png")
        
        # Plot viscosity field and comparison
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot inferred viscosity field
        im0 = axs[0].contourf(X_np, Y_np, nu_grid, 50, cmap='viridis')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title(f'Inferred viscosity field: ν(y) = {config.NU_BASE_TRUE} + {model.get_inferred_viscosity_param():.6f} * y')
        plt.colorbar(im0, ax=axs[0])
        
        # Plot true vs inferred viscosity profile
        y_line = np.linspace(config.Y_MIN, config.Y_MAX, 100)
        nu_true = config.NU_BASE_TRUE + config.A_TRUE * y_line
        nu_inferred = config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * y_line
        
        axs[1].plot(y_line, nu_true, 'r-', label=f'True: ν(y) = {config.NU_BASE_TRUE} + {config.A_TRUE} * y')
        axs[1].plot(y_line, nu_inferred, 'b--', label=f'Inferred: ν(y) = {config.NU_BASE_TRUE} + {model.get_inferred_viscosity_param():.6f} * y')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('viscosity ν')
        axs[1].set_title('Viscosity profile comparison')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'viscosity_comparison.png'), dpi=300, bbox_inches='tight')
        
        if config.SHOW_PLOTS_INTERACTIVE:
            plt.show()
        else:
            plt.close()
            
        print(f"Viscosity comparison plot saved to {config.OUTPUT_DIR}/viscosity_comparison.png")
        
        # Plot local Reynolds number
        plt.figure(figsize=(10, 8))
        im = plt.contourf(X_np, Y_np, Re_local, 50, cmap='viridis')
        plt.colorbar(im, label='Local Reynolds Number')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Local Reynolds Number (Global Re = {config.REYNOLDS_NUMBER})')
        
        # Save figure
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'reynolds_number.png'), dpi=300, bbox_inches='tight')
        
        if config.SHOW_PLOTS_INTERACTIVE:
            plt.show()
        else:
            plt.close()
            
        print(f"Reynolds number plot saved to {config.OUTPUT_DIR}/reynolds_number.png")
        
        # Calculate PDE residuals on a grid
        with torch.no_grad():
            # Sample points for residual evaluation
            n_sample = 1000
            x_sample = torch.rand(n_sample, 1, device=config.DEVICE, requires_grad=True) * (config.X_MAX - config.X_MIN) + config.X_MIN
            y_sample = torch.rand(n_sample, 1, device=config.DEVICE, requires_grad=True) * (config.Y_MAX - config.Y_MIN) + config.Y_MIN

            
            # Calculate residuals
            residuals = model.pde_residual(x_sample, y_sample)
            
            # Calculate mean absolute residuals
            mean_residuals = {
                'momentum_x': residuals['momentum_x'].abs().mean().item(),
                'momentum_y': residuals['momentum_y'].abs().mean().item(),
                'continuity': residuals['continuity'].abs().mean().item()
            }
            
            print("\nPDE Residuals (Mean Absolute):")
            print(f"  Momentum-x: {mean_residuals['momentum_x']:.6e}")
            print(f"  Momentum-y: {mean_residuals['momentum_y']:.6e}")
            print(f"  Continuity: {mean_residuals['continuity']:.6e}")
        
        # Calculate parameter inference accuracy
        inferred_a = model.get_inferred_viscosity_param()
        true_a = config.A_TRUE
        abs_error = abs(inferred_a - true_a)
        rel_error = abs_error / true_a * 100
        
        print("\nViscosity Parameter Inference:")
        print(f"  True a:      {true_a:.6f}")
        print(f"  Inferred a:  {inferred_a:.6f}")
        print(f"  Absolute error: {abs_error:.6f}")
        print(f"  Relative error: {rel_error:.2f}%")
        
        # Return evaluation metrics
        metrics = {
            'inferred_a': inferred_a,
            'true_a': true_a,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'pde_residuals': mean_residuals
        }
        
        print("\n" + "="*70)
        print("Model evaluation complete")
        print("="*70)
        
        return metrics

def evaluate_at_time(model, x_flat, y_flat, t_flat, X, Y, t_val, config):
    """
    Evaluate model at a specific time point for unsteady flow
    
    Args:
        model: Trained PINN model
        x_flat, y_flat, t_flat: Flattened coordinate tensors
        X, Y: Meshgrid tensors
        t_val: Time value
        config: Configuration
    """
    # Predict velocity and pressure fields at this time
    u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat, t_flat)
    
    # Reshape for plotting
    nx, ny = X.shape
    u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
    v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
    p_grid = p_pred.reshape(nx, ny).detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    
    # Calculate velocity magnitude
    vel_mag = np.sqrt(u_grid**2 + v_grid**2)
    
    # Plot velocity field at this time
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot u velocity
    im0 = axs[0, 0].contourf(X_np, Y_np, u_grid, 50, cmap='viridis')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title(f'u velocity (t = {t_val:.2f})')
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Plot v velocity
    im1 = axs[0, 1].contourf(X_np, Y_np, v_grid, 50, cmap='viridis')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_title(f'v velocity (t = {t_val:.2f})')
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot pressure
    im2 = axs[1, 0].contourf(X_np, Y_np, p_grid, 50, cmap='viridis')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_title(f'pressure (t = {t_val:.2f})')
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Plot velocity vectors with magnitude as background
    im3 = axs[1, 1].contourf(X_np, Y_np, vel_mag, 50, cmap='viridis')
    # Plot vectors (subsample for clarity)
    stride = 5
    axs[1, 1].quiver(X_np[::stride, ::stride], Y_np[::stride, ::stride], 
                     u_grid[::stride, ::stride], v_grid[::stride, ::stride],
                     color='white', scale=25)
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_title(f'velocity magnitude and direction (t = {t_val:.2f})')
    plt.colorbar(im3, ax=axs[1, 1])
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, f'velocity_pressure_t_{t_val:.2f}.png'), dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
        
    print(f"Velocity and pressure fields at t = {t_val:.2f} saved to {config.OUTPUT_DIR}/velocity_pressure_t_{t_val:.2f}.png")

def analyze_flow_features(model, config=None):
    """
    Analyze advanced flow features like vorticity, streamlines, and shear stress
    
    Args:
        model: Trained PINN model
        config: Project configuration (optional, uses global cfg if None)
    """
    if config is None:
        config = cfg
        
    print("\n" + "="*70)
    print("Analyzing advanced flow features")
    print("="*70)
    
    # Create a grid for visualization
    nx, ny = 200, 100  # Higher resolution for better feature visualization
    x = torch.linspace(config.X_MIN, config.X_MAX, nx, device=config.DEVICE)
    y = torch.linspace(config.Y_MIN, config.Y_MAX, ny, device=config.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    # Predict velocity and pressure fields
    u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat)
    
    # Reshape for plotting
    u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
    v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
    p_grid = p_pred.reshape(nx, ny).detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    
    # Calculate vorticity (curl of velocity field)
    # Vorticity = dv/dx - du/dy
    dx = (config.X_MAX - config.X_MIN) / (nx - 1)
    dy = (config.Y_MAX - config.Y_MIN) / (ny - 1)
    
    # Calculate gradients
    u_y = np.gradient(u_grid, dy, axis=1)
    v_x = np.gradient(v_grid, dx, axis=0)
    
    # Vorticity
    vorticity = v_x - u_y
    
    # Calculate shear stress
    # For a Newtonian fluid: tau = mu * (du/dy + dv/dx)
    # Get viscosity field
    nu_grid = (config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * Y_np)
    # Density is assumed to be 1 in non-dimensional units
    mu_grid = nu_grid  # For incompressible flow, mu = rho * nu, and rho = 1
    shear_stress = mu_grid * (u_y + v_x)
    
    # Plot vorticity
    plt.figure(figsize=(12, 8))
    im = plt.contourf(X_np, Y_np, vorticity, 100, cmap='RdBu_r')
    plt.colorbar(im, label='Vorticity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Vorticity Field (∇ × u)')
    
    # Save figure
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'vorticity_field.png'), dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
        
    print(f"Vorticity field plot saved to {config.OUTPUT_DIR}/vorticity_field.png")
    
    # Plot shear stress
    plt.figure(figsize=(12, 8))
    im = plt.contourf(X_np, Y_np, shear_stress, 100, cmap='viridis')
    plt.colorbar(im, label='Shear Stress')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Shear Stress Field')
    
    # Save figure
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'shear_stress_field.png'), dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
        
    print(f"Shear stress field plot saved to {config.OUTPUT_DIR}/shear_stress_field.png")
    
    # Plot streamlines
    plt.figure(figsize=(12, 8))
    plt.streamplot(X_np.T, Y_np.T, u_grid.T, v_grid.T, density=2, color='k', linewidth=0.5)
    plt.contourf(X_np, Y_np, np.sqrt(u_grid**2 + v_grid**2), 50, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Velocity Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Flow Streamlines')
    
    # Save figure
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'streamlines.png'), dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
        
    print(f"Streamlines plot saved to {config.OUTPUT_DIR}/streamlines.png")
    
    print("\n" + "="*70)
    print("Advanced flow feature analysis complete")
    print("="*70)

if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Please use main.py to evaluate the model.")

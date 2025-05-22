import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import cfg
from src.model.model import PINN

def interactive_query(model_path=None, config=None):
    """
    Interactive query interface for the trained PINN model
    
    Args:
        model_path: Path to the trained model (optional)
        config: Project configuration (optional, uses global cfg if None)
    """
    if config is None:
        config = cfg
        
    print("\n" + "="*70)
    print("PINN Interactive Query Interface")
    print("="*70)
    
    # Load model
    if model_path is None:
        model_path = os.path.join(config.OUTPUT_DIR, config.MODEL_SAVE_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please train a model first.")
        return
    
    model = PINN.load(model_path, config)
    print(f"Model loaded from {model_path}")
    print(f"Inferred viscosity parameter a: {model.get_inferred_viscosity_param():.6f}")
    
    # Create a grid for visualization
    nx, ny = 100, 50
    x = torch.linspace(config.X_MIN, config.X_MAX, nx, device=config.DEVICE)
    y = torch.linspace(config.Y_MIN, config.Y_MAX, ny, device=config.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    # For unsteady flow, allow time selection
    if config.UNSTEADY_FLOW:
        t_values = torch.linspace(config.T_MIN, config.T_MAX, 10, device=config.DEVICE)
        print(f"\nAvailable time points: {t_values.cpu().numpy()}")
        
        while True:
            try:
                t_input = input("\nEnter a time value (or 'q' to quit): ")
                if t_input.lower() == 'q':
                    break
                
                t_val = float(t_input)
                if t_val < config.T_MIN or t_val > config.T_MAX:
                    print(f"Time value must be between {config.T_MIN} and {config.T_MAX}")
                    continue
                
                # Create time tensor
                t_flat = torch.full_like(x_flat, t_val)
                
                # Query model
                u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat, t_flat)
                
                # Reshape for plotting
                u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
                v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
                p_grid = p_pred.reshape(nx, ny).detach().cpu().numpy()
                X_np = X.detach().cpu().numpy()
                Y_np = Y.detach().cpu().numpy()
                
                # Calculate velocity magnitude
                vel_mag = np.sqrt(u_grid**2 + v_grid**2)
                
                # Plot results
                plot_flow_field(X_np, Y_np, u_grid, v_grid, p_grid, vel_mag, t=t_val, config=config)
                
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Steady flow - just query once
        # Query model
        u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat)
        
        # Reshape for plotting
        u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
        v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
        p_grid = p_pred.reshape(nx, ny).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        
        # Calculate velocity magnitude
        vel_mag = np.sqrt(u_grid**2 + v_grid**2)
        
        # Plot results
        plot_flow_field(X_np, Y_np, u_grid, v_grid, p_grid, vel_mag, config=config)
    
    print("\n" + "="*70)
    print("Interactive query complete")
    print("="*70)

def plot_flow_field(X, Y, u, v, p, vel_mag, t=None, config=None):
    """
    Plot flow field results
    
    Args:
        X, Y: Meshgrid arrays
        u, v: Velocity components
        p: Pressure
        vel_mag: Velocity magnitude
        t: Time value (optional, for unsteady flow)
        config: Project configuration (optional)
    """
    if config is None:
        config = cfg
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Title suffix for unsteady flow
    title_suffix = f" (t = {t:.2f})" if t is not None else ""
    
    # Plot u velocity
    im0 = axs[0, 0].contourf(X, Y, u, 50, cmap='viridis')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].set_title(f'u velocity{title_suffix}')
    plt.colorbar(im0, ax=axs[0, 0])
    
    # Plot v velocity
    im1 = axs[0, 1].contourf(X, Y, v, 50, cmap='viridis')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    axs[0, 1].set_title(f'v velocity{title_suffix}')
    plt.colorbar(im1, ax=axs[0, 1])
    
    # Plot pressure
    im2 = axs[1, 0].contourf(X, Y, p, 50, cmap='viridis')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_title(f'pressure{title_suffix}')
    plt.colorbar(im2, ax=axs[1, 0])
    
    # Plot velocity vectors with magnitude as background
    im3 = axs[1, 1].contourf(X, Y, vel_mag, 50, cmap='viridis')
    # Plot vectors (subsample for clarity)
    stride = 5
    axs[1, 1].quiver(X[::stride, ::stride], Y[::stride, ::stride], 
                     u[::stride, ::stride], v[::stride, ::stride],
                     color='white', scale=25)
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].set_title(f'velocity magnitude and direction{title_suffix}')
    plt.colorbar(im3, ax=axs[1, 1])
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if t is not None:
        filename = f"{config.OUTPUT_DIR}/query_result_t_{t:.2f}.png"
    else:
        filename = f"{config.OUTPUT_DIR}/query_result.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
        
    print(f"Query result saved to {filename}")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Interactive query for PINN model")
    parser.add_argument("--model", type=str, help="Path to trained model file")
    parser.add_argument("--show-plots", action="store_true", help="Show interactive plots")
    
    args = parser.parse_args()
    
    # Update configuration
    if args.show_plots:
        cfg.SHOW_PLOTS_INTERACTIVE = True
    
    # Run interactive query
    interactive_query(model_path=args.model)

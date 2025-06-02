#!/usr/bin/env python3
"""
Interactive PINN Model Inference Script - 3D CSV Export

This script loads a trained PINN model and uses it to infer flow field information
for different simulation parameters, saving comprehensive 3D CSV data for plotting.

Model path is hardcoded to: results/trained_model.pth

Usage:
    python interactive.py
    python interactive.py --reynolds 100 --viscosity-param 0.05
    python interactive.py --interactive
    python interactive.py --model custom/path/to/model.pth  # Override default path
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import cfg, Config
from src.model.model import PINN
from src.generate_data import generate_collocation_points, generate_boundary_points, generate_sparse_data_points

def create_inference_config(reynolds_number=None, nu_base_true=None, a_true=None, 
                          u_max_inlet=None, x_max=None, y_max=None, x_min=None, y_min=None,
                          n_grid_x=100, n_grid_y=50, n_time_slices=5, name="Custom Inference"):
    """
    Create a configuration for inference with specified parameters
    
    Args:
        reynolds_number: Reynolds number for the flow
        nu_base_true: Base viscosity value
        a_true: Viscosity variation parameter (for comparison)
        u_max_inlet: Maximum inlet velocity
        x_max, y_max: Domain dimensions
        x_min, y_min: Domain origin (optional)
        n_grid_x, n_grid_y: Grid resolution for inference
        n_time_slices: Number of time slices (if unsteady)
        name: Configuration name
        
    Returns:
        Configured Config object with inference settings
    """
    # Create new config instance
    inference_config = Config()
    
    # Apply parameters if provided
    if reynolds_number is not None:
        inference_config.REYNOLDS_NUMBER = reynolds_number
    
    if nu_base_true is not None:
        inference_config.NU_BASE_TRUE = nu_base_true
    
    if a_true is not None:
        inference_config.A_TRUE = a_true
    
    if u_max_inlet is not None:
        inference_config.U_MAX_INLET = u_max_inlet
    
    if x_max is not None:
        inference_config.X_MAX = x_max
    
    if y_max is not None:
        inference_config.Y_MAX = y_max
    
    if x_min is not None:
        inference_config.X_MIN = x_min
        
    if y_min is not None:
        inference_config.Y_MIN = y_min
    
    # Store inference-specific parameters
    inference_config.N_GRID_X = n_grid_x
    inference_config.N_GRID_Y = n_grid_y
    inference_config.N_TIME_SLICES = n_time_slices
    inference_config.name = name
    
    return inference_config

def load_trained_model(model_path, inference_config):
    """
    Load trained model and prepare it for inference
    
    Args:
        model_path: Path to the saved model file
        inference_config: Inference configuration object
        
    Returns:
        Tuple of (loaded_model, updated_config)
    """
    print(f"\nLoading trained model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint to inspect saved configuration
    device = inference_config.DEVICE
    checkpoint = torch.load(model_path, map_location=device)
    
    print("Inspecting saved model configuration...")
    
    # Extract saved configuration and update inference config
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        print(f"Saved model configuration found:")
        for key, value in saved_config.items():
            print(f"  {key}: {value}")
        
        # Update critical parameters for model loading
        if 'layers' in saved_config:
            inference_config.PINN_LAYERS = saved_config['layers']
        if 'use_fourier_features' in saved_config:
            inference_config.USE_FOURIER_FEATURES = saved_config['use_fourier_features']
        if 'use_adaptive_weights' in saved_config:
            inference_config.USE_ADAPTIVE_WEIGHTS = saved_config['use_adaptive_weights']
        if 'nu_base' in saved_config:
            inference_config.NU_BASE_TRUE = saved_config['nu_base']
        if 'unsteady' in saved_config:
            inference_config.UNSTEADY_FLOW = saved_config['unsteady']
    else:
        print("Warning: No saved configuration found in checkpoint.")
    
    # Load the model
    model = PINN.load(model_path, inference_config)
    
    print(f"Model loaded successfully!")
    print(f"Model architecture: {inference_config.PINN_LAYERS}")
    print(f"Learned viscosity parameter: {model.get_inferred_viscosity_param():.6f}")
    print(f"Uses Fourier features: {inference_config.USE_FOURIER_FEATURES}")
    print(f"Uses adaptive weights: {inference_config.USE_ADAPTIVE_WEIGHTS}")
    
    return model, inference_config

def infer_3d_flow_field(model, inference_config, save_path):
    """
    Use the trained model to infer 3D flow field data and save as CSV
    
    Args:
        model: Trained PINN model
        inference_config: Inference configuration object
        save_path: Directory to save CSV files
        
    Returns:
        Dictionary containing inferred data information
    """
    print("\n" + "="*70)
    print("Inferring 3D Flow Field Data using Trained Model")
    print("="*70)
    
    # Create high-resolution grid for inference
    nx, ny = inference_config.N_GRID_X, inference_config.N_GRID_Y
    x = torch.linspace(inference_config.X_MIN, inference_config.X_MAX, nx, device=inference_config.DEVICE)
    y = torch.linspace(inference_config.Y_MIN, inference_config.Y_MAX, ny, device=inference_config.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    os.makedirs(save_path, exist_ok=True)
    
    # Get the learned viscosity parameter from the model
    learned_viscosity_param = model.get_inferred_viscosity_param()
    
    print(f"Using learned viscosity parameter: {learned_viscosity_param:.6f}")
    print(f"Grid resolution: {nx} x {ny} = {nx*ny} points")
    
    # For unsteady flow, create multiple time slices
    if inference_config.UNSTEADY_FLOW:
        n_time_slices = inference_config.N_TIME_SLICES
        time_values = torch.linspace(inference_config.T_MIN, inference_config.T_MAX, n_time_slices, device=inference_config.DEVICE)
        
        print(f"Inferring unsteady flow with {n_time_slices} time slices")
        
        # Create comprehensive 3D+time dataset
        all_data = []
        
        for i, t_val in enumerate(time_values):
            print(f"Processing time slice {i+1}/{n_time_slices}: t = {t_val:.3f}")
            
            t_flat = torch.full_like(x_flat, t_val)
            
            # Infer flow field using trained model
            with torch.no_grad():
                u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat, t_flat)
            
            # Convert to numpy
            x_np = x_flat.detach().cpu().numpy().flatten()
            y_np = y_flat.detach().cpu().numpy().flatten()
            t_np = t_flat.detach().cpu().numpy().flatten()
            u_np = u_pred.detach().cpu().numpy().flatten()
            v_np = v_pred.detach().cpu().numpy().flatten()
            p_np = p_pred.detach().cpu().numpy().flatten()
            
            # Calculate derived quantities using learned viscosity
            vel_mag = np.sqrt(u_np**2 + v_np**2)
            nu_np = inference_config.NU_BASE_TRUE + learned_viscosity_param * y_np
            
            # Calculate vorticity and other derived fields
            X_grid = X.detach().cpu().numpy()
            Y_grid = Y.detach().cpu().numpy()
            u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
            v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
            
            dx = (inference_config.X_MAX - inference_config.X_MIN) / (nx - 1)
            dy = (inference_config.Y_MAX - inference_config.Y_MIN) / (ny - 1)
            u_y = np.gradient(u_grid, dy, axis=1)
            v_x = np.gradient(v_grid, dx, axis=0)
            vorticity_grid = v_x - u_y
            vorticity_flat = vorticity_grid.flatten()
            
            # Create data for this time slice
            time_slice_data = {
                'x': x_np,
                'y': y_np,
                't': t_np,
                'u_velocity': u_np,
                'v_velocity': v_np,
                'pressure': p_np,
                'velocity_magnitude': vel_mag,
                'viscosity': nu_np,
                'vorticity': vorticity_flat,
                'reynolds_local': vel_mag * inference_config.U_MAX_INLET / nu_np,
                'time_slice': np.full_like(x_np, i),
                'learned_viscosity_param': np.full_like(x_np, learned_viscosity_param)
            }
            
            all_data.append(time_slice_data)
            
            # Save individual time slice
            df_time = pd.DataFrame(time_slice_data)
            filepath = os.path.join(save_path, f'inferred_flow_3d_t_{t_val:.3f}.csv')
            df_time.to_csv(filepath, index=False)
            print(f"  Saved: inferred_flow_3d_t_{t_val:.3f}.csv")
        
        # Combine all time slices into one comprehensive 4D dataset
        combined_data = {}
        for key in all_data[0].keys():
            combined_data[key] = np.concatenate([data[key] for data in all_data])
        
        df_combined = pd.DataFrame(combined_data)
        filepath = os.path.join(save_path, 'inferred_flow_4d_complete.csv')
        df_combined.to_csv(filepath, index=False)
        print(f"Saved combined 4D dataset: inferred_flow_4d_complete.csv")
        
        return {
            'grid_size': [nx, ny],
            'time_slices': n_time_slices,
            'learned_viscosity_param': learned_viscosity_param,
            'total_points': nx * ny * n_time_slices,
            'files_created': [f'inferred_flow_3d_t_{t:.3f}.csv' for t in time_values.cpu().numpy()] + ['inferred_flow_4d_complete.csv']
        }
        
    else:
        # Steady flow inference
        print("Inferring steady flow field")
        
        # Infer flow field using trained model
        with torch.no_grad():
            u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat)
        
        # Convert to numpy
        x_np = x_flat.detach().cpu().numpy().flatten()
        y_np = y_flat.detach().cpu().numpy().flatten()
        u_np = u_pred.detach().cpu().numpy().flatten()
        v_np = v_pred.detach().cpu().numpy().flatten()
        p_np = p_pred.detach().cpu().numpy().flatten()
        
        # Calculate derived quantities using learned viscosity
        vel_mag = np.sqrt(u_np**2 + v_np**2)
        nu_np = inference_config.NU_BASE_TRUE + learned_viscosity_param * y_np
        
        # Calculate vorticity and other derived fields on the grid
        X_grid = X.detach().cpu().numpy()
        Y_grid = Y.detach().cpu().numpy()
        u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
        v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
        p_grid = p_pred.reshape(nx, ny).detach().cpu().numpy()
        
        # Calculate gradients for derived quantities
        dx = (inference_config.X_MAX - inference_config.X_MIN) / (nx - 1)
        dy = (inference_config.Y_MAX - inference_config.Y_MIN) / (ny - 1)
        u_y = np.gradient(u_grid, dy, axis=1)
        v_x = np.gradient(v_grid, dx, axis=0)
        vorticity_grid = v_x - u_y
        vorticity_flat = vorticity_grid.flatten()
        
        # Calculate divergence (mass conservation check)
        u_x = np.gradient(u_grid, dx, axis=0)
        v_y = np.gradient(v_grid, dy, axis=1)
        divergence_grid = u_x + v_y
        divergence_flat = divergence_grid.flatten()
        
        # Calculate shear stress
        shear_stress_grid = nu_np.reshape(nx, ny) * (u_y + v_x)
        shear_stress_flat = shear_stress_grid.flatten()
        
        # Create comprehensive 3D dataset
        inferred_data = {
            'x': x_np,
            'y': y_np,
            'z_velocity_magnitude': vel_mag,  # Use velocity magnitude as 3rd dimension
            'u_velocity': u_np,
            'v_velocity': v_np,
            'pressure': p_np,
            'velocity_magnitude': vel_mag,
            'viscosity': nu_np,
            'vorticity': vorticity_flat,
            'divergence': divergence_flat,
            'shear_stress': shear_stress_flat,
            'reynolds_local': vel_mag * inference_config.U_MAX_INLET / nu_np,
            'learned_viscosity_param': np.full_like(x_np, learned_viscosity_param),
            'grid_i': np.repeat(np.arange(nx), ny),  # Grid indices for reconstruction
            'grid_j': np.tile(np.arange(ny), nx)
        }
        
        # Save main 3D flow field
        df = pd.DataFrame(inferred_data)
        filepath = os.path.join(save_path, 'inferred_flow_3d_complete.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved complete 3D dataset: inferred_flow_3d_complete.csv")
        
        # Save grid-structured data for easy plotting
        grid_data = {
            'x_coordinates': X_grid.tolist(),
            'y_coordinates': Y_grid.tolist(),
            'u_velocity_grid': u_grid.tolist(),
            'v_velocity_grid': v_grid.tolist(),
            'pressure_grid': p_grid.tolist(),
            'velocity_magnitude_grid': np.sqrt(u_grid**2 + v_grid**2).tolist(),
            'vorticity_grid': vorticity_grid.tolist(),
            'viscosity_grid': (inference_config.NU_BASE_TRUE + learned_viscosity_param * Y_grid).tolist()
        }
        
        with open(os.path.join(save_path, 'inferred_flow_3d_grid.json'), 'w') as f:
            json.dump(grid_data, f, indent=2)
        print(f"Saved grid-structured data: inferred_flow_3d_grid.json")
        
        # Save separate files for each field (for specialized plotting)
        field_files = []
        for field_name in ['u_velocity', 'v_velocity', 'pressure', 'velocity_magnitude', 'vorticity', 'viscosity']:
            field_df = pd.DataFrame({
                'x': x_np,
                'y': y_np,
                'z': inferred_data[field_name],
                'value': inferred_data[field_name],
                'field_name': field_name,
                'learned_viscosity_param': learned_viscosity_param
            })
            field_filepath = os.path.join(save_path, f'inferred_{field_name}_3d.csv')
            field_df.to_csv(field_filepath, index=False)
            field_files.append(f'inferred_{field_name}_3d.csv')
            print(f"  Saved field: inferred_{field_name}_3d.csv")
        
        return {
            'grid_size': [nx, ny],
            'steady_flow': True,
            'learned_viscosity_param': learned_viscosity_param,
            'total_points': nx * ny,
            'files_created': ['inferred_flow_3d_complete.csv', 'inferred_flow_3d_grid.json'] + field_files
        }

def infer_boundary_analysis(model, inference_config, save_path):
    """
    Use model to infer boundary condition behavior and save as CSV
    """
    print("\nInferring boundary behavior...")
    
    # Generate high-resolution boundary points
    n_boundary_points = 200
    
    # Inlet boundary analysis
    x_inlet = torch.full((n_boundary_points, 1), inference_config.X_MIN, device=inference_config.DEVICE)
    y_inlet = torch.linspace(inference_config.Y_MIN, inference_config.Y_MAX, n_boundary_points, device=inference_config.DEVICE).unsqueeze(1)
    
    with torch.no_grad():
        u_inlet, v_inlet, p_inlet = model.uvp(x_inlet, y_inlet)
    
    inlet_data = pd.DataFrame({
        'x': x_inlet.cpu().numpy().flatten(),
        'y': y_inlet.cpu().numpy().flatten(),
        'u_velocity': u_inlet.cpu().numpy().flatten(),
        'v_velocity': v_inlet.cpu().numpy().flatten(),
        'pressure': p_inlet.cpu().numpy().flatten(),
        'boundary_type': 'inlet',
        'learned_viscosity_param': model.get_inferred_viscosity_param()
    })
    
    # Outlet boundary analysis
    x_outlet = torch.full((n_boundary_points, 1), inference_config.X_MAX, device=inference_config.DEVICE)
    y_outlet = torch.linspace(inference_config.Y_MIN, inference_config.Y_MAX, n_boundary_points, device=inference_config.DEVICE).unsqueeze(1)
    
    with torch.no_grad():
        u_outlet, v_outlet, p_outlet = model.uvp(x_outlet, y_outlet)
    
    outlet_data = pd.DataFrame({
        'x': x_outlet.cpu().numpy().flatten(),
        'y': y_outlet.cpu().numpy().flatten(),
        'u_velocity': u_outlet.cpu().numpy().flatten(),
        'v_velocity': v_outlet.cpu().numpy().flatten(),
        'pressure': p_outlet.cpu().numpy().flatten(),
        'boundary_type': 'outlet',
        'learned_viscosity_param': model.get_inferred_viscosity_param()
    })
    
    # Wall boundaries
    x_wall_bottom = torch.linspace(inference_config.X_MIN, inference_config.X_MAX, n_boundary_points, device=inference_config.DEVICE).unsqueeze(1)
    y_wall_bottom = torch.full((n_boundary_points, 1), inference_config.Y_MIN, device=inference_config.DEVICE)
    
    with torch.no_grad():
        u_wall_bottom, v_wall_bottom, p_wall_bottom = model.uvp(x_wall_bottom, y_wall_bottom)
    
    wall_bottom_data = pd.DataFrame({
        'x': x_wall_bottom.cpu().numpy().flatten(),
        'y': y_wall_bottom.cpu().numpy().flatten(),
        'u_velocity': u_wall_bottom.cpu().numpy().flatten(),
        'v_velocity': v_wall_bottom.cpu().numpy().flatten(),
        'pressure': p_wall_bottom.cpu().numpy().flatten(),
        'boundary_type': 'wall_bottom',
        'learned_viscosity_param': model.get_inferred_viscosity_param()
    })
    
    # Combine all boundary data
    boundary_data = pd.concat([inlet_data, outlet_data, wall_bottom_data], ignore_index=True)
    
    # Save boundary analysis
    filepath = os.path.join(save_path, 'inferred_boundary_analysis.csv')
    boundary_data.to_csv(filepath, index=False)
    print(f"Saved boundary analysis: inferred_boundary_analysis.csv")
    
    return boundary_data

def infer_centerline_analysis(model, inference_config, save_path):
    """
    Use model to infer centerline flow behavior
    """
    print("\nInferring centerline flow behavior...")
    
    # Create centerline points
    n_points = 300
    x_centerline = torch.linspace(inference_config.X_MIN, inference_config.X_MAX, n_points, device=inference_config.DEVICE).unsqueeze(1)
    y_centerline = torch.full_like(x_centerline, (inference_config.Y_MIN + inference_config.Y_MAX) / 2)
    
    # Infer flow along centerline using trained model
    with torch.no_grad():
        u_pred, v_pred, p_pred = model.uvp(x_centerline, y_centerline)
    
    # Calculate viscosity along centerline using learned parameter
    learned_viscosity_param = model.get_inferred_viscosity_param()
    nu_centerline = inference_config.NU_BASE_TRUE + learned_viscosity_param * y_centerline
    
    # Create centerline analysis dataset
    centerline_data = pd.DataFrame({
        'x': x_centerline.detach().cpu().numpy().flatten(),
        'y': y_centerline.detach().cpu().numpy().flatten(),
        'u_velocity': u_pred.detach().cpu().numpy().flatten(),
        'v_velocity': v_pred.detach().cpu().numpy().flatten(),
        'pressure': p_pred.detach().cpu().numpy().flatten(),
        'velocity_magnitude': np.sqrt(u_pred.detach().cpu().numpy().flatten()**2 + v_pred.detach().cpu().numpy().flatten()**2),
        'viscosity': nu_centerline.detach().cpu().numpy().flatten(),
        'learned_viscosity_param': learned_viscosity_param,
        'distance_from_inlet': x_centerline.detach().cpu().numpy().flatten() - inference_config.X_MIN
    })
    
    filepath = os.path.join(save_path, 'inferred_centerline_analysis.csv')
    centerline_data.to_csv(filepath, index=False)
    print(f"Saved centerline analysis: inferred_centerline_analysis.csv")
    
    return centerline_data

def infer_viscosity_profile(model, inference_config, save_path):
    """
    Use model to infer viscosity profile across domain
    """
    print("\nInferring viscosity profile...")
    
    # Create y-coordinate range
    n_points = 200
    y_values = np.linspace(inference_config.Y_MIN, inference_config.Y_MAX, n_points)
    
    # Get learned viscosity parameter
    learned_viscosity_param = model.get_inferred_viscosity_param()
    
    # Calculate learned viscosity profile
    nu_learned = inference_config.NU_BASE_TRUE + learned_viscosity_param * y_values
    
    # Compare with reference (if a_true is provided)
    if hasattr(inference_config, 'A_TRUE') and inference_config.A_TRUE is not None:
        nu_reference = inference_config.NU_BASE_TRUE + inference_config.A_TRUE * y_values
        absolute_error = np.abs(nu_learned - nu_reference)
        relative_error = absolute_error / nu_reference * 100
    else:
        nu_reference = np.full_like(nu_learned, np.nan)
        absolute_error = np.full_like(nu_learned, np.nan)
        relative_error = np.full_like(nu_learned, np.nan)
    
    # Create viscosity profile dataset
    viscosity_data = pd.DataFrame({
        'y': y_values,
        'viscosity_learned': nu_learned,
        'viscosity_reference': nu_reference,
        'absolute_error': absolute_error,
        'relative_error_percent': relative_error,
        'learned_viscosity_param': learned_viscosity_param,
        'reference_viscosity_param': getattr(inference_config, 'A_TRUE', np.nan),
        'base_viscosity': inference_config.NU_BASE_TRUE
    })
    
    filepath = os.path.join(save_path, 'inferred_viscosity_profile.csv')
    viscosity_data.to_csv(filepath, index=False)
    print(f"Saved viscosity profile: inferred_viscosity_profile.csv")
    
    return viscosity_data

def export_inference_summary(model, inference_config, inference_results, save_path):
    """
    Export summary of inference results
    """
    learned_viscosity_param = model.get_inferred_viscosity_param()
    
    summary_data = {
        'inference_info': {
            'model_architecture': inference_config.PINN_LAYERS,
            'learned_viscosity_param': float(learned_viscosity_param),
            'base_viscosity': float(inference_config.NU_BASE_TRUE),
            'reference_viscosity_param': float(getattr(inference_config, 'A_TRUE', np.nan)),
            'reynolds_number': float(inference_config.REYNOLDS_NUMBER),
            'max_inlet_velocity': float(inference_config.U_MAX_INLET),
            'domain_bounds': {
                'x_min': float(inference_config.X_MIN),
                'x_max': float(inference_config.X_MAX),
                'y_min': float(inference_config.Y_MIN),
                'y_max': float(inference_config.Y_MAX)
            },
            'grid_resolution': inference_results['grid_size'],
            'total_inference_points': inference_results['total_points'],
            'inference_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration_name': inference_config.name
        },
        'files_created': inference_results['files_created'],
        'model_features': {
            'uses_fourier_features': inference_config.USE_FOURIER_FEATURES,
            'uses_adaptive_weights': inference_config.USE_ADAPTIVE_WEIGHTS,
            'unsteady_flow': inference_config.UNSTEADY_FLOW
        }
    }
    
    # Calculate error if reference is available
    if hasattr(inference_config, 'A_TRUE') and not np.isnan(inference_config.A_TRUE):
        error = abs(learned_viscosity_param - inference_config.A_TRUE)
        rel_error = error / inference_config.A_TRUE * 100
        summary_data['inference_accuracy'] = {
            'absolute_error': float(error),
            'relative_error_percent': float(rel_error)
        }
    
    # Save summary
    summary_path = os.path.join(save_path, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Saved inference summary: inference_summary.json")
    
    return summary_data

def run_inference_session(model, inference_config, model_path, output_suffix="inference"):
    """
    Run comprehensive inference session using trained model
    """
    print("\n" + "="*70)
    print(f"Running Inference Session: {inference_config.name}")
    print("="*70)
    
    # Print inference configuration
    print("\nInference Configuration:")
    config_params = [
        ('Model Path', model_path),
        ('Learned Viscosity Parameter', f"{model.get_inferred_viscosity_param():.6f}"),
        ('Reynolds Number', inference_config.REYNOLDS_NUMBER),
        ('Base Viscosity', inference_config.NU_BASE_TRUE),
        ('Reference Viscosity Parameter', getattr(inference_config, 'A_TRUE', 'Not provided')),
        ('Max Inlet Velocity', inference_config.U_MAX_INLET),
        ('Domain Width', inference_config.X_MAX),
        ('Domain Height', inference_config.Y_MAX),
        ('Grid Resolution', f"{inference_config.N_GRID_X} x {inference_config.N_GRID_Y}"),
        ('Fourier Features', inference_config.USE_FOURIER_FEATURES),
        ('Adaptive Weights', inference_config.USE_ADAPTIVE_WEIGHTS)
    ]
    
    for param_name, param_value in config_params:
        print(f"  {param_name}: {param_value}")
    
    # Create output directory
    base_output_dir = inference_config.OUTPUT_DIR
    inference_output_dir = os.path.join(base_output_dir, f"inference_{output_suffix}")
    inference_config.OUTPUT_DIR = inference_output_dir
    os.makedirs(inference_output_dir, exist_ok=True)
    
    print(f"\nInference results will be saved to: {inference_output_dir}")
    
    # Run comprehensive inference
    start_time = time.time()
    
    print("\n1. Inferring 3D flow field data...")
    flow_results = infer_3d_flow_field(model, inference_config, inference_output_dir)
    
    print("\n2. Inferring boundary behavior...")
    boundary_results = infer_boundary_analysis(model, inference_config, inference_output_dir)
    
    print("\n3. Inferring centerline flow...")
    centerline_results = infer_centerline_analysis(model, inference_config, inference_output_dir)
    
    print("\n4. Inferring viscosity profile...")
    viscosity_results = infer_viscosity_profile(model, inference_config, inference_output_dir)
    
    print("\n5. Exporting inference summary...")
    summary_results = export_inference_summary(model, inference_config, flow_results, inference_output_dir)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("Inference Session Complete!")
    print("="*70)
    
    # Print results summary
    print(f"\nInference Results:")
    print(f"  Learned viscosity parameter: {model.get_inferred_viscosity_param():.6f}")
    print(f"  Total inference points: {flow_results['total_points']:,}")
    print(f"  Grid resolution: {flow_results['grid_size'][0]} x {flow_results['grid_size'][1]}")
    print(f"  Processing time: {total_time:.2f}s")
    
    if 'inference_accuracy' in summary_results:
        print(f"  Inference accuracy: {summary_results['inference_accuracy']['relative_error_percent']:.2f}% error")
    
    print(f"\nGenerated CSV files for 3D plotting:")
    for filename in flow_results['files_created']:
        print(f"  - {filename}")
    
    return {
        'flow_results': flow_results,
        'boundary_results': boundary_results,
        'centerline_results': centerline_results,
        'viscosity_results': viscosity_results,
        'summary_results': summary_results,
        'total_time': total_time
    }

def get_interactive_parameters():
    """Interactive function to get inference parameters from user"""
    print("\n" + "="*60)
    print("Interactive Inference Parameter Setup")
    print("="*60)
    print("Enter parameters for inference (press Enter to use defaults)")
    
    params = {}
    
    # Physics parameters
    print("\n--- Physics Parameters ---")
    reynolds = input(f"Reynolds Number (default: {cfg.REYNOLDS_NUMBER}): ")
    if reynolds.strip():
        params['reynolds_number'] = float(reynolds)
    
    nu_base = input(f"Base Viscosity (default: {cfg.NU_BASE_TRUE}): ")
    if nu_base.strip():
        params['nu_base_true'] = float(nu_base)
    
    u_max = input(f"Maximum Inlet Velocity (default: {cfg.U_MAX_INLET}): ")
    if u_max.strip():
        params['u_max_inlet'] = float(u_max)
    
    # Domain parameters
    print("\n--- Domain Parameters ---")
    x_max = input(f"Domain Width X_MAX (default: {cfg.X_MAX}): ")
    if x_max.strip():
        params['x_max'] = float(x_max)
    
    y_max = input(f"Domain Height Y_MAX (default: {cfg.Y_MAX}): ")
    if y_max.strip():
        params['y_max'] = float(y_max)
    
    # Grid resolution
    print("\n--- Grid Resolution ---")
    n_grid_x = input(f"Grid points in X direction (default: 100): ")
    if n_grid_x.strip():
        params['n_grid_x'] = int(n_grid_x)
    
    n_grid_y = input(f"Grid points in Y direction (default: 50): ")
    if n_grid_y.strip():
        params['n_grid_y'] = int(n_grid_y)
    
    # Reference parameters (for comparison)
    print("\n--- Reference Parameters (for comparison) ---")
    a_true = input(f"Reference viscosity parameter 'a' (optional): ")
    if a_true.strip():
        params['a_true'] = float(a_true)
    
    # Session name
    session_name = input("\nInference session name (default: 'Interactive Inference'): ")
    if session_name.strip():
        params['name'] = session_name
    else:
        params['name'] = 'Interactive Inference'
    
    return params

def run_multiple_inference_scenarios(model_path, inference_scenarios, output_base="multi_inference"):
    """
    Run multiple inference scenarios on the same trained model
    """
    print(f"\n" + "="*70)
    print(f"Running Multiple Inference Scenarios")
    print(f"Model: {model_path}")
    print(f"Number of scenarios: {len(inference_scenarios)}")
    print("="*70)
    
    all_results = []
    
    for i, scenario_params in enumerate(inference_scenarios):
        print(f"\n{'='*50}")
        print(f"Inference Scenario {i+1}/{len(inference_scenarios)}")
        print(f"{'='*50}")
        
        # Create inference configuration
        inference_config = create_inference_config(**scenario_params)
        
        # Load model
        model, inference_config = load_trained_model(model_path, inference_config)
        
        # Run inference
        output_suffix = f"{output_base}_{i+1:02d}"
        if 'name' in scenario_params:
            output_suffix += f"_{scenario_params['name'].replace(' ', '_')}"
        
        results = run_inference_session(model, inference_config, model_path, output_suffix)
        
        # Store results
        scenario_result = {
            'scenario_index': i+1,
            'scenario_params': scenario_params,
            'inference_results': results,
            'output_suffix': output_suffix
        }
        all_results.append(scenario_result)
    
    # Generate summary report
    generate_multi_inference_summary(all_results, model_path, output_base)
    
    return all_results

def generate_multi_inference_summary(all_results, model_path, output_base):
    """Generate summary report for multiple inference scenarios"""
    summary_dir = os.path.join(cfg.OUTPUT_DIR, f"summary_{output_base}")
    os.makedirs(summary_dir, exist_ok=True)
    
    print(f"\n" + "="*70)
    print("Multi-Inference Summary")
    print("="*70)
    
    # Prepare summary data
    summary_data = []
    
    for result in all_results:
        scenario_summary = {
            'scenario_index': result['scenario_index'],
            'scenario_name': result['scenario_params'].get('name', f"Scenario {result['scenario_index']}"),
            'learned_viscosity_param': result['inference_results']['summary_results']['inference_info']['learned_viscosity_param'],
            'reynolds_number': result['inference_results']['summary_results']['inference_info']['reynolds_number'],
            'total_points': result['inference_results']['summary_results']['inference_info']['total_inference_points'],
            'grid_resolution': f"{result['inference_results']['flow_results']['grid_size'][0]}x{result['inference_results']['flow_results']['grid_size'][1]}",
            'processing_time': result['inference_results']['total_time'],
            'output_directory': result['output_suffix']
        }
        
        # Add accuracy if available
        if 'inference_accuracy' in result['inference_results']['summary_results']:
            scenario_summary['relative_error_percent'] = result['inference_results']['summary_results']['inference_accuracy']['relative_error_percent']
        else:
            scenario_summary['relative_error_percent'] = None
        
        summary_data.append(scenario_summary)
        
        print(f"Scenario {result['scenario_index']}: {scenario_summary['scenario_name']}")
        print(f"  Learned viscosity param: {scenario_summary['learned_viscosity_param']:.6f}")
        print(f"  Points processed: {scenario_summary['total_points']:,}")
    
    # Save summary as CSV
    df = pd.DataFrame(summary_data)
    summary_path = os.path.join(summary_dir, 'multi_inference_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV saved to: {summary_path}")
    
    # Save detailed JSON summary
    json_data = {
        'model_path': model_path,
        'inference_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_scenarios': len(all_results),
        'scenario_results': summary_data
    }
    
    json_path = os.path.join(summary_dir, 'multi_inference_summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Summary JSON saved to: {json_path}")

def main():
    """Main function for inference scenarios"""
    # Hardcoded model path
    DEFAULT_MODEL_PATH = "/home/brand/pinn_viscosity/backend/results/trained_model.pth"
    
    parser = argparse.ArgumentParser(description="Interactive PINN Model Inference - 3D CSV Export")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help=f"Path to the trained model file (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--interactive", action="store_true", help="Use interactive mode for parameter selection")
    
    # Physics parameters
    parser.add_argument("--reynolds", type=float, help="Reynolds number")
    parser.add_argument("--viscosity-base", type=float, help="Base viscosity")
    parser.add_argument("--viscosity-param-ref", type=float, help="Reference viscosity parameter 'a' (for comparison)")
    parser.add_argument("--inlet-velocity", type=float, help="Maximum inlet velocity")
    
    # Domain parameters
    parser.add_argument("--domain-width", type=float, help="Domain width (X_MAX)")
    parser.add_argument("--domain-height", type=float, help="Domain height (Y_MAX)")
    
    # Grid parameters
    parser.add_argument("--grid-x", type=int, default=100, help="Grid points in X direction")
    parser.add_argument("--grid-y", type=int, default=50, help="Grid points in Y direction")
    
    # Output options
    parser.add_argument("--output-suffix", default="inference", help="Output directory suffix")
    
    # Predefined scenarios
    parser.add_argument("--run-scenarios", action="store_true", help="Run predefined inference scenarios")
    
    args = parser.parse_args()
    
    print(f"Using trained model: {args.model}")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please ensure the model has been trained and saved to this location.")
        print("You can train a model by running: python main.py")
        return
    
    if args.interactive:
        # Interactive mode
        params = get_interactive_parameters()
        inference_config = create_inference_config(**params)
        model, inference_config = load_trained_model(args.model, inference_config)
        
        run_inference_session(model, inference_config, args.model, args.output_suffix)
    
    elif args.run_scenarios:
        # Run multiple predefined scenarios
        inference_scenarios = [
            # Low Reynolds number scenario
            {
                'reynolds_number': 10,
                'a_true': 0.02,
                'nu_base_true': 0.05,
                'u_max_inlet': 0.5,
                'n_grid_x': 150,
                'n_grid_y': 75,
                'name': 'Low_Re_HighRes'
            },
            # Medium Reynolds number scenario
            {
                'reynolds_number': 100,
                'a_true': 0.05,
                'nu_base_true': 0.01,
                'u_max_inlet': 1.0,
                'n_grid_x': 200,
                'n_grid_y': 100,
                'name': 'Medium_Re_HighRes'
            },
            # High Reynolds number scenario
            {
                'reynolds_number': 200,
                'a_true': 0.08,
                'nu_base_true': 0.005,
                'u_max_inlet': 2.0,
                'n_grid_x': 250,
                'n_grid_y': 125,
                'name': 'High_Re_UltraRes'
            },
            # Wide domain scenario
            {
                'reynolds_number': 50,
                'a_true': 0.05,
                'x_max': 4.0,
                'n_grid_x': 300,
                'n_grid_y': 75,
                'name': 'Wide_Domain_HighRes'
            },
            # Fine grid scenario
            {
                'reynolds_number': 75,
                'a_true': 0.06,
                'n_grid_x': 400,
                'n_grid_y': 200,
                'name': 'Ultra_Fine_Grid'
            }
        ]
        
        run_multiple_inference_scenarios(args.model, inference_scenarios, "scenarios")
    
    else:
        # Single inference with command line parameters
        params = {}
        
        # Map command line arguments to parameters
        if args.reynolds is not None:
            params['reynolds_number'] = args.reynolds
        if args.viscosity_base is not None:
            params['nu_base_true'] = args.viscosity_base
        if args.viscosity_param_ref is not None:
            params['a_true'] = args.viscosity_param_ref
        if args.inlet_velocity is not None:
            params['u_max_inlet'] = args.inlet_velocity
        if args.domain_width is not None:
            params['x_max'] = args.domain_width
        if args.domain_height is not None:
            params['y_max'] = args.domain_height
        
        params['n_grid_x'] = args.grid_x
        params['n_grid_y'] = args.grid_y
        params['name'] = 'Command_Line_Inference'
        
        # Create and run inference
        inference_config = create_inference_config(**params)
        model, inference_config = load_trained_model(args.model, inference_config)
        
        run_inference_session(model, inference_config, args.model, args.output_suffix)

if __name__ == "__main__":
    main()
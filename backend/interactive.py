#!/usr/bin/env python3
"""
Interactive PINN Model Testing Script - CSV Export Version

This script allows you to load a trained PINN model and test it with different
domain and physics parameters, saving results as CSV files instead of graphs.

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
from src.model.evaluate_model import evaluate_model
from src.generate_data import generate_collocation_points, generate_boundary_points, generate_sparse_data_points

def create_test_config(reynolds_number=None, nu_base_true=None, a_true=None, 
                      u_max_inlet=None, x_max=None, y_max=None, x_min=None, y_min=None,
                      use_fourier_features=None, fourier_scale=None, 
                      use_adaptive_weights=None, use_adaptive_sampling=None, 
                      use_curriculum_learning=None, n_collocation=None, 
                      n_boundary=None, n_data_sparse=None, pinn_layers=None,
                      learning_rate=None, epochs=None, name="Custom Test"):
    """
    Create a test configuration with specified parameters
    
    Args:
        reynolds_number: Reynolds number for the flow
        nu_base_true: Base viscosity value
        a_true: True viscosity variation parameter
        u_max_inlet: Maximum inlet velocity
        x_max, y_max: Domain dimensions
        x_min, y_min: Domain origin (optional)
        use_fourier_features: Enable Fourier feature embeddings
        fourier_scale: Scale for Fourier features
        use_adaptive_weights: Enable adaptive loss weighting
        use_adaptive_sampling: Enable adaptive collocation sampling
        use_curriculum_learning: Enable curriculum learning
        n_collocation: Number of collocation points
        n_boundary: Number of boundary points per edge
        n_data_sparse: Number of sparse data points
        pinn_layers: Network architecture
        learning_rate: Learning rate (for display)
        epochs: Training epochs (for display)
        name: Configuration name
        
    Returns:
        Configured Config object
    """
    # Create new config instance
    test_config = Config()
    
    # Apply parameters if provided
    if reynolds_number is not None:
        test_config.REYNOLDS_NUMBER = reynolds_number
    
    if nu_base_true is not None:
        test_config.NU_BASE_TRUE = nu_base_true
    
    if a_true is not None:
        test_config.A_TRUE = a_true
    
    if u_max_inlet is not None:
        test_config.U_MAX_INLET = u_max_inlet
    
    if x_max is not None:
        test_config.X_MAX = x_max
    
    if y_max is not None:
        test_config.Y_MAX = y_max
    
    if x_min is not None:
        test_config.X_MIN = x_min
        
    if y_min is not None:
        test_config.Y_MIN = y_min
    
    if use_fourier_features is not None:
        test_config.USE_FOURIER_FEATURES = use_fourier_features
    
    if fourier_scale is not None:
        test_config.FOURIER_SCALE = fourier_scale
    
    if use_adaptive_weights is not None:
        test_config.USE_ADAPTIVE_WEIGHTS = use_adaptive_weights
    
    if use_adaptive_sampling is not None:
        test_config.USE_ADAPTIVE_SAMPLING = use_adaptive_sampling
    
    if use_curriculum_learning is not None:
        test_config.USE_CURRICULUM_LEARNING = use_curriculum_learning
    
    if n_collocation is not None:
        test_config.N_COLLOCATION = n_collocation
    
    if n_boundary is not None:
        test_config.N_BOUNDARY = n_boundary
    
    if n_data_sparse is not None:
        test_config.N_DATA_SPARSE = n_data_sparse
    
    if pinn_layers is not None:
        test_config.PINN_LAYERS = pinn_layers
    
    if learning_rate is not None:
        test_config.LEARNING_RATE = learning_rate
    
    if epochs is not None:
        test_config.EPOCHS = epochs
    
    # Store name for reporting
    test_config.name = name
    
    return test_config

def inspect_saved_model(model_path):
    """
    Inspect a saved model's configuration without loading it
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dictionary containing model information
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_info = {
        'model_path': model_path,
        'has_config': 'config' in checkpoint,
        'has_model_state': 'model_state_dict' in checkpoint,
        'inferred_a_param': checkpoint.get('a_param', 'Not found')
    }
    
    if 'config' in checkpoint:
        model_info['saved_config'] = checkpoint['config']
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model_info['state_dict_keys'] = list(state_dict.keys())
        model_info['has_adaptive_weights'] = any('log_weight' in key for key in state_dict.keys())
        model_info['has_fourier_features'] = any('fourier_transform' in key for key in state_dict.keys())
        
        # Try to infer network architecture from state dict
        layer_keys = [key for key in state_dict.keys() if 'net.net.' in key and '.weight' in key]
        if layer_keys:
            model_info['inferred_layers'] = []
            for key in sorted(layer_keys):
                layer_shape = state_dict[key].shape
                model_info['inferred_layers'].append(f"{key}: {list(layer_shape)}")
    
    return model_info

def load_model_with_config(model_path, test_config):
    """
    Load model and prepare it for testing with the given configuration
    
    Args:
        model_path: Path to the saved model file
        test_config: Test configuration object
        
    Returns:
        Tuple of (loaded_model, test_config)
    """
    print(f"\nLoading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint to inspect saved configuration
    device = test_config.DEVICE
    checkpoint = torch.load(model_path, map_location=device)
    
    print("Inspecting saved model configuration...")
    
    # Extract saved configuration
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        print(f"Saved model configuration found:")
        for key, value in saved_config.items():
            print(f"  {key}: {value}")
        
        # Update test_config to match the saved model's architecture
        # These are critical for model loading
        if 'layers' in saved_config:
            test_config.PINN_LAYERS = saved_config['layers']
        if 'use_fourier_features' in saved_config:
            test_config.USE_FOURIER_FEATURES = saved_config['use_fourier_features']
        if 'use_adaptive_weights' in saved_config:
            test_config.USE_ADAPTIVE_WEIGHTS = saved_config['use_adaptive_weights']
        
        # Update domain bounds from saved model (these affect viscosity calculation)
        if 'nu_base' in saved_config:
            test_config.NU_BASE_TRUE = saved_config['nu_base']
        if 'x_min' in saved_config:
            test_config.X_MIN = saved_config['x_min']
        if 'x_max' in saved_config and not hasattr(test_config, '_user_set_x_max'):
            test_config.X_MAX = saved_config['x_max']
        if 'y_min' in saved_config:
            test_config.Y_MIN = saved_config['y_min']
        if 'y_max' in saved_config and not hasattr(test_config, '_user_set_y_max'):
            test_config.Y_MAX = saved_config['y_max']
        if 'unsteady' in saved_config:
            test_config.UNSTEADY_FLOW = saved_config['unsteady']
    else:
        print("Warning: No saved configuration found in checkpoint. Using current test config.")
    
    # Load the model with the updated configuration
    model = PINN.load(model_path, test_config)
    
    print(f"Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model architecture: {test_config.PINN_LAYERS}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Uses Fourier features: {test_config.USE_FOURIER_FEATURES}")
    print(f"Uses adaptive weights: {test_config.USE_ADAPTIVE_WEIGHTS}")
    
    return model, test_config

def export_flow_field_csv(model, test_config, save_path):
    """
    Export flow field data to CSV files
    
    Args:
        model: Trained PINN model
        test_config: Test configuration object
        save_path: Directory to save CSV files
        
    Returns:
        Dictionary containing exported data information
    """
    print("\n" + "="*70)
    print("Exporting Flow Field Data to CSV")
    print("="*70)
    
    # Create high-resolution grid for visualization
    nx, ny = 100, 50
    x = torch.linspace(test_config.X_MIN, test_config.X_MAX, nx, device=test_config.DEVICE)
    y = torch.linspace(test_config.Y_MIN, test_config.Y_MAX, ny, device=test_config.DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    os.makedirs(save_path, exist_ok=True)
    
    # For unsteady flow, create multiple time slices
    if test_config.UNSTEADY_FLOW:
        n_time_slices = 5
        time_values = torch.linspace(test_config.T_MIN, test_config.T_MAX, n_time_slices, device=test_config.DEVICE)
        
        for i, t_val in enumerate(time_values):
            t_flat = torch.full_like(x_flat, t_val)
            
            # Get predictions
            u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat, t_flat)
            
            # Convert to numpy
            x_np = x_flat.detach().cpu().numpy().flatten()
            y_np = y_flat.detach().cpu().numpy().flatten()
            t_np = t_flat.detach().cpu().numpy().flatten()
            u_np = u_pred.detach().cpu().numpy().flatten()
            v_np = v_pred.detach().cpu().numpy().flatten()
            p_np = p_pred.detach().cpu().numpy().flatten()
            
            # Calculate derived quantities
            vel_mag = np.sqrt(u_np**2 + v_np**2)
            nu_np = test_config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * y_np
            
            # Create DataFrame
            df = pd.DataFrame({
                'x': x_np,
                'y': y_np,
                't': t_np,
                'u_velocity': u_np,
                'v_velocity': v_np,
                'pressure': p_np,
                'velocity_magnitude': vel_mag,
                'viscosity': nu_np
            })
            
            # Save to CSV
            filename = f'flow_field_t_{t_val:.3f}.csv'
            filepath = os.path.join(save_path, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved time slice t={t_val:.3f} to {filename}")
            
    else:
        # Steady flow
        u_pred, v_pred, p_pred = model.uvp(x_flat, y_flat)
        
        # Convert to numpy
        x_np = x_flat.detach().cpu().numpy().flatten()
        y_np = y_flat.detach().cpu().numpy().flatten()
        u_np = u_pred.detach().cpu().numpy().flatten()
        v_np = v_pred.detach().cpu().numpy().flatten()
        p_np = p_pred.detach().cpu().numpy().flatten()
        
        # Calculate derived quantities
        vel_mag = np.sqrt(u_np**2 + v_np**2)
        nu_np = test_config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * y_np
        
        # Calculate vorticity on the grid
        X_grid = X.detach().cpu().numpy()
        Y_grid = Y.detach().cpu().numpy()
        u_grid = u_pred.reshape(nx, ny).detach().cpu().numpy()
        v_grid = v_pred.reshape(nx, ny).detach().cpu().numpy()
        
        # Calculate gradients for vorticity
        dx = (test_config.X_MAX - test_config.X_MIN) / (nx - 1)
        dy = (test_config.Y_MAX - test_config.Y_MIN) / (ny - 1)
        u_y = np.gradient(u_grid, dy, axis=1)
        v_x = np.gradient(v_grid, dx, axis=0)
        vorticity_grid = v_x - u_y
        vorticity_flat = vorticity_grid.flatten()
        
        # Calculate divergence (mass conservation check)
        u_x = np.gradient(u_grid, dx, axis=0)
        v_y = np.gradient(v_grid, dy, axis=1)
        divergence_grid = u_x + v_y
        divergence_flat = divergence_grid.flatten()
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': x_np,
            'y': y_np,
            'u_velocity': u_np,
            'v_velocity': v_np,
            'pressure': p_np,
            'velocity_magnitude': vel_mag,
            'viscosity': nu_np,
            'vorticity': vorticity_flat,
            'divergence': divergence_flat
        })
        
        # Save main flow field
        filepath = os.path.join(save_path, 'flow_field.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved steady flow field to flow_field.csv")
        
        # Save additional derived fields
        derived_df = pd.DataFrame({
            'x': x_np,
            'y': y_np,
            'vorticity': vorticity_flat,
            'divergence': divergence_flat,
            'shear_rate': np.sqrt(u_y.flatten()**2 + v_x.flatten()**2),
            'reynolds_local': vel_mag / nu_np
        })
        
        derived_filepath = os.path.join(save_path, 'derived_fields.csv')
        derived_df.to_csv(derived_filepath, index=False)
        print(f"Saved derived fields to derived_fields.csv")
    
    print(f"Flow field CSV export completed")
    
    return {
        'grid_size': [nx, ny],
        'steady_flow': not test_config.UNSTEADY_FLOW,
        'files_created': ['flow_field.csv', 'derived_fields.csv'] if not test_config.UNSTEADY_FLOW else [f'flow_field_t_{t:.3f}.csv' for t in time_values.cpu().numpy()]
    }

def export_boundary_data_csv(model, test_config, save_path):
    """
    Export boundary condition data to CSV files
    
    Args:
        model: Trained PINN model
        test_config: Test configuration object
        save_path: Directory to save CSV files
    """
    print("\nExporting boundary data to CSV...")
    
    # Generate boundary points
    boundary_points = generate_boundary_points(test_config)
    
    # Process each boundary
    for boundary_name, points in boundary_points.items():
        points_np = points.detach().cpu().numpy()
        
        if test_config.UNSTEADY_FLOW and points.shape[1] > 2:
            x_vals = points[:, 0:1]
            y_vals = points[:, 1:2]
            t_vals = points[:, 2:3]
            u_pred, v_pred, p_pred = model.uvp(x_vals, y_vals, t_vals)
            
            df = pd.DataFrame({
                'x': points_np[:, 0],
                'y': points_np[:, 1],
                't': points_np[:, 2],
                'u_velocity': u_pred.detach().cpu().numpy().flatten(),
                'v_velocity': v_pred.detach().cpu().numpy().flatten(),
                'pressure': p_pred.detach().cpu().numpy().flatten()
            })
        else:
            x_vals = points[:, 0:1]
            y_vals = points[:, 1:2]
            u_pred, v_pred, p_pred = model.uvp(x_vals, y_vals)
            
            df = pd.DataFrame({
                'x': points_np[:, 0],
                'y': points_np[:, 1],
                'u_velocity': u_pred.detach().cpu().numpy().flatten(),
                'v_velocity': v_pred.detach().cpu().numpy().flatten(),
                'pressure': p_pred.detach().cpu().numpy().flatten()
            })
        
        # Save boundary data
        filepath = os.path.join(save_path, f'boundary_{boundary_name}.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved {boundary_name} boundary data to boundary_{boundary_name}.csv")

def export_centerline_data_csv(model, test_config, save_path):
    """
    Export centerline velocity profile to CSV
    
    Args:
        model: Trained PINN model
        test_config: Test configuration object
        save_path: Directory to save CSV files
    """
    print("\nExporting centerline data to CSV...")
    
    # Create centerline points
    n_points = 100
    x_centerline = torch.linspace(test_config.X_MIN, test_config.X_MAX, n_points, device=test_config.DEVICE).unsqueeze(1)
    y_centerline = torch.full_like(x_centerline, (test_config.Y_MIN + test_config.Y_MAX) / 2)
    
    # Get predictions along centerline
    u_pred, v_pred, p_pred = model.uvp(x_centerline, y_centerline)
    
    # Calculate viscosity along centerline
    nu_centerline = test_config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * y_centerline
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x_centerline.detach().cpu().numpy().flatten(),
        'y': y_centerline.detach().cpu().numpy().flatten(),
        'u_velocity': u_pred.detach().cpu().numpy().flatten(),
        'v_velocity': v_pred.detach().cpu().numpy().flatten(),
        'pressure': p_pred.detach().cpu().numpy().flatten(),
        'viscosity': nu_centerline.detach().cpu().numpy().flatten()
    })
    
    filepath = os.path.join(save_path, 'centerline_profile.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved centerline profile to centerline_profile.csv")

def export_viscosity_profile_csv(model, test_config, save_path):
    """
    Export viscosity profile comparison to CSV
    
    Args:
        model: Trained PINN model
        test_config: Test configuration object
        save_path: Directory to save CSV files
    """
    print("\nExporting viscosity profile to CSV...")
    
    # Create y-coordinate range
    n_points = 100
    y_values = np.linspace(test_config.Y_MIN, test_config.Y_MAX, n_points)
    
    # Calculate true and inferred viscosity profiles
    nu_true = test_config.NU_BASE_TRUE + test_config.A_TRUE * y_values
    nu_inferred = test_config.NU_BASE_TRUE + model.get_inferred_viscosity_param() * y_values
    
    # Calculate error
    absolute_error = np.abs(nu_inferred - nu_true)
    relative_error = absolute_error / nu_true * 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'y': y_values,
        'viscosity_true': nu_true,
        'viscosity_inferred': nu_inferred,
        'absolute_error': absolute_error,
        'relative_error_percent': relative_error
    })
    
    filepath = os.path.join(save_path, 'viscosity_profile.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved viscosity profile comparison to viscosity_profile.csv")

def export_model_metrics_csv(model, test_config, metrics, save_path):
    """
    Export model evaluation metrics to CSV
    
    Args:
        model: Trained PINN model
        test_config: Test configuration object
        metrics: Evaluation metrics dictionary
        save_path: Directory to save CSV files
    """
    print("\nExporting model metrics to CSV...")
    
    # Create metrics summary
    metrics_data = {
        'metric': [
            'reynolds_number',
            'viscosity_base_true',
            'viscosity_param_true',
            'viscosity_param_inferred',
            'absolute_error',
            'relative_error_percent',
            'pde_residual_momentum_x',
            'pde_residual_momentum_y',
            'pde_residual_continuity',
            'model_total_parameters',
            'domain_x_min',
            'domain_x_max',
            'domain_y_min',
            'domain_y_max',
            'inlet_velocity_max',
            'use_fourier_features',
            'use_adaptive_weights'
        ],
        'value': [
            test_config.REYNOLDS_NUMBER,
            test_config.NU_BASE_TRUE,
            test_config.A_TRUE,
            model.get_inferred_viscosity_param(),
            metrics['abs_error'],
            metrics['rel_error'],
            metrics['pde_residuals']['momentum_x'],
            metrics['pde_residuals']['momentum_y'],
            metrics['pde_residuals']['continuity'],
            sum(p.numel() for p in model.parameters()),
            test_config.X_MIN,
            test_config.X_MAX,
            test_config.Y_MIN,
            test_config.Y_MAX,
            test_config.U_MAX_INLET,
            test_config.USE_FOURIER_FEATURES,
            test_config.USE_ADAPTIVE_WEIGHTS
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    filepath = os.path.join(save_path, 'model_metrics.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved model metrics to model_metrics.csv")

def export_pde_residuals_csv(model, test_config, save_path):
    """
    Export PDE residuals at sample points to CSV
    
    Args:
        model: Trained PINN model
        test_config: Test configuration object
        save_path: Directory to save CSV files
    """
    print("\nExporting PDE residuals to CSV...")
    
    # Create sample points for residual evaluation
    n_sample = 1000
    x_sample = torch.rand(n_sample, 1, device=test_config.DEVICE) * (test_config.X_MAX - test_config.X_MIN) + test_config.X_MIN
    y_sample = torch.rand(n_sample, 1, device=test_config.DEVICE) * (test_config.Y_MAX - test_config.Y_MIN) + test_config.Y_MIN
    
    x_sample.requires_grad_(True)
    y_sample.requires_grad_(True)
    
    # Calculate residuals
    residuals = model.pde_residual(x_sample, y_sample)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x_sample.detach().cpu().numpy().flatten(),
        'y': y_sample.detach().cpu().numpy().flatten(),
        'momentum_x_residual': residuals['momentum_x'].detach().cpu().numpy().flatten(),
        'momentum_y_residual': residuals['momentum_y'].detach().cpu().numpy().flatten(),
        'continuity_residual': residuals['continuity'].detach().cpu().numpy().flatten(),
        'total_residual_magnitude': (residuals['momentum_x']**2 + residuals['momentum_y']**2 + residuals['continuity']**2).sqrt().detach().cpu().numpy().flatten()
    })
    
    filepath = os.path.join(save_path, 'pde_residuals.csv')
    df.to_csv(filepath, index=False)
    print(f"Saved PDE residuals to pde_residuals.csv")

def run_model_test(model, test_config, model_path, output_suffix="test", 
                  export_flow_field=True, export_boundary=True, export_centerline=True,
                  export_viscosity=True, export_residuals=True):
    """
    Run comprehensive model testing with CSV export
    
    Args:
        model: Loaded PINN model
        test_config: Test configuration object
        model_path: Path to the model file (for reporting)
        output_suffix: Suffix for output directory
        export_flow_field: Whether to export flow field data
        export_boundary: Whether to export boundary data
        export_centerline: Whether to export centerline data
        export_viscosity: Whether to export viscosity profile
        export_residuals: Whether to export PDE residuals
        
    Returns:
        Tuple of (metrics, exported_files)
    """
    
    print("\n" + "="*70)
    print(f"Running Model Test: {getattr(test_config, 'name', 'Custom Configuration')}")
    print("="*70)
    
    # Print test configuration
    print("\nTest Configuration:")
    config_params = [
        ('Reynolds Number', 'REYNOLDS_NUMBER'),
        ('Base Viscosity', 'NU_BASE_TRUE'),
        ('Viscosity Parameter a', 'A_TRUE'),
        ('Max Inlet Velocity', 'U_MAX_INLET'),
        ('Domain Width (X_MAX)', 'X_MAX'),
        ('Domain Height (Y_MAX)', 'Y_MAX'),
        ('Collocation Points', 'N_COLLOCATION'),
        ('Boundary Points', 'N_BOUNDARY'),
        ('Sparse Data Points', 'N_DATA_SPARSE'),
        ('Fourier Features', 'USE_FOURIER_FEATURES'),
        ('Adaptive Weights', 'USE_ADAPTIVE_WEIGHTS'),
        ('Adaptive Sampling', 'USE_ADAPTIVE_SAMPLING'),
        ('Curriculum Learning', 'USE_CURRICULUM_LEARNING')
    ]
    
    for display_name, param_name in config_params:
        if hasattr(test_config, param_name):
            value = getattr(test_config, param_name)
            print(f"  {display_name}: {value}")
    
    # Create output directory for this test
    base_output_dir = test_config.OUTPUT_DIR
    test_output_dir = os.path.join(base_output_dir, f"interactive_{output_suffix}")
    test_config.OUTPUT_DIR = test_output_dir
    os.makedirs(test_output_dir, exist_ok=True)
    
    print(f"\nTest results will be saved to: {test_output_dir}")
    
    # Initialize return variables
    metrics = None
    exported_files = []
    
    try:
        # Generate test data with the new configuration
        print("\nGenerating test data...")
        start_time = time.time()
        
        collocation_points = generate_collocation_points(test_config)
        boundary_points = generate_boundary_points(test_config)
        sparse_data = generate_sparse_data_points(test_config)
        
        print(f"Data generation completed in {time.time() - start_time:.2f}s")
        
        # Evaluate model performance
        print("\nEvaluating model performance...")
        eval_start = time.time()
        
        metrics = evaluate_model(model, test_config)
        
        print(f"Model evaluation completed in {time.time() - eval_start:.2f}s")
        
        # Export CSV data
        print("\nExporting data to CSV files...")
        export_start = time.time()
        
        # Export flow field data
        if export_flow_field:
            flow_data = export_flow_field_csv(model, test_config, test_output_dir)
            exported_files.extend(flow_data['files_created'])
        
        # Export boundary data
        if export_boundary:
            export_boundary_data_csv(model, test_config, test_output_dir)
            exported_files.extend([f'boundary_{name}.csv' for name in ['inlet', 'outlet', 'walls']])
        
        # Export centerline data
        if export_centerline:
            export_centerline_data_csv(model, test_config, test_output_dir)
            exported_files.append('centerline_profile.csv')
        
        # Export viscosity profile
        if export_viscosity:
            export_viscosity_profile_csv(model, test_config, test_output_dir)
            exported_files.append('viscosity_profile.csv')
        
        # Export model metrics
        export_model_metrics_csv(model, test_config, metrics, test_output_dir)
        exported_files.append('model_metrics.csv')
        
        # Export PDE residuals
        if export_residuals:
            export_pde_residuals_csv(model, test_config, test_output_dir)
            exported_files.append('pde_residuals.csv')
        
        print(f"CSV export completed in {time.time() - export_start:.2f}s")
        
        # Generate test report
        generate_test_report(model, test_config, metrics, model_path, test_output_dir)
        exported_files.append('test_report.json')
        exported_files.append('test_report.txt')
        
        print("\n" + "="*70)
        print("Model Testing Complete!")
        print("="*70)
        
        # Print key results
        print_test_results(model, test_config, metrics)
        
        # Print exported files
        print(f"\nExported CSV files:")
        for filename in exported_files:
            print(f"  - {filename}")
        
        return metrics, exported_files
        
    except Exception as e:
        print(f"\nError during model testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return metrics, exported_files

def print_test_results(model, test_config, metrics):
    """Print key test results"""
    print(f"\nKey Results:")
    print(f"  True viscosity parameter 'a': {test_config.A_TRUE:.6f}")
    print(f"  Inferred viscosity parameter 'a': {model.get_inferred_viscosity_param():.6f}")
    print(f"  Absolute error: {metrics['abs_error']:.6f}")
    print(f"  Relative error: {metrics['rel_error']:.2f}%")
    print(f"  Reynolds number: {test_config.REYNOLDS_NUMBER}")
    
    print(f"\nPDE Residuals (Mean Absolute):")
    for key, value in metrics['pde_residuals'].items():
        print(f"  {key}: {value:.6e}")
    
    # Performance assessment
    rel_error = metrics['rel_error']
    if rel_error < 1.0:
        assessment = "Excellent"
    elif rel_error < 5.0:
        assessment = "Good"
    elif rel_error < 15.0:
        assessment = "Acceptable"
    else:
        assessment = "Poor"
    
    print(f"\nPerformance Assessment: {assessment} ({rel_error:.2f}% error)")

def get_interactive_config():
    """Interactive function to get configuration from user input"""
    print("\n" + "="*60)
    print("Interactive Configuration Setup")
    print("="*60)
    print("Enter parameters (press Enter to use default values)")
    
    config_params = {}
    
    # Physics parameters
    print("\n--- Physics Parameters ---")
    
    reynolds = input(f"Reynolds Number (default: {cfg.REYNOLDS_NUMBER}): ")
    if reynolds.strip():
        config_params['reynolds_number'] = float(reynolds)
    
    nu_base = input(f"Base Viscosity (default: {cfg.NU_BASE_TRUE}): ")
    if nu_base.strip():
        config_params['nu_base_true'] = float(nu_base)
    
    a_true = input(f"Viscosity Variation Parameter 'a' (default: {cfg.A_TRUE}): ")
    if a_true.strip():
        config_params['a_true'] = float(a_true)
    
    u_max = input(f"Maximum Inlet Velocity (default: {cfg.U_MAX_INLET}): ")
    if u_max.strip():
        config_params['u_max_inlet'] = float(u_max)
    
    # Domain parameters
    print("\n--- Domain Parameters ---")
    
    x_max = input(f"Domain Width X_MAX (default: {cfg.X_MAX}): ")
    if x_max.strip():
        config_params['x_max'] = float(x_max)
    
    y_max = input(f"Domain Height Y_MAX (default: {cfg.Y_MAX}): ")
    if y_max.strip():
        config_params['y_max'] = float(y_max)
    
    # Computational parameters
    print("\n--- Computational Parameters ---")
    
    n_collocation = input(f"Collocation Points (default: {cfg.N_COLLOCATION}): ")
    if n_collocation.strip():
        config_params['n_collocation'] = int(n_collocation)
    
    n_boundary = input(f"Boundary Points per edge (default: {cfg.N_BOUNDARY}): ")
    if n_boundary.strip():
        config_params['n_boundary'] = int(n_boundary)
    
    n_data_sparse = input(f"Sparse Data Points (default: {cfg.N_DATA_SPARSE}): ")
    if n_data_sparse.strip():
        config_params['n_data_sparse'] = int(n_data_sparse)
    
    # Advanced features
    print("\n--- Advanced Features ---")
    
    fourier = input(f"Use Fourier Features? (y/n, default: {'y' if cfg.USE_FOURIER_FEATURES else 'n'}): ")
    if fourier.lower() in ['y', 'yes']:
        config_params['use_fourier_features'] = True
        fourier_scale = input(f"Fourier Scale (default: {cfg.FOURIER_SCALE}): ")
        if fourier_scale.strip():
            config_params['fourier_scale'] = float(fourier_scale)
    elif fourier.lower() in ['n', 'no']:
        config_params['use_fourier_features'] = False
    
    adaptive_weights = input(f"Use Adaptive Weights? (y/n, default: {'y' if cfg.USE_ADAPTIVE_WEIGHTS else 'n'}): ")
    if adaptive_weights.lower() in ['y', 'yes']:
        config_params['use_adaptive_weights'] = True
    elif adaptive_weights.lower() in ['n', 'no']:
        config_params['use_adaptive_weights'] = False
    
    adaptive_sampling = input(f"Use Adaptive Sampling? (y/n, default: {'y' if cfg.USE_ADAPTIVE_SAMPLING else 'n'}): ")
    if adaptive_sampling.lower() in ['y', 'yes']:
        config_params['use_adaptive_sampling'] = True
    elif adaptive_sampling.lower() in ['n', 'no']:
        config_params['use_adaptive_sampling'] = False
    
    # Test name
    test_name = input("\nTest Name (default: 'Interactive Test'): ")
    if test_name.strip():
        config_params['name'] = test_name
    else:
        config_params['name'] = 'Interactive Test'
    
    return config_params

def run_multiple_tests(model_path, test_configurations, output_base="multi_test"):
    """
    Run multiple test configurations on the same model
    
    Args:
        model_path: Path to the model file
        test_configurations: List of configuration parameter dictionaries
        output_base: Base name for output directories
        
    Returns:
        List of test results
    """
    print(f"\n" + "="*70)
    print(f"Running Multiple Tests on Model: {model_path}")
    print(f"Number of test configurations: {len(test_configurations)}")
    print("="*70)
    
    all_results = []
    
    for i, config_params in enumerate(test_configurations):
        print(f"\n{'='*50}")
        print(f"Test {i+1}/{len(test_configurations)}")
        print(f"{'='*50}")
        
        # Create test configuration
        test_config = create_test_config(**config_params)
        
        # Load model
        model, test_config = load_model_with_config(model_path, test_config)
        
        # Run test
        output_suffix = f"{output_base}_{i+1:02d}"
        if 'name' in config_params:
            output_suffix += f"_{config_params['name'].replace(' ', '_')}"
        
        metrics, exported_files = run_model_test(model, test_config, model_path, output_suffix)
        
        # Store results
        result = {
            'test_index': i+1,
            'config_params': config_params,
            'metrics': metrics,
            'exported_files': exported_files,
            'output_suffix': output_suffix
        }
        all_results.append(result)
    
    # Generate summary report
    generate_multi_test_summary(all_results, model_path, output_base)
    
    return all_results

def generate_multi_test_summary(all_results, model_path, output_base):
    """Generate summary report for multiple tests as CSV"""
    summary_dir = os.path.join(cfg.OUTPUT_DIR, f"summary_{output_base}")
    os.makedirs(summary_dir, exist_ok=True)
    
    print(f"\n" + "="*70)
    print("Multi-Test Summary")
    print("="*70)
    
    # Prepare data for CSV
    summary_data = []
    
    for result in all_results:
        if result['metrics'] is not None:
            test_summary = {
                'test_index': result['test_index'],
                'test_name': result['config_params'].get('name', f"Test {result['test_index']}"),
                'reynolds_number': result['config_params'].get('reynolds_number', cfg.REYNOLDS_NUMBER),
                'a_true': result['config_params'].get('a_true', cfg.A_TRUE),
                'relative_error_percent': result['metrics']['rel_error'],
                'absolute_error': result['metrics']['abs_error'],
                'inferred_a': result['metrics']['inferred_a'],
                'momentum_x_residual': result['metrics']['pde_residuals']['momentum_x'],
                'momentum_y_residual': result['metrics']['pde_residuals']['momentum_y'],
                'continuity_residual': result['metrics']['pde_residuals']['continuity'],
                'output_directory': result['output_suffix']
            }
            summary_data.append(test_summary)
            
            print(f"Test {result['test_index']}: {test_summary['test_name']}")
            print(f"  Re={test_summary['reynolds_number']}, a_true={test_summary['a_true']:.4f}")
            print(f"  Error: {test_summary['relative_error_percent']:.2f}%")
    
    # Save summary as CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        summary_path = os.path.join(summary_dir, 'multi_test_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"\nSummary CSV saved to: {summary_path}")
        
        # Also save as JSON for backward compatibility
        json_data = {
            'model_path': model_path,
            'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(all_results),
            'test_results': summary_data
        }
        
        json_path = os.path.join(summary_dir, 'multi_test_summary.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Summary JSON saved to: {json_path}")

def generate_test_report(model, test_config, metrics, model_path, output_dir):
    """Generate a comprehensive test report"""
    
    report_data = {
        'test_info': {
            'model_path': model_path,
            'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_name': getattr(test_config, 'name', 'Custom Test'),
            'output_directory': output_dir
        },
        'configuration': {
            'reynolds_number': test_config.REYNOLDS_NUMBER,
            'nu_base_true': test_config.NU_BASE_TRUE,
            'a_true': test_config.A_TRUE,
            'u_max_inlet': test_config.U_MAX_INLET,
            'domain_x_max': test_config.X_MAX,
            'domain_y_max': test_config.Y_MAX,
            'domain_x_min': test_config.X_MIN,
            'domain_y_min': test_config.Y_MIN,
            'use_fourier_features': test_config.USE_FOURIER_FEATURES,
            'use_adaptive_weights': test_config.USE_ADAPTIVE_WEIGHTS,
            'use_adaptive_sampling': test_config.USE_ADAPTIVE_SAMPLING,
            'use_curriculum_learning': test_config.USE_CURRICULUM_LEARNING,
            'collocation_points': test_config.N_COLLOCATION,
            'boundary_points': test_config.N_BOUNDARY,
            'sparse_data_points': test_config.N_DATA_SPARSE
        },
        'results': {
            'inferred_viscosity_param': model.get_inferred_viscosity_param(),
            'true_viscosity_param': test_config.A_TRUE,
            'absolute_error': metrics['abs_error'],
            'relative_error_percent': metrics['rel_error'],
            'pde_residuals': metrics['pde_residuals']
        },
        'model_info': {
            'architecture': test_config.PINN_LAYERS,
            'device': str(model.device),
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
    }
    
    # Save report as JSON
    report_path = os.path.join(output_dir, 'test_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Save report as text
    text_report_path = os.path.join(output_dir, 'test_report.txt')
    with open(text_report_path, 'w') as f:
        f.write("PINN Model Interactive Test Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Name: {report_data['test_info']['test_name']}\n")
        f.write(f"Test Date: {report_data['test_info']['test_time']}\n")
        f.write(f"Model Path: {report_data['test_info']['model_path']}\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 20 + "\n")
        for key, value in report_data['configuration'].items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nResults:\n")
        f.write("-" * 20 + "\n")
        f.write(f"True viscosity parameter: {report_data['results']['true_viscosity_param']:.6f}\n")
        f.write(f"Inferred viscosity parameter: {report_data['results']['inferred_viscosity_param']:.6f}\n")
        f.write(f"Absolute error: {report_data['results']['absolute_error']:.6f}\n")
        f.write(f"Relative error: {report_data['results']['relative_error_percent']:.2f}%\n")
        
        f.write(f"\nPDE Residuals:\n")
        f.write("-" * 20 + "\n")
        for key, value in report_data['results']['pde_residuals'].items():
            f.write(f"{key}: {value:.6e}\n")
    
    print(f"Test report saved to: {report_path}")

def main():
    """Main function with example test scenarios"""
    # Hardcoded model path
    DEFAULT_MODEL_PATH = "/home/brand/pinn_viscosity/backend/results/trained_model.pth"
    
    parser = argparse.ArgumentParser(description="Interactive PINN Model Testing - CSV Export Version")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help=f"Path to the trained model file (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--interactive", action="store_true", help="Use interactive mode for configuration")
    parser.add_argument("--inspect", action="store_true", help="Just inspect the saved model configuration and exit")
    
    # Physics parameters
    parser.add_argument("--reynolds", type=float, help="Reynolds number")
    parser.add_argument("--viscosity-base", type=float, help="Base viscosity")
    parser.add_argument("--viscosity-param", type=float, help="Viscosity variation parameter 'a'")
    parser.add_argument("--inlet-velocity", type=float, help="Maximum inlet velocity")
    
    # Domain parameters
    parser.add_argument("--domain-width", type=float, help="Domain width (X_MAX)")
    parser.add_argument("--domain-height", type=float, help="Domain height (Y_MAX)")
    
    # Computational parameters
    parser.add_argument("--collocation-points", type=int, help="Number of collocation points")
    parser.add_argument("--boundary-points", type=int, help="Number of boundary points per edge")
    parser.add_argument("--sparse-data-points", type=int, help="Number of sparse data points")
    
    # Advanced features
    parser.add_argument("--fourier", action="store_true", help="Use Fourier features")
    parser.add_argument("--adaptive-weights", action="store_true", help="Use adaptive weights")
    parser.add_argument("--adaptive-sampling", action="store_true", help="Use adaptive sampling")
    
    # Output options
    parser.add_argument("--output-suffix", default="test", help="Output directory suffix")
    parser.add_argument("--no-flow-field", action="store_true", help="Skip flow field export")
    parser.add_argument("--no-boundary", action="store_true", help="Skip boundary data export")
    parser.add_argument("--no-centerline", action="store_true", help="Skip centerline export")
    parser.add_argument("--no-viscosity", action="store_true", help="Skip viscosity profile export")
    parser.add_argument("--no-residuals", action="store_true", help="Skip PDE residuals export")
    
    # Test scenarios
    parser.add_argument("--run-scenarios", action="store_true", help="Run predefined test scenarios")
    
    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please ensure the model has been trained and saved to this location.")
        print("You can train a model by running: python main.py")
        return
    
    if args.inspect:
        # Just inspect the model and exit
        print("\n" + "="*60)
        print("Model Inspection")
        print("="*60)
        
        try:
            model_info = inspect_saved_model(args.model)
            
            print(f"Model Path: {model_info['model_path']}")
            print(f"Has Configuration: {model_info['has_config']}")
            print(f"Has Model State: {model_info['has_model_state']}")
            print(f"Inferred 'a' Parameter: {model_info['inferred_a_param']}")
            print(f"Has Adaptive Weights: {model_info.get('has_adaptive_weights', 'Unknown')}")
            print(f"Has Fourier Features: {model_info.get('has_fourier_features', 'Unknown')}")
            
            if 'saved_config' in model_info:
                print(f"\nSaved Configuration:")
                for key, value in model_info['saved_config'].items():
                    print(f"  {key}: {value}")
            
            if 'inferred_layers' in model_info:
                print(f"\nInferred Network Architecture:")
                for layer_info in model_info['inferred_layers']:
                    print(f"  {layer_info}")
                    
        except Exception as e:
            print(f"Error inspecting model: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    if args.interactive:
        # Interactive mode
        config_params = get_interactive_config()
        test_config = create_test_config(**config_params)
        model, test_config = load_model_with_config(args.model, test_config)
        
        run_model_test(
            model, test_config, args.model, args.output_suffix,
            export_flow_field=not args.no_flow_field,
            export_boundary=not args.no_boundary,
            export_centerline=not args.no_centerline,
            export_viscosity=not args.no_viscosity,
            export_residuals=not args.no_residuals
        )
    
    elif args.run_scenarios:
        # Run multiple predefined scenarios
        test_scenarios = [
            # Low Reynolds number test
            {
                'reynolds_number': 10,
                'a_true': 0.02,
                'nu_base_true': 0.05,
                'u_max_inlet': 0.5,
                'name': 'Low_Re_Test'
            },
            # Medium Reynolds number test
            {
                'reynolds_number': 100,
                'a_true': 0.05,
                'nu_base_true': 0.01,
                'u_max_inlet': 1.0,
                'name': 'Medium_Re_Test'
            },
            # High Reynolds number test
            {
                'reynolds_number': 200,
                'a_true': 0.08,
                'nu_base_true': 0.005,
                'u_max_inlet': 2.0,
                'name': 'High_Re_Test'
            },
            # Wide domain test
            {
                'reynolds_number': 50,
                'a_true': 0.05,
                'x_max': 4.0,
                'name': 'Wide_Domain_Test'
            },
            # Tall domain test
            {
                'reynolds_number': 75,
                'a_true': 0.06,
                'y_max': 2.0,
                'name': 'Tall_Domain_Test'
            }
        ]
        
        run_multiple_tests(args.model, test_scenarios, "scenarios")
    
    else:
        # Single test with command line parameters
        config_params = {}
        
        # Map command line arguments to config parameters
        if args.reynolds is not None:
            config_params['reynolds_number'] = args.reynolds
        if args.viscosity_base is not None:
            config_params['nu_base_true'] = args.viscosity_base
        if args.viscosity_param is not None:
            config_params['a_true'] = args.viscosity_param
        if args.inlet_velocity is not None:
            config_params['u_max_inlet'] = args.inlet_velocity
        if args.domain_width is not None:
            config_params['x_max'] = args.domain_width
        if args.domain_height is not None:
            config_params['y_max'] = args.domain_height
        if args.collocation_points is not None:
            config_params['n_collocation'] = args.collocation_points
        if args.boundary_points is not None:
            config_params['n_boundary'] = args.boundary_points
        if args.sparse_data_points is not None:
            config_params['n_data_sparse'] = args.sparse_data_points
        if args.fourier:
            config_params['use_fourier_features'] = True
        if args.adaptive_weights:
            config_params['use_adaptive_weights'] = True
        if args.adaptive_sampling:
            config_params['use_adaptive_sampling'] = True
        
        config_params['name'] = 'Command_Line_Test'
        
        # Create and run test
        test_config = create_test_config(**config_params)
        model, test_config = load_model_with_config(args.model, test_config)
        
        run_model_test(
            model, test_config, args.model, args.output_suffix,
            export_flow_field=not args.no_flow_field,
            export_boundary=not args.no_boundary,
            export_centerline=not args.no_centerline,
            export_viscosity=not args.no_viscosity,
            export_residuals=not args.no_residuals
        )

if __name__ == "__main__":
    main()
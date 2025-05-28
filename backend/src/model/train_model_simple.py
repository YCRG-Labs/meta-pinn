import torch
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import cfg
from src.model.model import PINN

def train_model(model, collocation_points, boundary_points, sparse_data, config=None):
    """
    Train the PINN model with advanced features
    
    Args:
        model: PINN model
        collocation_points: Tensor of collocation points for PDE residual
        boundary_points: Dictionary of boundary points
        sparse_data: Tuple of (coords, u, v, p) for sparse data
        config: Project configuration (optional, uses global cfg if None)
        
    Returns:
        Trained model and training history
    """
    if config is None:
        config = cfg
        
    # Ensure all inputs are on the correct device
    device = config.DEVICE
    
    # Unpack sparse data and ensure it's on the correct device
    sparse_coords, sparse_u, sparse_v, sparse_p = sparse_data
    sparse_coords = sparse_coords.to(device)
    sparse_u = sparse_u.to(device)
    sparse_v = sparse_v.to(device)
    sparse_p = sparse_p.to(device)
    
    # Ensure collocation points are on the correct device
    collocation_points = collocation_points.to(device)
    
    # Ensure boundary points are on the correct device
    for key in boundary_points:
        boundary_points[key] = boundary_points[key].to(device)
    
    # Setup optimizer
    if config.OPTIMIZER_TYPE == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER_TYPE == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=config.LEARNING_RATE)
    else:
        raise ValueError(f"Unknown optimizer type: {config.OPTIMIZER_TYPE}")
        
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA
    )
    
    # Initialize adaptive collocation sampler if enabled
    if config.USE_ADAPTIVE_SAMPLING:
        from src.model.train_model import AdaptiveCollocationSampler
        adaptive_sampler = AdaptiveCollocationSampler(model, config, initial_points=collocation_points)
        collocation_points = adaptive_sampler.get_collocation_points()
    
    # Initialize curriculum learning if enabled
    if config.USE_CURRICULUM_LEARNING:
        from src.model.train_model import CurriculumLearning
        curriculum = CurriculumLearning(config)
    
    # Prepare collocation points for PDE residual
    x_collocation = collocation_points[:, 0:1]
    y_collocation = collocation_points[:, 1:2]
    
    # For unsteady flow
    if config.UNSTEADY_FLOW and collocation_points.shape[1] > 2:
        t_collocation = collocation_points[:, 2:3]
    else:
        t_collocation = None
    
    # Training history
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_pde': [],
        'loss_bc': [],
        'loss_data': [],
        'inferred_a': [],
        'adaptive_weights': [] if config.USE_ADAPTIVE_WEIGHTS else None,
        'curriculum_weights': [] if config.USE_CURRICULUM_LEARNING else None,
        'reinitialization_epochs': []
    }
    
    # For re-initialization strategy
    reinit_counter = 0
    reinit_patience = config.REINIT_PATIENCE
    reinit_threshold = config.REINIT_THRESHOLD
    previous_loss = float('inf')
    
    # Training loop
    import time
    start_time = time.time()
    
    def closure():
        optimizer.zero_grad()
        
        # PDE residuals at collocation points
        if t_collocation is not None:
            pde_residuals = model.pde_residual(x_collocation, y_collocation, t_collocation)
        else:
            pde_residuals = model.pde_residual(x_collocation, y_collocation)
            
        momentum_x_residual = pde_residuals['momentum_x']
        momentum_y_residual = pde_residuals['momentum_y']
        continuity_residual = pde_residuals['continuity']
        
        # Mean squared PDE residuals
        loss_momentum_x = torch.mean(momentum_x_residual**2)
        loss_momentum_y = torch.mean(momentum_y_residual**2)
        loss_continuity = torch.mean(continuity_residual**2)
        
        # Get weights (adaptive or fixed)
        if config.USE_ADAPTIVE_WEIGHTS:
            weights = model.get_adaptive_weights()
        elif config.USE_CURRICULUM_LEARNING:
            weights = curriculum.get_weights(epoch)
        else:
            weights = {
                'momentum_x': 1.0,
                'momentum_y': 1.0,
                'continuity': 1.0,
                'bc': config.WEIGHT_BC,
                'data_u': config.WEIGHT_DATA_U,
                'data_v': config.WEIGHT_DATA_V,
                'data_p': config.WEIGHT_DATA_P
            }
        
        # Combined PDE loss with weights
        loss_pde = (
            weights['momentum_x'] * loss_momentum_x + 
            weights['momentum_y'] * loss_momentum_y + 
            weights['continuity'] * loss_continuity
        )
        
        # Boundary condition residuals
        bc_residuals = model.boundary_conditions(boundary_points)
        loss_bc_inlet_u = torch.mean(bc_residuals['inlet_u']**2)
        loss_bc_inlet_v = torch.mean(bc_residuals['inlet_v']**2)
        loss_bc_wall_u = torch.mean(bc_residuals['wall_u']**2)
        loss_bc_wall_v = torch.mean(bc_residuals['wall_v']**2)
        
        # Combined BC loss
        loss_bc = weights['bc'] * (loss_bc_inlet_u + loss_bc_inlet_v + loss_bc_wall_u + loss_bc_wall_v)
        
        # Data loss for sparse measurements
        if t_collocation is not None and sparse_coords.shape[1] > 2:
            u_pred, v_pred, p_pred = model.uvp(
                sparse_coords[:, 0:1], sparse_coords[:, 1:2], sparse_coords[:, 2:3]
            )
        else:
            u_pred, v_pred, p_pred = model.uvp(sparse_coords[:, 0:1], sparse_coords[:, 1:2])
            
        loss_data_u = torch.mean((u_pred - sparse_u)**2)
        loss_data_v = torch.mean((v_pred - sparse_v)**2)
        loss_data_p = torch.mean((p_pred - sparse_p)**2)
        
        # Combined data loss
        loss_data = (
            weights['data_u'] * loss_data_u + 
            weights['data_v'] * loss_data_v + 
            weights['data_p'] * loss_data_p
        )
        
        # Total loss
        loss = loss_pde + loss_bc + loss_data
        
        loss.backward()
        
        # Store for logging
        closure.loss_total = loss.item()
        closure.loss_pde = loss_pde.item()
        closure.loss_bc = loss_bc.item()
        closure.loss_data = loss_data.item()
        closure.weights = weights
        
        return loss
    
    print("\n" + "="*70)
    print(f"Starting training for {config.EPOCHS} epochs")
    if config.USE_FOURIER_FEATURES:
        print("Using Fourier feature embeddings")
    if config.USE_ADAPTIVE_WEIGHTS:
        print("Using adaptive loss weighting")
    if config.USE_ADAPTIVE_SAMPLING:
        print("Using adaptive collocation sampling")
    if config.USE_CURRICULUM_LEARNING:
        print("Using curriculum learning")
    if config.USE_REINIT_STRATEGY:
        print(f"Using re-initialization strategy (patience={reinit_patience}, threshold={reinit_threshold})")
    print("="*70)
    
    for epoch in range(config.EPOCHS):
        # For LBFGS, we need to use the closure
        if config.OPTIMIZER_TYPE == 'LBFGS':
            optimizer.step(closure)
        else:
            # For Adam, we can call closure() directly
            loss = closure()
            optimizer.step()
            
        # Step the scheduler
        scheduler.step()
        
        # Update adaptive collocation points if enabled
        if config.USE_ADAPTIVE_SAMPLING and (epoch + 1) % config.ADAPTIVE_SAMPLING_FREQUENCY == 0:
            collocation_points = adaptive_sampler.update_points()
            x_collocation = collocation_points[:, 0:1]
            y_collocation = collocation_points[:, 1:2]
            if t_collocation is not None:
                t_collocation = collocation_points[:, 2:3]
        
        # Check for re-initialization
        if config.USE_REINIT_STRATEGY:
            current_loss = closure.loss_total
            if previous_loss - current_loss < reinit_threshold * previous_loss:
                reinit_counter += 1
            else:
                reinit_counter = 0
                
            if reinit_counter >= reinit_patience:
                print(f"Epoch {epoch}: Re-initializing parameters to escape local minimum")
                model.reinitialize_parameters(scale=0.1)
                reinit_counter = 0
                history['reinitialization_epochs'].append(epoch)
                
            previous_loss = current_loss
        
        # Log progress
        if epoch % config.LOG_FREQUENCY == 0 or epoch == config.EPOCHS - 1:
            inferred_a = model.get_inferred_viscosity_param()
            
            # Update history
            history['epoch'].append(epoch)
            history['loss_total'].append(closure.loss_total)
            history['loss_pde'].append(closure.loss_pde)
            history['loss_bc'].append(closure.loss_bc)
            history['loss_data'].append(closure.loss_data)
            history['inferred_a'].append(inferred_a)
            
            if config.USE_ADAPTIVE_WEIGHTS:
                history['adaptive_weights'].append(closure.weights)
            
            if config.USE_CURRICULUM_LEARNING:
                history['curriculum_weights'].append(closure.weights)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{config.EPOCHS} [{elapsed:.2f}s] - "
                  f"Loss: {closure.loss_total:.6e}, "
                  f"PDE: {closure.loss_pde:.6e}, "
                  f"BC: {closure.loss_bc:.6e}, "
                  f"Data: {closure.loss_data:.6e}, "
                  f"a: {inferred_a:.6f}")
            
    # Final inferred parameter
    final_a = model.get_inferred_viscosity_param()
    print("\n" + "="*70)
    print(f"Training completed in {time.time() - start_time:.2f}s")
    print(f"Final inferred viscosity parameter a: {final_a:.6f}")
    print(f"True viscosity parameter a: {config.A_TRUE:.6f}")
    print(f"Relative error: {abs(final_a - config.A_TRUE) / config.A_TRUE * 100:.2f}%")
    print("="*70)
    
    # Save the trained model
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model.save(os.path.join(config.OUTPUT_DIR, config.MODEL_SAVE_FILENAME))
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from src.model.train_model import plot_training_history
        plot_training_history(history, config)
    except ImportError:
        print("Matplotlib not available, skipping history plotting")
    
    # Plot additional diagnostics
    if config.USE_ADAPTIVE_SAMPLING:
        try:
            adaptive_sampler.plot_residual_history(
                os.path.join(config.OUTPUT_DIR, 'adaptive_sampling_residuals.png')
            )
        except Exception as e:
            print(f"Error plotting adaptive sampling residuals: {e}")
    
    if config.USE_CURRICULUM_LEARNING:
        try:
            curriculum.plot_curriculum(
                os.path.join(config.OUTPUT_DIR, 'curriculum_schedule.png')
            )
        except Exception as e:
            print(f"Error plotting curriculum schedule: {e}")
    
    return model, history

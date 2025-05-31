import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import cfg
from src.model.model import PINN

class AdaptiveCollocationSampler:
    """
    Adaptive collocation point sampler that focuses on regions with high PDE residuals
    """
    def __init__(self, model, config=None, initial_points=None, update_frequency=50):
        if config is None:
            config = cfg
            
        self.config = config
        self.model = model
        self.device = config.DEVICE
        
        # Domain bounds
        self.x_min, self.x_max = config.X_MIN, config.X_MAX
        self.y_min, self.y_max = config.Y_MIN, config.Y_MAX
        
        # For unsteady flow
        self.unsteady = config.UNSTEADY_FLOW
        if self.unsteady:
            self.t_min, self.t_max = config.T_MIN, config.T_MAX
        
        # Sampling parameters
        self.n_points = config.N_COLLOCATION
        self.n_initial = int(self.n_points * 0.8)  # Initial uniform points
        self.n_adaptive = self.n_points - self.n_initial  # Adaptive points
        
        # Residual history for adaptive sampling
        self.residual_history = []
        self.points_history = []
        
        # Initialize points
        if initial_points is not None:
            self.current_points = initial_points
        else:
            self.current_points = self._generate_initial_points()
        
        self.update_frequency = update_frequency
    
    def _generate_initial_points(self):
        """
        Generate initial uniform collocation points
        
        Returns:
            Tensor of initial collocation points
        """
        # Random sampling in the domain
        x_collocation = torch.rand(self.n_initial, 1, device=self.device) * (self.x_max - self.x_min) + self.x_min
        y_collocation = torch.rand(self.n_initial, 1, device=self.device) * (self.y_max - self.y_min) + self.y_min
        
        if self.unsteady:
            t_collocation = torch.rand(self.n_initial, 1, device=self.device) * (self.t_max - self.t_min) + self.t_min
            collocation_points = torch.cat([x_collocation, y_collocation, t_collocation], dim=1)
        else:
            collocation_points = torch.cat([x_collocation, y_collocation], dim=1)
        
        return collocation_points
    
    def _compute_residuals(self, points):
        """
        Compute PDE residuals at given points
        
        Args:
            points: Tensor of collocation points
            
        Returns:
            Tensor of residual magnitudes
        """
        if self.unsteady:
            x = points[:, 0:1]
            y = points[:, 1:2]
            t = points[:, 2:3]
            residuals = self.model.pde_residual(x, y, t)
        else:
            x = points[:, 0:1]
            y = points[:, 1:2]
            residuals = self.model.pde_residual(x, y)
        
        # Compute total residual magnitude
        total_residual = (
            residuals['momentum_x']**2 + 
            residuals['momentum_y']**2 + 
            residuals['continuity']**2
        ).detach()
        
        return total_residual
    
    def _sample_adaptive_points(self, n_samples):
        """
        Sample new points with probability proportional to residual magnitude
        
        Args:
            n_samples: Number of new points to sample
            
        Returns:
            Tensor of new adaptive points
        """
        # Compute residuals at current points
        residuals = self._compute_residuals(self.current_points)
        
        # Store for history
        self.residual_history.append(residuals.mean().item())
        self.points_history.append(self.current_points.clone())
        
        # Normalize residuals to get sampling probabilities
        # Add small epsilon to prevent zero probabilities
        probs = (residuals + 1e-10) / (residuals.sum() + 1e-10)
        
        # Ensure probabilities sum to 1
        probs = probs / probs.sum()
        
        # Sample indices based on probabilities
        try:
            indices = torch.multinomial(probs.flatten(), n_samples, replacement=True)
        except RuntimeError:
            # Fallback to uniform sampling if multinomial fails
            indices = torch.randint(0, len(probs), (n_samples,), device=self.device)
        
        # Get selected points
        selected_points = self.current_points[indices]
        
        # Add small random perturbations to explore nearby regions
        if self.unsteady:
            perturb_x = torch.randn(n_samples, 1, device=self.device) * 0.05 * (self.x_max - self.x_min)
            perturb_y = torch.randn(n_samples, 1, device=self.device) * 0.05 * (self.y_max - self.y_min)
            perturb_t = torch.randn(n_samples, 1, device=self.device) * 0.05 * (self.t_max - self.t_min)
            
            new_x = (selected_points[:, 0:1] + perturb_x).clamp(self.x_min, self.x_max)
            new_y = (selected_points[:, 1:2] + perturb_y).clamp(self.y_min, self.y_max)
            new_t = (selected_points[:, 2:3] + perturb_t).clamp(self.t_min, self.t_max)
            
            new_points = torch.cat([new_x, new_y, new_t], dim=1)
        else:
            perturb_x = torch.randn(n_samples, 1, device=self.device) * 0.05 * (self.x_max - self.x_min)
            perturb_y = torch.randn(n_samples, 1, device=self.device) * 0.05 * (self.y_max - self.y_min)
            
            new_x = (selected_points[:, 0:1] + perturb_x).clamp(self.x_min, self.x_max)
            new_y = (selected_points[:, 1:2] + perturb_y).clamp(self.y_min, self.y_max)
            
            new_points = torch.cat([new_x, new_y], dim=1)
        
        return new_points
    
    def get_collocation_points(self):
        """
        Get current collocation points
        
        Returns:
            Tensor of collocation points
        """
        return self.current_points
    
    def update_points(self):
        """
        Update collocation points using adaptive sampling
        
        Returns:
            Tensor of updated collocation points
        """
        # Keep some of the current points (uniform)
        keep_indices = torch.randperm(self.current_points.size(0))[:self.n_initial]
        uniform_points = self.current_points[keep_indices]
        
        # Sample new adaptive points
        adaptive_points = self._sample_adaptive_points(self.n_adaptive)
        
        # Combine uniform and adaptive points
        self.current_points = torch.cat([uniform_points, adaptive_points], dim=0)
        
        return self.current_points
    
    def plot_residual_history(self, save_path=None):
        """
        Plot the history of mean residuals
        
        Args:
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.residual_history, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Residual')
        plt.title('Adaptive Sampling Residual History')
        plt.yscale('log')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual history plot saved to {save_path}")
        
        if self.config.SHOW_PLOTS_INTERACTIVE:
            plt.show()
        else:
            plt.close()

class CurriculumLearning:
    """
    Curriculum learning strategy for PINNs
    """
    def __init__(self, config=None):
        if config is None:
            config = cfg
            
        self.config = config
        self.current_epoch = 0
        self.total_epochs = config.EPOCHS
        
        # Curriculum stages
        self.stages = [
            {'epoch': 0, 'pde_weight': 0.01, 'bc_weight': 50.0, 'data_weight': 100.0},
            {'epoch': int(0.1 * self.total_epochs), 'pde_weight': 0.1, 'bc_weight': 20.0, 'data_weight': 50.0},
            {'epoch': int(0.3 * self.total_epochs), 'pde_weight': 0.5, 'bc_weight': 10.0, 'data_weight': 20.0},
            {'epoch': int(0.6 * self.total_epochs), 'pde_weight': 1.0, 'bc_weight': 5.0, 'data_weight': 10.0},
            {'epoch': int(0.8 * self.total_epochs), 'pde_weight': 1.0, 'bc_weight': 1.0, 'data_weight': 1.0}
        ]
    
    def get_weights(self, epoch):
        """
        Get weights for the current epoch
        
        Args:
            epoch: Current epoch
            
        Returns:
            Dictionary of weights
        """
        self.current_epoch = epoch
        
        # Find the current stage
        current_stage = self.stages[0]
        for stage in self.stages:
            if epoch >= stage['epoch']:
                current_stage = stage
        
        # Get weights from current stage
        return {
            'pde': current_stage['pde_weight'],
            'bc': current_stage['bc_weight'],
            'data_u': current_stage['data_weight'],
            'data_v': current_stage['data_weight'],
            'data_p': current_stage['data_weight'] * 0.4  # Lower weight for pressure
        }
    
    def plot_curriculum(self, save_path=None):
        """
        Plot the curriculum learning schedule
        
        Args:
            save_path: Path to save the plot (optional)
        """
        epochs = [stage['epoch'] for stage in self.stages] + [self.total_epochs]
        pde_weights = [stage['pde_weight'] for stage in self.stages] + [self.stages[-1]['pde_weight']]
        bc_weights = [stage['bc_weight'] for stage in self.stages] + [self.stages[-1]['bc_weight']]
        data_weights = [stage['data_weight'] for stage in self.stages] + [self.stages[-1]['data_weight']]
        
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, pde_weights, 'o-', label='PDE Weight')
        plt.plot(epochs, bc_weights, 's-', label='BC Weight')
        plt.plot(epochs, data_weights, '^-', label='Data Weight')
        
        plt.axvline(x=self.current_epoch, color='r', linestyle='--', label='Current Epoch')
        
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.title('Curriculum Learning Schedule')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curriculum schedule plot saved to {save_path}")
        
        if self.config.SHOW_PLOTS_INTERACTIVE:
            plt.show()
        else:
            plt.close()

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
        
    # Unpack sparse data
    sparse_coords, sparse_u, sparse_v, sparse_p = sparse_data
    
    # Setup optimizer with gradient clipping
    if config.OPTIMIZER_TYPE == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Setup scheduler with warmup for Adam
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE,
            total_steps=config.EPOCHS,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    elif config.OPTIMIZER_TYPE == 'LBFGS':
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=config.LEARNING_RATE,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50
        )
        # Simple step scheduler for LBFGS
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1000,
            gamma=0.5
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.OPTIMIZER_TYPE}")
        
    # Initialize adaptive collocation sampler if enabled
    if config.USE_ADAPTIVE_SAMPLING:
        adaptive_sampler = AdaptiveCollocationSampler(
            model, 
            config, 
            initial_points=collocation_points,
            update_frequency=50  # More frequent updates
        )
        collocation_points = adaptive_sampler.get_collocation_points()
    
    # Initialize curriculum learning if enabled
    if config.USE_CURRICULUM_LEARNING:
        curriculum = CurriculumLearning(config)
        # Adjust curriculum stages for better progression
        curriculum.stages = [
            {'epoch': 0, 'pde_weight': 0.01, 'bc_weight': 50.0, 'data_weight': 100.0},
            {'epoch': int(0.1 * config.EPOCHS), 'pde_weight': 0.1, 'bc_weight': 20.0, 'data_weight': 50.0},
            {'epoch': int(0.3 * config.EPOCHS), 'pde_weight': 0.5, 'bc_weight': 10.0, 'data_weight': 20.0},
            {'epoch': int(0.6 * config.EPOCHS), 'pde_weight': 1.0, 'bc_weight': 5.0, 'data_weight': 10.0},
            {'epoch': int(0.8 * config.EPOCHS), 'pde_weight': 1.0, 'bc_weight': 1.0, 'data_weight': 1.0}
        ]
    
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
    
    # Plot training history
    plot_training_history(history, config)
    
    # Plot additional diagnostics
    if config.USE_ADAPTIVE_SAMPLING:
        adaptive_sampler.plot_residual_history(
            os.path.join(config.OUTPUT_DIR, 'adaptive_sampling_residuals.png')
        )
    
    if config.USE_CURRICULUM_LEARNING:
        curriculum.plot_curriculum(
            os.path.join(config.OUTPUT_DIR, 'curriculum_schedule.png')
        )
    
    return model, history

def plot_training_history(history, config=None):
    """
    Plot training history with advanced diagnostics
    
    Args:
        history: Training history dictionary
        config: Project configuration (optional, uses global cfg if None)
    """
    if config is None:
        config = cfg
        
    # Create figure with multiple subplots
    n_rows = 3 if config.USE_ADAPTIVE_WEIGHTS or config.USE_CURRICULUM_LEARNING or config.USE_REINIT_STRATEGY else 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    
    # Plot total loss
    axs[0, 0].semilogy(history['epoch'], history['loss_total'])
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Total Loss')
    axs[0, 0].set_title('Total Loss vs. Epoch')
    axs[0, 0].grid(True)
    
    # Plot component losses
    axs[0, 1].semilogy(history['epoch'], history['loss_pde'], label='PDE')
    axs[0, 1].semilogy(history['epoch'], history['loss_bc'], label='BC')
    axs[0, 1].semilogy(history['epoch'], history['loss_data'], label='Data')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Component Losses vs. Epoch')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot inferred parameter a
    axs[1, 0].plot(history['epoch'], history['inferred_a'])
    axs[1, 0].axhline(y=config.A_TRUE, color='r', linestyle='--', label=f'True a = {config.A_TRUE}')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Inferred a')
    axs[1, 0].set_title('Inferred Viscosity Parameter a vs. Epoch')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot relative error in a
    rel_error = [abs(a - config.A_TRUE) / config.A_TRUE * 100 for a in history['inferred_a']]
    axs[1, 1].semilogy(history['epoch'], rel_error)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Relative Error (%)')
    axs[1, 1].set_title('Relative Error in Inferred a vs. Epoch')
    axs[1, 1].grid(True)
    
    # Plot additional diagnostics if available
    if n_rows > 2:
        row_idx = 2
        
        # Plot adaptive weights if used
        if config.USE_ADAPTIVE_WEIGHTS and history['adaptive_weights'] is not None:
            momentum_x_weights = [w['momentum_x'] for w in history['adaptive_weights']]
            momentum_y_weights = [w['momentum_y'] for w in history['adaptive_weights']]
            continuity_weights = [w['continuity'] for w in history['adaptive_weights']]
            bc_weights = [w['bc'] for w in history['adaptive_weights']]
            data_weights = [w['data_u'] for w in history['adaptive_weights']]
            
            axs[row_idx, 0].semilogy(history['epoch'], momentum_x_weights, label='Momentum-x')
            axs[row_idx, 0].semilogy(history['epoch'], momentum_y_weights, label='Momentum-y')
            axs[row_idx, 0].semilogy(history['epoch'], continuity_weights, label='Continuity')
            axs[row_idx, 0].semilogy(history['epoch'], bc_weights, label='BC')
            axs[row_idx, 0].semilogy(history['epoch'], data_weights, label='Data')
            axs[row_idx, 0].set_xlabel('Epoch')
            axs[row_idx, 0].set_ylabel('Weight')
            axs[row_idx, 0].set_title('Adaptive Weights vs. Epoch')
            axs[row_idx, 0].legend()
            axs[row_idx, 0].grid(True)
        
        # Plot curriculum weights if used
        elif config.USE_CURRICULUM_LEARNING and history['curriculum_weights'] is not None:
            pde_weights = [w['pde'] for w in history['curriculum_weights']]
            bc_weights = [w['bc'] for w in history['curriculum_weights']]
            data_weights = [w['data_u'] for w in history['curriculum_weights']]
            
            axs[row_idx, 0].plot(history['epoch'], pde_weights, label='PDE')
            axs[row_idx, 0].plot(history['epoch'], bc_weights, label='BC')
            axs[row_idx, 0].plot(history['epoch'], data_weights, label='Data')
            axs[row_idx, 0].set_xlabel('Epoch')
            axs[row_idx, 0].set_ylabel('Weight')
            axs[row_idx, 0].set_title('Curriculum Weights vs. Epoch')
            axs[row_idx, 0].legend()
            axs[row_idx, 0].grid(True)
        
        # Plot re-initialization epochs if used
        if config.USE_REINIT_STRATEGY and history['reinitialization_epochs']:
            # Create a scatter plot of loss with vertical lines at reinit epochs
            axs[row_idx, 1].semilogy(history['epoch'], history['loss_total'])
            for reinit_epoch in history['reinitialization_epochs']:
                axs[row_idx, 1].axvline(x=reinit_epoch, color='r', linestyle='--')
            axs[row_idx, 1].set_xlabel('Epoch')
            axs[row_idx, 1].set_ylabel('Total Loss')
            axs[row_idx, 1].set_title('Loss with Re-initialization Events')
            axs[row_idx, 1].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    
    if config.SHOW_PLOTS_INTERACTIVE:
        plt.show()
    else:
        plt.close()
        
    print(f"Training history plot saved to {config.OUTPUT_DIR}/training_history.png")

if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Please use main.py to train the model.")

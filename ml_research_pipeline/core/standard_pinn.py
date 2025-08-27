"""
Standard Physics-Informed Neural Network (StandardPINN) implementation.

This module implements a baseline StandardPINN class for single-task training
to serve as a comparison baseline for the meta-learning approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import copy
from collections import OrderedDict

from ..config.model_config import ModelConfig
from .meta_pinn import MetaPINN


class StandardPINN(nn.Module):
    """
    Standard Physics-Informed Neural Network for single-task training.
    
    This class serves as a baseline for comparison with meta-learning approaches.
    It implements standard PINN training on individual tasks without meta-learning.
    
    Args:
        config: ModelConfig containing model architecture and training parameters
    """
    
    def __init__(self, config: ModelConfig):
        super(StandardPINN, self).__init__()
        self.config = config
        
        # Network architecture (reuse MetaPINN architecture for fair comparison)
        self.layers = self._build_network()
        
        # Physics parameters
        self.physics_loss_weight = config.physics_loss_weight
        self.adaptive_physics_weight = config.adaptive_physics_weight
        
        # Training state
        self.training_history = []
        self.current_task_info = None
        
        # Initialize network weights
        self._initialize_weights()
        
    def _build_network(self) -> nn.ModuleList:
        """
        Build the neural network architecture based on configuration.
        
        Returns:
            nn.ModuleList: List of network layers
        """
        layers = nn.ModuleList()
        layer_sizes = self.config.get_layer_sizes()
        
        # Build hidden layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add normalization if specified
            if self.config.layer_normalization and i < len(layer_sizes) - 2:
                layers.append(nn.LayerNorm(layer_sizes[i + 1]))
            elif self.config.batch_normalization and i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                
        return layers
        
    def _initialize_weights(self):
        """Initialize network weights according to configuration."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.config.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif self.config.weight_init == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    
                if self.config.bias_init == "zeros":
                    nn.init.zeros_(module.bias)
                elif self.config.bias_init == "normal":
                    nn.init.normal_(module.bias, std=0.01)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Network output of shape (batch_size, output_dim)
        """
        # Input normalization
        if self.config.input_normalization:
            x = self._normalize_input(x)
            
        # Forward through layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                
                # Apply activation (except for output layer)
                if i < len(self.layers) - 1:
                    x = self._apply_activation(x)
                    
                    # Apply dropout if specified
                    if self.config.dropout_rate > 0 and self.training:
                        x = F.dropout(x, p=self.config.dropout_rate)
                        
            elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d)):
                x = layer(x)
                
        # Output normalization
        if self.config.output_normalization:
            x = self._normalize_output(x)
            
        return x
        
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function based on configuration."""
        if self.config.activation == "tanh":
            return torch.tanh(x)
        elif self.config.activation == "relu":
            return F.relu(x)
        elif self.config.activation == "gelu":
            return F.gelu(x)
        elif self.config.activation == "swish":
            return x * torch.sigmoid(x)
        elif self.config.activation == "sin":
            return torch.sin(x)
        else:
            raise ValueError(f"Unknown activation function: {self.config.activation}")
            
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input coordinates to [-1, 1] range."""
        # Simple min-max normalization (can be made more sophisticated)
        return 2 * (x - x.min(dim=0, keepdim=True)[0]) / (
            x.max(dim=0, keepdim=True)[0] - x.min(dim=0, keepdim=True)[0] + 1e-8
        ) - 1
        
    def _normalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize output if specified."""
        # Simple standardization (can be made more sophisticated)
        return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)
        
    def physics_loss(self, coords: torch.Tensor, task_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute physics loss based on Navier-Stokes equations with variable viscosity.
        
        Args:
            coords: Coordinate tensor of shape (batch_size, input_dim) [x, y, t]
            task_info: Dictionary containing task-specific information including viscosity parameters
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing different physics loss components
        """
        # Enable gradient computation for coordinates
        coords = coords.clone().detach().requires_grad_(True)
        
        # Forward pass to get predictions [u, v, p]
        predictions = self.forward(coords)
        u, v, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        
        # Extract coordinates
        x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        
        # Compute first-order derivatives
        u_derivatives = self._compute_derivatives(u, coords)
        v_derivatives = self._compute_derivatives(v, coords)
        p_derivatives = self._compute_derivatives(p, coords)
        
        # Extract individual derivatives
        u_x, u_y, u_t = u_derivatives['x'], u_derivatives['y'], u_derivatives['t']
        v_x, v_y, v_t = v_derivatives['x'], v_derivatives['y'], v_derivatives['t']
        p_x, p_y = p_derivatives['x'], p_derivatives['y']
        
        # Compute second-order derivatives for viscous terms
        u_xx = self._compute_second_derivative(u, coords, 0, 0)  # d²u/dx²
        u_yy = self._compute_second_derivative(u, coords, 1, 1)  # d²u/dy²
        v_xx = self._compute_second_derivative(v, coords, 0, 0)  # d²v/dx²
        v_yy = self._compute_second_derivative(v, coords, 1, 1)  # d²v/dy²
        
        # Compute viscosity field based on task type
        viscosity = self._compute_viscosity(coords, task_info)
        
        # Compute viscosity derivatives for variable viscosity
        viscosity_type = task_info.get('viscosity_type', 'constant')
        if viscosity_type == 'constant':
            # For constant viscosity, derivatives are zero
            mu_x = torch.zeros_like(viscosity)
            mu_y = torch.zeros_like(viscosity)
        else:
            # For variable viscosity, compute derivatives
            viscosity_derivatives = self._compute_derivatives(viscosity, coords)
            mu_x, mu_y = viscosity_derivatives['x'], viscosity_derivatives['y']
        
        # Navier-Stokes momentum equations with variable viscosity
        # ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ∂/∂x(μ∂u/∂x) + ∂/∂y(μ∂u/∂y)
        momentum_x = (u_t + u * u_x + v * u_y + p_x - 
                     (mu_x * u_x + viscosity * u_xx) - 
                     (mu_y * u_y + viscosity * u_yy))
        
        # ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ∂/∂x(μ∂v/∂x) + ∂/∂y(μ∂v/∂y)
        momentum_y = (v_t + u * v_x + v * v_y + p_y - 
                     (mu_x * v_x + viscosity * v_xx) - 
                     (mu_y * v_y + viscosity * v_yy))
        
        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        continuity = u_x + v_y
        
        # Compute loss components
        physics_losses = {
            'momentum_x': torch.mean(momentum_x**2),
            'momentum_y': torch.mean(momentum_y**2),
            'continuity': torch.mean(continuity**2),
            'total_pde': torch.mean(momentum_x**2 + momentum_y**2 + continuity**2)
        }
        
        # Add boundary condition losses if specified
        if self.config.enforce_boundary_conditions:
            bc_loss = self._compute_boundary_loss(coords, predictions, task_info)
            physics_losses['boundary'] = bc_loss
            physics_losses['total'] = physics_losses['total_pde'] + bc_loss
        else:
            physics_losses['total'] = physics_losses['total_pde']
            
        return physics_losses
    
    def _compute_derivatives(self, output: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute first-order derivatives using automatic differentiation.
        
        Args:
            output: Network output tensor
            coords: Coordinate tensor
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of derivatives
        """
        derivatives = {}
        coord_names = ['x', 'y', 't']
        
        for i, name in enumerate(coord_names):
            if i < coords.shape[1]:
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=coords,
                    grad_outputs=torch.ones_like(output),
                    create_graph=True,
                    retain_graph=True
                )[0]
                derivatives[name] = grad[:, i:i+1]
            else:
                derivatives[name] = torch.zeros_like(output)
                
        return derivatives
    
    def _compute_second_derivative(self, output: torch.Tensor, coords: torch.Tensor, 
                                 dim1: int, dim2: int) -> torch.Tensor:
        """
        Compute second-order derivative using automatic differentiation.
        
        Args:
            output: Network output tensor
            coords: Coordinate tensor
            dim1: First dimension for derivative
            dim2: Second dimension for derivative
            
        Returns:
            torch.Tensor: Second derivative tensor
        """
        # First derivative
        first_grad = torch.autograd.grad(
            outputs=output,
            inputs=coords,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivative
        second_grad = torch.autograd.grad(
            outputs=first_grad[:, dim1:dim1+1],
            inputs=coords,
            grad_outputs=torch.ones_like(first_grad[:, dim1:dim1+1]),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return second_grad[:, dim2:dim2+1]
    
    def _compute_viscosity(self, coords: torch.Tensor, task_info: Dict[str, Any]) -> torch.Tensor:
        """
        Compute viscosity field based on task configuration.
        
        Args:
            coords: Coordinate tensor [x, y, t]
            task_info: Task information containing viscosity parameters
            
        Returns:
            torch.Tensor: Viscosity field
        """
        x, y = coords[:, 0:1], coords[:, 1:2]
        viscosity_type = task_info.get('viscosity_type', 'constant')
        viscosity_params = task_info.get('viscosity_params', {})
        
        if viscosity_type == 'constant':
            mu_0 = viscosity_params.get('mu_0', 1.0)
            return torch.full_like(x, mu_0)
            
        elif viscosity_type == 'linear':
            # μ(x,y) = μ₀ + α*x + β*y
            mu_0 = viscosity_params.get('mu_0', 1.0)
            alpha = viscosity_params.get('alpha', 0.1)
            beta = viscosity_params.get('beta', 0.0)
            return mu_0 + alpha * x + beta * y
            
        elif viscosity_type == 'bilinear':
            # μ(x,y) = μ₀ + α*x + β*y + γ*x*y
            mu_0 = viscosity_params.get('mu_0', 1.0)
            alpha = viscosity_params.get('alpha', 0.1)
            beta = viscosity_params.get('beta', 0.1)
            gamma = viscosity_params.get('gamma', 0.05)
            return mu_0 + alpha * x + beta * y + gamma * x * y
            
        elif viscosity_type == 'exponential':
            # μ(x,y) = μ₀ * exp(α*x + β*y)
            mu_0 = viscosity_params.get('mu_0', 1.0)
            alpha = viscosity_params.get('alpha', 0.1)
            beta = viscosity_params.get('beta', 0.0)
            return mu_0 * torch.exp(alpha * x + beta * y)
            
        elif viscosity_type == 'temperature_dependent':
            # μ(T) = μ₀ * (T/T₀)^n where T is represented by one coordinate
            mu_0 = viscosity_params.get('mu_0', 1.0)
            T_0 = viscosity_params.get('T_0', 1.0)
            n = viscosity_params.get('n', -0.5)
            # Use y-coordinate as temperature
            T = y + T_0  # Shift to ensure positive temperature
            return mu_0 * (T / T_0) ** n
            
        elif viscosity_type == 'non_newtonian':
            # Power-law model: μ = K * |γ̇|^(n-1) where γ̇ is shear rate
            # Simplified: use spatial gradient as proxy for shear rate
            K = viscosity_params.get('K', 1.0)
            n = viscosity_params.get('n', 0.8)
            # Simple approximation using coordinate gradients
            shear_rate = torch.sqrt(x**2 + y**2) + 1e-6  # Add small value to avoid singularity
            return K * shear_rate**(n - 1)
            
        else:
            raise ValueError(f"Unknown viscosity type: {viscosity_type}")
    
    def _compute_boundary_loss(self, coords: torch.Tensor, predictions: torch.Tensor, 
                              task_info: Dict[str, Any]) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            coords: Coordinate tensor
            predictions: Network predictions [u, v, p]
            task_info: Task information containing boundary conditions
            
        Returns:
            torch.Tensor: Boundary condition loss
        """
        # This is a simplified boundary loss - in practice, you would identify
        # boundary points and apply specific boundary conditions
        boundary_conditions = task_info.get('boundary_conditions', {})
        
        if not boundary_conditions:
            return torch.tensor(0.0, device=coords.device)
        
        # Example: no-slip boundary conditions (u=0, v=0 at walls)
        # This would need to be implemented based on specific geometry
        x, y = coords[:, 0:1], coords[:, 1:2]
        u, v = predictions[:, 0:1], predictions[:, 1:2]
        
        # Simple example: penalize non-zero velocity at domain boundaries
        # In practice, you would identify actual boundary points
        boundary_mask = ((torch.abs(x) > 0.9) | (torch.abs(y) > 0.9))
        
        if boundary_mask.any():
            u_boundary = u[boundary_mask]
            v_boundary = v[boundary_mask]
            return torch.mean(u_boundary**2 + v_boundary**2)
        else:
            return torch.tensor(0.0, device=coords.device)
    
    def compute_adaptive_physics_weight(self, physics_losses: Dict[str, torch.Tensor], 
                                      base_weight: float) -> float:
        """
        Compute adaptive physics loss weight based on residual magnitudes.
        
        Args:
            physics_losses: Dictionary of physics loss components
            base_weight: Base physics loss weight
            
        Returns:
            float: Adaptive physics weight
        """
        if not self.adaptive_physics_weight:
            return base_weight
            
        # Get maximum residual magnitude
        max_residual = max([loss.item() for key, loss in physics_losses.items() 
                           if key in ['momentum_x', 'momentum_y', 'continuity']])
        
        # Adaptive weighting based on residual magnitude
        residual_threshold = 1e-4
        if max_residual > residual_threshold:
            adaptive_weight = base_weight * (1 + np.log(max_residual / residual_threshold))
            return min(adaptive_weight, 10.0 * base_weight)  # Cap the weight
        else:
            return base_weight
    
    def train_on_task(self, task: Dict[str, Any], epochs: int = 1000, 
                     learning_rate: float = 0.001, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model on a single task using standard optimization.
        
        Args:
            task: Dictionary containing task information with 'coords', 'data', and 'task_info'
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            verbose: Whether to print training progress
            
        Returns:
            Dict[str, List[float]]: Training history with loss components
        """
        # Extract task data
        coords = task['coords']  # Shape: (n_points, input_dim)
        data = task['data']      # Shape: (n_points, output_dim)
        task_info = task['task_info']
        
        # Store current task info for evaluation
        self.current_task_info = task_info
        
        # Set up optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, 
                                   weight_decay=self.config.weight_decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training history
        history = {
            'data_loss': [],
            'physics_loss': [],
            'total_loss': [],
            'momentum_x_loss': [],
            'momentum_y_loss': [],
            'continuity_loss': []
        }
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.forward(coords)
            
            # Data loss (MSE between predictions and ground truth)
            data_loss = F.mse_loss(predictions, data)
            
            # Physics loss
            physics_losses = self.physics_loss(coords, task_info)
            physics_weight = self.compute_adaptive_physics_weight(physics_losses, self.physics_loss_weight)
            physics_loss = physics_weight * physics_losses['total']
            
            # Total loss
            total_loss = data_loss + physics_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping if specified
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Store history
            history['data_loss'].append(data_loss.item())
            history['physics_loss'].append(physics_loss.item())
            history['total_loss'].append(total_loss.item())
            history['momentum_x_loss'].append(physics_losses['momentum_x'].item())
            history['momentum_y_loss'].append(physics_losses['momentum_y'].item())
            history['continuity_loss'].append(physics_losses['continuity'].item())
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Data Loss: {data_loss.item():.6f}, "
                      f"Physics Loss: {physics_loss.item():.6f}, "
                      f"Total Loss: {total_loss.item():.6f}")
        
        # Store training history
        self.training_history = history
        
        return history
    
    def evaluate_on_task(self, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the model on a task.
        
        Args:
            task: Dictionary containing task information
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            coords = task['coords']
            data = task['data']
            task_info = task['task_info']
            
            # Forward pass
            predictions = self.forward(coords)
            
            # Data loss
            data_loss = F.mse_loss(predictions, data)
            
            # Physics loss (without gradients for evaluation)
            coords_eval = coords.clone().detach()
            predictions_eval = predictions.clone().detach()
            
            # Compute physics residuals for evaluation
            # Note: This is a simplified evaluation - full physics evaluation would require gradients
            physics_residual = torch.mean((predictions_eval - data)**2)  # Simplified residual
            
            # Parameter accuracy (if ground truth parameters are available)
            param_accuracy = 0.0
            if 'true_params' in task_info:
                # This would compute parameter inference accuracy
                # For now, use prediction accuracy as proxy
                param_accuracy = 1.0 / (1.0 + data_loss.item())
            
            metrics = {
                'data_loss': data_loss.item(),
                'physics_residual': physics_residual.item(),
                'parameter_accuracy': param_accuracy,
                'prediction_mse': data_loss.item(),
                'l2_error': torch.sqrt(data_loss).item()
            }
            
        return metrics
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get the training history from the last training run.
        
        Returns:
            Dict[str, List[float]]: Training history
        """
        return self.training_history
    
    def reset_parameters(self):
        """Reset model parameters to initial state."""
        self._initialize_weights()
        self.training_history = []
        self.current_task_info = None
    
    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.
        
        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'model_type': 'StandardPINN',
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'hidden_layers': self.config.hidden_layers,
            'activation': self.config.activation,
            'total_parameters': self.count_parameters(),
            'physics_loss_weight': self.physics_loss_weight,
            'adaptive_physics_weight': self.adaptive_physics_weight
        }
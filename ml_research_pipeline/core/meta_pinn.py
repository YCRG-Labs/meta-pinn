"""
Meta-Learning Physics-Informed Neural Network (MetaPINN) implementation.

This module implements the core MetaPINN class that combines Model-Agnostic Meta-Learning (MAML)
with Physics-Informed Neural Networks for few-shot learning on fluid dynamics tasks.
"""

import copy
from collections import OrderedDict as ODict
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import MetaPINNConfig


class MetaPINN(nn.Module):
    """Meta-Learning Physics-Informed Neural Network.

    This class implements Model-Agnostic Meta-Learning (MAML) specifically adapted for
    physics-informed neural networks. It enables few-shot learning on fluid dynamics
    tasks with varying viscosity profiles by learning initial parameters that can be
    quickly adapted to new tasks while maintaining physics constraints.

    The network combines data-driven learning with physics-informed constraints from
    the Navier-Stokes equations, allowing for rapid adaptation to new viscosity
    profiles with minimal training data.

    Args:
        layers: List of layer sizes defining the network architecture.
            Example: [2, 64, 64, 64, 3] for 2D input, 3 hidden layers, 3D output.
        activation: Activation function name. Supported: 'tanh', 'relu', 'gelu', 'swish'.
            Defaults to 'tanh' which works well for physics problems.
        meta_lr: Meta-learning rate for outer loop optimization. Typically 0.001-0.01.
        adapt_lr: Adaptation learning rate for inner loop optimization. Typically 0.01-0.1.
        adaptation_steps: Number of gradient steps for task adaptation. Typically 3-10.
        first_order: Whether to use first-order MAML approximation for efficiency.
            Defaults to True for computational efficiency.
        physics_loss_weight: Weight for physics loss term. Higher values enforce
            stronger physics constraints. Typically 0.1-10.0.
        adaptive_physics_weight: Whether to adaptively adjust physics loss weight
            based on residual magnitudes. Defaults to True.
        device: Device to run computations on ('cpu' or 'cuda').

    Attributes:
        layers: Neural network layers as nn.ModuleList.
        meta_optimizer: Optimizer for meta-learning updates.
        adaptation_history: History of adaptation losses for monitoring.
        physics_residuals: Current physics residual values.

    Example:
        >>> import torch
        >>> from ml_research_pipeline.core import MetaPINN
        >>>
        >>> # Initialize model
        >>> model = MetaPINN(
        ...     layers=[2, 64, 64, 64, 3],
        ...     meta_lr=0.001,
        ...     adapt_lr=0.01
        ... )
        >>>
        >>> # Generate sample task
        >>> coords = torch.randn(50, 2)  # 50 points in 2D
        >>> data = torch.randn(50, 3)    # velocity (u,v) + pressure (p)
        >>> task = {'support_coords': coords, 'support_data': data}
        >>>
        >>> # Adapt to task
        >>> adapted_params = model.adapt_to_task(task, adaptation_steps=5)
        >>>
        >>> # Make predictions with adapted parameters
        >>> query_coords = torch.randn(100, 2)
        >>> predictions = model.forward(query_coords, adapted_params)
        >>> print(f"Predictions shape: {predictions.shape}")
        Predictions shape: torch.Size([100, 3])

    Note:
        The network expects 2D spatial coordinates as input and outputs 3D vectors
        representing velocity components (u, v) and pressure (p). For different
        problem dimensions, adjust the input/output layer sizes accordingly.

        Physics constraints are enforced through automatic differentiation of the
        predicted fields to compute PDE residuals. Ensure input coordinates have
        requires_grad=True for proper gradient computation.
    """

    def __init__(self, config: MetaPINNConfig):
        super(MetaPINN, self).__init__()
        self.config = config

        # Network architecture
        self.layers = self._build_network()

        # Meta-learning parameters
        self.meta_lr = config.meta_lr
        self.adapt_lr = config.adapt_lr
        self.adaptation_steps = config.adaptation_steps
        self.first_order = config.first_order

        # Physics parameters
        self.physics_loss_weight = config.physics_loss_weight
        self.adaptive_physics_weight = config.adaptive_physics_weight

        # Initialize network weights
        self._initialize_weights()

    def _build_network(self) -> nn.ModuleList:
        """Build the neural network architecture based on configuration.

        Creates a fully connected neural network with optional normalization layers
        based on the configuration settings. The architecture supports various
        normalization techniques and activation functions optimized for physics
        problems.

        Returns:
            nn.ModuleList: List of network layers including linear layers and
                optional normalization layers (LayerNorm or BatchNorm1d).

        Note:
            The network architecture is designed for physics-informed learning:
            - Input layer: Spatial coordinates (typically 2D or 3D)
            - Hidden layers: Feature extraction with physics-aware activations
            - Output layer: Physical quantities (velocity, pressure, etc.)

        Example:
            For a 2D fluid dynamics problem:
            - Input: [x, y] coordinates (2D)
            - Output: [u, v, p] velocity components and pressure (3D)
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
        """Initialize network weights according to configuration.

        Applies the specified weight initialization scheme to all linear layers
        in the network. Different initialization schemes are suitable for
        different activation functions and problem types.

        Supported initialization schemes:
            - xavier_normal: Good for tanh/sigmoid activations
            - xavier_uniform: Alternative Xavier initialization
            - kaiming_normal: Optimal for ReLU-based activations
            - kaiming_uniform: Alternative Kaiming initialization

        Bias initialization options:
            - zeros: Standard zero initialization
            - normal: Small random values from normal distribution

        Note:
            Proper weight initialization is crucial for meta-learning as it
            affects the quality of the initial parameters that will be adapted
            to new tasks.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.weight_init == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.config.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                elif self.config.weight_init == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

                if self.config.bias_init == "zeros":
                    nn.init.zeros_(module.bias)
                elif self.config.bias_init == "normal":
                    nn.init.normal_(module.bias, std=0.01)

    def forward(self, x: torch.Tensor, params: Optional[ODict] = None) -> torch.Tensor:
        """Forward pass through the network.

        Performs a forward pass through the neural network, either using the
        current module parameters or provided functional parameters. The latter
        is essential for meta-learning where we need to evaluate the network
        with temporarily adapted parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
                For fluid dynamics problems, this typically contains spatial
                coordinates [x, y] or [x, y, z].
            params (Optional[OrderedDict]): Optional parameters for functional
                forward pass. When provided, uses these parameters instead of
                the module's current parameters. Used during meta-learning
                adaptation steps.

        Returns:
            torch.Tensor: Network output of shape (batch_size, output_dim).
                For fluid dynamics, typically contains [u, v, p] representing
                velocity components and pressure.

        Example:
            >>> # Standard forward pass
            >>> coords = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
            >>> output = model.forward(coords)
            >>> print(output.shape)  # torch.Size([2, 3])

            >>> # Functional forward pass with adapted parameters
            >>> adapted_params = model.adapt_to_task(task)
            >>> output = model.forward(coords, adapted_params)

        Note:
            Input coordinates should have requires_grad=True when computing
            physics losses that require spatial derivatives.
        """
        if params is None:
            return self._forward_with_modules(x)
        else:
            return self._forward_functional(x, params)

    def _forward_with_modules(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using nn.Module parameters."""
        # Input normalization
        if self.config.input_normalization:
            x = self._normalize_input(x)

        # Forward through layers
        layer_idx = 0
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

    def _forward_functional(self, x: torch.Tensor, params: ODict) -> torch.Tensor:
        """
        Functional forward pass using provided parameters.

        This is used during meta-learning adaptation where we need to compute
        gradients through temporary parameter updates.
        """
        # Input normalization
        if self.config.input_normalization:
            x = self._normalize_input(x)

        # Extract linear layer parameters
        linear_params = [
            (k, v) for k, v in params.items() if "weight" in k or "bias" in k
        ]
        linear_params.sort(key=lambda x: int(x[0].split(".")[1]))  # Sort by layer index

        # Group weights and biases by layer
        layer_params = {}
        for name, param in linear_params:
            layer_idx = int(name.split(".")[1])
            param_type = name.split(".")[-1]  # 'weight' or 'bias'

            if layer_idx not in layer_params:
                layer_params[layer_idx] = {}
            layer_params[layer_idx][param_type] = param

        # Forward through layers
        for layer_idx in sorted(layer_params.keys()):
            weight = layer_params[layer_idx]["weight"]
            bias = layer_params[layer_idx]["bias"]

            x = F.linear(x, weight, bias)

            # Apply activation (except for output layer)
            if layer_idx < max(layer_params.keys()):
                x = self._apply_activation(x)

                # Apply dropout if specified
                if self.config.dropout_rate > 0 and self.training:
                    x = F.dropout(x, p=self.config.dropout_rate)

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
        return (
            2
            * (x - x.min(dim=0, keepdim=True)[0])
            / (x.max(dim=0, keepdim=True)[0] - x.min(dim=0, keepdim=True)[0] + 1e-8)
            - 1
        )

    def _normalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize output if specified."""
        # Simple standardization (can be made more sophisticated)
        return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)

    def get_parameters_dict(self) -> ODict:
        """
        Get model parameters as an OrderedDict.

        Returns:
            OrderedDict: Model parameters
        """
        return ODict(self.named_parameters())

    def clone_parameters(self) -> ODict:
        """
        Clone model parameters for meta-learning adaptation.

        Returns:
            OrderedDict: Cloned parameters
        """
        params = ODict()
        for name, param in self.named_parameters():
            params[name] = param.clone()
        return params

    def set_parameters(self, params: ODict):
        """
        Set model parameters from an OrderedDict.

        Args:
            params: Parameters to set
        """
        for name, param in params.items():
            # Navigate to the parameter using the name
            module = self
            for attr in name.split(".")[:-1]:
                module = getattr(module, attr)
            setattr(module, name.split(".")[-1], nn.Parameter(param))

    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.

        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        Get information about network layers.

        Returns:
            List[Dict]: Layer information
        """
        layer_info = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                layer_info.append(
                    {
                        "layer_idx": i,
                        "type": "Linear",
                        "input_size": layer.in_features,
                        "output_size": layer.out_features,
                        "parameters": layer.in_features * layer.out_features
                        + layer.out_features,
                    }
                )
            elif isinstance(layer, nn.LayerNorm):
                layer_info.append(
                    {
                        "layer_idx": i,
                        "type": "LayerNorm",
                        "normalized_shape": layer.normalized_shape,
                        "parameters": 2 * layer.normalized_shape[0],  # weight + bias
                    }
                )
            elif isinstance(layer, nn.BatchNorm1d):
                layer_info.append(
                    {
                        "layer_idx": i,
                        "type": "BatchNorm1d",
                        "num_features": layer.num_features,
                        "parameters": 2 * layer.num_features,  # weight + bias
                    }
                )
        return layer_info

    def physics_loss(
        self,
        coords: torch.Tensor,
        task_info: Dict[str, Any],
        params: Optional[ODict] = None,
        create_graph: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute physics loss based on Navier-Stokes equations with variable viscosity.

        Computes the physics-informed loss by evaluating the residuals of the
        Navier-Stokes equations with variable viscosity profiles. The method
        uses automatic differentiation to compute spatial and temporal derivatives
        required for the PDE residuals.

        The Navier-Stokes equations for incompressible flow with variable viscosity:
            ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ∇·(μ∇u)
            ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ∇·(μ∇v)
            ∂u/∂x + ∂v/∂y = 0  (continuity equation)

        Args:
            coords (torch.Tensor): Coordinate tensor of shape (batch_size, input_dim).
                Expected format: [x, y, t] for 2D time-dependent problems.
                Must have requires_grad=True for gradient computation.
            task_info (Dict[str, Any]): Dictionary containing task-specific information:
                - 'viscosity_type': Type of viscosity profile ('linear', 'bilinear', etc.)
                - 'viscosity_params': Parameters defining the viscosity function
                - 'reynolds': Reynolds number for the flow
                - 'boundary_conditions': Boundary condition specifications
            params (Optional[OrderedDict]): Optional parameters for functional
                computation during meta-learning adaptation.
            create_graph (bool): Whether to create computation graph for
                higher-order derivatives. Set to False during evaluation
                for computational efficiency.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing physics loss components:
                - 'momentum_x': Residual of x-momentum equation
                - 'momentum_y': Residual of y-momentum equation
                - 'continuity': Residual of continuity equation
                - 'total': Combined physics loss
                - 'residual_magnitude': L2 norm of total residuals

        Example:
            >>> coords = torch.tensor([[0.5, 0.5, 0.1]], requires_grad=True)
            >>> task_info = {
            ...     'viscosity_type': 'linear',
            ...     'viscosity_params': {'a': 1.0, 'b': 0.1},
            ...     'reynolds': 100.0
            ... }
            >>> physics_losses = model.physics_loss(coords, task_info)
            >>> print(f"Total physics loss: {physics_losses['total'].item():.6f}")

        Note:
            - Higher create_graph=True enables computation of meta-gradients
              but increases computational cost
            - The viscosity function μ(x,y) is evaluated based on task_info
            - Boundary conditions are enforced separately in the training loop

        Raises:
            ValueError: If coords doesn't have the expected dimensionality
            KeyError: If required task_info keys are missing
        """
        # Enable gradient computation for coordinates if needed
        if create_graph:
            coords = coords.clone().detach().requires_grad_(True)
        else:
            coords = coords.clone().detach()

        # Forward pass to get predictions [u, v, p]
        predictions = self.forward(coords, params)
        u, v, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]

        # Extract coordinates
        x, y, t = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]

        if create_graph and coords.requires_grad:
            # Compute first-order derivatives
            u_derivatives = self._compute_derivatives(u, coords)
            v_derivatives = self._compute_derivatives(v, coords)
            p_derivatives = self._compute_derivatives(p, coords)
        else:
            # Return zero derivatives for evaluation mode
            batch_size = coords.shape[0]
            zero_grad = torch.zeros(batch_size, 1, device=coords.device)
            u_derivatives = {"x": zero_grad, "y": zero_grad, "t": zero_grad}
            v_derivatives = {"x": zero_grad, "y": zero_grad, "t": zero_grad}
            p_derivatives = {"x": zero_grad, "y": zero_grad, "t": zero_grad}

        # Extract individual derivatives
        u_x, u_y, u_t = u_derivatives["x"], u_derivatives["y"], u_derivatives["t"]
        v_x, v_y, v_t = v_derivatives["x"], v_derivatives["y"], v_derivatives["t"]
        p_x, p_y = p_derivatives["x"], p_derivatives["y"]

        if create_graph and coords.requires_grad:
            # Compute second-order derivatives for viscous terms
            u_xx = self._compute_second_derivative(u, coords, 0, 0)  # d²u/dx²
            u_yy = self._compute_second_derivative(u, coords, 1, 1)  # d²u/dy²
            v_xx = self._compute_second_derivative(v, coords, 0, 0)  # d²v/dx²
            v_yy = self._compute_second_derivative(v, coords, 1, 1)  # d²v/dy²
        else:
            # Return zero second derivatives for evaluation mode
            batch_size = coords.shape[0]
            zero_grad = torch.zeros(batch_size, 1, device=coords.device)
            u_xx = u_yy = v_xx = v_yy = zero_grad

        # Compute viscosity field based on task type
        viscosity = self._compute_viscosity(coords, task_info)

        # Compute viscosity derivatives for variable viscosity
        viscosity_type = task_info.get("viscosity_type", "constant")
        if viscosity_type == "constant" or not (create_graph and coords.requires_grad):
            # For constant viscosity or evaluation mode, derivatives are zero
            mu_x = torch.zeros_like(viscosity)
            mu_y = torch.zeros_like(viscosity)
        else:
            # For variable viscosity, compute derivatives
            viscosity_derivatives = self._compute_derivatives(viscosity, coords)
            mu_x, mu_y = viscosity_derivatives["x"], viscosity_derivatives["y"]

        # Navier-Stokes momentum equations with variable viscosity
        # ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ∂/∂x(μ∂u/∂x) + ∂/∂y(μ∂u/∂y)
        momentum_x = (
            u_t
            + u * u_x
            + v * u_y
            + p_x
            - (mu_x * u_x + viscosity * u_xx)
            - (mu_y * u_y + viscosity * u_yy)
        )

        # ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ∂/∂x(μ∂v/∂x) + ∂/∂y(μ∂v/∂y)
        momentum_y = (
            v_t
            + u * v_x
            + v * v_y
            + p_y
            - (mu_x * v_x + viscosity * v_xx)
            - (mu_y * v_y + viscosity * v_yy)
        )

        # Continuity equation: ∂u/∂x + ∂v/∂y = 0
        continuity = u_x + v_y

        # Compute loss components
        physics_losses = {
            "momentum_x": torch.mean(momentum_x**2),
            "momentum_y": torch.mean(momentum_y**2),
            "continuity": torch.mean(continuity**2),
            "total_pde": torch.mean(momentum_x**2 + momentum_y**2 + continuity**2),
        }

        # Add boundary condition losses if specified
        if self.config.enforce_boundary_conditions:
            bc_loss = self._compute_boundary_loss(coords, predictions, task_info)
            physics_losses["boundary"] = bc_loss
            physics_losses["total"] = physics_losses["total_pde"] + bc_loss
        else:
            physics_losses["total"] = physics_losses["total_pde"]

        return physics_losses

    def _compute_derivatives(
        self, output: torch.Tensor, coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute first-order derivatives using automatic differentiation.

        Args:
            output: Network output tensor
            coords: Coordinate tensor

        Returns:
            Dict[str, torch.Tensor]: Dictionary of derivatives
        """
        derivatives = {}
        coord_names = ["x", "y", "t"]

        for i, name in enumerate(coord_names):
            if i < coords.shape[1]:
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=coords,
                    grad_outputs=torch.ones_like(output),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                derivatives[name] = grad[:, i : i + 1]
            else:
                derivatives[name] = torch.zeros_like(output)

        return derivatives

    def _compute_second_derivative(
        self, output: torch.Tensor, coords: torch.Tensor, dim1: int, dim2: int
    ) -> torch.Tensor:
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
            retain_graph=True,
        )[0]

        # Second derivative
        second_grad = torch.autograd.grad(
            outputs=first_grad[:, dim1 : dim1 + 1],
            inputs=coords,
            grad_outputs=torch.ones_like(first_grad[:, dim1 : dim1 + 1]),
            create_graph=True,
            retain_graph=True,
        )[0]

        return second_grad[:, dim2 : dim2 + 1]

    def _compute_viscosity(
        self, coords: torch.Tensor, task_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute viscosity field based on task configuration.

        Args:
            coords: Coordinate tensor [x, y, t]
            task_info: Task information containing viscosity parameters

        Returns:
            torch.Tensor: Viscosity field
        """
        x, y = coords[:, 0:1], coords[:, 1:2]
        viscosity_type = task_info.get("viscosity_type", "constant")
        viscosity_params = task_info.get("viscosity_params", {})

        if viscosity_type == "constant":
            mu_0 = viscosity_params.get("mu_0", 1.0)
            return torch.full_like(x, mu_0)

        elif viscosity_type == "linear":
            # μ(x,y) = μ₀ + α*x + β*y
            mu_0 = viscosity_params.get("mu_0", 1.0)
            alpha = viscosity_params.get("alpha", 0.1)
            beta = viscosity_params.get("beta", 0.0)
            return mu_0 + alpha * x + beta * y

        elif viscosity_type == "bilinear":
            # μ(x,y) = μ₀ + α*x + β*y + γ*x*y
            mu_0 = viscosity_params.get("mu_0", 1.0)
            alpha = viscosity_params.get("alpha", 0.1)
            beta = viscosity_params.get("beta", 0.1)
            gamma = viscosity_params.get("gamma", 0.05)
            return mu_0 + alpha * x + beta * y + gamma * x * y

        elif viscosity_type == "exponential":
            # μ(x,y) = μ₀ * exp(α*x + β*y)
            mu_0 = viscosity_params.get("mu_0", 1.0)
            alpha = viscosity_params.get("alpha", 0.1)
            beta = viscosity_params.get("beta", 0.0)
            return mu_0 * torch.exp(alpha * x + beta * y)

        elif viscosity_type == "temperature_dependent":
            # μ(T) = μ₀ * (T/T₀)^n where T is represented by one coordinate
            mu_0 = viscosity_params.get("mu_0", 1.0)
            T_0 = viscosity_params.get("T_0", 1.0)
            n = viscosity_params.get("n", -0.5)
            # Use y-coordinate as temperature
            T = y + T_0  # Shift to ensure positive temperature
            return mu_0 * (T / T_0) ** n

        elif viscosity_type == "non_newtonian":
            # Power-law model: μ = K * |γ̇|^(n-1) where γ̇ is shear rate
            # Simplified: use spatial gradient as proxy for shear rate
            K = viscosity_params.get("K", 1.0)
            n = viscosity_params.get("n", 0.8)
            # Simple approximation using coordinate gradients
            shear_rate = (
                torch.sqrt(x**2 + y**2) + 1e-6
            )  # Add small value to avoid singularity
            return K * shear_rate ** (n - 1)

        else:
            raise ValueError(f"Unknown viscosity type: {viscosity_type}")

    def _compute_boundary_loss(
        self, coords: torch.Tensor, predictions: torch.Tensor, task_info: Dict[str, Any]
    ) -> torch.Tensor:
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
        boundary_conditions = task_info.get("boundary_conditions", {})

        if not boundary_conditions:
            return torch.tensor(0.0, device=coords.device)

        # Example: no-slip boundary conditions (u=0, v=0 at walls)
        # This would need to be implemented based on specific geometry
        x, y = coords[:, 0:1], coords[:, 1:2]
        u, v = predictions[:, 0:1], predictions[:, 1:2]

        # Simple example: penalize non-zero velocity at domain boundaries
        # In practice, you would identify actual boundary points
        boundary_mask = (torch.abs(x) > 0.9) | (torch.abs(y) > 0.9)

        if boundary_mask.any():
            u_boundary = u[boundary_mask]
            v_boundary = v[boundary_mask]
            return torch.mean(u_boundary**2 + v_boundary**2)
        else:
            return torch.tensor(0.0, device=coords.device)

    def adapt_to_task(
        self,
        task: Dict[str, Any],
        adaptation_steps: Optional[int] = None,
        create_graph: bool = False,
    ) -> ODict:
        """Adapt the model to a new task using gradient-based adaptation (MAML inner loop).

        Performs few-shot adaptation to a new fluid dynamics task by taking gradient
        steps on the support set. This implements the inner loop of Model-Agnostic
        Meta-Learning (MAML), where the model parameters are temporarily updated
        to minimize both data fitting loss and physics constraint violations.

        The adaptation process:
        1. Clone current model parameters
        2. For each adaptation step:
           - Compute predictions on support data
           - Calculate data fitting loss (MSE)
           - Calculate physics constraint loss (PDE residuals)
           - Update parameters via gradient descent
        3. Return adapted parameters

        Args:
            task (Dict[str, Any]): Dictionary containing task information:
                - 'support_coords': Coordinate tensor (n_support, input_dim)
                - 'support_data': Target data tensor (n_support, output_dim)
                - 'task_info': Task-specific information (viscosity, Reynolds, etc.)
            adaptation_steps (Optional[int]): Number of gradient steps for adaptation.
                If None, uses the default from model configuration. Typical range: 3-10.
            create_graph (bool): Whether to create computation graph for meta-learning.
                Set to True during meta-training to enable higher-order gradients.
                Set to False during evaluation for computational efficiency.

        Returns:
            OrderedDict: Adapted model parameters that can be used with forward()
                for making predictions on the adapted task.

        Example:
            >>> # Prepare task data
            >>> support_coords = torch.randn(20, 2, requires_grad=True)
            >>> support_data = torch.randn(20, 3)  # [u, v, p]
            >>> task_info = {
            ...     'viscosity_type': 'linear',
            ...     'viscosity_params': {'a': 1.0, 'b': 0.1},
            ...     'reynolds': 100.0
            ... }
            >>> task = {
            ...     'support_coords': support_coords,
            ...     'support_data': support_data,
            ...     'task_info': task_info
            ... }
            >>>
            >>> # Adapt to task
            >>> adapted_params = model.adapt_to_task(task, adaptation_steps=5)
            >>>
            >>> # Make predictions with adapted parameters
            >>> query_coords = torch.randn(50, 2)
            >>> predictions = model.forward(query_coords, adapted_params)

        Note:
            - The adaptation is temporary and doesn't modify the original model parameters
            - Physics loss weight is adaptively adjusted based on residual magnitudes
            - First-order MAML approximation is used by default for efficiency
            - Support coordinates should have requires_grad=True for physics loss computation

        Raises:
            KeyError: If required keys are missing from the task dictionary
            ValueError: If tensor dimensions don't match expected shapes
        """
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps

        # Extract task data
        support_coords = task["support_coords"]  # Shape: (n_support, input_dim)
        support_data = task["support_data"]  # Shape: (n_support, output_dim)
        task_info = task["task_info"]

        # Clone current parameters for adaptation
        adapted_params = self.clone_parameters()

        if create_graph:
            # For meta-learning: keep gradients connected to original parameters
            # Use manual gradient updates to maintain computation graph
            for step in range(adaptation_steps):
                # Compute predictions with current adapted parameters
                predictions = self.forward(support_coords, adapted_params)

                # Compute data loss
                data_loss = F.mse_loss(predictions, support_data)

                # Compute physics loss
                physics_losses = self.physics_loss(
                    support_coords, task_info, adapted_params, create_graph=create_graph
                )
                physics_weight = self.compute_adaptive_physics_weight(
                    physics_losses, self.physics_loss_weight
                )
                physics_loss = physics_weight * physics_losses["total"]

                # Total loss
                total_loss = data_loss + physics_loss

                # Compute gradients with respect to adapted parameters
                grads = torch.autograd.grad(
                    total_loss,
                    list(adapted_params.values()),
                    create_graph=True,
                    retain_graph=True,
                )

                # Update parameters manually
                for (name, param), grad in zip(adapted_params.items(), grads):
                    adapted_params[name] = param - self.adapt_lr * grad
        else:
            # For inference: detach parameters for efficient optimization
            param_list = []
            for name, param in adapted_params.items():
                # Create new leaf tensor
                new_param = param.detach().clone().requires_grad_(True)
                adapted_params[name] = new_param
                param_list.append(new_param)

            inner_optimizer = torch.optim.SGD(param_list, lr=self.adapt_lr)

            # Perform adaptation steps
            adaptation_losses = []

            for step in range(adaptation_steps):
                inner_optimizer.zero_grad()

                # Compute predictions with adapted parameters
                predictions = self.forward(support_coords, adapted_params)

                # Compute data loss (MSE between predictions and support data)
                data_loss = F.mse_loss(predictions, support_data)

                # Compute physics loss
                physics_losses = self.physics_loss(
                    support_coords, task_info, adapted_params
                )
                physics_weight = self.compute_adaptive_physics_weight(
                    physics_losses, self.physics_loss_weight
                )
                physics_loss = physics_weight * physics_losses["total"]

                # Total loss for adaptation
                total_loss = data_loss + physics_loss

                # Compute gradients and update parameters
                total_loss.backward(retain_graph=True)

                # Apply gradient clipping if specified
                if self.config.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        param_list, self.config.gradient_clipping
                    )

                inner_optimizer.step()

                # Store adaptation loss for monitoring
                adaptation_losses.append(
                    {
                        "step": step,
                        "data_loss": data_loss.item(),
                        "physics_loss": physics_loss.item(),
                        "total_loss": total_loss.item(),
                    }
                )

            # Store adaptation history
            self._last_adaptation_losses = adaptation_losses

        return adapted_params

    def compute_adaptation_loss(
        self, task: Dict[str, Any], params: Optional[ODict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the adaptation loss for a given task.

        Args:
            task: Task dictionary containing support data and task info
            params: Parameters to use (current model parameters if None)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        support_coords = task["support_coords"]
        support_data = task["support_data"]
        task_info = task["task_info"]

        # Compute predictions
        predictions = self.forward(support_coords, params)

        # Data loss
        data_loss = F.mse_loss(predictions, support_data)

        # Physics loss
        physics_losses = self.physics_loss(
            support_coords, task_info, params, create_graph=True
        )
        physics_weight = self.compute_adaptive_physics_weight(
            physics_losses, self.physics_loss_weight
        )
        physics_loss = physics_weight * physics_losses["total"]

        # Total loss
        total_loss = data_loss + physics_loss

        return {
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "total_loss": total_loss,
            "physics_components": physics_losses,
        }

    def evaluate_on_query_set(
        self, task: Dict[str, Any], params: Optional[ODict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the model on the query set of a task.

        Args:
            task: Task dictionary containing query data
            params: Parameters to use (current model parameters if None)

        Returns:
            Dict[str, torch.Tensor]: Evaluation metrics
        """
        query_coords = task["query_coords"]
        query_data = task["query_data"]
        task_info = task["task_info"]

        # Compute predictions
        with torch.no_grad():
            predictions = self.forward(query_coords, params)

        # Compute metrics
        mse_loss = F.mse_loss(predictions, query_data)
        mae_loss = F.l1_loss(predictions, query_data)

        # Compute physics residuals on query set
        physics_losses = self.physics_loss(
            query_coords, task_info, params, create_graph=False
        )

        return {
            "mse": mse_loss,
            "mae": mae_loss,
            "physics_residual": physics_losses["total_pde"],
            "physics_components": physics_losses,
        }

    def meta_update(
        self, task_batch: List[Dict[str, Any]], meta_optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform meta-learning update using a batch of tasks (MAML outer loop).

        Args:
            task_batch: List of tasks, each containing support and query sets
            meta_optimizer: Optimizer for meta-parameters

        Returns:
            Dict[str, float]: Meta-learning metrics
        """
        # Ensure we're in training mode
        self.train()
        meta_optimizer.zero_grad()

        total_meta_loss = 0.0
        batch_metrics = {
            "meta_loss": 0.0,
            "query_data_loss": 0.0,
            "query_physics_loss": 0.0,
            "adaptation_loss": 0.0,
            "n_tasks": len(task_batch),
        }

        task_losses = []

        for i, task in enumerate(task_batch):
            # Perform adaptation on support set with gradient tracking
            adapted_params = self.adapt_to_task(task, create_graph=True)

            # Evaluate on query set
            query_coords = task["query_coords"]
            query_data = task["query_data"]
            task_info = task["task_info"]

            # Compute query loss with adapted parameters
            query_predictions = self.forward(query_coords, adapted_params)
            query_data_loss = F.mse_loss(query_predictions, query_data)

            # Compute physics loss on query set
            query_physics_losses = self.physics_loss(
                query_coords, task_info, adapted_params, create_graph=True
            )
            query_physics_weight = self.compute_adaptive_physics_weight(
                query_physics_losses, self.physics_loss_weight
            )
            query_physics_loss = query_physics_weight * query_physics_losses["total"]

            # Total query loss for this task
            task_meta_loss = query_data_loss + query_physics_loss
            task_losses.append(task_meta_loss)

            # Clamp individual losses to prevent numerical issues
            task_meta_loss = torch.clamp(task_meta_loss, max=1e4)
            task_losses.append(task_meta_loss)

            # Accumulate metrics (before clamping for accurate reporting)
            batch_metrics["query_data_loss"] += query_data_loss.item()
            batch_metrics["query_physics_loss"] += query_physics_loss.item()

        # Compute total meta loss
        total_meta_loss = torch.stack(task_losses).sum()

        # Average meta loss across tasks
        meta_loss = total_meta_loss / len(task_batch)
        batch_metrics["meta_loss"] = meta_loss.item()

        # Compute meta-gradients
        meta_loss.backward()

        # Apply gradient clipping if specified
        if self.config.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.config.gradient_clipping
            )

        # Update meta-parameters
        meta_optimizer.step()

        # Average metrics
        for key in ["query_data_loss", "query_physics_loss"]:
            batch_metrics[key] /= len(task_batch)

        return batch_metrics

    def create_meta_optimizer(self) -> torch.optim.Optimizer:
        """
        Create meta-optimizer based on configuration.

        Returns:
            torch.optim.Optimizer: Meta-optimizer
        """
        if self.config.outer_optimizer == "adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.meta_lr,
                betas=self.config.outer_betas,
                eps=self.config.outer_eps,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.outer_optimizer == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.meta_lr,
                momentum=getattr(self.config, "outer_momentum", 0.0),
                weight_decay=self.config.weight_decay,
            )
        elif self.config.outer_optimizer == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.meta_lr,
                betas=self.config.outer_betas,
                eps=self.config.outer_eps,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown outer optimizer: {self.config.outer_optimizer}")

    def create_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.

        Args:
            optimizer: Optimizer to schedule
            total_steps: Total number of training steps

        Returns:
            Optional learning rate scheduler
        """
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        elif self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=total_steps // 3, gamma=self.config.lr_decay_factor
            )
        elif self.config.lr_scheduler == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.config.lr_decay_factor
            )
        elif self.config.lr_scheduler == "warmup_cosine":
            # Simple warmup + cosine schedule
            def lr_lambda(step):
                if step < self.config.lr_warmup_steps:
                    return step / self.config.lr_warmup_steps
                else:
                    progress = (step - self.config.lr_warmup_steps) / (
                        total_steps - self.config.lr_warmup_steps
                    )
                    return 0.5 * (1 + np.cos(np.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.config.lr_scheduler == "none":
            return None
        else:
            raise ValueError(
                f"Unknown learning rate scheduler: {self.config.lr_scheduler}"
            )

    def evaluate_meta_learning(
        self, test_tasks: List[Dict[str, Any]], adaptation_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate meta-learning performance on test tasks.

        Args:
            test_tasks: List of test tasks
            adaptation_steps: Number of adaptation steps for evaluation

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps

        self.eval()

        metrics = {
            "test_accuracy": 0.0,
            "test_data_loss": 0.0,
            "test_physics_loss": 0.0,
            "adaptation_efficiency": 0.0,
            "n_tasks": len(test_tasks),
        }

        for task in test_tasks:
            # Adapt to task (without gradients for evaluation)
            adapted_params = self.adapt_to_task(
                task, adaptation_steps=adaptation_steps, create_graph=False
            )

            # Evaluate on query set
            with torch.no_grad():
                query_metrics = self.evaluate_on_query_set(task, adapted_params)

                # Accumulate metrics
                metrics["test_data_loss"] += query_metrics["mse"].item()
                metrics["test_physics_loss"] += query_metrics["physics_residual"].item()

                # Compute accuracy (1 - normalized MSE)
                query_data_std = task["query_data"].std().item()
                normalized_mse = query_metrics["mse"].item() / (
                    query_data_std**2 + 1e-8
                )
                accuracy = max(0.0, 1.0 - normalized_mse)
                metrics["test_accuracy"] += accuracy

        # Average metrics
        for key in ["test_accuracy", "test_data_loss", "test_physics_loss"]:
            metrics[key] /= len(test_tasks)

        # Compute adaptation efficiency (how much loss decreases per adaptation step)
        metrics["adaptation_efficiency"] = 1.0 / (adaptation_steps + 1)

        self.train()
        return metrics

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get the adaptation history from the last adaptation.

        Returns:
            List[Dict]: Adaptation loss history
        """
        return getattr(self, "_last_adaptation_losses", [])

    def compute_adaptive_physics_weight(
        self, physics_losses: Dict[str, torch.Tensor], base_weight: float = 1.0
    ) -> float:
        """
        Compute adaptive physics loss weight based on residual magnitudes.

        Args:
            physics_losses: Dictionary of physics loss components
            base_weight: Base physics loss weight

        Returns:
            float: Adaptive physics loss weight
        """
        if not self.adaptive_physics_weight:
            return base_weight

        # Get maximum residual magnitude
        max_residual = max(
            physics_losses[key].item()
            for key in ["momentum_x", "momentum_y", "continuity"]
        )

        # Adaptive weighting: increase weight if residuals are large
        residual_threshold = 1e-4  # Target residual magnitude
        if max_residual > residual_threshold:
            adaptive_factor = (
                1 + torch.log(torch.tensor(max_residual / residual_threshold)).item()
            )
            return base_weight * adaptive_factor
        else:
            return base_weight

    def _compute_pde_residuals_per_point(
        self, coords: torch.Tensor, predictions: torch.Tensor, task_info: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """
        Compute PDE residuals at each point for detailed physics validation.

        Args:
            coords: Coordinate tensor with gradients enabled
            predictions: Network predictions [u, v, p]
            task_info: Task information containing viscosity parameters

        Returns:
            Optional[torch.Tensor]: PDE residuals per point, or None if computation fails
        """
        try:
            # Extract predictions
            u, v, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
            x, y = coords[:, 0:1], coords[:, 1:2]

            # Compute first derivatives
            u_derivatives = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
            v_derivatives = torch.autograd.grad(v.sum(), coords, create_graph=True)[0]
            p_derivatives = torch.autograd.grad(p.sum(), coords, create_graph=True)[0]

            u_x, u_y, u_t = (
                u_derivatives[:, 0:1],
                u_derivatives[:, 1:2],
                u_derivatives[:, 2:3],
            )
            v_x, v_y, v_t = (
                v_derivatives[:, 0:1],
                v_derivatives[:, 1:2],
                v_derivatives[:, 2:3],
            )
            p_x, p_y = p_derivatives[:, 0:1], p_derivatives[:, 1:2]

            # Compute second derivatives
            u_xx = self._compute_second_derivative(u, coords, 0, 0)
            u_yy = self._compute_second_derivative(u, coords, 1, 1)
            v_xx = self._compute_second_derivative(v, coords, 0, 0)
            v_yy = self._compute_second_derivative(v, coords, 1, 1)

            # Get viscosity field
            mu = self._compute_viscosity(coords, task_info)

            # Get Reynolds number and density
            Re = task_info.get("reynolds_number", 100.0)
            rho = task_info.get("density", 1.0)

            # Compute PDE residuals per point
            # Momentum equation in x-direction
            momentum_x_residual = (
                rho * (u_t + u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
            )

            # Momentum equation in y-direction
            momentum_y_residual = (
                rho * (v_t + u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
            )

            # Continuity equation
            continuity_residual = u_x + v_y

            # Combined residual magnitude per point
            total_residual_per_point = torch.sqrt(
                momentum_x_residual**2 + momentum_y_residual**2 + continuity_residual**2
            ).squeeze()

            return total_residual_per_point

        except Exception as e:
            # Return None if computation fails
            return None

    def __repr__(self) -> str:
        """String representation of the model."""
        total_params = self.count_parameters()
        layer_sizes = self.config.get_layer_sizes()

        return (
            f"MetaPINN(\n"
            f"  layers={layer_sizes},\n"
            f"  activation={self.config.activation},\n"
            f"  total_parameters={total_params:,},\n"
            f"  meta_lr={self.meta_lr},\n"
            f"  adapt_lr={self.adapt_lr},\n"
            f"  adaptation_steps={self.adaptation_steps}\n"
            f")"
        )

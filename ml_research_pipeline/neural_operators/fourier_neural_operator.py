"""
Inverse Fourier Neural Operator for Parameter Inference

This module implements a Fourier Neural Operator (FNO) designed for inverse problems
in physics-informed learning, specifically for mapping sparse observations to
parameter fields in fluid dynamics problems.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer using FFT

    Implements the core spectral convolution operation in Fourier space
    with learnable complex-valued weights for specified modes.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Initialize spectral convolution layer

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Initialize complex-valued weights for Fourier modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication in Fourier space

        Args:
            input: Input tensor in Fourier domain
            weights: Complex-valued weight tensor

        Returns:
            Result of complex multiplication
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor after spectral convolution
        """
        batchsize = x.shape[0]

        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Ensure we don't exceed the available modes
        modes1_actual = min(self.modes1, x_ft.size(-2))
        modes2_actual = min(self.modes2, x_ft.size(-1))

        out_ft[:, :, :modes1_actual, :modes2_actual] = self.compl_mul2d(
            x_ft[:, :, :modes1_actual, :modes2_actual],
            self.weights1[:, :, :modes1_actual, :modes2_actual],
        )

        if modes1_actual < x_ft.size(-2):
            out_ft[:, :, -modes1_actual:, :modes2_actual] = self.compl_mul2d(
                x_ft[:, :, -modes1_actual:, :modes2_actual],
                self.weights2[:, :, :modes1_actual, :modes2_actual],
            )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FourierLayer(nn.Module):
    """
    Complete Fourier layer combining spectral convolution with local convolution
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Initialize Fourier layer

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super(FourierLayer, self).__init__()

        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.local_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining spectral and local convolutions

        Args:
            x: Input tensor

        Returns:
            Output after Fourier layer processing
        """
        return self.spectral_conv(x) + self.local_conv(x)


class InverseFourierNeuralOperator(nn.Module):
    """
    Inverse Fourier Neural Operator for parameter inference from sparse observations

    Maps sparse velocity/pressure observations to viscosity parameter fields
    for physics-informed learning applications.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 64,
        input_channels: int = 3,  # x, y, observations
        output_channels: int = 1,  # viscosity field
        n_layers: int = 4,
        grid_size: Tuple[int, int] = (64, 64),
    ):
        """
        Initialize Inverse FNO

        Args:
            modes1: Number of Fourier modes in x-direction
            modes2: Number of Fourier modes in y-direction
            width: Width of hidden layers
            input_channels: Number of input channels (coordinates + observations)
            output_channels: Number of output channels (parameter fields)
            n_layers: Number of Fourier layers
            grid_size: Size of computational grid
        """
        super(InverseFourierNeuralOperator, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.grid_size = grid_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Input projection
        self.fc0 = nn.Linear(input_channels, self.width)

        # Fourier layers
        self.fourier_layers = nn.ModuleList(
            [
                FourierLayer(self.width, self.width, self.modes1, self.modes2)
                for _ in range(self.n_layers)
            ]
        )

        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_channels)

        # Activation function
        self.activation = F.gelu

    def grid_to_tensor(
        self, sparse_observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert sparse observations to grid tensor format

        Args:
            sparse_observations: Dictionary containing:
                - 'coords': Observation coordinates (N, 2)
                - 'values': Observation values (N, n_vars)
                - 'grid_coords': Grid coordinates (H, W, 2)

        Returns:
            Grid tensor of shape (batch, channels, height, width)
        """
        coords = sparse_observations["coords"]  # (N, 2)
        values = sparse_observations["values"]  # (N, n_vars)
        grid_coords = sparse_observations["grid_coords"]  # (H, W, 2)

        batch_size = values.shape[0] if len(values.shape) == 3 else 1
        if len(values.shape) == 2:
            values = values.unsqueeze(0)
            coords = coords.unsqueeze(0)

        H, W = self.grid_size
        device = values.device

        # Initialize grid tensor
        grid_tensor = torch.zeros(batch_size, self.input_channels, H, W, device=device)

        # Fill coordinate channels
        grid_tensor[:, 0, :, :] = (
            grid_coords[:, :, 0].unsqueeze(0).expand(batch_size, -1, -1)
        )
        grid_tensor[:, 1, :, :] = (
            grid_coords[:, :, 1].unsqueeze(0).expand(batch_size, -1, -1)
        )

        # Interpolate sparse observations to grid
        for b in range(batch_size):
            for i, coord in enumerate(coords[b]):
                # Find nearest grid points
                x_idx = torch.clamp(torch.round(coord[0] * (W - 1)).long(), 0, W - 1)
                y_idx = torch.clamp(torch.round(coord[1] * (H - 1)).long(), 0, H - 1)

                # Assign observation values to grid
                if len(values.shape) == 3 and values.shape[2] > 0:
                    grid_tensor[b, 2 : 2 + values.shape[2], y_idx, x_idx] = values[
                        b, i, :
                    ]

        return grid_tensor

    def forward(self, sparse_observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass mapping sparse observations to parameter field

        Args:
            sparse_observations: Either tensor of shape (batch, channels, H, W) or
                                dict with sparse observation data

        Returns:
            Parameter field tensor of shape (batch, output_channels, H, W)
        """
        if isinstance(sparse_observations, dict):
            x = self.grid_to_tensor(sparse_observations)
        else:
            x = sparse_observations

        # Input projection
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, channels)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, H, W)

        # Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)
            x = self.activation(x)

        # Output projection
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, channels)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # (batch, output_channels, H, W)

        return x

    def reconstruct_parameter_field(
        self,
        sparse_coords: torch.Tensor,
        sparse_values: torch.Tensor,
        grid_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct parameter field from sparse observations

        Args:
            sparse_coords: Sparse observation coordinates (N, 2)
            sparse_values: Sparse observation values (N, n_vars)
            grid_coords: Grid coordinates for reconstruction (H, W, 2)

        Returns:
            Reconstructed parameter field (1, output_channels, H, W)
        """
        sparse_obs = {
            "coords": sparse_coords.unsqueeze(0),
            "values": sparse_values.unsqueeze(0),
            "grid_coords": grid_coords,
        }

        return self.forward(sparse_obs)

    def compute_reconstruction_loss(
        self,
        predicted_field: torch.Tensor,
        target_field: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between predicted and target parameter fields

        Args:
            predicted_field: Predicted parameter field
            target_field: Target parameter field
            loss_type: Type of loss ('mse', 'l1', 'huber')

        Returns:
            Reconstruction loss
        """
        if loss_type == "mse":
            return F.mse_loss(predicted_field, target_field)
        elif loss_type == "l1":
            return F.l1_loss(predicted_field, target_field)
        elif loss_type == "huber":
            return F.huber_loss(predicted_field, target_field)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def get_fourier_modes_info(self) -> Dict[str, Any]:
        """
        Get information about Fourier modes and model configuration

        Returns:
            Dictionary with model configuration information
        """
        return {
            "modes1": self.modes1,
            "modes2": self.modes2,
            "width": self.width,
            "n_layers": self.n_layers,
            "grid_size": self.grid_size,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

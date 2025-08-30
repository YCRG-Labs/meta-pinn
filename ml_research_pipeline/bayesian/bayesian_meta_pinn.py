"""
Bayesian Meta-Learning Physics-Informed Neural Network implementation.

This module implements a Bayesian extension of MetaPINN using variational inference
to quantify epistemic and aleatoric uncertainty in physics-informed predictions.
"""

import math
from collections import OrderedDict as ODict
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import MetaPINNConfig
from ..core.meta_pinn import MetaPINN


class VariationalLinear(nn.Module):
    """
    Variational linear layer with learnable mean and log-variance parameters.

    Implements Bayes by Backprop for weight uncertainty quantification.
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super(VariationalLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Weight parameters (mean and log-variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias parameters (mean and log-variance)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize variational parameters."""
        # Initialize means with Xavier normal
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.zeros_(self.bias_mu)

        # Initialize log-variances to small negative values (small initial variance)
        nn.init.constant_(self.weight_logvar, -5.0)
        nn.init.constant_(self.bias_logvar, -5.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass with weight sampling.

        Args:
            x: Input tensor
            sample: Whether to sample weights (True) or use mean (False)

        Returns:
            torch.Tensor: Output tensor
        """
        if sample and self.training:
            # Sample weights from variational distribution
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)

            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)

            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean parameters (deterministic)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between variational posterior and prior.

        Returns:
            torch.Tensor: KL divergence
        """
        # Prior: N(0, prior_std²)
        # Posterior: N(μ, σ²) where σ² = exp(log_var)

        # KL[q(w)||p(w)] = 0.5 * (σ²/σ₀² + μ²/σ₀² - 1 - log(σ²/σ₀²))
        # where σ₀² is prior variance

        prior_var = self.prior_std**2

        # Weight KL divergence
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * (
            weight_var / prior_var
            + self.weight_mu**2 / prior_var
            - 1
            - torch.log(weight_var / prior_var)
        )

        # Bias KL divergence
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * (
            bias_var / prior_var
            + self.bias_mu**2 / prior_var
            - 1
            - torch.log(bias_var / prior_var)
        )

        return torch.sum(weight_kl) + torch.sum(bias_kl)


class BayesianMetaPINN(MetaPINN):
    """
    Bayesian Meta-Learning Physics-Informed Neural Network.

    Extends MetaPINN with variational inference for uncertainty quantification.
    Provides both epistemic (model) and aleatoric (data) uncertainty estimates.
    """

    def __init__(
        self,
        config: MetaPINNConfig,
        prior_std: float = 1.0,
        kl_weight: float = 1e-4,
        n_mc_samples: int = 100,
    ):
        """
        Initialize Bayesian MetaPINN.

        Args:
            config: MetaPINN configuration
            prior_std: Standard deviation of weight priors
            kl_weight: Weight for KL divergence regularization
            n_mc_samples: Number of Monte Carlo samples for uncertainty estimation
        """
        # Initialize parent class but don't build network yet
        nn.Module.__init__(self)
        self.config = config

        # Bayesian-specific parameters
        self.prior_std = prior_std
        self.kl_weight = kl_weight
        self.n_mc_samples = n_mc_samples

        # Meta-learning parameters
        self.meta_lr = config.meta_lr
        self.adapt_lr = config.adapt_lr
        self.adaptation_steps = config.adaptation_steps
        self.first_order = config.first_order

        # Physics parameters
        self.physics_loss_weight = config.physics_loss_weight
        self.adaptive_physics_weight = config.adaptive_physics_weight

        # Build variational network
        self.layers = self._build_variational_network()

        # Initialize weights
        self._initialize_weights()

    def _build_variational_network(self) -> nn.ModuleList:
        """
        Build variational neural network with uncertainty quantification.

        Returns:
            nn.ModuleList: List of variational layers
        """
        layers = nn.ModuleList()
        layer_sizes = self.config.get_layer_sizes()

        # Build variational linear layers
        for i in range(len(layer_sizes) - 1):
            layers.append(
                VariationalLinear(
                    layer_sizes[i], layer_sizes[i + 1], prior_std=self.prior_std
                )
            )

            # Add normalization layers (deterministic)
            if self.config.layer_normalization and i < len(layer_sizes) - 2:
                layers.append(nn.LayerNorm(layer_sizes[i + 1]))
            elif self.config.batch_normalization and i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))

        return layers

    def forward(
        self, x: torch.Tensor, params: Optional[ODict] = None, sample: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through variational network.

        Args:
            x: Input tensor
            params: Optional parameters for functional forward pass
            sample: Whether to sample from variational distributions

        Returns:
            torch.Tensor: Network output
        """
        if params is None:
            return self._forward_variational(x, sample=sample)
        else:
            # For meta-learning with provided parameters, use deterministic forward
            return self._forward_functional(x, params)

    def _forward_variational(
        self, x: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """Forward pass using variational layers."""
        # Input normalization
        if self.config.input_normalization:
            x = self._normalize_input(x)

        # Forward through variational layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, VariationalLinear):
                x = layer(x, sample=sample)

                # Apply activation (except for output layer)
                if (
                    i
                    < len([l for l in self.layers if isinstance(l, VariationalLinear)])
                    - 1
                ):
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

    def forward_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None,
        params: Optional[ODict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty quantification using Monte Carlo sampling.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            n_samples: Number of Monte Carlo samples (uses default if None)
            params: Optional parameters for functional forward pass

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mean_predictions, uncertainty_std)
                - mean_predictions: Shape (batch_size, output_dim)
                - uncertainty_std: Shape (batch_size, output_dim)
        """
        if n_samples is None:
            n_samples = self.n_mc_samples

        if params is not None:
            # For meta-learning case, use deterministic forward pass
            # Uncertainty comes from adaptation process rather than weight sampling
            predictions = self._forward_functional(x, params)
            # Return zero uncertainty for deterministic case
            uncertainty = torch.zeros_like(predictions)
            return predictions, uncertainty

        # Monte Carlo sampling for uncertainty estimation
        self.train()  # Ensure sampling is enabled

        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                sample_pred = self._forward_variational(x, sample=True)
                samples.append(sample_pred)

        # Stack samples and compute statistics
        samples = torch.stack(
            samples, dim=0
        )  # Shape: (n_samples, batch_size, output_dim)

        # Compute mean and standard deviation
        mean_predictions = torch.mean(samples, dim=0)
        uncertainty_std = torch.std(samples, dim=0)

        return mean_predictions, uncertainty_std

    def sample_weights(self, n_samples: int = 1) -> List[ODict]:
        """
        Sample weight configurations from variational distributions.

        Args:
            n_samples: Number of weight samples to generate

        Returns:
            List[OrderedDict]: List of sampled weight dictionaries
        """
        weight_samples = []

        for _ in range(n_samples):
            sample_dict = ODict()

            for name, module in self.named_modules():
                if isinstance(module, VariationalLinear):
                    # Sample weights and biases
                    weight_std = torch.exp(0.5 * module.weight_logvar)
                    bias_std = torch.exp(0.5 * module.bias_logvar)

                    weight_eps = torch.randn_like(module.weight_mu)
                    bias_eps = torch.randn_like(module.bias_mu)

                    sampled_weight = (
                        module.weight_mu + weight_std * weight_eps
                    ).detach()
                    sampled_bias = (module.bias_mu + bias_std * bias_eps).detach()

                    sample_dict[f"{name}.weight"] = sampled_weight
                    sample_dict[f"{name}.bias"] = sampled_bias

            weight_samples.append(sample_dict)

        return weight_samples

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute total KL divergence for all variational layers.

        Returns:
            torch.Tensor: Total KL divergence
        """
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)

        for module in self.modules():
            if isinstance(module, VariationalLinear):
                total_kl += module.kl_divergence()

        return total_kl

    def compute_epistemic_uncertainty(
        self, x: torch.Tensor, n_samples: int = 100
    ) -> torch.Tensor:
        """
        Compute epistemic (model) uncertainty using weight sampling.

        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples

        Returns:
            torch.Tensor: Epistemic uncertainty (standard deviation)
        """
        _, uncertainty = self.forward_with_uncertainty(x, n_samples=n_samples)
        return uncertainty

    def compute_aleatoric_uncertainty(
        self, x: torch.Tensor, task_info: Dict[str, Any], params: Optional[ODict] = None
    ) -> torch.Tensor:
        """
        Compute aleatoric (data) uncertainty from physics residuals.

        Args:
            x: Input coordinates
            task_info: Task information
            params: Optional parameters

        Returns:
            torch.Tensor: Aleatoric uncertainty estimate
        """
        try:
            # Compute physics residuals
            physics_losses = self.physics_loss(x, task_info, params, create_graph=False)

            # Use physics residuals as proxy for aleatoric uncertainty
            # Higher residuals indicate higher data uncertainty
            total_residual = physics_losses["total_pde"]

            # Ensure residual is non-negative and finite
            total_residual = torch.clamp(total_residual, min=0.0)
            if not torch.isfinite(total_residual):
                total_residual = torch.tensor(1e-3, device=x.device)

            # Convert to per-point uncertainty estimate
            batch_size = x.shape[0]
            output_dim = self.config.get_layer_sizes()[-1]

            # Scale residual to reasonable uncertainty range
            uncertainty_scale = torch.sqrt(total_residual + 1e-8)

            # Ensure uncertainty scale is finite
            if not torch.isfinite(uncertainty_scale):
                uncertainty_scale = torch.tensor(1e-3, device=x.device)

            aleatoric_uncertainty = uncertainty_scale.expand(batch_size, output_dim)

            return aleatoric_uncertainty

        except Exception as e:
            # Fallback to small constant uncertainty if physics computation fails
            batch_size = x.shape[0]
            output_dim = self.config.get_layer_sizes()[-1]
            return torch.full((batch_size, output_dim), 1e-3, device=x.device)

    def decompose_uncertainty(
        self,
        x: torch.Tensor,
        task_info: Dict[str, Any],
        params: Optional[ODict] = None,
        n_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose total uncertainty into epistemic and aleatoric components.

        Args:
            x: Input tensor
            task_info: Task information for physics uncertainty
            params: Optional parameters
            n_samples: Number of Monte Carlo samples for epistemic uncertainty

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing uncertainty components
        """
        # Compute epistemic uncertainty
        epistemic_uncertainty = self.compute_epistemic_uncertainty(x, n_samples)

        # Compute aleatoric uncertainty
        aleatoric_uncertainty = self.compute_aleatoric_uncertainty(x, task_info, params)

        # Total uncertainty (assuming independence)
        total_uncertainty = torch.sqrt(
            epistemic_uncertainty**2 + aleatoric_uncertainty**2
        )

        return {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
            "total": total_uncertainty,
        }

    def variational_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        coords: torch.Tensor,
        task_info: Dict[str, Any],
        params: Optional[ODict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute variational loss including data loss, physics loss, and KL divergence.

        Args:
            predictions: Model predictions
            targets: Target values
            coords: Input coordinates
            task_info: Task information
            params: Optional parameters

        Returns:
            Dict[str, torch.Tensor]: Loss components
        """
        # Data loss (negative log-likelihood)
        data_loss = F.mse_loss(predictions, targets)

        # Physics loss
        physics_losses = self.physics_loss(coords, task_info, params, create_graph=True)
        physics_weight = self.compute_adaptive_physics_weight(
            physics_losses, self.physics_loss_weight
        )
        physics_loss = physics_weight * physics_losses["total"]

        # KL divergence (only for variational parameters, not meta-learning params)
        if params is None:
            kl_loss = self.kl_weight * self.kl_divergence()
        else:
            kl_loss = torch.tensor(0.0, device=predictions.device)

        # Total variational loss
        total_loss = data_loss + physics_loss + kl_loss

        return {
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "physics_components": physics_losses,
        }

    def adapt_to_task_bayesian(
        self,
        task: Dict[str, Any],
        adaptation_steps: Optional[int] = None,
        create_graph: bool = False,
        use_uncertainty: bool = True,
    ) -> Tuple[ODict, Dict[str, Any]]:
        """
        Bayesian adaptation to a new task with uncertainty-aware optimization.

        Args:
            task: Task dictionary
            adaptation_steps: Number of adaptation steps
            create_graph: Whether to create computation graph
            use_uncertainty: Whether to use uncertainty in adaptation

        Returns:
            Tuple[OrderedDict, Dict]: (adapted_parameters, uncertainty_info)
        """
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps

        # Extract task data
        support_coords = task["support_coords"]
        support_data = task["support_data"]
        task_info = task["task_info"]

        # Get initial uncertainty estimate
        if use_uncertainty:
            initial_uncertainty = self.decompose_uncertainty(support_coords, task_info)
        else:
            initial_uncertainty = None

        # Perform standard adaptation
        adapted_params = self.adapt_to_task(task, adaptation_steps, create_graph)

        # Compute final uncertainty with adapted parameters
        if use_uncertainty:
            final_uncertainty = self.decompose_uncertainty(
                support_coords, task_info, adapted_params
            )
        else:
            final_uncertainty = None

        uncertainty_info = {
            "initial_uncertainty": initial_uncertainty,
            "final_uncertainty": final_uncertainty,
            "uncertainty_reduction": None,
        }

        # Compute uncertainty reduction if available
        if initial_uncertainty is not None and final_uncertainty is not None:
            uncertainty_reduction = {}
            for key in initial_uncertainty:
                reduction = torch.mean(
                    initial_uncertainty[key] - final_uncertainty[key]
                )
                uncertainty_reduction[key] = reduction
            uncertainty_info["uncertainty_reduction"] = uncertainty_reduction

        return adapted_params, uncertainty_info

    def predict_with_confidence(
        self,
        x: torch.Tensor,
        task_info: Dict[str, Any],
        params: Optional[ODict] = None,
        confidence_level: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence intervals.

        Args:
            x: Input tensor
            task_info: Task information
            params: Optional parameters
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)

        Returns:
            Dict[str, torch.Tensor]: Predictions with confidence intervals
        """
        # Get mean predictions and uncertainty
        mean_pred, uncertainty = self.forward_with_uncertainty(x, params=params)

        # Compute confidence intervals assuming Gaussian uncertainty
        z_score = torch.tensor(
            1.96 if confidence_level == 0.95 else 2.576
        )  # 95% or 99%

        lower_bound = mean_pred - z_score * uncertainty
        upper_bound = mean_pred + z_score * uncertainty

        return {
            "mean": mean_pred,
            "uncertainty": uncertainty,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": confidence_level,
        }

    def get_variational_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get variational parameters (means and log-variances) for all layers.

        Returns:
            Dict: Variational parameters organized by layer
        """
        var_params = {}

        for name, module in self.named_modules():
            if isinstance(module, VariationalLinear):
                var_params[name] = {
                    "weight_mu": module.weight_mu,
                    "weight_logvar": module.weight_logvar,
                    "bias_mu": module.bias_mu,
                    "bias_logvar": module.bias_logvar,
                }

        return var_params

    def set_variational_parameters(
        self, var_params: Dict[str, Dict[str, torch.Tensor]]
    ):
        """
        Set variational parameters from a dictionary.

        Args:
            var_params: Variational parameters organized by layer
        """
        for name, module in self.named_modules():
            if isinstance(module, VariationalLinear) and name in var_params:
                params = var_params[name]
                module.weight_mu.data = params["weight_mu"].data
                module.weight_logvar.data = params["weight_logvar"].data
                module.bias_mu.data = params["bias_mu"].data
                module.bias_logvar.data = params["bias_logvar"].data

    def clone_parameters(self) -> ODict:
        """
        Clone model parameters for meta-learning adaptation.

        For BayesianMetaPINN, we clone the mean parameters for adaptation.

        Returns:
            OrderedDict: Cloned parameters
        """
        params = ODict()

        # For variational layers, use mean parameters for adaptation
        for name, module in self.named_modules():
            if isinstance(module, VariationalLinear):
                # Use mean parameters for adaptation
                params[f"{name}.weight"] = module.weight_mu.clone()
                params[f"{name}.bias"] = module.bias_mu.clone()

        # Add any non-variational parameters
        for name, param in self.named_parameters():
            if (
                "weight_mu" not in name
                and "weight_logvar" not in name
                and "bias_mu" not in name
                and "bias_logvar" not in name
            ):
                params[name] = param.clone()

        return params

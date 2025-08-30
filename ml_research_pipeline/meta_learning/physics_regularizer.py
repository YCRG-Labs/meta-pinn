"""
Physics-informed regularization for meta-learning systems.

This module implements physics-informed loss functions and regularization terms
that can be integrated into meta-learning algorithms to enforce physical constraints
and improve generalization to new physics tasks.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ConservationLaw:
    """Represents a conservation law constraint."""

    name: str
    equation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    weight: float = 1.0
    tolerance: float = 1e-6


@dataclass
class CausalConstraint:
    """Represents a causal relationship constraint."""

    cause_vars: List[str]
    effect_vars: List[str]
    constraint_type: str  # 'correlation', 'mutual_info', 'granger'
    strength: float
    confidence: float
    weight: float = 1.0


@dataclass
class SymbolicConstraint:
    """Represents a symbolic mathematical constraint."""

    expression: str
    variables: List[str]
    expected_form: str
    complexity_penalty: float = 0.1
    weight: float = 1.0


class PhysicsRegularizer(nn.Module):
    """
    Physics-informed regularization for meta-learning systems.

    This class implements various physics-informed loss functions and regularization
    terms that enforce conservation laws, causal relationships, and symbolic
    mathematical constraints during meta-learning.

    Args:
        conservation_laws: List of conservation law constraints
        causal_constraints: List of causal relationship constraints
        symbolic_constraints: List of symbolic mathematical constraints
        adaptive_weighting: Whether to adaptively adjust constraint weights
        temperature: Temperature parameter for adaptive weighting
        device: Device to run computations on

    Example:
        >>> # Define conservation laws
        >>> mass_conservation = ConservationLaw(
        ...     name="mass_conservation",
        ...     equation=lambda u, coords: torch.autograd.grad(u, coords)[0].sum(dim=-1),
        ...     weight=1.0
        ... )
        >>>
        >>> # Create regularizer
        >>> regularizer = PhysicsRegularizer(
        ...     conservation_laws=[mass_conservation],
        ...     adaptive_weighting=True
        ... )
        >>>
        >>> # Compute physics loss
        >>> predictions = model(coords)
        >>> physics_loss = regularizer.compute_physics_loss(predictions, coords, task_info)
    """

    def __init__(
        self,
        conservation_laws: Optional[List[ConservationLaw]] = None,
        causal_constraints: Optional[List[CausalConstraint]] = None,
        symbolic_constraints: Optional[List[SymbolicConstraint]] = None,
        adaptive_weighting: bool = True,
        temperature: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()

        self.conservation_laws = conservation_laws or []
        self.causal_constraints = causal_constraints or []
        self.symbolic_constraints = symbolic_constraints or []
        self.adaptive_weighting = adaptive_weighting
        self.temperature = temperature
        self.device = device

        # Initialize adaptive weights
        if self.adaptive_weighting:
            self._init_adaptive_weights()

        # Track constraint violations for monitoring
        self.violation_history = {"conservation": [], "causal": [], "symbolic": []}

    def _init_adaptive_weights(self):
        """Initialize adaptive weight parameters."""
        n_conservation = len(self.conservation_laws)
        n_causal = len(self.causal_constraints)
        n_symbolic = len(self.symbolic_constraints)

        if n_conservation > 0:
            self.conservation_weights = nn.Parameter(
                torch.ones(n_conservation, device=self.device)
            )
        if n_causal > 0:
            self.causal_weights = nn.Parameter(torch.ones(n_causal, device=self.device))
        if n_symbolic > 0:
            self.symbolic_weights = nn.Parameter(
                torch.ones(n_symbolic, device=self.device)
            )

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        task_info: Dict[str, Any],
        create_graph: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive physics-informed loss.

        Args:
            predictions: Model predictions [batch_size, output_dim]
            coordinates: Input coordinates [batch_size, input_dim]
            task_info: Task-specific information
            create_graph: Whether to create computation graph for gradients

        Returns:
            Dict containing individual and total physics losses
        """
        losses = {}
        loss_components = []

        # Ensure coordinates require gradients for physics computations
        if create_graph and not coordinates.requires_grad:
            coordinates = coordinates.clone().detach().requires_grad_(True)

        # Ensure predictions require gradients for physics computations
        if create_graph and not predictions.requires_grad:
            predictions = predictions.clone().detach().requires_grad_(True)

        # Conservation law losses
        if self.conservation_laws:
            conservation_loss, conservation_violations = (
                self._compute_conservation_loss(
                    predictions, coordinates, task_info, create_graph
                )
            )
            losses["conservation"] = conservation_loss
            loss_components.append(conservation_loss)
            self.violation_history["conservation"].append(conservation_violations)

        # Causal constraint losses
        if self.causal_constraints:
            causal_loss, causal_violations = self._compute_causal_loss(
                predictions, coordinates, task_info
            )
            losses["causal"] = causal_loss
            loss_components.append(causal_loss)
            self.violation_history["causal"].append(causal_violations)

        # Symbolic constraint losses
        if self.symbolic_constraints:
            symbolic_loss, symbolic_violations = self._compute_symbolic_loss(
                predictions, coordinates, task_info, create_graph
            )
            losses["symbolic"] = symbolic_loss
            loss_components.append(symbolic_loss)
            self.violation_history["symbolic"].append(symbolic_violations)

        # Compute total loss
        if loss_components:
            total_loss = sum(loss_components)
        else:
            total_loss = torch.tensor(
                0.0, device=self.device, requires_grad=create_graph
            )

        losses["total"] = total_loss
        return losses

    def _compute_conservation_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        task_info: Dict[str, Any],
        create_graph: bool = True,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Compute conservation law constraint losses."""
        total_loss = torch.tensor(0.0, device=self.device)
        violations = []

        for i, law in enumerate(self.conservation_laws):
            try:
                # Compute conservation law residual
                if law.name == "mass_conservation":
                    residual = self._mass_conservation_residual(
                        predictions, coordinates, create_graph
                    )
                elif law.name == "momentum_conservation":
                    residual = self._momentum_conservation_residual(
                        predictions, coordinates, task_info, create_graph
                    )
                elif law.name == "energy_conservation":
                    residual = self._energy_conservation_residual(
                        predictions, coordinates, task_info, create_graph
                    )
                else:
                    # Custom conservation law
                    residual = law.equation(predictions, coordinates)

                # Compute loss with adaptive weighting
                if self.adaptive_weighting and hasattr(self, "conservation_weights"):
                    weight = torch.softmax(
                        self.conservation_weights / self.temperature, dim=0
                    )[i]
                else:
                    weight = law.weight

                law_loss = weight * torch.mean(residual**2)
                total_loss += law_loss

                # Track violation magnitude
                violation_magnitude = torch.mean(torch.abs(residual)).item()
                violations.append(violation_magnitude)

            except Exception as e:
                warnings.warn(f"Error computing conservation law {law.name}: {e}")
                violations.append(float("inf"))

        return total_loss, violations

    def _compute_causal_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        task_info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, List[float]]:
        """Compute causal constraint losses."""
        total_loss = torch.tensor(0.0, device=self.device)
        violations = []

        for i, constraint in enumerate(self.causal_constraints):
            try:
                # Extract relevant variables
                cause_data = self._extract_variables(
                    predictions, coordinates, constraint.cause_vars, task_info
                )
                effect_data = self._extract_variables(
                    predictions, coordinates, constraint.effect_vars, task_info
                )

                # Compute causal constraint violation
                if constraint.constraint_type == "correlation":
                    violation = self._correlation_constraint_loss(
                        cause_data, effect_data, constraint.strength
                    )
                elif constraint.constraint_type == "mutual_info":
                    violation = self._mutual_info_constraint_loss(
                        cause_data, effect_data, constraint.strength
                    )
                elif constraint.constraint_type == "granger":
                    violation = self._granger_constraint_loss(
                        cause_data, effect_data, constraint.strength
                    )
                else:
                    violation = torch.tensor(0.0, device=self.device)

                # Apply confidence weighting
                confidence_weight = constraint.confidence

                # Apply adaptive weighting
                if self.adaptive_weighting and hasattr(self, "causal_weights"):
                    adaptive_weight = torch.softmax(
                        self.causal_weights / self.temperature, dim=0
                    )[i]
                else:
                    adaptive_weight = constraint.weight

                constraint_loss = adaptive_weight * confidence_weight * violation
                total_loss += constraint_loss

                violations.append(violation.item())

            except Exception as e:
                warnings.warn(f"Error computing causal constraint {i}: {e}")
                violations.append(float("inf"))

        return total_loss, violations

    def _compute_symbolic_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        task_info: Dict[str, Any],
        create_graph: bool = True,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Compute symbolic constraint losses."""
        total_loss = torch.tensor(0.0, device=self.device)
        violations = []

        for i, constraint in enumerate(self.symbolic_constraints):
            try:
                # Evaluate symbolic expression
                violation = self._evaluate_symbolic_constraint(
                    constraint, predictions, coordinates, task_info, create_graph
                )

                # Apply complexity penalty
                complexity_penalty = constraint.complexity_penalty * len(
                    constraint.expression
                )

                # Apply adaptive weighting
                if self.adaptive_weighting and hasattr(self, "symbolic_weights"):
                    adaptive_weight = torch.softmax(
                        self.symbolic_weights / self.temperature, dim=0
                    )[i]
                else:
                    adaptive_weight = constraint.weight

                constraint_loss = adaptive_weight * (violation + complexity_penalty)
                total_loss += constraint_loss

                violations.append(violation.item())

            except Exception as e:
                warnings.warn(f"Error computing symbolic constraint {i}: {e}")
                violations.append(float("inf"))

        return total_loss, violations

    def _mass_conservation_residual(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Compute mass conservation (continuity equation) residual."""
        if not create_graph or not coordinates.requires_grad:
            return torch.zeros(predictions.shape[0], 1, device=self.device)

        # Ensure predictions require gradients for physics computations
        if not predictions.requires_grad:
            predictions = predictions.clone().detach().requires_grad_(True)

        # Assume predictions are [u, v, p] for 2D flow
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]

        # Compute divergence: ∂u/∂x + ∂v/∂y = 0
        u_grad = torch.autograd.grad(
            u,
            coordinates,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        v_grad = torch.autograd.grad(
            v,
            coordinates,
            torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if u_grad is None or v_grad is None:
            return torch.zeros(predictions.shape[0], 1, device=self.device)

        # Extract spatial derivatives
        u_x = u_grad[:, 0:1]  # ∂u/∂x
        v_y = v_grad[:, 1:2]  # ∂v/∂y

        # Continuity equation residual
        residual = u_x + v_y
        return residual

    def _momentum_conservation_residual(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        task_info: Dict[str, Any],
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Compute momentum conservation residual."""
        if not create_graph or not coordinates.requires_grad:
            return torch.zeros(predictions.shape[0], 2, device=self.device)

        # Ensure predictions require gradients for physics computations
        if not predictions.requires_grad:
            predictions = predictions.clone().detach().requires_grad_(True)

        # Extract velocity and pressure
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]
        p = predictions[:, 2:3]

        # Get Reynolds number from task info
        reynolds = task_info.get("reynolds", 100.0)

        # Compute gradients
        u_grad = torch.autograd.grad(
            u,
            coordinates,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        v_grad = torch.autograd.grad(
            v,
            coordinates,
            torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        p_grad = torch.autograd.grad(
            p,
            coordinates,
            torch.ones_like(p),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if u_grad is None or v_grad is None or p_grad is None:
            return torch.zeros(predictions.shape[0], 2, device=self.device)

        # Extract derivatives
        u_x, u_y, u_t = (
            u_grad[:, 0:1],
            u_grad[:, 1:2],
            u_grad[:, 2:3] if coordinates.shape[1] > 2 else torch.zeros_like(u),
        )
        v_x, v_y, v_t = (
            v_grad[:, 0:1],
            v_grad[:, 1:2],
            v_grad[:, 2:3] if coordinates.shape[1] > 2 else torch.zeros_like(v),
        )
        p_x, p_y = p_grad[:, 0:1], p_grad[:, 1:2]

        # Compute second derivatives for viscous terms
        u_xx_grad = torch.autograd.grad(
            u_x,
            coordinates,
            torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        u_yy_grad = torch.autograd.grad(
            u_y,
            coordinates,
            torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        v_xx_grad = torch.autograd.grad(
            v_x,
            coordinates,
            torch.ones_like(v_x),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        v_yy_grad = torch.autograd.grad(
            v_y,
            coordinates,
            torch.ones_like(v_y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if (
            u_xx_grad is None
            or u_yy_grad is None
            or v_xx_grad is None
            or v_yy_grad is None
        ):
            return torch.zeros(predictions.shape[0], 2, device=self.device)

        u_xx = u_xx_grad[:, 0:1]
        u_yy = u_yy_grad[:, 1:2]
        v_xx = v_xx_grad[:, 0:1]
        v_yy = v_yy_grad[:, 1:2]

        # Navier-Stokes momentum equations
        momentum_x = u_t + u * u_x + v * u_y + p_x - (1.0 / reynolds) * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + p_y - (1.0 / reynolds) * (v_xx + v_yy)

        return torch.cat([momentum_x, momentum_y], dim=1)

    def _energy_conservation_residual(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        task_info: Dict[str, Any],
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Compute energy conservation residual."""
        if not create_graph or not coordinates.requires_grad:
            return torch.zeros(predictions.shape[0], 1, device=self.device)

        # Ensure predictions require gradients for physics computations
        if not predictions.requires_grad:
            predictions = predictions.clone().detach().requires_grad_(True)

        # For incompressible flow, kinetic energy equation
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]

        kinetic_energy = 0.5 * (u**2 + v**2)

        # Compute time derivative of kinetic energy
        if coordinates.shape[1] > 2:  # Has time dimension
            ke_grad = torch.autograd.grad(
                kinetic_energy,
                coordinates,
                torch.ones_like(kinetic_energy),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            if ke_grad is None:
                return torch.zeros(predictions.shape[0], 1, device=self.device)

            ke_t = ke_grad[:, 2:3]

            # Energy dissipation rate (simplified)
            reynolds = task_info.get("reynolds", 100.0)
            dissipation = self._compute_dissipation_rate(
                predictions, coordinates, reynolds
            )

            # Energy conservation: ∂E/∂t + dissipation = 0
            residual = ke_t + dissipation
        else:
            residual = torch.zeros_like(kinetic_energy)

        return residual

    def _compute_dissipation_rate(
        self, predictions: torch.Tensor, coordinates: torch.Tensor, reynolds: float
    ) -> torch.Tensor:
        """Compute viscous dissipation rate."""
        # Ensure predictions require gradients for physics computations
        if not predictions.requires_grad:
            predictions = predictions.clone().detach().requires_grad_(True)

        u = predictions[:, 0:1]
        v = predictions[:, 1:2]

        # Compute velocity gradients
        u_grad = torch.autograd.grad(
            u,
            coordinates,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        v_grad = torch.autograd.grad(
            v,
            coordinates,
            torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if u_grad is None or v_grad is None:
            return torch.zeros(predictions.shape[0], 1, device=self.device)

        u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]
        v_x, v_y = v_grad[:, 0:1], v_grad[:, 1:2]

        # Strain rate tensor components
        S11 = u_x
        S22 = v_y
        S12 = 0.5 * (u_y + v_x)

        # Dissipation rate: 2μ/ρ * S:S
        dissipation = (2.0 / reynolds) * (S11**2 + S22**2 + 2 * S12**2)

        return dissipation

    def _extract_variables(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        var_names: List[str],
        task_info: Dict[str, Any],
    ) -> torch.Tensor:
        """Extract specified variables from predictions and coordinates."""
        variables = []

        for var_name in var_names:
            if var_name == "u":
                variables.append(predictions[:, 0:1])
            elif var_name == "v":
                variables.append(predictions[:, 1:2])
            elif var_name == "p":
                variables.append(predictions[:, 2:3])
            elif var_name == "x":
                variables.append(coordinates[:, 0:1])
            elif var_name == "y":
                variables.append(coordinates[:, 1:2])
            elif var_name == "t":
                if coordinates.shape[1] > 2:
                    variables.append(coordinates[:, 2:3])
                else:
                    variables.append(
                        torch.zeros(coordinates.shape[0], 1, device=self.device)
                    )
            else:
                # Try to get from task_info
                if var_name in task_info:
                    var_data = task_info[var_name]
                    if isinstance(var_data, torch.Tensor):
                        variables.append(var_data)
                    else:
                        variables.append(
                            torch.full(
                                (coordinates.shape[0], 1), var_data, device=self.device
                            )
                        )
                else:
                    variables.append(
                        torch.zeros(coordinates.shape[0], 1, device=self.device)
                    )

        return (
            torch.cat(variables, dim=1)
            if variables
            else torch.zeros(coordinates.shape[0], 1, device=self.device)
        )

    def _correlation_constraint_loss(
        self,
        cause_data: torch.Tensor,
        effect_data: torch.Tensor,
        expected_strength: float,
    ) -> torch.Tensor:
        """Compute correlation constraint loss."""
        # Compute correlation coefficient
        cause_centered = cause_data - torch.mean(cause_data, dim=0, keepdim=True)
        effect_centered = effect_data - torch.mean(effect_data, dim=0, keepdim=True)

        numerator = torch.sum(cause_centered * effect_centered, dim=0)
        denominator = torch.sqrt(
            torch.sum(cause_centered**2, dim=0) * torch.sum(effect_centered**2, dim=0)
        )

        correlation = numerator / (denominator + 1e-8)

        # Loss is difference from expected correlation
        loss = torch.mean((correlation - expected_strength) ** 2)
        return loss

    def _mutual_info_constraint_loss(
        self,
        cause_data: torch.Tensor,
        effect_data: torch.Tensor,
        expected_strength: float,
    ) -> torch.Tensor:
        """Compute mutual information constraint loss (simplified)."""
        # Simplified mutual information using correlation as proxy
        correlation_loss = self._correlation_constraint_loss(
            cause_data, effect_data, expected_strength
        )

        # Add entropy regularization
        cause_entropy = -torch.mean(torch.log(torch.abs(cause_data) + 1e-8))
        effect_entropy = -torch.mean(torch.log(torch.abs(effect_data) + 1e-8))

        # Mutual information approximation
        mi_loss = correlation_loss - 0.1 * (cause_entropy + effect_entropy)
        return mi_loss

    def _granger_constraint_loss(
        self,
        cause_data: torch.Tensor,
        effect_data: torch.Tensor,
        expected_strength: float,
    ) -> torch.Tensor:
        """Compute Granger causality constraint loss (simplified)."""
        # Simplified Granger causality using lagged correlation
        if cause_data.shape[0] > 1:
            # Use simple lag-1 correlation as proxy
            cause_lagged = cause_data[:-1]
            effect_current = effect_data[1:]

            granger_loss = self._correlation_constraint_loss(
                cause_lagged, effect_current, expected_strength
            )
        else:
            granger_loss = torch.tensor(0.0, device=self.device)

        return granger_loss

    def _evaluate_symbolic_constraint(
        self,
        constraint: SymbolicConstraint,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        task_info: Dict[str, Any],
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Evaluate symbolic mathematical constraint."""
        try:
            # Extract variables for the constraint
            variables = self._extract_variables(
                predictions, coordinates, constraint.variables, task_info
            )

            # Simple symbolic constraint evaluation
            # This is a simplified implementation - in practice, you'd use a symbolic math library
            if constraint.expression == "u^2 + v^2":
                # Kinetic energy constraint
                u, v = variables[:, 0:1], variables[:, 1:2]
                result = u**2 + v**2
                expected = torch.ones_like(result)  # Normalized kinetic energy
                violation = torch.mean((result - expected) ** 2)

            elif constraint.expression == "du/dx + dv/dy":
                # Continuity equation
                if create_graph and coordinates.requires_grad:
                    residual = self._mass_conservation_residual(
                        predictions, coordinates, create_graph
                    )
                    violation = torch.mean(residual**2)
                else:
                    violation = torch.tensor(
                        0.0, device=self.device, requires_grad=create_graph
                    )

            else:
                # Default: assume constraint should be zero
                violation = torch.mean(variables**2)

            return violation

        except Exception as e:
            warnings.warn(f"Error evaluating symbolic constraint: {e}")
            return torch.tensor(0.0, device=self.device)

    def get_constraint_violations(self) -> Dict[str, List[float]]:
        """Get history of constraint violations for monitoring."""
        return {
            "conservation": self.violation_history["conservation"][
                -10:
            ],  # Last 10 values
            "causal": self.violation_history["causal"][-10:],
            "symbolic": self.violation_history["symbolic"][-10:],
        }

    def update_adaptive_weights(self, violation_magnitudes: Dict[str, List[float]]):
        """Update adaptive weights based on constraint violation magnitudes."""
        if not self.adaptive_weighting:
            return

        # Update conservation weights
        if (
            hasattr(self, "conservation_weights")
            and "conservation" in violation_magnitudes
        ):
            violations = torch.tensor(
                violation_magnitudes["conservation"], device=self.device
            )
            # Increase weights for constraints with higher violations
            self.conservation_weights.data = torch.log(violations + 1e-8)

        # Update causal weights
        if hasattr(self, "causal_weights") and "causal" in violation_magnitudes:
            violations = torch.tensor(
                violation_magnitudes["causal"], device=self.device
            )
            self.causal_weights.data = torch.log(violations + 1e-8)

        # Update symbolic weights
        if hasattr(self, "symbolic_weights") and "symbolic" in violation_magnitudes:
            violations = torch.tensor(
                violation_magnitudes["symbolic"], device=self.device
            )
            self.symbolic_weights.data = torch.log(violations + 1e-8)

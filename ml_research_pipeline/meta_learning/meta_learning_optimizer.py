"""
Physics-constrained optimization algorithms for meta-learning.

This module implements optimization algorithms that enforce physics constraints
during meta-learning, including gradient projection methods and Lagrangian
optimization for constraint satisfaction.
"""

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .physics_regularizer import (
    CausalConstraint,
    ConservationLaw,
    PhysicsRegularizer,
    SymbolicConstraint,
)


@dataclass
class OptimizationConfig:
    """Configuration for physics-constrained optimization."""

    learning_rate: float = 1e-3
    constraint_penalty: float = 1.0
    lagrange_lr: float = 1e-2
    projection_tolerance: float = 1e-6
    max_projection_iterations: int = 100
    constraint_violation_tolerance: float = 1e-4
    adaptive_penalty: bool = True
    penalty_increase_factor: float = 2.0
    penalty_decrease_factor: float = 0.5
    gradient_clipping: float = 1.0
    use_second_order: bool = False


@dataclass
class ConstraintFunction:
    """Represents a constraint function for optimization."""

    name: str
    function: Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
    jacobian: Optional[Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]] = None
    tolerance: float = 1e-6
    weight: float = 1.0
    constraint_type: str = "equality"  # 'equality' or 'inequality'


class PhysicsConstrainedOptimizer(ABC):
    """Abstract base class for physics-constrained optimizers."""

    @abstractmethod
    def step(
        self,
        loss_fn: Callable,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        **kwargs,
    ) -> Dict[str, float]:
        """Perform one optimization step with physics constraints."""
        pass

    @abstractmethod
    def project_to_constraints(
        self,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        **kwargs,
    ) -> List[torch.Tensor]:
        """Project parameters to satisfy constraints."""
        pass


class GradientProjectionOptimizer(PhysicsConstrainedOptimizer):
    """
    Gradient projection optimizer for physics-constrained meta-learning.

    This optimizer uses gradient projection to enforce physics constraints
    during optimization by projecting gradients onto the constraint manifold.

    Args:
        config: Optimization configuration
        base_optimizer: Base optimizer (e.g., Adam, SGD)

    Example:
        >>> config = OptimizationConfig(learning_rate=1e-3)
        >>> optimizer = GradientProjectionOptimizer(config)
        >>>
        >>> # Define constraints
        >>> mass_constraint = ConstraintFunction(
        ...     name="mass_conservation",
        ...     function=lambda params, data: compute_mass_residual(params, data),
        ...     constraint_type='equality'
        ... )
        >>>
        >>> # Optimization step
        >>> metrics = optimizer.step(loss_fn, model.parameters(), [mass_constraint])
    """

    def __init__(
        self,
        config: OptimizationConfig,
        base_optimizer: Optional[optim.Optimizer] = None,
    ):
        self.config = config
        self.base_optimizer = base_optimizer

        # Track optimization history
        self.constraint_violations = defaultdict(list)
        self.gradient_norms = []
        self.projection_iterations = []

    def step(
        self,
        loss_fn: Callable,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Perform gradient projection optimization step.

        Args:
            loss_fn: Loss function to minimize
            parameters: Model parameters
            constraints: Physics constraints to enforce
            data: Additional data for constraint evaluation

        Returns:
            Dictionary of optimization metrics
        """
        if data is None:
            data = {}

        # Compute loss and gradients
        loss = loss_fn()
        loss.backward()

        # Store original gradients
        original_grads = [
            p.grad.clone() if p.grad is not None else None for p in parameters
        ]

        # Project gradients to constraint manifold
        projected_grads = self._project_gradients(parameters, constraints, data)

        # Update parameters with projected gradients
        for param, proj_grad in zip(parameters, projected_grads):
            if proj_grad is not None:
                param.grad = proj_grad

        # Apply gradient clipping
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clipping)

        # Perform optimization step
        if self.base_optimizer is not None:
            self.base_optimizer.step()
        else:
            # Simple gradient descent
            with torch.no_grad():
                for param in parameters:
                    if param.grad is not None:
                        param -= self.config.learning_rate * param.grad

        # Project parameters to satisfy constraints
        projected_params = self.project_to_constraints(parameters, constraints, data)

        # Update parameters
        with torch.no_grad():
            for param, proj_param in zip(parameters, projected_params):
                param.copy_(proj_param)

        # Compute metrics
        metrics = self._compute_metrics(
            parameters, constraints, original_grads, projected_grads, data
        )

        return metrics

    def _project_gradients(
        self,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """Project gradients onto constraint manifold."""
        if not constraints:
            return [p.grad.clone() if p.grad is not None else None for p in parameters]

        projected_grads = []

        for param in parameters:
            if param.grad is None:
                projected_grads.append(None)
                continue

            grad = param.grad.clone()

            # Project gradient for each constraint
            for constraint in constraints:
                if constraint.constraint_type == "equality":
                    grad = self._project_gradient_equality(
                        grad, param, constraint, data
                    )
                elif constraint.constraint_type == "inequality":
                    grad = self._project_gradient_inequality(
                        grad, param, constraint, data
                    )

            projected_grads.append(grad)

        return projected_grads

    def _project_gradient_equality(
        self,
        gradient: torch.Tensor,
        parameter: torch.Tensor,
        constraint: ConstraintFunction,
        data: Dict[str, Any],
    ) -> torch.Tensor:
        """Project gradient for equality constraint using nullspace projection."""
        try:
            # Compute constraint Jacobian
            if constraint.jacobian is not None:
                jacobian = constraint.jacobian(parameter, data)
            else:
                # Numerical Jacobian
                jacobian = self._compute_numerical_jacobian(constraint, parameter, data)

            if jacobian is None or jacobian.numel() == 0:
                return gradient

            # Reshape for matrix operations
            grad_flat = gradient.view(-1)
            jac_flat = jacobian.view(jacobian.shape[0], -1)

            # Nullspace projection: P = I - J^T(JJ^T)^{-1}J
            try:
                jjt = torch.mm(jac_flat, jac_flat.t())
                jjt_inv = torch.inverse(
                    jjt + 1e-6 * torch.eye(jjt.shape[0], device=jjt.device)
                )
                projection = torch.eye(
                    grad_flat.shape[0], device=grad_flat.device
                ) - torch.mm(jac_flat.t(), torch.mm(jjt_inv, jac_flat))

                projected_grad = torch.mv(projection, grad_flat)
                return projected_grad.view_as(gradient)

            except RuntimeError:
                # Fallback to pseudo-inverse
                jac_pinv = torch.pinverse(jac_flat)
                projection = torch.eye(
                    grad_flat.shape[0], device=grad_flat.device
                ) - torch.mm(jac_flat.t(), jac_pinv)
                projected_grad = torch.mv(projection, grad_flat)
                return projected_grad.view_as(gradient)

        except Exception as e:
            warnings.warn(f"Error in gradient projection for {constraint.name}: {e}")
            return gradient

    def _project_gradient_inequality(
        self,
        gradient: torch.Tensor,
        parameter: torch.Tensor,
        constraint: ConstraintFunction,
        data: Dict[str, Any],
    ) -> torch.Tensor:
        """Project gradient for inequality constraint."""
        try:
            # Evaluate constraint
            constraint_value = constraint.function(parameter, data)

            # Only project if constraint is violated
            if constraint_value.item() <= constraint.tolerance:
                return gradient

            # Compute constraint gradient
            if constraint.jacobian is not None:
                constraint_grad = constraint.jacobian(parameter, data)
            else:
                constraint_grad = self._compute_numerical_jacobian(
                    constraint, parameter, data
                )

            if constraint_grad is None:
                return gradient

            # Project gradient away from constraint boundary
            grad_flat = gradient.view(-1)
            cgrad_flat = constraint_grad.view(-1)

            # Remove component in constraint gradient direction
            dot_product = torch.dot(grad_flat, cgrad_flat)
            if dot_product > 0:  # Gradient points into constraint violation
                cgrad_norm_sq = torch.dot(cgrad_flat, cgrad_flat)
                if cgrad_norm_sq > 1e-12:
                    projected_grad = (
                        grad_flat - (dot_product / cgrad_norm_sq) * cgrad_flat
                    )
                    return projected_grad.view_as(gradient)

            return gradient

        except Exception as e:
            warnings.warn(
                f"Error in inequality gradient projection for {constraint.name}: {e}"
            )
            return gradient

    def project_to_constraints(
        self,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Optional[Dict[str, Any]] = None,
    ) -> List[torch.Tensor]:
        """Project parameters to satisfy constraints using iterative projection."""
        if not constraints or data is None:
            return [p.clone() for p in parameters]

        projected_params = [p.clone() for p in parameters]

        for iteration in range(self.config.max_projection_iterations):
            max_violation = 0.0

            for i, param in enumerate(projected_params):
                for constraint in constraints:
                    # Evaluate constraint
                    constraint_value = constraint.function(param, data)
                    violation = torch.abs(constraint_value).max().item()
                    max_violation = max(max_violation, violation)

                    # Project if constraint is violated
                    if violation > constraint.tolerance:
                        projected_params[i] = self._project_parameter_to_constraint(
                            param, constraint, data
                        )

            # Check convergence
            if max_violation < self.config.constraint_violation_tolerance:
                break

        self.projection_iterations.append(iteration + 1)
        return projected_params

    def _project_parameter_to_constraint(
        self,
        parameter: torch.Tensor,
        constraint: ConstraintFunction,
        data: Dict[str, Any],
    ) -> torch.Tensor:
        """Project single parameter to satisfy constraint."""
        try:
            param = parameter.clone()

            for _ in range(10):  # Inner projection iterations
                constraint_value = constraint.function(param, data)

                if torch.abs(constraint_value).max().item() < constraint.tolerance:
                    break

                # Compute constraint gradient
                if constraint.jacobian is not None:
                    constraint_grad = constraint.jacobian(param, data)
                else:
                    constraint_grad = self._compute_numerical_jacobian(
                        constraint, param, data
                    )

                if constraint_grad is None:
                    break

                # Newton-like update
                grad_flat = constraint_grad.view(-1)
                grad_norm_sq = torch.dot(grad_flat, grad_flat)

                if grad_norm_sq > 1e-12:
                    update = (constraint_value.item() / grad_norm_sq) * grad_flat
                    param.view(-1).sub_(update)
                else:
                    break

            return param

        except Exception as e:
            warnings.warn(f"Error in parameter projection for {constraint.name}: {e}")
            return parameter.clone()

    def _compute_numerical_jacobian(
        self,
        constraint: ConstraintFunction,
        parameter: torch.Tensor,
        data: Dict[str, Any],
        eps: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        """Compute numerical Jacobian of constraint function."""
        try:
            param_flat = parameter.view(-1).detach()
            jacobian = torch.zeros(1, param_flat.shape[0], device=parameter.device)

            # Central difference
            for i in range(param_flat.shape[0]):
                param_plus = param_flat.clone()
                param_minus = param_flat.clone()
                param_plus[i] += eps
                param_minus[i] -= eps

                param_plus_reshaped = param_plus.view_as(parameter).requires_grad_(True)
                param_minus_reshaped = param_minus.view_as(parameter).requires_grad_(
                    True
                )

                # Create fresh data copies to avoid graph conflicts
                data_plus = {
                    k: (
                        v.detach().clone().requires_grad_(True)
                        if isinstance(v, torch.Tensor) and v.requires_grad
                        else v
                    )
                    for k, v in data.items()
                }
                data_minus = {
                    k: (
                        v.detach().clone().requires_grad_(True)
                        if isinstance(v, torch.Tensor) and v.requires_grad
                        else v
                    )
                    for k, v in data.items()
                }

                with torch.no_grad():
                    constraint_plus = constraint.function(
                        param_plus_reshaped, data_plus
                    )
                    constraint_minus = constraint.function(
                        param_minus_reshaped, data_minus
                    )

                jacobian[0, i] = (constraint_plus - constraint_minus) / (2 * eps)

            return jacobian

        except Exception as e:
            warnings.warn(f"Error computing numerical Jacobian: {e}")
            return None

    def _compute_metrics(
        self,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        original_grads: List[torch.Tensor],
        projected_grads: List[torch.Tensor],
        data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute optimization metrics."""
        metrics = {}

        # Constraint violations
        total_violation = 0.0
        for constraint in constraints:
            for param in parameters:
                try:
                    violation = constraint.function(param, data)
                    violation_magnitude = torch.abs(violation).max().item()
                    self.constraint_violations[constraint.name].append(
                        violation_magnitude
                    )
                    total_violation += violation_magnitude
                except Exception:
                    pass

        metrics["total_constraint_violation"] = total_violation

        # Gradient norms
        original_norm = 0.0
        projected_norm = 0.0

        for orig_grad, proj_grad in zip(original_grads, projected_grads):
            if orig_grad is not None:
                original_norm += torch.norm(orig_grad).item() ** 2
            if proj_grad is not None:
                projected_norm += torch.norm(proj_grad).item() ** 2

        original_norm = np.sqrt(original_norm)
        projected_norm = np.sqrt(projected_norm)

        self.gradient_norms.append((original_norm, projected_norm))

        metrics["original_gradient_norm"] = original_norm
        metrics["projected_gradient_norm"] = projected_norm
        metrics["gradient_projection_ratio"] = projected_norm / (original_norm + 1e-12)

        # Projection iterations
        if self.projection_iterations:
            metrics["projection_iterations"] = self.projection_iterations[-1]

        return metrics


class LagrangianOptimizer(PhysicsConstrainedOptimizer):
    """
    Lagrangian optimizer for physics-constrained meta-learning.

    This optimizer uses the method of Lagrange multipliers to enforce
    physics constraints during optimization.

    Args:
        config: Optimization configuration

    Example:
        >>> config = OptimizationConfig(learning_rate=1e-3, lagrange_lr=1e-2)
        >>> optimizer = LagrangianOptimizer(config)
        >>>
        >>> # Optimization with Lagrangian method
        >>> metrics = optimizer.step(loss_fn, model.parameters(), constraints)
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

        # Lagrange multipliers
        self.lagrange_multipliers = {}

        # Penalty parameters for augmented Lagrangian
        self.penalty_parameters = {}

        # Track optimization history
        self.constraint_violations = defaultdict(list)
        self.multiplier_history = defaultdict(list)
        self.penalty_history = defaultdict(list)

    def step(
        self,
        loss_fn: Callable,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Perform Lagrangian optimization step.

        Args:
            loss_fn: Loss function to minimize
            parameters: Model parameters
            constraints: Physics constraints to enforce
            data: Additional data for constraint evaluation

        Returns:
            Dictionary of optimization metrics
        """
        if data is None:
            data = {}

        # Initialize multipliers and penalties if needed
        self._initialize_multipliers_and_penalties(constraints)

        # Compute augmented Lagrangian
        augmented_loss = self._compute_augmented_lagrangian(
            loss_fn, parameters, constraints, data
        )

        # Compute gradients
        augmented_loss.backward()

        # Apply gradient clipping
        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clipping)

        # Update parameters
        with torch.no_grad():
            for param in parameters:
                if param.grad is not None:
                    param -= self.config.learning_rate * param.grad

        # Update Lagrange multipliers
        self._update_lagrange_multipliers(parameters, constraints, data)

        # Update penalty parameters
        self._update_penalty_parameters(parameters, constraints, data)

        # Compute metrics
        metrics = self._compute_lagrangian_metrics(parameters, constraints, data)

        return metrics

    def _initialize_multipliers_and_penalties(
        self, constraints: List[ConstraintFunction]
    ):
        """Initialize Lagrange multipliers and penalty parameters."""
        for constraint in constraints:
            if constraint.name not in self.lagrange_multipliers:
                self.lagrange_multipliers[constraint.name] = torch.tensor(
                    0.0, requires_grad=False
                )
                self.penalty_parameters[constraint.name] = (
                    self.config.constraint_penalty
                )

    def _compute_augmented_lagrangian(
        self,
        loss_fn: Callable,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute augmented Lagrangian objective."""
        # Original loss
        loss = loss_fn()

        # Add constraint terms
        for constraint in constraints:
            for param in parameters:
                try:
                    constraint_value = constraint.function(param, data)
                    multiplier = self.lagrange_multipliers[constraint.name]
                    penalty = self.penalty_parameters[constraint.name]

                    if constraint.constraint_type == "equality":
                        # Augmented Lagrangian for equality constraints
                        # L = f(x) + λ*c(x) + (ρ/2)*c(x)^2
                        loss += multiplier * constraint_value
                        loss += 0.5 * penalty * constraint_value**2

                    elif constraint.constraint_type == "inequality":
                        # Augmented Lagrangian for inequality constraints c(x) <= 0
                        # L = f(x) + max(0, λ + ρ*c(x))*c(x) + (ρ/2)*max(0, c(x))^2
                        constraint_violation = torch.max(
                            torch.tensor(0.0, device=constraint_value.device),
                            constraint_value,
                        )

                        effective_multiplier = torch.max(
                            torch.tensor(0.0, device=multiplier.device),
                            multiplier + penalty * constraint_value,
                        )

                        loss += effective_multiplier * constraint_value
                        loss += 0.5 * penalty * constraint_violation**2

                except Exception as e:
                    warnings.warn(
                        f"Error in augmented Lagrangian for {constraint.name}: {e}"
                    )

        return loss

    def _update_lagrange_multipliers(
        self,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Dict[str, Any],
    ):
        """Update Lagrange multipliers."""
        for constraint in constraints:
            constraint_violations = []

            for param in parameters:
                try:
                    constraint_value = constraint.function(param, data)
                    constraint_violations.append(constraint_value.item())
                except Exception:
                    constraint_violations.append(0.0)

            avg_violation = np.mean(constraint_violations)

            # Update multiplier
            multiplier = self.lagrange_multipliers[constraint.name]
            penalty = self.penalty_parameters[constraint.name]

            if constraint.constraint_type == "equality":
                # λ_{k+1} = λ_k + ρ*c(x)
                new_multiplier = (
                    multiplier + self.config.lagrange_lr * penalty * avg_violation
                )

            elif constraint.constraint_type == "inequality":
                # λ_{k+1} = max(0, λ_k + ρ*c(x))
                new_multiplier = max(
                    0.0, multiplier + self.config.lagrange_lr * penalty * avg_violation
                )

            if isinstance(new_multiplier, torch.Tensor):
                self.lagrange_multipliers[constraint.name] = (
                    new_multiplier.detach().clone()
                )
            else:
                self.lagrange_multipliers[constraint.name] = torch.tensor(
                    new_multiplier, dtype=torch.float32
                )
            self.multiplier_history[constraint.name].append(new_multiplier)

    def _update_penalty_parameters(
        self,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Dict[str, Any],
    ):
        """Update penalty parameters for augmented Lagrangian."""
        if not self.config.adaptive_penalty:
            return

        for constraint in constraints:
            constraint_violations = []

            for param in parameters:
                try:
                    constraint_value = constraint.function(param, data)
                    constraint_violations.append(abs(constraint_value.item()))
                except Exception:
                    constraint_violations.append(0.0)

            max_violation = max(constraint_violations) if constraint_violations else 0.0

            # Adaptive penalty update
            current_penalty = self.penalty_parameters[constraint.name]

            if max_violation > self.config.constraint_violation_tolerance:
                # Increase penalty if constraints are violated
                new_penalty = current_penalty * self.config.penalty_increase_factor
            else:
                # Decrease penalty if constraints are satisfied
                new_penalty = max(
                    self.config.constraint_penalty,
                    current_penalty * self.config.penalty_decrease_factor,
                )

            self.penalty_parameters[constraint.name] = new_penalty
            self.penalty_history[constraint.name].append(new_penalty)

    def project_to_constraints(
        self,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Optional[Dict[str, Any]] = None,
    ) -> List[torch.Tensor]:
        """
        Project parameters to constraints (not typically used in Lagrangian method).

        The Lagrangian method handles constraints through multipliers rather than
        explicit projection, but this method is provided for interface compatibility.
        """
        # For Lagrangian method, we typically don't project explicitly
        # The constraints are handled through the multipliers
        return [p.clone() for p in parameters]

    def _compute_lagrangian_metrics(
        self,
        parameters: List[torch.Tensor],
        constraints: List[ConstraintFunction],
        data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute Lagrangian optimization metrics."""
        metrics = {}

        # Constraint violations
        total_violation = 0.0
        for constraint in constraints:
            constraint_violations = []

            for param in parameters:
                try:
                    violation = constraint.function(param, data)
                    violation_magnitude = abs(violation.item())
                    constraint_violations.append(violation_magnitude)
                    total_violation += violation_magnitude
                except Exception:
                    pass

            if constraint_violations:
                avg_violation = np.mean(constraint_violations)
                max_violation = max(constraint_violations)
                self.constraint_violations[constraint.name].append(avg_violation)

                metrics[f"{constraint.name}_avg_violation"] = avg_violation
                metrics[f"{constraint.name}_max_violation"] = max_violation

        metrics["total_constraint_violation"] = total_violation

        # Multiplier magnitudes
        total_multiplier_magnitude = 0.0
        for name, multiplier in self.lagrange_multipliers.items():
            multiplier_mag = abs(multiplier.item())
            metrics[f"{name}_multiplier"] = multiplier_mag
            total_multiplier_magnitude += multiplier_mag

        metrics["total_multiplier_magnitude"] = total_multiplier_magnitude

        # Penalty parameters
        for name, penalty in self.penalty_parameters.items():
            metrics[f"{name}_penalty"] = penalty

        return metrics


class MetaLearningOptimizer(nn.Module):
    """
    Main physics-constrained optimizer for meta-learning.

    This class integrates gradient projection and Lagrangian optimization
    methods for physics-constrained meta-learning optimization.

    Args:
        config: Optimization configuration
        physics_regularizer: Physics regularizer for constraint definition
        method: Optimization method ('gradient_projection' or 'lagrangian')

    Example:
        >>> # Create physics regularizer with constraints
        >>> regularizer = PhysicsRegularizer(conservation_laws=[mass_law])
        >>>
        >>> # Create optimizer
        >>> config = OptimizationConfig(learning_rate=1e-3)
        >>> optimizer = MetaLearningOptimizer(config, regularizer, method='gradient_projection')
        >>>
        >>> # Optimization step
        >>> metrics = optimizer.optimize_step(loss_fn, model.parameters(), data)
    """

    def __init__(
        self,
        config: OptimizationConfig,
        physics_regularizer: Optional[PhysicsRegularizer] = None,
        method: str = "gradient_projection",
        base_optimizer: Optional[optim.Optimizer] = None,
    ):
        super().__init__()

        self.config = config
        self.physics_regularizer = physics_regularizer
        self.method = method

        # Initialize optimizer based on method
        if method == "gradient_projection":
            self.optimizer = GradientProjectionOptimizer(config, base_optimizer)
        elif method == "lagrangian":
            self.optimizer = LagrangianOptimizer(config)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Track optimization history
        self.optimization_history = []

    def optimize_step(
        self,
        loss_fn: Callable,
        parameters: List[torch.Tensor],
        data: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Perform physics-constrained optimization step.

        Args:
            loss_fn: Loss function to minimize
            parameters: Model parameters to optimize
            data: Data for constraint evaluation

        Returns:
            Dictionary of optimization metrics
        """
        # Convert physics regularizer constraints to constraint functions
        constraints = self._convert_physics_constraints(data)

        # Perform optimization step
        metrics = self.optimizer.step(loss_fn, parameters, constraints, data, **kwargs)

        # Add physics loss metrics if regularizer is available
        if self.physics_regularizer is not None:
            try:
                # Compute physics losses for monitoring
                predictions = data.get("predictions")
                coordinates = data.get("coordinates")
                task_info = data.get("task_info", {})

                if predictions is not None and coordinates is not None:
                    physics_losses = self.physics_regularizer.compute_physics_loss(
                        predictions, coordinates, task_info, create_graph=False
                    )

                    for loss_name, loss_value in physics_losses.items():
                        metrics[f"physics_{loss_name}"] = loss_value.item()

            except Exception as e:
                warnings.warn(f"Error computing physics losses: {e}")

        # Store optimization history
        self.optimization_history.append(metrics)

        return metrics

    def _convert_physics_constraints(
        self, data: Dict[str, Any]
    ) -> List[ConstraintFunction]:
        """Convert physics regularizer constraints to constraint functions."""
        constraints = []

        if self.physics_regularizer is None:
            return constraints

        predictions = data.get("predictions")
        coordinates = data.get("coordinates")
        task_info = data.get("task_info", {})

        if predictions is None or coordinates is None:
            return constraints

        # Convert conservation laws
        for law in self.physics_regularizer.conservation_laws:

            def constraint_fn(params, data_dict, law_name=law.name):
                pred = data_dict.get("predictions")
                coord = data_dict.get("coordinates")
                if pred is None or coord is None:
                    return torch.tensor(
                        0.0,
                        device=params.device if hasattr(params, "device") else "cpu",
                    )

                # Ensure fresh tensors for gradient computation
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().clone().requires_grad_(True)
                if isinstance(coord, torch.Tensor):
                    coord = coord.detach().clone().requires_grad_(True)

                try:
                    if law_name == "mass_conservation":
                        residual = self.physics_regularizer._mass_conservation_residual(
                            pred, coord, create_graph=False
                        )
                        return residual.mean().detach()
                    elif law_name == "momentum_conservation":
                        residual = (
                            self.physics_regularizer._momentum_conservation_residual(
                                pred, coord, task_info, create_graph=False
                            )
                        )
                        return residual.mean().detach()
                    elif law_name == "energy_conservation":
                        residual = (
                            self.physics_regularizer._energy_conservation_residual(
                                pred, coord, task_info, create_graph=False
                            )
                        )
                        return residual.mean().detach()
                    else:
                        result = law.equation(pred, coord)
                        return result.mean().detach()
                except Exception as e:
                    warnings.warn(f"Error in constraint function {law_name}: {e}")
                    return torch.tensor(
                        0.0,
                        device=params.device if hasattr(params, "device") else "cpu",
                    )

            constraints.append(
                ConstraintFunction(
                    name=law.name,
                    function=constraint_fn,
                    tolerance=law.tolerance,
                    weight=law.weight,
                    constraint_type="equality",
                )
            )

        # Convert causal constraints (simplified)
        for constraint in self.physics_regularizer.causal_constraints:

            def causal_constraint_fn(params, data_dict, constraint_obj=constraint):
                pred = data_dict.get("predictions")
                coord = data_dict.get("coordinates")
                if pred is None or coord is None:
                    return torch.tensor(
                        0.0,
                        device=params.device if hasattr(params, "device") else "cpu",
                    )

                # Ensure fresh tensors for gradient computation
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().clone()
                if isinstance(coord, torch.Tensor):
                    coord = coord.detach().clone()

                try:
                    # Simplified causal constraint - enforce correlation structure
                    cause_data = self.physics_regularizer._extract_variables(
                        pred, coord, constraint_obj.cause_vars, task_info
                    )
                    effect_data = self.physics_regularizer._extract_variables(
                        pred, coord, constraint_obj.effect_vars, task_info
                    )

                    result = self.physics_regularizer._correlation_constraint_loss(
                        cause_data, effect_data, constraint_obj.strength
                    )
                    return result.detach()
                except Exception as e:
                    warnings.warn(f"Error in causal constraint: {e}")
                    return torch.tensor(
                        0.0,
                        device=params.device if hasattr(params, "device") else "cpu",
                    )

            constraints.append(
                ConstraintFunction(
                    name=f"causal_{constraint.cause_vars[0] if constraint.cause_vars else 'unknown'}_{constraint.effect_vars[0] if constraint.effect_vars else 'unknown'}",
                    function=causal_constraint_fn,
                    weight=constraint.weight,
                    constraint_type="equality",
                )
            )

        return constraints

    def get_optimization_history(self) -> List[Dict[str, float]]:
        """Get optimization history for analysis."""
        return self.optimization_history

    def get_constraint_violations(self) -> Dict[str, List[float]]:
        """Get constraint violation history."""
        if hasattr(self.optimizer, "constraint_violations"):
            return dict(self.optimizer.constraint_violations)
        return {}

    def reset_history(self):
        """Reset optimization history."""
        self.optimization_history = []
        if hasattr(self.optimizer, "constraint_violations"):
            self.optimizer.constraint_violations.clear()
        if hasattr(self.optimizer, "gradient_norms"):
            self.optimizer.gradient_norms.clear()
        if hasattr(self.optimizer, "projection_iterations"):
            self.optimizer.projection_iterations.clear()

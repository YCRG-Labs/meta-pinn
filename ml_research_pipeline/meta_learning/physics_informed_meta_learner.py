"""
Physics-Informed Meta-Learning System.

This module implements a comprehensive physics-informed meta-learning system that
integrates physics constraints, adaptive constraint weighting, and physics-guided
optimization for improved meta-learning performance on physics tasks.
"""

import copy
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_constraint_integration import (
    AdaptiveConstraintIntegration,
    ConstraintConfig,
)
from .meta_learning_optimizer import (
    ConstraintFunction,
    GradientProjectionOptimizer,
    LagrangianOptimizer,
    MetaLearningOptimizer,
    OptimizationConfig,
)
from .physics_regularizer import (
    CausalConstraint,
    ConservationLaw,
    PhysicsRegularizer,
    SymbolicConstraint,
)


@dataclass
class PhysicsMetaLearningConfig:
    """Configuration for physics-informed meta-learning."""

    # Meta-learning parameters
    inner_lr: float = 1e-3
    outer_lr: float = 1e-4
    num_inner_steps: int = 5
    num_outer_steps: int = 1000

    # Physics constraint parameters
    physics_weight: float = 1.0
    adaptive_physics_weight: bool = True
    constraint_tolerance: float = 1e-4

    # Initialization parameters
    physics_guided_init: bool = True
    init_physics_weight: float = 10.0
    init_decay_steps: int = 100

    # Fast adaptation parameters
    fast_adaptation: bool = True
    physics_adaptation_rate: float = 0.1
    constraint_adaptation_steps: int = 3

    # Optimization parameters
    optimizer_type: str = "gradient_projection"  # 'gradient_projection' or 'lagrangian'
    use_second_order: bool = False
    gradient_clipping: float = 1.0

    # Monitoring parameters
    track_physics_metrics: bool = True
    validation_frequency: int = 10
    early_stopping_patience: int = 50


@dataclass
class Task:
    """Represents a physics task for meta-learning."""

    name: str
    data: Dict[str, torch.Tensor]
    physics_info: Dict[str, Any]
    support_set: Dict[str, torch.Tensor]
    query_set: Dict[str, torch.Tensor]
    task_type: str = "regression"


@dataclass
class MetaLearningResult:
    """Results from meta-learning training or adaptation."""

    model_state: Dict[str, torch.Tensor]
    loss_history: List[float]
    physics_loss_history: List[float]
    constraint_violations: Dict[str, List[float]]
    adaptation_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]


class PhysicsInformedMetaLearner(nn.Module):
    """
    Physics-Informed Meta-Learning System.

    This class integrates all physics-informed components into a comprehensive
    meta-learning system that can effectively utilize discovered physics relationships
    for improved performance and faster adaptation.

    Args:
        model: Base neural network model
        physics_regularizer: Physics constraint regularizer
        config: Meta-learning configuration
        device: Device for computations

    Example:
        >>> # Create physics constraints
        >>> mass_conservation = ConservationLaw(
        ...     name="mass_conservation",
        ...     equation=lambda u, coords: compute_divergence(u, coords),
        ...     weight=1.0
        ... )
        >>>
        >>> # Create physics regularizer
        >>> physics_reg = PhysicsRegularizer(
        ...     conservation_laws=[mass_conservation]
        ... )
        >>>
        >>> # Create meta-learner
        >>> config = PhysicsMetaLearningConfig()
        >>> meta_learner = PhysicsInformedMetaLearner(
        ...     model=neural_network,
        ...     physics_regularizer=physics_reg,
        ...     config=config
        ... )
        >>>
        >>> # Meta-train on physics tasks
        >>> result = meta_learner.meta_train(tasks)
        >>>
        >>> # Fast adaptation to new task
        >>> adapted_model = meta_learner.adapt(new_task, result.model_state)
    """

    def __init__(
        self,
        model: nn.Module,
        physics_regularizer: PhysicsRegularizer,
        config: PhysicsMetaLearningConfig = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.model = model
        self.physics_regularizer = physics_regularizer
        self.config = config or PhysicsMetaLearningConfig()
        self.device = device

        # Move components to device
        self.model = self.model.to(device)
        self.physics_regularizer = self.physics_regularizer.to(device)

        # Initialize adaptive constraint integration
        constraint_config = ConstraintConfig(
            adaptation_rate=self.config.physics_adaptation_rate,
            violation_threshold=self.config.constraint_tolerance,
        )
        self.constraint_integrator = AdaptiveConstraintIntegration(
            physics_regularizer=self.physics_regularizer,
            config=constraint_config,
            device=device,
        )

        # Initialize physics-constrained optimizer
        opt_config = OptimizationConfig(
            learning_rate=self.config.inner_lr,
            constraint_violation_tolerance=self.config.constraint_tolerance,
            gradient_clipping=self.config.gradient_clipping,
            use_second_order=self.config.use_second_order,
        )

        if self.config.optimizer_type == "gradient_projection":
            self.physics_optimizer = GradientProjectionOptimizer(opt_config)
        elif self.config.optimizer_type == "lagrangian":
            self.physics_optimizer = LagrangianOptimizer(opt_config)
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

        # Initialize meta-optimizer for outer loop
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.outer_lr
        )

        # Initialize physics-guided initialization strategy
        self.physics_initializer = PhysicsGuidedInitializer(
            physics_regularizer=self.physics_regularizer,
            config=self.config,
            device=device,
        )

        # Tracking and monitoring
        self.training_history = {
            "meta_loss": [],
            "physics_loss": [],
            "constraint_violations": defaultdict(list),
            "adaptation_metrics": defaultdict(list),
        }

        self.best_model_state = None
        self.best_validation_score = float("inf")
        self.patience_counter = 0

    def meta_train(
        self,
        tasks: List[Task],
        validation_tasks: Optional[List[Task]] = None,
        verbose: bool = True,
    ) -> MetaLearningResult:
        """
        Perform physics-informed meta-training.

        Args:
            tasks: List of training tasks
            validation_tasks: Optional validation tasks
            verbose: Whether to print progress

        Returns:
            Meta-learning results including trained model state
        """
        if verbose:
            print("Starting physics-informed meta-training...")

        # Initialize model with physics-guided initialization
        if self.config.physics_guided_init:
            self._physics_guided_initialization(tasks[:5])  # Use first 5 tasks for init

        # Meta-training loop
        for outer_step in range(self.config.num_outer_steps):
            # Sample batch of tasks
            batch_tasks = self._sample_task_batch(tasks)

            # Compute meta-gradients
            meta_loss, physics_metrics = self._compute_meta_gradients(batch_tasks)

            # Update meta-parameters
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

            # Track training progress
            self.training_history["meta_loss"].append(meta_loss.item())
            self.training_history["physics_loss"].append(
                physics_metrics["total_physics_loss"]
            )

            for constraint_type, violations in physics_metrics[
                "constraint_violations"
            ].items():
                self.training_history["constraint_violations"][constraint_type].extend(
                    violations
                )

            # Validation and early stopping
            if validation_tasks and outer_step % self.config.validation_frequency == 0:
                val_score = self._validate(validation_tasks)

                if val_score < self.best_validation_score:
                    self.best_validation_score = val_score
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at step {outer_step}")
                    break

            # Progress reporting
            if verbose and outer_step % 100 == 0:
                print(
                    f"Step {outer_step}: Meta Loss = {meta_loss.item():.6f}, "
                    f"Physics Loss = {physics_metrics['total_physics_loss']:.6f}"
                )

        # Prepare results
        final_model_state = (
            self.best_model_state if self.best_model_state else self.model.state_dict()
        )

        result = MetaLearningResult(
            model_state=final_model_state,
            loss_history=self.training_history["meta_loss"],
            physics_loss_history=self.training_history["physics_loss"],
            constraint_violations=dict(self.training_history["constraint_violations"]),
            adaptation_metrics=dict(self.training_history["adaptation_metrics"]),
            convergence_info={
                "converged": self.patience_counter
                < self.config.early_stopping_patience,
                "final_step": outer_step,
                "best_validation_score": self.best_validation_score,
            },
        )

        if verbose:
            print("Meta-training completed!")
            print(f"Final validation score: {self.best_validation_score:.6f}")

        return result

    def adapt(
        self,
        task: Task,
        meta_model_state: Dict[str, torch.Tensor],
        num_adaptation_steps: Optional[int] = None,
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Perform fast adaptation to a new task with physics constraints.

        Args:
            task: Target task for adaptation
            meta_model_state: Meta-learned model parameters
            num_adaptation_steps: Number of adaptation steps (uses config default if None)

        Returns:
            Tuple of (adapted_model, adaptation_metrics)
        """
        # Load meta-learned parameters
        adapted_model = copy.deepcopy(self.model)
        adapted_model.load_state_dict(meta_model_state)
        adapted_model.train()

        # Ensure model parameters require gradients
        for param in adapted_model.parameters():
            param.requires_grad_(True)

        # Reset constraint integrator for new task
        self.constraint_integrator.reset_adaptation()

        # Extract physics constraints for this task
        physics_constraints = self._extract_task_physics_constraints(task)

        # Adaptation loop
        num_steps = num_adaptation_steps or self.config.num_inner_steps
        adaptation_metrics = {
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "physics_improvement": 0.0,
            "constraint_satisfaction": 0.0,
            "adaptation_efficiency": 0.0,
        }

        # Compute initial loss
        with torch.no_grad():
            initial_predictions = adapted_model(task.support_set["coordinates"])
            initial_loss = F.mse_loss(initial_predictions, task.support_set["targets"])
            adaptation_metrics["initial_loss"] = initial_loss.item()

        # Physics-informed adaptation steps
        for step in range(num_steps):
            # Ensure coordinates require gradients for physics computations
            coordinates = (
                task.support_set["coordinates"].clone().detach().requires_grad_(True)
            )

            # Forward pass
            predictions = adapted_model(coordinates)

            # Compute task loss
            task_loss = F.mse_loss(predictions, task.support_set["targets"])

            # Compute physics loss with adaptive weighting
            physics_result = self.constraint_integrator.integrate_constraints(
                predictions=predictions,
                coordinates=coordinates,
                task_info=task.physics_info,
                confidence_scores=self._compute_constraint_confidence(
                    task, predictions
                ),
                create_graph=True,
            )

            # Combine losses with adaptive physics weighting
            physics_weight = self._get_adaptive_physics_weight(step, num_steps)
            total_loss = task_loss + physics_weight * physics_result["total_loss"]

            # Check if total_loss has gradients before calling backward
            if total_loss.requires_grad:
                # Standard gradient descent step
                total_loss.backward()

                # Apply gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        adapted_model.parameters(), self.config.gradient_clipping
                    )

                # Update parameters
                with torch.no_grad():
                    for param in adapted_model.parameters():
                        if param.grad is not None:
                            param -= self.config.inner_lr * param.grad
                            param.grad.zero_()
            else:
                # Fallback: just use task loss if it has gradients
                if task_loss.requires_grad:
                    task_loss.backward()

                    # Apply gradient clipping
                    if self.config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(
                            adapted_model.parameters(), self.config.gradient_clipping
                        )

                    # Update parameters
                    with torch.no_grad():
                        for param in adapted_model.parameters():
                            if param.grad is not None:
                                param -= self.config.inner_lr * param.grad
                                param.grad.zero_()
                else:
                    # Skip this step if no gradients available
                    pass

            # Track adaptation progress
            if step == 0:
                adaptation_metrics["initial_physics_loss"] = physics_result[
                    "total_loss"
                ].item()
            elif step == num_steps - 1:
                adaptation_metrics["final_physics_loss"] = physics_result[
                    "total_loss"
                ].item()

        # Compute final metrics
        with torch.no_grad():
            final_predictions = adapted_model(task.support_set["coordinates"])
            final_loss = F.mse_loss(final_predictions, task.support_set["targets"])
            adaptation_metrics["final_loss"] = final_loss.item()

            # Compute adaptation efficiency
            loss_improvement = (
                adaptation_metrics["initial_loss"] - adaptation_metrics["final_loss"]
            )
            adaptation_metrics["adaptation_efficiency"] = loss_improvement / (
                adaptation_metrics["initial_loss"] + 1e-8
            )

            # Compute physics improvement
            if (
                "initial_physics_loss" in adaptation_metrics
                and "final_physics_loss" in adaptation_metrics
            ):
                physics_improvement = (
                    adaptation_metrics["initial_physics_loss"]
                    - adaptation_metrics["final_physics_loss"]
                )
                adaptation_metrics["physics_improvement"] = physics_improvement / (
                    adaptation_metrics["initial_physics_loss"] + 1e-8
                )

            # Compute constraint satisfaction
            final_physics_result = self.constraint_integrator.integrate_constraints(
                predictions=final_predictions,
                coordinates=task.support_set["coordinates"],
                task_info=task.physics_info,
                create_graph=False,
            )

            constraint_metrics = final_physics_result["constraint_metrics"]
            total_violation = sum(
                metrics.violation_magnitude for metrics in constraint_metrics.values()
            )
            adaptation_metrics["constraint_satisfaction"] = 1.0 / (
                1.0 + total_violation
            )

        return adapted_model, adaptation_metrics

    def _physics_guided_initialization(self, sample_tasks: List[Task]):
        """Initialize model parameters using physics-guided strategies."""
        self.physics_initializer.initialize_model(self.model, sample_tasks)

    def _sample_task_batch(self, tasks: List[Task], batch_size: int = 4) -> List[Task]:
        """Sample a batch of tasks for meta-training."""
        indices = np.random.choice(
            len(tasks), size=min(batch_size, len(tasks)), replace=False
        )
        return [tasks[i] for i in indices]

    def _compute_meta_gradients(
        self, batch_tasks: List[Task]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute meta-gradients for a batch of tasks."""
        meta_loss = torch.tensor(0.0, device=self.device)
        physics_metrics = {
            "total_physics_loss": 0.0,
            "constraint_violations": defaultdict(list),
        }

        for task in batch_tasks:
            # Create task-specific model copy
            task_model = copy.deepcopy(self.model)
            task_model.train()

            # Inner loop adaptation
            for inner_step in range(self.config.num_inner_steps):
                # Forward pass
                predictions = task_model(task.support_set["coordinates"])

                # Task loss
                task_loss = F.mse_loss(predictions, task.support_set["targets"])

                # Physics loss
                physics_result = self.constraint_integrator.integrate_constraints(
                    predictions=predictions,
                    coordinates=task.support_set["coordinates"],
                    task_info=task.physics_info,
                    create_graph=True,
                )

                # Combined loss
                physics_weight = self._get_adaptive_physics_weight(
                    inner_step, self.config.num_inner_steps
                )

                # Only add physics loss if it has gradients
                if physics_result["total_loss"].requires_grad:
                    inner_loss = (
                        task_loss + physics_weight * physics_result["total_loss"]
                    )
                else:
                    inner_loss = task_loss

                # Compute gradients and update
                try:
                    grads = torch.autograd.grad(
                        inner_loss,
                        task_model.parameters(),
                        create_graph=True,
                        allow_unused=True,
                    )

                    # Update task model parameters
                    with torch.no_grad():
                        for param, grad in zip(task_model.parameters(), grads):
                            if grad is not None:
                                param -= self.config.inner_lr * grad
                except RuntimeError:
                    # Fallback to standard backward pass
                    inner_loss.backward()
                    with torch.no_grad():
                        for param in task_model.parameters():
                            if param.grad is not None:
                                param -= self.config.inner_lr * param.grad
                                param.grad.zero_()

                # Track physics metrics
                physics_metrics["total_physics_loss"] += physics_result[
                    "total_loss"
                ].item()

                for constraint_type, metrics in physics_result[
                    "constraint_metrics"
                ].items():
                    physics_metrics["constraint_violations"][constraint_type].append(
                        metrics.violation_magnitude
                    )

            # Compute query loss for meta-gradient
            query_predictions = task_model(task.query_set["coordinates"])
            query_loss = F.mse_loss(query_predictions, task.query_set["targets"])

            # Add to meta-loss
            meta_loss += query_loss

        # Average over batch
        meta_loss /= len(batch_tasks)
        physics_metrics["total_physics_loss"] /= (
            len(batch_tasks) * self.config.num_inner_steps
        )

        # Compute meta-gradients
        meta_loss.backward()

        return meta_loss, physics_metrics

    def _validate(self, validation_tasks: List[Task]) -> float:
        """Validate meta-learned model on validation tasks."""
        total_loss = 0.0
        num_tasks = 0

        with torch.no_grad():
            for task in validation_tasks:
                # Adapt to task
                adapted_model, _ = self.adapt(task, self.model.state_dict())

                # Evaluate on query set
                adapted_model.eval()
                predictions = adapted_model(task.query_set["coordinates"])
                loss = F.mse_loss(predictions, task.query_set["targets"])

                total_loss += loss.item()
                num_tasks += 1

        return total_loss / num_tasks if num_tasks > 0 else float("inf")

    def _extract_task_physics_constraints(self, task: Task) -> Dict[str, Any]:
        """Extract physics constraints relevant to a specific task."""
        constraints = {
            "conservation_laws": [],
            "causal_constraints": [],
            "symbolic_constraints": [],
        }

        # Extract from physics_info
        if "conservation_laws" in task.physics_info:
            constraints["conservation_laws"] = task.physics_info["conservation_laws"]

        if "causal_constraints" in task.physics_info:
            constraints["causal_constraints"] = task.physics_info["causal_constraints"]

        if "symbolic_constraints" in task.physics_info:
            constraints["symbolic_constraints"] = task.physics_info[
                "symbolic_constraints"
            ]

        # Add default physics constraints based on task type
        if task.task_type == "fluid_dynamics":
            # Add mass conservation for fluid dynamics
            if not any(
                law.name == "mass_conservation"
                for law in constraints["conservation_laws"]
            ):
                mass_conservation = ConservationLaw(
                    name="mass_conservation",
                    equation=lambda u, coords: self._compute_divergence(u, coords),
                    weight=1.0,
                )
                constraints["conservation_laws"].append(mass_conservation)

        return constraints

    def _compute_constraint_confidence(
        self, task: Task, predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute confidence scores for physics constraints."""
        confidence_scores = {}

        # Simple confidence based on prediction consistency
        pred_std = torch.std(predictions, dim=0)
        base_confidence = torch.exp(-pred_std)  # Higher confidence for lower variance

        # Conservation law confidence
        if self.physics_regularizer.conservation_laws:
            n_conservation = len(self.physics_regularizer.conservation_laws)
            confidence_scores["conservation"] = base_confidence.mean().repeat(
                n_conservation
            )

        # Causal constraint confidence
        if self.physics_regularizer.causal_constraints:
            n_causal = len(self.physics_regularizer.causal_constraints)
            confidence_scores["causal"] = base_confidence.mean().repeat(n_causal)

        # Symbolic constraint confidence
        if self.physics_regularizer.symbolic_constraints:
            n_symbolic = len(self.physics_regularizer.symbolic_constraints)
            confidence_scores["symbolic"] = base_confidence.mean().repeat(n_symbolic)

        return confidence_scores

    def _get_adaptive_physics_weight(self, step: int, total_steps: int) -> float:
        """Get adaptive physics weight based on training progress."""
        if not self.config.adaptive_physics_weight:
            return self.config.physics_weight

        # Start with high physics weight and decay
        if step < self.config.init_decay_steps:
            decay_factor = step / self.config.init_decay_steps
            weight = (
                self.config.init_physics_weight * (1 - decay_factor)
                + self.config.physics_weight * decay_factor
            )
        else:
            weight = self.config.physics_weight

        return weight

    def _convert_to_constraint_functions(
        self, physics_constraints: Dict[str, Any], task: Task
    ) -> List[ConstraintFunction]:
        """Convert physics constraints to constraint functions for optimizer."""
        constraint_functions = []

        # Conservation law constraints
        for law in physics_constraints.get("conservation_laws", []):

            def constraint_fn(params, data, law=law):
                # This is a simplified constraint function
                # In practice, you'd evaluate the conservation law
                return torch.tensor(0.0, device=self.device)

            constraint_func = ConstraintFunction(
                name=law.name,
                function=constraint_fn,
                constraint_type="equality",
                tolerance=self.config.constraint_tolerance,
                weight=law.weight,
            )
            constraint_functions.append(constraint_func)

        return constraint_functions

    def _compute_divergence(
        self, velocity: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Compute divergence for mass conservation."""
        if not coordinates.requires_grad:
            coordinates = coordinates.clone().detach().requires_grad_(True)

        u = velocity[:, 0:1]
        v = velocity[:, 1:2]

        try:
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

            if u_grad is None or v_grad is None:
                return torch.zeros(velocity.shape[0], 1, device=velocity.device)

            # Divergence: ∂u/∂x + ∂v/∂y
            divergence = u_grad[:, 0:1] + v_grad[:, 1:2]
            return divergence

        except RuntimeError:
            # Fallback if gradient computation fails
            return torch.zeros(velocity.shape[0], 1, device=velocity.device)

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        return {
            "loss_history": self.training_history["meta_loss"],
            "physics_loss_history": self.training_history["physics_loss"],
            "constraint_violations": dict(
                self.training_history["constraint_violations"]
            ),
            "best_validation_score": self.best_validation_score,
            "convergence_info": {
                "patience_counter": self.patience_counter,
                "converged": self.patience_counter
                < self.config.early_stopping_patience,
            },
        }

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "meta_optimizer_state_dict": self.meta_optimizer.state_dict(),
            "training_history": self.training_history,
            "best_validation_score": self.best_validation_score,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state_dict"])
        self.training_history = checkpoint["training_history"]
        self.best_validation_score = checkpoint["best_validation_score"]


class PhysicsGuidedInitializer:
    """Physics-guided initialization strategies for meta-learning models."""

    def __init__(
        self,
        physics_regularizer: PhysicsRegularizer,
        config: PhysicsMetaLearningConfig,
        device: str = "cpu",
    ):
        self.physics_regularizer = physics_regularizer
        self.config = config
        self.device = device

    def initialize_model(self, model: nn.Module, sample_tasks: List[Task]):
        """Initialize model using physics-guided strategies."""
        # Strategy 1: Initialize based on physics symmetries
        self._initialize_with_symmetries(model)

        # Strategy 2: Pre-train on physics constraints
        if len(sample_tasks) > 0:
            self._pretrain_on_physics(model, sample_tasks)

    def _initialize_with_symmetries(self, model: nn.Module):
        """Initialize weights to respect physics symmetries."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with physics-informed scaling
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv2d):
                # He initialization for convolutional layers
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pretrain_on_physics(self, model: nn.Module, sample_tasks: List[Task]):
        """Pre-train model to satisfy physics constraints."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(50):  # Limited pre-training
            total_loss = 0.0

            for task in sample_tasks:
                # Forward pass
                predictions = model(task.support_set["coordinates"])

                # Compute physics loss only
                physics_losses = self.physics_regularizer.compute_physics_loss(
                    predictions=predictions,
                    coordinates=task.support_set["coordinates"],
                    task_info=task.physics_info,
                    create_graph=True,
                )

                loss = physics_losses["total"]
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early stopping if physics constraints are satisfied
            if total_loss < len(sample_tasks) * self.config.constraint_tolerance:
                break

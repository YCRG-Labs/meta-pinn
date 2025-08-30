"""
Adaptive constraint integration for physics-informed meta-learning.

This module implements dynamic constraint weighting and integration strategies
that adapt constraint strength based on confidence scores and training progress.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .physics_regularizer import PhysicsRegularizer


@dataclass
class ConstraintConfig:
    """Configuration for adaptive constraint integration."""

    adaptation_rate: float = 0.01
    confidence_threshold: float = 0.5
    violation_threshold: float = 1e-3
    smoothing_window: int = 10
    min_weight: float = 1e-6
    max_weight: float = 10.0
    temperature_schedule: str = "exponential"
    initial_temperature: float = 1.0
    final_temperature: float = 0.1


@dataclass
class ConstraintMetrics:
    """Metrics for constraint monitoring."""

    violation_magnitude: float
    confidence_score: float
    weight: float
    adaptation_rate: float
    violation_trend: float


class AdaptiveConstraintIntegration(nn.Module):
    """
    Adaptive constraint integration for physics-informed meta-learning.

    This class implements dynamic constraint weighting based on confidence scores,
    violation magnitudes, and training progress.
    """

    def __init__(
        self,
        physics_regularizer: PhysicsRegularizer,
        config: ConstraintConfig = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.physics_regularizer = physics_regularizer
        self.config = config or ConstraintConfig()
        self.device = device

        # Initialize adaptive weights and tracking
        self._init_adaptive_weights()
        self._init_monitoring()

        # Training step counter for scheduling
        self.training_step = 0

    def _init_adaptive_weights(self):
        """Initialize adaptive weight parameters."""
        # Conservation law weights
        n_conservation = len(self.physics_regularizer.conservation_laws)
        if n_conservation > 0:
            self.conservation_weights = nn.Parameter(
                torch.ones(n_conservation, device=self.device)
            )
            self.conservation_confidence = torch.ones(
                n_conservation, device=self.device
            )

        # Causal constraint weights
        n_causal = len(self.physics_regularizer.causal_constraints)
        if n_causal > 0:
            self.causal_weights = nn.Parameter(torch.ones(n_causal, device=self.device))
            self.causal_confidence = torch.ones(n_causal, device=self.device)

        # Symbolic constraint weights
        n_symbolic = len(self.physics_regularizer.symbolic_constraints)
        if n_symbolic > 0:
            self.symbolic_weights = nn.Parameter(
                torch.ones(n_symbolic, device=self.device)
            )
            self.symbolic_confidence = torch.ones(n_symbolic, device=self.device)

    def _init_monitoring(self):
        """Initialize constraint violation monitoring."""
        self.violation_history = {
            "conservation": deque(maxlen=self.config.smoothing_window),
            "causal": deque(maxlen=self.config.smoothing_window),
            "symbolic": deque(maxlen=self.config.smoothing_window),
        }

        self.weight_history = {
            "conservation": deque(maxlen=self.config.smoothing_window),
            "causal": deque(maxlen=self.config.smoothing_window),
            "symbolic": deque(maxlen=self.config.smoothing_window),
        }

    def integrate_constraints(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        task_info: Dict[str, Any],
        confidence_scores: Optional[Dict[str, torch.Tensor]] = None,
        create_graph: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate physics constraints with adaptive weighting.
        """
        # Update training step
        self.training_step += 1

        # Compute base physics losses
        physics_losses = self.physics_regularizer.compute_physics_loss(
            predictions, coordinates, task_info, create_graph
        )

        # Update confidence scores
        if confidence_scores is not None:
            self._update_confidence_scores(confidence_scores)

        # Adapt constraint weights
        adapted_losses = self._adapt_constraint_weights(physics_losses)

        # Monitor constraint violations
        self._monitor_violations(physics_losses)

        # Apply temperature scheduling
        temperature = self._get_current_temperature()

        # Compute final integrated loss
        integrated_loss = self._compute_integrated_loss(adapted_losses, temperature)

        # Prepare output with metrics
        output = {
            "total_loss": integrated_loss,
            "physics_losses": physics_losses,
            "adapted_losses": adapted_losses,
            "constraint_metrics": self._get_constraint_metrics(),
            "temperature": temperature,
        }

        return output

    def _update_confidence_scores(self, confidence_scores: Dict[str, torch.Tensor]):
        """Update confidence scores for constraints."""
        if "conservation" in confidence_scores and hasattr(
            self, "conservation_confidence"
        ):
            # Clamp confidence scores to valid range [0, 1]
            clamped_scores = torch.clamp(confidence_scores["conservation"], 0.0, 1.0)
            self.conservation_confidence = clamped_scores.to(self.device)

        if "causal" in confidence_scores and hasattr(self, "causal_confidence"):
            clamped_scores = torch.clamp(confidence_scores["causal"], 0.0, 1.0)
            self.causal_confidence = clamped_scores.to(self.device)

        if "symbolic" in confidence_scores and hasattr(self, "symbolic_confidence"):
            clamped_scores = torch.clamp(confidence_scores["symbolic"], 0.0, 1.0)
            self.symbolic_confidence = clamped_scores.to(self.device)

    def _adapt_constraint_weights(
        self, physics_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Adapt constraint weights based on violations and confidence."""
        adapted_losses = {}

        # Adapt conservation law weights
        if "conservation" in physics_losses and hasattr(self, "conservation_weights"):
            adapted_conservation = self._adapt_conservation_weights(
                physics_losses["conservation"]
            )
            adapted_losses["conservation"] = adapted_conservation

        # Adapt causal constraint weights
        if "causal" in physics_losses and hasattr(self, "causal_weights"):
            adapted_causal = self._adapt_causal_weights(physics_losses["causal"])
            adapted_losses["causal"] = adapted_causal

        # Adapt symbolic constraint weights
        if "symbolic" in physics_losses and hasattr(self, "symbolic_weights"):
            adapted_symbolic = self._adapt_symbolic_weights(physics_losses["symbolic"])
            adapted_losses["symbolic"] = adapted_symbolic

        # Copy other losses unchanged
        for key, value in physics_losses.items():
            if key not in adapted_losses:
                adapted_losses[key] = value

        return adapted_losses

    def _adapt_conservation_weights(
        self, conservation_loss: torch.Tensor
    ) -> torch.Tensor:
        """Adapt conservation law constraint weights."""
        if not hasattr(self, "conservation_weights"):
            return conservation_loss

        # Get current violation magnitude
        violation_magnitude = conservation_loss.detach()

        # Compute adaptation based on violation and confidence
        confidence_factor = torch.clamp(self.conservation_confidence, 0.0, 1.0)

        # Increase weight if violation is high and confidence is high
        violation_factor = torch.sigmoid(
            violation_magnitude / self.config.violation_threshold - 1.0
        )

        # Compute weight adaptation
        weight_adaptation = (
            self.config.adaptation_rate * confidence_factor * violation_factor
        )

        # Update weights with momentum
        momentum = 0.9
        self.conservation_weights.data = (
            momentum * self.conservation_weights.data
            + (1 - momentum) * weight_adaptation
        )

        # Clamp weights to valid range
        self.conservation_weights.data = torch.clamp(
            self.conservation_weights.data,
            self.config.min_weight,
            self.config.max_weight,
        )

        # Apply adapted weights
        adapted_loss = torch.mean(self.conservation_weights) * conservation_loss

        # Store weight history
        self.weight_history["conservation"].append(
            torch.mean(self.conservation_weights).item()
        )

        return adapted_loss

    def _adapt_causal_weights(self, causal_loss: torch.Tensor) -> torch.Tensor:
        """Adapt causal constraint weights."""
        if not hasattr(self, "causal_weights"):
            return causal_loss

        # Get current violation magnitude
        violation_magnitude = causal_loss.detach()

        # Compute adaptation based on violation and confidence
        confidence_factor = torch.clamp(self.causal_confidence, 0.0, 1.0)

        # Increase weight if violation is high and confidence is high
        violation_factor = torch.sigmoid(
            violation_magnitude / self.config.violation_threshold - 1.0
        )

        # Compute weight adaptation
        weight_adaptation = (
            self.config.adaptation_rate * confidence_factor * violation_factor
        )

        # Update weights with momentum
        momentum = 0.9
        self.causal_weights.data = (
            momentum * self.causal_weights.data + (1 - momentum) * weight_adaptation
        )

        # Clamp weights to valid range
        self.causal_weights.data = torch.clamp(
            self.causal_weights.data, self.config.min_weight, self.config.max_weight
        )

        # Apply adapted weights
        adapted_loss = torch.mean(self.causal_weights) * causal_loss

        # Store weight history
        self.weight_history["causal"].append(torch.mean(self.causal_weights).item())

        return adapted_loss

    def _adapt_symbolic_weights(self, symbolic_loss: torch.Tensor) -> torch.Tensor:
        """Adapt symbolic constraint weights."""
        if not hasattr(self, "symbolic_weights"):
            return symbolic_loss

        # Get current violation magnitude
        violation_magnitude = symbolic_loss.detach()

        # Compute adaptation based on violation and confidence
        confidence_factor = torch.clamp(self.symbolic_confidence, 0.0, 1.0)

        # Increase weight if violation is high and confidence is high
        violation_factor = torch.sigmoid(
            violation_magnitude / self.config.violation_threshold - 1.0
        )

        # Compute weight adaptation
        weight_adaptation = (
            self.config.adaptation_rate * confidence_factor * violation_factor
        )

        # Update weights with momentum
        momentum = 0.9
        self.symbolic_weights.data = (
            momentum * self.symbolic_weights.data + (1 - momentum) * weight_adaptation
        )

        # Clamp weights to valid range
        self.symbolic_weights.data = torch.clamp(
            self.symbolic_weights.data, self.config.min_weight, self.config.max_weight
        )

        # Apply adapted weights
        adapted_loss = torch.mean(self.symbolic_weights) * symbolic_loss

        # Store weight history
        self.weight_history["symbolic"].append(torch.mean(self.symbolic_weights).item())

        return adapted_loss

    def _monitor_violations(self, physics_losses: Dict[str, torch.Tensor]):
        """Monitor constraint violations for trend analysis."""
        for constraint_type in ["conservation", "causal", "symbolic"]:
            if constraint_type in physics_losses:
                violation = physics_losses[constraint_type].detach().item()
                self.violation_history[constraint_type].append(violation)

    def _get_current_temperature(self) -> float:
        """Get current temperature based on scheduling."""
        if self.config.temperature_schedule == "exponential":
            decay_rate = 0.99
            temperature = self.config.initial_temperature * (
                decay_rate**self.training_step
            )
            temperature = max(temperature, self.config.final_temperature)
        elif self.config.temperature_schedule == "linear":
            max_steps = 10000
            progress = min(self.training_step / max_steps, 1.0)
            temperature = (
                self.config.initial_temperature * (1 - progress)
                + self.config.final_temperature * progress
            )
        elif self.config.temperature_schedule == "cosine":
            max_steps = 10000
            progress = min(self.training_step / max_steps, 1.0)
            temperature = self.config.final_temperature + 0.5 * (
                self.config.initial_temperature - self.config.final_temperature
            ) * (1 + np.cos(np.pi * progress))
        else:
            temperature = self.config.initial_temperature

        return temperature

    def _compute_integrated_loss(
        self, adapted_losses: Dict[str, torch.Tensor], temperature: float
    ) -> torch.Tensor:
        """Compute final integrated loss with temperature scaling."""
        loss_components = []

        for constraint_type in ["conservation", "causal", "symbolic"]:
            if constraint_type in adapted_losses:
                scaled_loss = adapted_losses[constraint_type] / temperature
                loss_components.append(scaled_loss)

        if loss_components:
            total_loss = sum(loss_components)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_loss

    def _get_constraint_metrics(self) -> Dict[str, ConstraintMetrics]:
        """Get current constraint metrics for monitoring."""
        metrics = {}

        for constraint_type in ["conservation", "causal", "symbolic"]:
            if len(self.violation_history[constraint_type]) > 0:
                violation_magnitude = self.violation_history[constraint_type][-1]

                if len(self.violation_history[constraint_type]) >= 2:
                    recent_violations = list(self.violation_history[constraint_type])[
                        -5:
                    ]
                    violation_trend = np.polyfit(
                        range(len(recent_violations)), recent_violations, 1
                    )[0]
                else:
                    violation_trend = 0.0

                if len(self.weight_history[constraint_type]) > 0:
                    current_weight = self.weight_history[constraint_type][-1]
                else:
                    current_weight = 1.0

                if constraint_type == "conservation" and hasattr(
                    self, "conservation_confidence"
                ):
                    confidence_score = torch.mean(self.conservation_confidence).item()
                elif constraint_type == "causal" and hasattr(self, "causal_confidence"):
                    confidence_score = torch.mean(self.causal_confidence).item()
                elif constraint_type == "symbolic" and hasattr(
                    self, "symbolic_confidence"
                ):
                    confidence_score = torch.mean(self.symbolic_confidence).item()
                else:
                    confidence_score = 1.0

                metrics[constraint_type] = ConstraintMetrics(
                    violation_magnitude=violation_magnitude,
                    confidence_score=confidence_score,
                    weight=current_weight,
                    adaptation_rate=self.config.adaptation_rate,
                    violation_trend=violation_trend,
                )

        return metrics

    def reset_adaptation(self):
        """Reset adaptation state for new task."""
        self._init_adaptive_weights()
        self._init_monitoring()
        self.training_step = 0

    def get_constraint_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all constraints."""
        status = {}

        for constraint_type in ["conservation", "causal", "symbolic"]:
            if len(self.violation_history[constraint_type]) > 0:
                recent_violations = list(self.violation_history[constraint_type])
                recent_weights = list(self.weight_history[constraint_type])

                status[constraint_type] = {
                    "current_violation": (
                        recent_violations[-1] if recent_violations else 0.0
                    ),
                    "average_violation": (
                        np.mean(recent_violations) if recent_violations else 0.0
                    ),
                    "violation_std": (
                        np.std(recent_violations) if len(recent_violations) > 1 else 0.0
                    ),
                    "current_weight": recent_weights[-1] if recent_weights else 1.0,
                    "weight_range": (
                        (min(recent_weights), max(recent_weights))
                        if recent_weights
                        else (1.0, 1.0)
                    ),
                    "is_active": (
                        recent_violations[-1] > self.config.violation_threshold
                        if recent_violations
                        else False
                    ),
                    "num_violations": sum(
                        1
                        for v in recent_violations
                        if v > self.config.violation_threshold
                    ),
                }

        return status

    def get_violation_trends(self) -> Dict[str, List[float]]:
        """Get violation trends for all constraint types."""
        trends = {}
        for constraint_type in ["conservation", "causal", "symbolic"]:
            trends[constraint_type] = list(self.violation_history[constraint_type])
        return trends

    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for constraint activation."""
        self.config.confidence_threshold = threshold

    def set_adaptation_rate(self, rate: float):
        """Set adaptation rate for weight updates."""
        self.config.adaptation_rate = rate

"""
Hyperparameter Optimization Framework for physics discovery pipeline.

This module provides a comprehensive framework for automated hyperparameter
optimization across all discovery components with early stopping, resource
allocation, and importance analysis.
"""

import json
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .bayesian_optimizer import (
    BayesianOptimizer,
    OptimizationBounds,
    OptimizationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""

    name: str
    bounds: OptimizationBounds
    parameter_names: List[str]
    parameter_types: List[str] = field(
        default_factory=list
    )  # 'continuous', 'integer', 'categorical'
    categorical_values: Dict[str, List[Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate hyperparameter space."""
        if len(self.parameter_names) != len(self.bounds.lower):
            raise ValueError("Number of parameter names must match bounds dimensions")

        if not self.parameter_types:
            self.parameter_types = ["continuous"] * len(self.parameter_names)

        if len(self.parameter_types) != len(self.parameter_names):
            raise ValueError("Number of parameter types must match parameter names")


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    max_iterations: int = 100
    max_time_seconds: Optional[float] = None
    n_initial_points: int = 10
    acquisition_function: str = "ei"
    early_stopping_patience: int = 20
    convergence_threshold: float = 1e-4
    n_parallel_jobs: int = 1
    random_state: Optional[int] = None
    save_results: bool = True
    results_dir: str = "optimization_results"


@dataclass
class ComponentConfig:
    """Configuration for a discovery component."""

    name: str
    hyperparameter_space: HyperparameterSpace
    objective_function: Callable[[Dict[str, Any]], float]
    weight: float = 1.0
    timeout_seconds: Optional[float] = None
    enabled: bool = True


@dataclass
class OptimizationHistory:
    """History of optimization process."""

    iteration: int
    parameters: Dict[str, Any]
    objective_value: float
    component_scores: Dict[str, float]
    evaluation_time: float
    timestamp: float


@dataclass
class ImportanceAnalysis:
    """Hyperparameter importance analysis results."""

    parameter_importance: Dict[str, float]
    parameter_correlations: Dict[str, Dict[str, float]]
    sensitivity_analysis: Dict[str, List[float]]
    interaction_effects: Dict[Tuple[str, str], float]


class EarlyStopping:
    """Early stopping mechanism for optimization."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Check if optimization should stop early."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True

        return False


class ResourceManager:
    """Manages computational resources during optimization."""

    def __init__(self, max_parallel_jobs: int = 1, max_memory_gb: float = 8.0):
        self.max_parallel_jobs = max_parallel_jobs
        self.max_memory_gb = max_memory_gb
        self.active_jobs = 0
        self.memory_usage = 0.0

    def can_start_job(self, estimated_memory_gb: float = 1.0) -> bool:
        """Check if a new job can be started."""
        return (
            self.active_jobs < self.max_parallel_jobs
            and self.memory_usage + estimated_memory_gb <= self.max_memory_gb
        )

    def start_job(self, estimated_memory_gb: float = 1.0):
        """Start a new job."""
        self.active_jobs += 1
        self.memory_usage += estimated_memory_gb

    def finish_job(self, estimated_memory_gb: float = 1.0):
        """Finish a job."""
        self.active_jobs = max(0, self.active_jobs - 1)
        self.memory_usage = max(0.0, self.memory_usage - estimated_memory_gb)


class HyperparameterOptimizationFramework:
    """
    Comprehensive framework for hyperparameter optimization across all
    physics discovery components.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize optimization framework.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.components: Dict[str, ComponentConfig] = {}
        self.optimizers: Dict[str, BayesianOptimizer] = {}
        self.history: List[OptimizationHistory] = []
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.convergence_threshold,
        )
        self.resource_manager = ResourceManager(
            max_parallel_jobs=config.n_parallel_jobs
        )

        # Create results directory
        if config.save_results:
            Path(config.results_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized hyperparameter optimization framework")

    def register_component(self, component_config: ComponentConfig):
        """
        Register a discovery component for optimization.

        Args:
            component_config: Component configuration
        """
        if not component_config.enabled:
            logger.info(f"Component {component_config.name} is disabled, skipping")
            return

        self.components[component_config.name] = component_config

        # Create Bayesian optimizer for this component
        self.optimizers[component_config.name] = BayesianOptimizer(
            bounds=component_config.hyperparameter_space.bounds,
            acquisition_function=self.config.acquisition_function,
            random_state=self.config.random_state,
        )

        logger.info(f"Registered component: {component_config.name}")

    def _convert_parameters(
        self, raw_params: np.ndarray, hyperparameter_space: HyperparameterSpace
    ) -> Dict[str, Any]:
        """Convert raw parameter array to named dictionary."""
        params = {}

        for i, (name, param_type) in enumerate(
            zip(
                hyperparameter_space.parameter_names,
                hyperparameter_space.parameter_types,
            )
        ):
            value = raw_params[i]

            if param_type == "integer":
                params[name] = int(round(value))
            elif param_type == "categorical":
                # Map continuous value to categorical choice
                choices = hyperparameter_space.categorical_values[name]
                idx = int(round(value * (len(choices) - 1)))
                idx = max(0, min(len(choices) - 1, idx))
                params[name] = choices[idx]
            else:  # continuous
                params[name] = float(value)

        return params

    def _evaluate_component(
        self, component_name: str, raw_params: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate a single component with given parameters.

        Returns:
            Tuple of (objective_value, evaluation_time)
        """
        component = self.components[component_name]

        # Convert parameters
        params = self._convert_parameters(raw_params, component.hyperparameter_space)

        # Evaluate with timeout
        start_time = time.time()

        try:
            if component.timeout_seconds:
                # TODO: Implement timeout mechanism
                objective_value = component.objective_function(params)
            else:
                objective_value = component.objective_function(params)
        except Exception as e:
            logger.warning(f"Component {component_name} evaluation failed: {e}")
            objective_value = -np.inf

        evaluation_time = time.time() - start_time

        return objective_value, evaluation_time

    def _evaluate_all_components(self, parameter_dict: Dict[str, np.ndarray]) -> float:
        """
        Evaluate all components with their respective parameters.

        Args:
            parameter_dict: Dictionary mapping component names to parameter arrays

        Returns:
            Weighted average objective value
        """
        component_scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        if self.config.n_parallel_jobs > 1:
            # Parallel evaluation
            with ThreadPoolExecutor(
                max_workers=self.config.n_parallel_jobs
            ) as executor:
                future_to_component = {
                    executor.submit(self._evaluate_component, name, params): name
                    for name, params in parameter_dict.items()
                    if name in self.components
                }

                for future in as_completed(future_to_component):
                    component_name = future_to_component[future]
                    try:
                        objective_value, eval_time = future.result()
                        component = self.components[component_name]

                        component_scores[component_name] = objective_value
                        weighted_sum += objective_value * component.weight
                        total_weight += component.weight

                        logger.debug(
                            f"{component_name}: {objective_value:.6f} ({eval_time:.2f}s)"
                        )
                    except Exception as e:
                        logger.error(f"Component {component_name} failed: {e}")
                        component_scores[component_name] = -np.inf
        else:
            # Sequential evaluation
            for component_name, params in parameter_dict.items():
                if component_name not in self.components:
                    continue

                objective_value, eval_time = self._evaluate_component(
                    component_name, params
                )
                component = self.components[component_name]

                component_scores[component_name] = objective_value
                weighted_sum += objective_value * component.weight
                total_weight += component.weight

                logger.debug(
                    f"{component_name}: {objective_value:.6f} ({eval_time:.2f}s)"
                )

        # Calculate weighted average
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = -np.inf

        # Store history
        self.history.append(
            OptimizationHistory(
                iteration=len(self.history),
                parameters={
                    name: self._convert_parameters(
                        params, self.components[name].hyperparameter_space
                    )
                    for name, params in parameter_dict.items()
                    if name in self.components
                },
                objective_value=overall_score,
                component_scores=component_scores,
                evaluation_time=sum(
                    eval_time
                    for _, eval_time in [
                        self._evaluate_component(name, params)
                        for name, params in parameter_dict.items()
                        if name in self.components
                    ]
                ),
                timestamp=time.time(),
            )
        )

        return overall_score

    def optimize_single_component(self, component_name: str) -> OptimizationResult:
        """
        Optimize hyperparameters for a single component.

        Args:
            component_name: Name of component to optimize

        Returns:
            Optimization result
        """
        if component_name not in self.components:
            raise ValueError(f"Component {component_name} not registered")

        component = self.components[component_name]
        optimizer = self.optimizers[component_name]

        logger.info(f"Starting optimization for component: {component_name}")

        def objective_wrapper(raw_params: np.ndarray) -> float:
            objective_value, eval_time = self._evaluate_component(
                component_name, raw_params
            )

            # Add to history for single component optimization
            converted_params = self._convert_parameters(
                raw_params, component.hyperparameter_space
            )
            self.history.append(
                OptimizationHistory(
                    iteration=len(self.history),
                    parameters={component_name: converted_params},
                    objective_value=objective_value,
                    component_scores={component_name: objective_value},
                    evaluation_time=eval_time,
                    timestamp=time.time(),
                )
            )

            return objective_value

        result = optimizer.optimize(
            objective_func=objective_wrapper,
            n_iterations=self.config.max_iterations,
            n_initial=self.config.n_initial_points,
            convergence_threshold=self.config.convergence_threshold,
            patience=self.config.early_stopping_patience,
        )

        # Convert best parameters
        best_params = self._convert_parameters(
            result.best_params, component.hyperparameter_space
        )

        logger.info(
            f"Optimization completed for {component_name}. Best score: {result.best_value:.6f}"
        )
        logger.info(f"Best parameters: {best_params}")

        if self.config.save_results:
            self._save_component_results(component_name, result, best_params)

        return result

    def optimize_all_components(self) -> Dict[str, OptimizationResult]:
        """
        Optimize hyperparameters for all registered components jointly.

        Returns:
            Dictionary of optimization results for each component
        """
        if not self.components:
            raise ValueError("No components registered for optimization")

        logger.info(
            f"Starting joint optimization for {len(self.components)} components"
        )

        # Create joint parameter space
        joint_bounds_lower = []
        joint_bounds_upper = []
        component_param_indices = {}

        current_idx = 0
        for name, component in self.components.items():
            space = component.hyperparameter_space
            n_params = len(space.parameter_names)

            component_param_indices[name] = (current_idx, current_idx + n_params)
            joint_bounds_lower.extend(space.bounds.lower)
            joint_bounds_upper.extend(space.bounds.upper)

            current_idx += n_params

        joint_bounds = OptimizationBounds(
            lower=np.array(joint_bounds_lower), upper=np.array(joint_bounds_upper)
        )

        # Create joint optimizer
        joint_optimizer = BayesianOptimizer(
            bounds=joint_bounds,
            acquisition_function=self.config.acquisition_function,
            random_state=self.config.random_state,
        )

        def joint_objective(joint_params: np.ndarray) -> float:
            # Split joint parameters for each component
            parameter_dict = {}
            for name, (start_idx, end_idx) in component_param_indices.items():
                parameter_dict[name] = joint_params[start_idx:end_idx]

            return self._evaluate_all_components(parameter_dict)

        # Run optimization
        start_time = time.time()

        result = joint_optimizer.optimize(
            objective_func=joint_objective,
            n_iterations=self.config.max_iterations,
            n_initial=self.config.n_initial_points,
            convergence_threshold=self.config.convergence_threshold,
            patience=self.config.early_stopping_patience,
        )

        optimization_time = time.time() - start_time

        # Extract individual component results
        component_results = {}
        for name, (start_idx, end_idx) in component_param_indices.items():
            component_params = result.best_params[start_idx:end_idx]
            component_best_params = self._convert_parameters(
                component_params, self.components[name].hyperparameter_space
            )

            # Create individual result (simplified)
            component_results[name] = OptimizationResult(
                best_params=component_params,
                best_value=result.best_value,  # Joint score
                best_values=result.best_values,
                all_params=[],  # Not tracked individually
                all_values=[],
                n_iterations=result.n_iterations,
                convergence_history=result.convergence_history,
            )

            logger.info(f"Best parameters for {name}: {component_best_params}")

        logger.info(f"Joint optimization completed in {optimization_time:.2f}s")
        logger.info(f"Best joint score: {result.best_value:.6f}")

        if self.config.save_results:
            self._save_joint_results(component_results, result)

        return component_results

    def analyze_hyperparameter_importance(
        self, component_name: str, n_samples: int = 1000
    ) -> ImportanceAnalysis:
        """
        Analyze hyperparameter importance for a component.

        Args:
            component_name: Name of component to analyze
            n_samples: Number of samples for analysis

        Returns:
            Importance analysis results
        """
        if component_name not in self.components:
            raise ValueError(f"Component {component_name} not registered")

        component = self.components[component_name]
        space = component.hyperparameter_space

        logger.info(f"Analyzing hyperparameter importance for {component_name}")

        # Generate random samples
        rng = np.random.RandomState(self.config.random_state)
        samples = []
        objectives = []

        for _ in range(n_samples):
            # Random sample in parameter space
            sample = rng.uniform(space.bounds.lower, space.bounds.upper)
            samples.append(sample)

            # Evaluate objective
            objective_value, _ = self._evaluate_component(component_name, sample)
            objectives.append(objective_value)

        samples = np.array(samples)
        objectives = np.array(objectives)

        # Calculate parameter importance (variance-based)
        parameter_importance = {}
        for i, param_name in enumerate(space.parameter_names):
            # Calculate correlation with objective
            param_values = samples[:, i]
            correlation = np.corrcoef(param_values, objectives)[0, 1]
            parameter_importance[param_name] = abs(correlation)

        # Calculate parameter correlations
        parameter_correlations = {}
        for i, param1 in enumerate(space.parameter_names):
            parameter_correlations[param1] = {}
            for j, param2 in enumerate(space.parameter_names):
                if i != j:
                    corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
                    parameter_correlations[param1][param2] = corr

        # Sensitivity analysis (partial derivatives approximation)
        sensitivity_analysis = {}
        for i, param_name in enumerate(space.parameter_names):
            # Calculate finite differences
            sensitivities = []
            for j in range(min(100, len(samples) - 1)):
                if j + 1 < len(samples):
                    param_diff = samples[j + 1, i] - samples[j, i]
                    obj_diff = objectives[j + 1] - objectives[j]
                    if abs(param_diff) > 1e-8:
                        sensitivity = obj_diff / param_diff
                        sensitivities.append(sensitivity)

            sensitivity_analysis[param_name] = sensitivities

        # Interaction effects (simplified)
        interaction_effects = {}
        for i, param1 in enumerate(space.parameter_names):
            for j, param2 in enumerate(space.parameter_names):
                if i < j:  # Avoid duplicates
                    # Calculate interaction as correlation of product with objective
                    interaction_term = samples[:, i] * samples[:, j]
                    interaction_corr = np.corrcoef(interaction_term, objectives)[0, 1]
                    interaction_effects[(param1, param2)] = abs(interaction_corr)

        logger.info(f"Importance analysis completed for {component_name}")

        return ImportanceAnalysis(
            parameter_importance=parameter_importance,
            parameter_correlations=parameter_correlations,
            sensitivity_analysis=sensitivity_analysis,
            interaction_effects=interaction_effects,
        )

    def visualize_optimization_results(
        self, component_name: Optional[str] = None, save_plots: bool = True
    ):
        """
        Create visualization of optimization results.

        Args:
            component_name: Specific component to visualize (None for all)
            save_plots: Whether to save plots to disk
        """
        if not self.history:
            logger.warning("No optimization history available for visualization")
            return

        # Extract data for plotting
        iterations = [h.iteration for h in self.history]
        objective_values = [h.objective_value for h in self.history]

        # Create convergence plot
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(iterations, objective_values, "b-", alpha=0.7, label="Objective Value")

        # Add best value line
        best_values = []
        current_best = -np.inf
        for obj_val in objective_values:
            if obj_val > current_best:
                current_best = obj_val
            best_values.append(current_best)

        plt.plot(iterations, best_values, "r-", linewidth=2, label="Best Value")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title("Optimization Convergence")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Component scores over time
        if component_name and len(self.history) > 0:
            component_scores = [
                h.component_scores.get(component_name, 0) for h in self.history
            ]

            plt.subplot(2, 2, 2)
            plt.plot(iterations, component_scores, "g-", alpha=0.7)
            plt.xlabel("Iteration")
            plt.ylabel("Component Score")
            plt.title(f"{component_name} Score Over Time")
            plt.grid(True, alpha=0.3)

        # Evaluation time distribution
        eval_times = [h.evaluation_time for h in self.history]

        plt.subplot(2, 2, 3)
        plt.hist(eval_times, bins=20, alpha=0.7, color="orange")
        plt.xlabel("Evaluation Time (s)")
        plt.ylabel("Frequency")
        plt.title("Evaluation Time Distribution")
        plt.grid(True, alpha=0.3)

        # Parameter importance (if available)
        if component_name and component_name in self.components:
            try:
                importance = self.analyze_hyperparameter_importance(
                    component_name, n_samples=100
                )

                plt.subplot(2, 2, 4)
                params = list(importance.parameter_importance.keys())
                importances = list(importance.parameter_importance.values())

                plt.barh(params, importances, alpha=0.7, color="purple")
                plt.xlabel("Importance")
                plt.title(f"{component_name} Parameter Importance")
                plt.grid(True, alpha=0.3)
            except Exception as e:
                logger.warning(f"Could not create importance plot: {e}")

        plt.tight_layout()

        if save_plots and self.config.save_results:
            plot_path = (
                Path(self.config.results_dir)
                / f"optimization_results_{component_name or 'all'}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved optimization plots to {plot_path}")

        plt.show()

    def _save_component_results(
        self,
        component_name: str,
        result: OptimizationResult,
        best_params: Dict[str, Any],
    ):
        """Save optimization results for a single component."""
        results_data = {
            "component_name": component_name,
            "best_parameters": best_params,
            "best_value": float(result.best_value),
            "n_iterations": result.n_iterations,
            "convergence_history": [float(x) for x in result.convergence_history],
            "config": {
                "max_iterations": self.config.max_iterations,
                "acquisition_function": self.config.acquisition_function,
                "early_stopping_patience": self.config.early_stopping_patience,
            },
        }

        results_path = (
            Path(self.config.results_dir) / f"{component_name}_optimization.json"
        )
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved results for {component_name} to {results_path}")

    def _save_joint_results(
        self,
        component_results: Dict[str, OptimizationResult],
        joint_result: OptimizationResult,
    ):
        """Save joint optimization results."""
        results_data = {
            "joint_optimization": {
                "best_value": float(joint_result.best_value),
                "n_iterations": joint_result.n_iterations,
                "convergence_history": [
                    float(x) for x in joint_result.convergence_history
                ],
            },
            "component_results": {
                name: {
                    "best_value": float(result.best_value),
                    "n_iterations": result.n_iterations,
                }
                for name, result in component_results.items()
            },
            "optimization_history": [
                {
                    "iteration": h.iteration,
                    "objective_value": h.objective_value,
                    "component_scores": h.component_scores,
                    "evaluation_time": h.evaluation_time,
                }
                for h in self.history
            ],
            "config": {
                "max_iterations": self.config.max_iterations,
                "acquisition_function": self.config.acquisition_function,
                "n_parallel_jobs": self.config.n_parallel_jobs,
            },
        }

        results_path = Path(self.config.results_dir) / "joint_optimization.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Saved joint optimization results to {results_path}")

    def get_best_parameters(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get best parameters for a component from history."""
        if not self.history:
            return None

        # Find best iteration
        best_iteration = max(self.history, key=lambda h: h.objective_value)

        if component_name in best_iteration.parameters:
            return best_iteration.parameters[component_name]

        return None

    def reset(self):
        """Reset optimization state."""
        self.history.clear()
        for optimizer in self.optimizers.values():
            optimizer.X_observed.clear()
            optimizer.y_observed.clear()

        logger.info("Reset optimization framework state")

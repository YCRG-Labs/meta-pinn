"""
Robust Symbolic Regression Orchestrator

This module integrates multiple symbolic regression methods, implements expression
ensemble and voting mechanisms, and provides automatic hyperparameter optimization.
"""

import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from .expression_validation import (
    DimensionalAnalyzer,
    DimensionalVector,
    ExpressionValidator,
)
from .neural_symbolic_regression import NeuralSymbolicRegression as TransformerNSR

# Import our symbolic regression components
from .symbolic_regression import NeuralSymbolicRegression, SymbolicExpression


class SymbolicRegressionMethod(Enum):
    """Enumeration of available symbolic regression methods."""

    GENETIC_PROGRAMMING = "genetic_programming"
    NEURAL_SYMBOLIC = "neural_symbolic"
    TRANSFORMER_BASED = "transformer_based"
    HYBRID = "hybrid"


@dataclass
class MethodConfig:
    """Configuration for a symbolic regression method."""

    method: SymbolicRegressionMethod
    enabled: bool = True
    weight: float = 1.0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    validation_threshold: float = 0.8


@dataclass
class EnsembleResult:
    """Result from ensemble symbolic regression."""

    best_expression: sp.Expr
    confidence_score: float
    method_contributions: Dict[str, float]
    individual_results: List[Tuple[sp.Expr, float, str]]
    ensemble_fitness: float
    validation_metrics: Dict[str, Any]
    hyperparameter_history: List[Dict[str, Any]]


class ExpressionEnsemble:
    """Manages ensemble of symbolic expressions with voting mechanisms."""

    def __init__(
        self,
        voting_strategy: str = "weighted_average",
        confidence_threshold: float = 0.7,
        diversity_weight: float = 0.2,
    ):
        """
        Initialize expression ensemble.

        Args:
            voting_strategy: Strategy for combining expressions ('weighted_average', 'majority_vote', 'best_performer')
            confidence_threshold: Minimum confidence for accepting ensemble result
            diversity_weight: Weight for promoting diversity in ensemble
        """
        self.voting_strategy = voting_strategy
        self.confidence_threshold = confidence_threshold
        self.diversity_weight = diversity_weight

        self.expressions = []
        self.fitnesses = []
        self.methods = []
        self.weights = []

    def add_expression(
        self, expression: sp.Expr, fitness: float, method: str, weight: float = 1.0
    ):
        """Add an expression to the ensemble."""
        self.expressions.append(expression)
        self.fitnesses.append(fitness)
        self.methods.append(method)
        self.weights.append(weight)

    def compute_ensemble_prediction(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute ensemble prediction using voting strategy.

        Args:
            data: Input data dictionary

        Returns:
            Ensemble prediction array
        """
        if not self.expressions:
            raise ValueError("No expressions in ensemble")

        predictions = []
        valid_weights = []

        # Evaluate each expression
        for expr, fitness, method, weight in zip(
            self.expressions, self.fitnesses, self.methods, self.weights
        ):
            try:
                # Convert to numerical function
                variables = list(expr.free_symbols)
                expr_func = sp.lambdify(variables, expr, "numpy")

                # Prepare input data
                input_values = [data[str(var)] for var in variables if str(var) in data]

                if input_values:
                    pred = expr_func(*input_values)

                    # Handle scalar predictions
                    if np.isscalar(pred):
                        pred = np.full_like(list(data.values())[0], pred)

                    if np.isfinite(pred).all():
                        predictions.append(pred)
                        valid_weights.append(weight * fitness)

            except Exception:
                continue  # Skip invalid expressions

        if not predictions:
            # Fallback to zero prediction
            return np.zeros_like(list(data.values())[0])

        predictions = np.array(predictions)
        valid_weights = np.array(valid_weights)

        # Apply voting strategy
        if self.voting_strategy == "weighted_average":
            # Weighted average of predictions
            if valid_weights.sum() > 0:
                weights_norm = valid_weights / valid_weights.sum()
                ensemble_pred = np.average(predictions, axis=0, weights=weights_norm)
            else:
                ensemble_pred = np.mean(predictions, axis=0)

        elif self.voting_strategy == "majority_vote":
            # Simple majority vote (median)
            ensemble_pred = np.median(predictions, axis=0)

        elif self.voting_strategy == "best_performer":
            # Use prediction from best performing method
            best_idx = np.argmax(valid_weights)
            ensemble_pred = predictions[best_idx]

        else:
            # Default to simple average
            ensemble_pred = np.mean(predictions, axis=0)

        return ensemble_pred

    def compute_diversity_score(self) -> float:
        """Compute diversity score of the ensemble."""
        if len(self.expressions) < 2:
            return 0.0

        # Compute pairwise expression differences
        diversity_scores = []

        for i in range(len(self.expressions)):
            for j in range(i + 1, len(self.expressions)):
                expr1_str = str(self.expressions[i])
                expr2_str = str(self.expressions[j])

                # Simple string-based diversity measure
                # In practice, could use more sophisticated measures
                max_len = max(len(expr1_str), len(expr2_str))
                if max_len > 0:
                    # Normalized edit distance approximation
                    common_chars = sum(
                        1 for c1, c2 in zip(expr1_str, expr2_str) if c1 == c2
                    )
                    diversity = 1.0 - (common_chars / max_len)
                    diversity_scores.append(diversity)

        return np.mean(diversity_scores) if diversity_scores else 0.0

    def get_best_expression(self) -> Tuple[sp.Expr, float, str]:
        """Get the best expression from the ensemble."""
        if not self.expressions:
            raise ValueError("No expressions in ensemble")

        best_idx = np.argmax(self.fitnesses)
        return (
            self.expressions[best_idx],
            self.fitnesses[best_idx],
            self.methods[best_idx],
        )

    def compute_confidence_score(self) -> float:
        """Compute confidence score for the ensemble."""
        if not self.fitnesses:
            return 0.0

        # Base confidence on fitness statistics
        mean_fitness = np.mean(self.fitnesses)
        std_fitness = np.std(self.fitnesses)
        max_fitness = np.max(self.fitnesses)

        # High confidence if:
        # 1. High maximum fitness
        # 2. Low variance in fitness (consensus)
        # 3. Good diversity

        fitness_confidence = max_fitness
        consensus_confidence = 1.0 / (1.0 + std_fitness) if std_fitness > 0 else 1.0
        diversity_confidence = self.compute_diversity_score()

        # Weighted combination
        confidence = (
            0.5 * fitness_confidence
            + 0.3 * consensus_confidence
            + 0.2 * diversity_confidence
        )

        return min(1.0, max(0.0, confidence))


class HyperparameterOptimizer:
    """Optimizes hyperparameters for symbolic regression methods."""

    def __init__(
        self,
        optimization_method: str = "optuna",
        n_trials: int = 50,
        timeout: Optional[float] = None,
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            optimization_method: Optimization method ('optuna', 'grid_search', 'random_search')
            n_trials: Number of optimization trials
            timeout: Timeout for optimization in seconds
        """
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.timeout = timeout

        self.optimization_history = []

    def optimize_genetic_programming(
        self,
        data: Dict[str, np.ndarray],
        target: np.ndarray,
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for genetic programming.

        Args:
            data: Training data
            target: Target values
            base_config: Base configuration

        Returns:
            Optimized hyperparameters
        """

        def objective(trial):
            # Define hyperparameter search space
            config = base_config.copy()
            config.update(
                {
                    "population_size": trial.suggest_int("population_size", 50, 200),
                    "max_generations": trial.suggest_int("max_generations", 20, 100),
                    "mutation_rate": trial.suggest_float("mutation_rate", 0.05, 0.3),
                    "crossover_rate": trial.suggest_float("crossover_rate", 0.5, 0.9),
                    "complexity_penalty": trial.suggest_float(
                        "complexity_penalty", 0.001, 0.1
                    ),
                    "max_expression_depth": trial.suggest_int(
                        "max_expression_depth", 3, 8
                    ),
                }
            )

            try:
                # Create and train model
                variables = list(data.keys())
                nsr = NeuralSymbolicRegression(
                    variables=variables,
                    population_size=config["population_size"],
                    max_generations=config["max_generations"],
                    mutation_rate=config["mutation_rate"],
                    crossover_rate=config["crossover_rate"],
                    complexity_penalty=config["complexity_penalty"],
                    max_expression_depth=config["max_expression_depth"],
                )

                # Discover expression
                result = nsr.discover_viscosity_law(data, target)

                # Return fitness (to maximize)
                return result.r2_score

            except Exception:
                return -1.0  # Poor fitness for failed trials

        if self.optimization_method == "optuna" and OPTUNA_AVAILABLE:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

            best_params = study.best_params
            best_params.update(base_config)

            # Store optimization history
            self.optimization_history.append(
                {
                    "method": "genetic_programming",
                    "best_params": best_params,
                    "best_value": study.best_value,
                    "n_trials": len(study.trials),
                }
            )

            return best_params

        else:
            # Fallback to base config
            return base_config

    def optimize_neural_symbolic(
        self,
        data: Dict[str, np.ndarray],
        target: np.ndarray,
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for neural symbolic regression.

        Args:
            data: Training data
            target: Target values
            base_config: Base configuration

        Returns:
            Optimized hyperparameters
        """

        def objective(trial):
            config = base_config.copy()
            config.update(
                {
                    "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
                    "nhead": trial.suggest_categorical("nhead", [4, 8, 16]),
                    "num_layers": trial.suggest_int("num_layers", 2, 8),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-2, log=True
                    ),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                    "num_epochs": trial.suggest_int("num_epochs", 10, 100),
                }
            )

            try:
                # Create and train model
                variables = list(data.keys())
                nsr = TransformerNSR(
                    variables=variables,
                    d_model=config["d_model"],
                    nhead=config["nhead"],
                    num_layers=config["num_layers"],
                    learning_rate=config["learning_rate"],
                    batch_size=config["batch_size"],
                    num_epochs=config["num_epochs"],
                )

                # Generate training data and train
                expressions, data_points, targets = nsr.generate_training_data(
                    num_expressions=50, num_data_points=100
                )

                if expressions:
                    nsr.train(expressions, data_points, targets)

                    # Test discovery
                    candidates = nsr.discover_expression(data, target, num_candidates=5)

                    if candidates:
                        return candidates[0][1]  # Best fitness

                return -1.0

            except Exception:
                return -1.0

        if self.optimization_method == "optuna" and OPTUNA_AVAILABLE:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective, n_trials=min(self.n_trials, 20), timeout=self.timeout
            )

            best_params = study.best_params
            best_params.update(base_config)

            self.optimization_history.append(
                {
                    "method": "neural_symbolic",
                    "best_params": best_params,
                    "best_value": study.best_value,
                    "n_trials": len(study.trials),
                }
            )

            return best_params

        else:
            return base_config


class RobustSymbolicRegression:
    """
    Robust symbolic regression orchestrator that integrates multiple methods.

    This class implements Requirements 1.1 and 1.3 by providing a unified interface
    for multiple symbolic regression approaches with ensemble learning and
    automatic hyperparameter optimization.
    """

    def __init__(
        self,
        variables: List[str],
        method_configs: Optional[Dict[str, MethodConfig]] = None,
        ensemble_strategy: str = "weighted_average",
        enable_hyperparameter_optimization: bool = True,
        optimization_budget: int = 100,
        parallel_execution: bool = True,
        validation_split: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize robust symbolic regression orchestrator.

        Args:
            variables: List of variable names
            method_configs: Configuration for each method
            ensemble_strategy: Strategy for ensemble learning
            enable_hyperparameter_optimization: Whether to optimize hyperparameters
            optimization_budget: Budget for hyperparameter optimization
            parallel_execution: Whether to run methods in parallel
            validation_split: Fraction of data for validation
            random_state: Random seed
        """
        self.variables = variables
        self.ensemble_strategy = ensemble_strategy
        self.enable_hyperparameter_optimization = enable_hyperparameter_optimization
        self.optimization_budget = optimization_budget
        self.parallel_execution = parallel_execution
        self.validation_split = validation_split
        self.random_state = random_state

        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)

        # Initialize method configurations
        if method_configs is None:
            self.method_configs = self._get_default_method_configs()
        else:
            self.method_configs = method_configs

        # Initialize components
        self.ensemble = ExpressionEnsemble(voting_strategy=ensemble_strategy)
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            n_trials=optimization_budget // len(self.method_configs)
        )
        self.expression_validator = ExpressionValidator()

        # Results tracking
        self.discovery_history = []
        self.performance_metrics = {}

    def _get_default_method_configs(self) -> Dict[str, MethodConfig]:
        """Get default method configurations."""
        return {
            "genetic_programming": MethodConfig(
                method=SymbolicRegressionMethod.GENETIC_PROGRAMMING,
                weight=1.0,
                hyperparameters={
                    "population_size": 100,
                    "max_generations": 50,
                    "mutation_rate": 0.1,
                    "crossover_rate": 0.7,
                    "complexity_penalty": 0.01,
                    "max_expression_depth": 5,
                },
            ),
            "neural_symbolic": MethodConfig(
                method=SymbolicRegressionMethod.NEURAL_SYMBOLIC,
                weight=0.8,
                hyperparameters={
                    "d_model": 128,
                    "nhead": 8,
                    "num_layers": 4,
                    "learning_rate": 1e-4,
                    "batch_size": 32,
                    "num_epochs": 50,
                },
            ),
        }

    def discover_physics_law(
        self,
        data: Dict[str, np.ndarray],
        target: np.ndarray,
        target_dimension: Optional[DimensionalVector] = None,
        variable_dimensions: Optional[Dict[str, DimensionalVector]] = None,
        max_time: Optional[float] = None,
    ) -> EnsembleResult:
        """
        Discover physics law using ensemble of symbolic regression methods.

        Args:
            data: Input data dictionary
            target: Target values
            target_dimension: Expected dimension of target
            variable_dimensions: Dimensions of input variables
            max_time: Maximum time for discovery in seconds

        Returns:
            Ensemble result with best discovered expression
        """
        start_time = time.time()

        # Split data for validation
        train_data, val_data, train_target, val_target = self._split_data(
            data, target, self.validation_split
        )

        # Initialize ensemble
        self.ensemble = ExpressionEnsemble(voting_strategy=self.ensemble_strategy)

        # Run symbolic regression methods
        method_results = {}

        if self.parallel_execution:
            method_results = self._run_methods_parallel(
                train_data, train_target, max_time
            )
        else:
            method_results = self._run_methods_sequential(
                train_data, train_target, max_time
            )

        # Add results to ensemble
        for method_name, (expr, fitness, config) in method_results.items():
            if expr is not None:
                self.ensemble.add_expression(
                    expr, fitness, method_name, self.method_configs[method_name].weight
                )

        # Validate expressions if dimensions provided
        if target_dimension and variable_dimensions:
            self._validate_ensemble_expressions(target_dimension, variable_dimensions)

        # Compute ensemble metrics
        ensemble_fitness = self._evaluate_ensemble_fitness(val_data, val_target)
        confidence_score = self.ensemble.compute_confidence_score()

        # Get best expression
        if self.ensemble.expressions:
            best_expr, best_fitness, best_method = self.ensemble.get_best_expression()
        else:
            # Fallback expression
            best_expr = (
                sp.Symbol(self.variables[0]) if self.variables else sp.Symbol("x")
            )
            best_fitness = 0.0
            best_method = "fallback"

        # Compute method contributions
        method_contributions = self._compute_method_contributions()

        # Validation metrics
        validation_metrics = self._compute_validation_metrics(
            val_data, val_target, target_dimension, variable_dimensions
        )

        # Create result
        result = EnsembleResult(
            best_expression=best_expr,
            confidence_score=confidence_score,
            method_contributions=method_contributions,
            individual_results=[
                (expr, fit, method) for method, (expr, fit, _) in method_results.items()
            ],
            ensemble_fitness=ensemble_fitness,
            validation_metrics=validation_metrics,
            hyperparameter_history=self.hyperparameter_optimizer.optimization_history,
        )

        # Store in history
        self.discovery_history.append(
            {
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "result": result,
                "data_shape": {var: arr.shape for var, arr in data.items()},
                "target_shape": target.shape,
            }
        )

        return result

    def _split_data(
        self, data: Dict[str, np.ndarray], target: np.ndarray, validation_split: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        n_samples = len(target)
        n_train = int(n_samples * (1 - validation_split))

        # Random indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Split data
        train_data = {var: arr[train_indices] for var, arr in data.items()}
        val_data = {var: arr[val_indices] for var, arr in data.items()}
        train_target = target[train_indices]
        val_target = target[val_indices]

        return train_data, val_data, train_target, val_target

    def _run_methods_parallel(
        self, data: Dict[str, np.ndarray], target: np.ndarray, max_time: Optional[float]
    ) -> Dict[str, Tuple[sp.Expr, float, Dict]]:
        """Run symbolic regression methods in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=len(self.method_configs)) as executor:
            # Submit tasks
            future_to_method = {}
            for method_name, config in self.method_configs.items():
                if config.enabled:
                    future = executor.submit(
                        self._run_single_method, method_name, config, data, target
                    )
                    future_to_method[future] = method_name

            # Collect results
            for future in as_completed(future_to_method, timeout=max_time):
                method_name = future_to_method[future]
                try:
                    result = future.result()
                    results[method_name] = result
                except Exception as e:
                    warnings.warn(f"Method {method_name} failed: {e}")
                    results[method_name] = (None, 0.0, {})

        return results

    def _run_methods_sequential(
        self, data: Dict[str, np.ndarray], target: np.ndarray, max_time: Optional[float]
    ) -> Dict[str, Tuple[sp.Expr, float, Dict]]:
        """Run symbolic regression methods sequentially."""
        results = {}
        start_time = time.time()

        for method_name, config in self.method_configs.items():
            if config.enabled:
                # Check time limit
                if max_time and (time.time() - start_time) > max_time:
                    break

                try:
                    result = self._run_single_method(method_name, config, data, target)
                    results[method_name] = result
                except Exception as e:
                    warnings.warn(f"Method {method_name} failed: {e}")
                    results[method_name] = (None, 0.0, {})

        return results

    def _run_single_method(
        self,
        method_name: str,
        config: MethodConfig,
        data: Dict[str, np.ndarray],
        target: np.ndarray,
    ) -> Tuple[sp.Expr, float, Dict]:
        """Run a single symbolic regression method."""
        # Optimize hyperparameters if enabled
        if self.enable_hyperparameter_optimization:
            if config.method == SymbolicRegressionMethod.GENETIC_PROGRAMMING:
                optimized_params = (
                    self.hyperparameter_optimizer.optimize_genetic_programming(
                        data, target, config.hyperparameters
                    )
                )
            elif config.method == SymbolicRegressionMethod.NEURAL_SYMBOLIC:
                optimized_params = (
                    self.hyperparameter_optimizer.optimize_neural_symbolic(
                        data, target, config.hyperparameters
                    )
                )
            else:
                optimized_params = config.hyperparameters
        else:
            optimized_params = config.hyperparameters

        # Run the method
        if config.method == SymbolicRegressionMethod.GENETIC_PROGRAMMING:
            return self._run_genetic_programming(data, target, optimized_params)
        elif config.method == SymbolicRegressionMethod.NEURAL_SYMBOLIC:
            return self._run_neural_symbolic(data, target, optimized_params)
        else:
            # Fallback
            return sp.Symbol("x"), 0.0, optimized_params

    def _run_genetic_programming(
        self, data: Dict[str, np.ndarray], target: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[sp.Expr, float, Dict]:
        """Run genetic programming method."""
        nsr = NeuralSymbolicRegression(variables=self.variables, **params)

        result = nsr.discover_viscosity_law(data, target)
        return result.expression, result.r2_score, params

    def _run_neural_symbolic(
        self, data: Dict[str, np.ndarray], target: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[sp.Expr, float, Dict]:
        """Run neural symbolic regression method."""
        nsr = TransformerNSR(variables=self.variables, **params)

        # Generate training data and train
        expressions, data_points, targets = nsr.generate_training_data(
            num_expressions=100, num_data_points=len(target)
        )

        if expressions:
            nsr.train(expressions, data_points, targets)
            candidates = nsr.discover_expression(data, target, num_candidates=1)

            if candidates:
                expr, fitness = candidates[0]
                return expr, fitness, params

        # Fallback
        return sp.Symbol(self.variables[0]), 0.0, params

    def _validate_ensemble_expressions(
        self,
        target_dimension: DimensionalVector,
        variable_dimensions: Dict[str, DimensionalVector],
    ):
        """Validate expressions in ensemble using dimensional analysis."""
        validated_expressions = []
        validated_fitnesses = []
        validated_methods = []
        validated_weights = []

        for expr, fitness, method, weight in zip(
            self.ensemble.expressions,
            self.ensemble.fitnesses,
            self.ensemble.methods,
            self.ensemble.weights,
        ):
            # Validate expression
            validation_result = self.expression_validator.validate_expression(
                expr, target_dimension, variable_dimensions, self.variables
            )

            # Keep expressions that pass validation or have high fitness
            if validation_result["overall_validation_score"] > 0.5 or fitness > 0.8:
                validated_expressions.append(expr)
                validated_fitnesses.append(fitness)
                validated_methods.append(method)
                validated_weights.append(weight)

        # Update ensemble
        self.ensemble.expressions = validated_expressions
        self.ensemble.fitnesses = validated_fitnesses
        self.ensemble.methods = validated_methods
        self.ensemble.weights = validated_weights

    def _evaluate_ensemble_fitness(
        self, data: Dict[str, np.ndarray], target: np.ndarray
    ) -> float:
        """Evaluate fitness of the ensemble."""
        try:
            ensemble_pred = self.ensemble.compute_ensemble_prediction(data)
            r2 = r2_score(target, ensemble_pred)
            return max(0.0, r2)
        except Exception:
            return 0.0

    def _compute_method_contributions(self) -> Dict[str, float]:
        """Compute contribution of each method to the ensemble."""
        contributions = {}

        for method, fitness, weight in zip(
            self.ensemble.methods, self.ensemble.fitnesses, self.ensemble.weights
        ):
            contribution = fitness * weight
            if method in contributions:
                contributions[method] += contribution
            else:
                contributions[method] = contribution

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def _compute_validation_metrics(
        self,
        data: Dict[str, np.ndarray],
        target: np.ndarray,
        target_dimension: Optional[DimensionalVector],
        variable_dimensions: Optional[Dict[str, DimensionalVector]],
    ) -> Dict[str, Any]:
        """Compute comprehensive validation metrics."""
        metrics = {}

        # Ensemble performance
        try:
            ensemble_pred = self.ensemble.compute_ensemble_prediction(data)
            metrics["ensemble_r2"] = r2_score(target, ensemble_pred)
            metrics["ensemble_mse"] = mean_squared_error(target, ensemble_pred)
        except Exception:
            metrics["ensemble_r2"] = 0.0
            metrics["ensemble_mse"] = np.inf

        # Individual method performance
        method_metrics = {}
        for expr, fitness, method in zip(
            self.ensemble.expressions, self.ensemble.fitnesses, self.ensemble.methods
        ):
            if method not in method_metrics:
                method_metrics[method] = []
            method_metrics[method].append(fitness)

        # Aggregate method metrics
        for method, fitnesses in method_metrics.items():
            metrics[f"{method}_mean_fitness"] = np.mean(fitnesses)
            metrics[f"{method}_std_fitness"] = np.std(fitnesses)

        # Dimensional validation if available
        if target_dimension and variable_dimensions:
            dimensional_scores = []
            for expr in self.ensemble.expressions:
                validation = self.expression_validator.validate_expression(
                    expr, target_dimension, variable_dimensions, self.variables
                )
                dimensional_scores.append(validation["overall_validation_score"])

            if dimensional_scores:
                metrics["mean_dimensional_score"] = np.mean(dimensional_scores)
                metrics["std_dimensional_score"] = np.std(dimensional_scores)

        # Ensemble diversity
        metrics["ensemble_diversity"] = self.ensemble.compute_diversity_score()
        metrics["ensemble_confidence"] = self.ensemble.compute_confidence_score()

        return metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.discovery_history:
            return {}

        # Aggregate statistics across all discoveries
        durations = [entry["duration"] for entry in self.discovery_history]
        confidence_scores = [
            entry["result"].confidence_score for entry in self.discovery_history
        ]
        ensemble_fitnesses = [
            entry["result"].ensemble_fitness for entry in self.discovery_history
        ]

        summary = {
            "total_discoveries": len(self.discovery_history),
            "mean_duration": np.mean(durations),
            "std_duration": np.std(durations),
            "mean_confidence": np.mean(confidence_scores),
            "std_confidence": np.std(confidence_scores),
            "mean_ensemble_fitness": np.mean(ensemble_fitnesses),
            "std_ensemble_fitness": np.std(ensemble_fitnesses),
            "hyperparameter_optimization_history": self.hyperparameter_optimizer.optimization_history,
        }

        # Method-specific statistics
        all_contributions = {}
        for entry in self.discovery_history:
            for method, contribution in entry["result"].method_contributions.items():
                if method not in all_contributions:
                    all_contributions[method] = []
                all_contributions[method].append(contribution)

        for method, contributions in all_contributions.items():
            summary[f"{method}_mean_contribution"] = np.mean(contributions)
            summary[f"{method}_std_contribution"] = np.std(contributions)

        return summary

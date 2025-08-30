"""
Bayesian Optimizer for hyperparameter optimization.

This module implements Gaussian process-based Bayesian optimization with multiple
acquisition functions and multi-objective optimization capabilities.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

logger = logging.getLogger(__name__)


@dataclass
class OptimizationBounds:
    """Bounds for optimization parameters."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        """Validate bounds."""
        if len(self.lower) != len(self.upper):
            raise ValueError("Lower and upper bounds must have same length")
        if np.any(self.lower >= self.upper):
            raise ValueError("Lower bounds must be less than upper bounds")


@dataclass
class OptimizationResult:
    """Result from Bayesian optimization."""

    best_params: np.ndarray
    best_value: float
    best_values: List[float]
    all_params: List[np.ndarray]
    all_values: List[float]
    n_iterations: int
    convergence_history: List[float]
    pareto_front: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None


class AcquisitionFunction:
    """Base class for acquisition functions."""

    def __init__(self, gp: GaussianProcessRegressor, xi: float = 0.01):
        self.gp = gp
        self.xi = xi

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate acquisition function."""
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Calculate expected improvement."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = self.gp.predict(X, return_std=True)

        # Get current best value
        y_best = np.max(self.gp.y_train_)

        # Calculate expected improvement
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        z = (mu - y_best - self.xi) / sigma
        ei = (mu - y_best - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)

        return ei.flatten()


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""

    def __init__(self, gp: GaussianProcessRegressor, kappa: float = 2.576):
        super().__init__(gp)
        self.kappa = kappa

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Calculate upper confidence bound."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = self.gp.predict(X, return_std=True)

        ucb = mu + self.kappa * sigma
        return ucb.flatten()


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function."""

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Calculate probability of improvement."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = self.gp.predict(X, return_std=True)

        # Get current best value
        y_best = np.max(self.gp.y_train_)

        # Calculate probability of improvement
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        z = (mu - y_best - self.xi) / sigma
        pi = norm.cdf(z)

        return pi.flatten()


class BayesianOptimizer:
    """
    Gaussian process-based Bayesian optimizer with multiple acquisition functions
    and multi-objective optimization capabilities.
    """

    def __init__(
        self,
        bounds: OptimizationBounds,
        acquisition_function: str = "ei",
        kernel: Optional[Any] = None,
        n_restarts: int = 10,
        random_state: Optional[int] = None,
        normalize_y: bool = True,
        alpha: float = 1e-6,
    ):
        """
        Initialize Bayesian optimizer.

        Args:
            bounds: Parameter bounds for optimization
            acquisition_function: Type of acquisition function ('ei', 'ucb', 'pi')
            kernel: GP kernel (default: Matern with nu=2.5)
            n_restarts: Number of restarts for acquisition optimization
            random_state: Random seed for reproducibility
            normalize_y: Whether to normalize target values
            alpha: Noise level in GP
        """
        self.bounds = bounds
        self.n_dims = len(bounds.lower)
        self.acquisition_type = acquisition_function
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.normalize_y = normalize_y
        self.alpha = alpha

        # Set up random number generator
        self.rng = np.random.RandomState(random_state)

        # Initialize kernel
        if kernel is None:
            self.kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(
                noise_level=alpha
            )
        else:
            self.kernel = kernel

        # Initialize GP
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=normalize_y,
            random_state=random_state,
            alpha=alpha,
        )

        # Storage for optimization history
        self.X_observed = []
        self.y_observed = []
        self.acquisition_func = None

    def _create_acquisition_function(self) -> AcquisitionFunction:
        """Create acquisition function based on type."""
        if self.acquisition_type == "ei":
            return ExpectedImprovement(self.gp)
        elif self.acquisition_type == "ucb":
            return UpperConfidenceBound(self.gp)
        elif self.acquisition_type == "pi":
            return ProbabilityOfImprovement(self.gp)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_type}")

    def _optimize_acquisition(self) -> np.ndarray:
        """Optimize acquisition function to find next point."""

        def objective(x):
            return -self.acquisition_func(x.reshape(1, -1))[0]

        # Multiple random restarts
        best_x = None
        best_val = np.inf

        for _ in range(self.n_restarts):
            # Random starting point
            x0 = self.rng.uniform(self.bounds.lower, self.bounds.upper)

            # Optimize
            result = minimize(
                objective,
                x0,
                bounds=list(zip(self.bounds.lower, self.bounds.upper)),
                method="L-BFGS-B",
            )

            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x

        return best_x

    def suggest(self, n_points: int = 1) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Suggest next point(s) to evaluate.

        Args:
            n_points: Number of points to suggest

        Returns:
            Next point(s) to evaluate
        """
        if len(self.X_observed) == 0:
            # Random initialization
            if n_points == 1:
                return self.rng.uniform(self.bounds.lower, self.bounds.upper)
            else:
                return [
                    self.rng.uniform(self.bounds.lower, self.bounds.upper)
                    for _ in range(n_points)
                ]

        # Fit GP to observed data
        X_array = np.array(self.X_observed)
        y_array = np.array(self.y_observed)
        self.gp.fit(X_array, y_array)

        # Create acquisition function
        self.acquisition_func = self._create_acquisition_function()

        # Suggest points
        if n_points == 1:
            return self._optimize_acquisition()
        else:
            points = []
            for _ in range(n_points):
                point = self._optimize_acquisition()
                points.append(point)
                # Add temporary observation to avoid suggesting same point
                self.X_observed.append(point)
                self.y_observed.append(np.mean(self.y_observed))

            # Remove temporary observations
            for _ in range(n_points):
                self.X_observed.pop()
                self.y_observed.pop()

            return points

    def tell(
        self, X: Union[np.ndarray, List[np.ndarray]], y: Union[float, List[float]]
    ) -> None:
        """
        Update optimizer with new observations.

        Args:
            X: Parameter values
            y: Objective values
        """
        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = [X]
        if isinstance(y, (int, float)):
            y = [y]

        for x_i, y_i in zip(X, y):
            self.X_observed.append(np.array(x_i))
            self.y_observed.append(float(y_i))

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        n_iterations: int = 50,
        n_initial: int = 5,
        convergence_threshold: float = 1e-6,
        patience: int = 10,
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.

        Args:
            objective_func: Function to optimize
            n_iterations: Maximum number of iterations
            n_initial: Number of initial random evaluations
            convergence_threshold: Convergence threshold for early stopping
            patience: Number of iterations without improvement before stopping

        Returns:
            Optimization result
        """
        logger.info(f"Starting Bayesian optimization with {n_iterations} iterations")

        # Clear previous observations
        self.X_observed = []
        self.y_observed = []

        all_params = []
        all_values = []
        convergence_history = []

        # Initial random evaluations
        for i in range(n_initial):
            x = self.rng.uniform(self.bounds.lower, self.bounds.upper)
            y = objective_func(x)

            self.tell(x, y)
            all_params.append(x.copy())
            all_values.append(y)

            logger.debug(f"Initial evaluation {i+1}/{n_initial}: {y:.6f}")

        best_value = max(all_values)
        no_improvement_count = 0

        # Initialize convergence history with initial evaluations
        for i in range(n_initial):
            current_best = max(all_values[: i + 1])
            convergence_history.append(current_best)

        # Bayesian optimization iterations
        for iteration in range(n_iterations - n_initial):
            # Suggest next point
            x_next = self.suggest()
            y_next = objective_func(x_next)

            # Update optimizer
            self.tell(x_next, y_next)
            all_params.append(x_next.copy())
            all_values.append(y_next)

            # Check for improvement
            current_best = max(all_values)
            if current_best > best_value + convergence_threshold:
                best_value = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            convergence_history.append(current_best)

            logger.debug(
                f"Iteration {iteration+1}: {y_next:.6f} (best: {current_best:.6f})"
            )

            # Early stopping
            if no_improvement_count >= patience:
                logger.info(f"Early stopping after {iteration+1} iterations")
                break

        # Find best result
        best_idx = np.argmax(all_values)
        best_params = all_params[best_idx]
        best_value = all_values[best_idx]

        logger.info(f"Optimization completed. Best value: {best_value:.6f}")

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            best_values=[max(all_values[: i + 1]) for i in range(len(all_values))],
            all_params=all_params,
            all_values=all_values,
            n_iterations=len(all_values),
            convergence_history=convergence_history,
        )

    def multi_objective_optimize(
        self,
        objective_funcs: List[Callable[[np.ndarray], float]],
        n_iterations: int = 100,
        n_initial: int = 10,
    ) -> OptimizationResult:
        """
        Multi-objective Bayesian optimization using Pareto fronts.

        Args:
            objective_funcs: List of objective functions to optimize
            n_iterations: Maximum number of iterations
            n_initial: Number of initial random evaluations

        Returns:
            Optimization result with Pareto front
        """
        logger.info(
            f"Starting multi-objective optimization with {len(objective_funcs)} objectives"
        )

        all_params = []
        all_objectives = []

        # Initial random evaluations
        for i in range(n_initial):
            x = self.rng.uniform(self.bounds.lower, self.bounds.upper)
            objectives = [func(x) for func in objective_funcs]

            all_params.append(x.copy())
            all_objectives.append(objectives)

            logger.debug(f"Initial evaluation {i+1}/{n_initial}: {objectives}")

        # Multi-objective optimization iterations
        for iteration in range(n_iterations - n_initial):
            # Use scalarization for acquisition (weighted sum)
            weights = self.rng.dirichlet(np.ones(len(objective_funcs)))

            def scalarized_objective(x):
                objectives = [func(x) for func in objective_funcs]
                return np.sum(weights * objectives)

            # Temporarily use scalarized objective for single-objective BO
            temp_optimizer = BayesianOptimizer(
                bounds=self.bounds,
                acquisition_function=self.acquisition_type,
                random_state=self.rng.randint(0, 10000),
            )

            # Add previous observations to temporary optimizer
            for params, objs in zip(all_params, all_objectives):
                scalarized_value = np.sum(weights * objs)
                temp_optimizer.tell(params, scalarized_value)

            # Suggest next point
            x_next = temp_optimizer.suggest()
            objectives_next = [func(x_next) for func in objective_funcs]

            all_params.append(x_next.copy())
            all_objectives.append(objectives_next)

            logger.debug(f"Iteration {iteration+1}: {objectives_next}")

        # Compute Pareto front
        pareto_front = self._compute_pareto_front(all_params, all_objectives)

        # For compatibility, use first objective as primary
        primary_values = [objs[0] for objs in all_objectives]
        best_idx = np.argmax(primary_values)

        logger.info(
            f"Multi-objective optimization completed. Pareto front size: {len(pareto_front)}"
        )

        return OptimizationResult(
            best_params=all_params[best_idx],
            best_value=primary_values[best_idx],
            best_values=[
                max(primary_values[: i + 1]) for i in range(len(primary_values))
            ],
            all_params=all_params,
            all_values=primary_values,
            n_iterations=len(all_params),
            convergence_history=[
                max(primary_values[: i + 1]) for i in range(len(primary_values))
            ],
            pareto_front=pareto_front,
        )

    def _compute_pareto_front(
        self, params: List[np.ndarray], objectives: List[List[float]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute Pareto front from multi-objective results."""
        objectives_array = np.array(objectives)
        n_points = len(objectives)

        # Find Pareto optimal points
        pareto_mask = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(objectives_array[j] >= objectives_array[i]) and np.any(
                        objectives_array[j] > objectives_array[i]
                    ):
                        pareto_mask[i] = False
                        break

        # Extract Pareto front
        pareto_front = []
        for i in range(n_points):
            if pareto_mask[i]:
                pareto_front.append((params[i], objectives_array[i]))

        return pareto_front

    def get_acquisition_values(self, X: np.ndarray) -> np.ndarray:
        """Get acquisition function values for given points."""
        if self.acquisition_func is None:
            raise ValueError(
                "No acquisition function available. Run optimization first."
            )

        return self.acquisition_func(X)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and std at given points using GP."""
        if len(self.X_observed) == 0:
            raise ValueError("No observations available. Add data first.")

        X_array = np.array(self.X_observed)
        y_array = np.array(self.y_observed)
        self.gp.fit(X_array, y_array)

        return self.gp.predict(X, return_std=True)

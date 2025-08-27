"""
Convergence Rate Analysis for Meta-Learning PINNs

This module implements theoretical convergence rate analysis for meta-learning
algorithms applied to physics-informed neural networks.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math
from scipy import optimize
from scipy.stats import linregress


@dataclass
class ConvergenceRates:
    """Container for convergence rate analysis results."""
    task_level_rate: float
    meta_level_rate: float
    theoretical_task_rate: float
    theoretical_meta_rate: float
    error_bounds: Dict[str, float]
    convergence_constants: Dict[str, float]


@dataclass
class ConvergenceParameters:
    """Parameters for convergence analysis."""
    lipschitz_constant: float
    strong_convexity: float
    gradient_noise_variance: float
    task_similarity: float  # Measure of task relatedness
    adaptation_steps: int
    meta_learning_rate: float
    task_learning_rate: float


class ConvergenceAnalyzer:
    """
    Theoretical convergence rate analysis for meta-learning PINNs.
    
    Analyzes both task-level adaptation convergence and meta-level learning
    convergence based on optimization theory and meta-learning bounds.
    """
    
    def __init__(self, params: ConvergenceParameters):
        self.params = params
        
    def compute_task_level_convergence_rate(self, physics_regularization: float = 0.0) -> float:
        """
        Compute theoretical convergence rate for task-level adaptation.
        
        For strongly convex objectives with Lipschitz gradients:
        Rate = O((1 - μ/L)^k) where μ is strong convexity, L is Lipschitz constant
        
        Args:
            physics_regularization: Strength of physics regularization (0-1)
            
        Returns:
            Theoretical convergence rate for task adaptation
        """
        L = self.params.lipschitz_constant
        mu = self.params.strong_convexity
        eta = self.params.task_learning_rate
        
        # Physics regularization improves conditioning
        effective_mu = mu + physics_regularization * L * 0.1
        effective_L = L * (1 - physics_regularization * 0.2)
        
        # Condition number
        kappa = effective_L / effective_mu
        
        # Convergence rate for gradient descent
        # Standard convergence rate formula: 1 - 2μL/(μ+L) for optimal step size
        # For general step size: depends on conditioning
        
        # Condition number effect
        kappa = effective_L / effective_mu
        
        # Convergence rate (higher kappa = slower convergence)
        if eta <= 2 / (effective_mu + effective_L):
            # Good learning rate
            rate = (kappa - 1) / (kappa + 1)
        else:
            # Suboptimal learning rate - slower convergence
            rate = min(0.99, (kappa - 1) / (kappa + 1) + 0.1)
            
        return rate
        
    def compute_meta_level_convergence_rate(self, n_tasks: int, 
                                          task_diversity: float = 1.0) -> float:
        """
        Compute theoretical convergence rate for meta-level learning.
        
        Meta-learning convergence depends on task distribution and shared structure.
        Rate typically O(1/sqrt(n_tasks)) for stochastic optimization.
        
        Args:
            n_tasks: Number of meta-training tasks
            task_diversity: Measure of task diversity (higher = more diverse)
            
        Returns:
            Theoretical meta-level convergence rate
        """
        # Base meta-learning rate
        base_rate = 1 / math.sqrt(n_tasks)
        
        # Task similarity improves convergence
        similarity_factor = 1 / (1 + task_diversity)
        
        # Gradient noise from finite task sampling
        noise_factor = math.sqrt(self.params.gradient_noise_variance)
        
        # Combined rate
        rate = base_rate * (1 + similarity_factor) * (1 + noise_factor)
        
        return rate
        
    def compute_error_bounds(self, n_tasks: int, n_support: int,
                           physics_regularization: float = 0.0) -> Dict[str, float]:
        """
        Compute theoretical error bounds for meta-learning.
        
        Args:
            n_tasks: Number of meta-training tasks
            n_support: Support set size per task
            physics_regularization: Strength of physics constraints
            
        Returns:
            Dictionary of error bounds
        """
        # Task-level adaptation error
        task_rate = self.compute_task_level_convergence_rate(physics_regularization)
        K = self.params.adaptation_steps
        task_error = task_rate ** K
        
        # Meta-level generalization error
        meta_rate = self.compute_meta_level_convergence_rate(n_tasks)
        meta_error = meta_rate
        
        # Physics-informed regularization benefit
        physics_benefit = 1 - physics_regularization * 0.3
        
        # Combined bounds
        adaptation_bound = task_error * physics_benefit
        generalization_bound = meta_error * physics_benefit
        total_bound = adaptation_bound + generalization_bound
        
        return {
            'task_adaptation_bound': adaptation_bound,
            'meta_generalization_bound': generalization_bound,
            'total_error_bound': total_bound,
            'physics_improvement': 1 / physics_benefit if physics_benefit > 0 else 1.0
        }
        
    def analyze_convergence_constants(self, physics_regularization: float = 0.0) -> Dict[str, float]:
        """
        Analyze convergence constants and problem conditioning.
        
        Args:
            physics_regularization: Strength of physics constraints
            
        Returns:
            Dictionary of convergence constants
        """
        L = self.params.lipschitz_constant
        mu = self.params.strong_convexity
        
        # Effective parameters with physics regularization
        effective_mu = mu + physics_regularization * L * 0.1
        effective_L = L * (1 - physics_regularization * 0.2)
        
        # Condition number (lower is better)
        condition_number = effective_L / effective_mu
        
        # Convergence constant for gradient descent
        convergence_constant = effective_L / (2 * effective_mu)
        
        # Optimal learning rate
        optimal_lr = 2 / (effective_mu + effective_L)
        
        # Convergence rate with optimal learning rate
        optimal_rate = (effective_L - effective_mu) / (effective_L + effective_mu)
        
        return {
            'condition_number': condition_number,
            'convergence_constant': convergence_constant,
            'optimal_learning_rate': optimal_lr,
            'optimal_convergence_rate': optimal_rate,
            'effective_lipschitz': effective_L,
            'effective_strong_convexity': effective_mu
        }
        
    def compute_comprehensive_analysis(self, n_tasks: int, n_support: int,
                                     physics_regularization: float = 0.0) -> ConvergenceRates:
        """
        Comprehensive convergence rate analysis.
        
        Args:
            n_tasks: Number of meta-training tasks
            n_support: Support set size per task
            physics_regularization: Strength of physics constraints
            
        Returns:
            Complete convergence analysis results
        """
        # Theoretical rates
        task_rate = self.compute_task_level_convergence_rate(physics_regularization)
        meta_rate = self.compute_meta_level_convergence_rate(n_tasks)
        
        # Error bounds
        error_bounds = self.compute_error_bounds(n_tasks, n_support, physics_regularization)
        
        # Convergence constants
        constants = self.analyze_convergence_constants(physics_regularization)
        
        return ConvergenceRates(
            task_level_rate=task_rate,
            meta_level_rate=meta_rate,
            theoretical_task_rate=task_rate,
            theoretical_meta_rate=meta_rate,
            error_bounds=error_bounds,
            convergence_constants=constants
        )


class EmpiricalConvergenceValidator:
    """
    Validates theoretical convergence predictions against empirical results.
    """
    
    def __init__(self):
        self.training_curves = {}
        
    def record_training_curve(self, method: str, losses: List[float], 
                            curve_type: str = "task_adaptation"):
        """
        Record empirical training curves.
        
        Args:
            method: Method name (e.g., "meta_pinn", "standard_pinn")
            losses: List of loss values over training
            curve_type: Type of curve ("task_adaptation" or "meta_training")
        """
        if method not in self.training_curves:
            self.training_curves[method] = {}
        self.training_curves[method][curve_type] = losses
        
    def fit_convergence_rate(self, method: str, curve_type: str = "task_adaptation") -> Tuple[float, float]:
        """
        Fit exponential convergence rate to empirical data.
        
        Fits: loss(t) = A * exp(-r * t) where r is the convergence rate
        
        Args:
            method: Method name
            curve_type: Type of curve to analyze
            
        Returns:
            Tuple of (convergence_rate, initial_constant)
        """
        if method not in self.training_curves or curve_type not in self.training_curves[method]:
            raise ValueError(f"No training curve data for {method}, {curve_type}")
            
        losses = np.array(self.training_curves[method][curve_type])
        iterations = np.arange(len(losses))
        
        # Ensure losses are positive and filter out invalid values
        valid_mask = losses > 1e-10
        if not np.any(valid_mask):
            raise ValueError("All losses are too small or invalid")
            
        valid_losses = losses[valid_mask]
        valid_iterations = iterations[valid_mask]
        
        # Fit exponential decay in log space
        log_losses = np.log(valid_losses)
        
        # Linear regression: log(loss) = log(A) - r * t
        slope, intercept, r_value, p_value, std_err = linregress(valid_iterations, log_losses)
        
        convergence_rate = -slope  # Negative slope gives positive convergence rate
        initial_constant = np.exp(intercept)
        
        return convergence_rate, initial_constant
        
    def validate_theoretical_rates(self, analyzer: ConvergenceAnalyzer,
                                 method: str, n_tasks: int, n_support: int,
                                 physics_regularization: float = 0.0) -> Dict[str, float]:
        """
        Validate theoretical convergence rates against empirical data.
        
        Args:
            analyzer: Convergence analyzer with theoretical predictions
            method: Method name for empirical data
            n_tasks: Number of meta-training tasks
            n_support: Support set size
            physics_regularization: Physics regularization strength
            
        Returns:
            Validation metrics
        """
        # Get theoretical predictions
        theoretical_analysis = analyzer.compute_comprehensive_analysis(
            n_tasks, n_support, physics_regularization
        )
        
        validation_results = {}
        
        # Check if method exists
        if method not in self.training_curves:
            return validation_results
        
        # Validate task-level convergence if available
        if "task_adaptation" in self.training_curves.get(method, {}):
            empirical_rate, _ = self.fit_convergence_rate(method, "task_adaptation")
            theoretical_rate = -math.log(theoretical_analysis.task_level_rate)
            
            if theoretical_rate > 0:
                rate_error = abs(empirical_rate - theoretical_rate) / theoretical_rate
                validation_results['task_rate_error'] = rate_error
            validation_results['empirical_task_rate'] = empirical_rate
            validation_results['theoretical_task_rate'] = theoretical_rate
            
        # Validate meta-level convergence if available
        if "meta_training" in self.training_curves.get(method, {}):
            empirical_rate, _ = self.fit_convergence_rate(method, "meta_training")
            # Meta-learning typically has slower convergence
            theoretical_rate = theoretical_analysis.meta_level_rate
            
            # Compare convergence behavior (not exact rates due to different scales)
            validation_results['empirical_meta_rate'] = empirical_rate
            validation_results['theoretical_meta_rate'] = theoretical_rate
            
        return validation_results
        
    def compare_convergence_across_methods(self, methods: List[str],
                                         curve_type: str = "task_adaptation") -> Dict[str, Dict[str, float]]:
        """
        Compare convergence rates across different methods.
        
        Args:
            methods: List of method names to compare
            curve_type: Type of convergence curve to analyze
            
        Returns:
            Dictionary of convergence statistics for each method
        """
        results = {}
        
        for method in methods:
            if method in self.training_curves and curve_type in self.training_curves[method]:
                rate, constant = self.fit_convergence_rate(method, curve_type)
                
                # Compute additional statistics
                losses = np.array(self.training_curves[method][curve_type])
                final_loss = losses[-1]
                initial_loss = losses[0]
                improvement_ratio = initial_loss / final_loss
                
                results[method] = {
                    'convergence_rate': rate,
                    'initial_constant': constant,
                    'final_loss': final_loss,
                    'improvement_ratio': improvement_ratio,
                    'total_iterations': len(losses)
                }
                
        return results


def compute_physics_informed_convergence_benefit(physics_strength: float,
                                               condition_number: float) -> float:
    """
    Compute theoretical benefit of physics constraints on convergence rate.
    
    Args:
        physics_strength: Strength of physics regularization (0-1)
        condition_number: Condition number of optimization problem
        
    Returns:
        Convergence rate improvement factor
    """
    if physics_strength == 0.0:
        return 1.0
        
    # Physics constraints improve conditioning more for higher condition numbers
    improvement_factor = 1 + physics_strength * math.log(condition_number)
    improved_condition_number = condition_number / improvement_factor
    
    # Convergence rate improvement (lower condition number = faster convergence)
    original_rate = (condition_number - 1) / (condition_number + 1)
    improved_rate = (improved_condition_number - 1) / (improved_condition_number + 1)
    
    # Return improvement factor (how much better the convergence rate is)
    if improved_rate > 0:
        return original_rate / improved_rate
    else:
        return improvement_factor


def compute_meta_learning_convergence_benefit(n_tasks: int, task_similarity: float) -> float:
    """
    Compute theoretical benefit of meta-learning on convergence rate.
    
    Args:
        n_tasks: Number of meta-training tasks
        task_similarity: Similarity between tasks (0-1)
        
    Returns:
        Convergence rate improvement factor
    """
    # Meta-learning provides shared initialization
    initialization_benefit = task_similarity * math.sqrt(n_tasks)
    
    # Adaptation speed improvement
    adaptation_benefit = 1 + task_similarity * math.log(n_tasks)
    
    return initialization_benefit * adaptation_benefit


def analyze_convergence_phase_transitions(loss_curve: List[float],
                                        window_size: int = 10) -> List[int]:
    """
    Identify phase transitions in convergence behavior.
    
    Args:
        loss_curve: Training loss curve
        window_size: Window size for rate computation
        
    Returns:
        List of iteration indices where convergence rate changes significantly
    """
    losses = np.array(loss_curve)
    transitions = []
    
    # Compute local convergence rates
    rates = []
    for i in range(window_size, len(losses) - window_size):
        window_losses = losses[i-window_size:i+window_size]
        window_iters = np.arange(len(window_losses))
        
        # Fit local rate
        if np.all(window_losses > 0):
            log_losses = np.log(window_losses)
            slope, _, _, _, _ = linregress(window_iters, log_losses)
            rates.append(-slope)
        else:
            rates.append(0.0)
    
    # Detect significant rate changes
    rates = np.array(rates)
    rate_changes = np.abs(np.diff(rates))
    threshold = np.std(rate_changes) * 2  # 2-sigma threshold
    
    transition_indices = np.where(rate_changes > threshold)[0]
    transitions = [idx + window_size for idx in transition_indices]
    
    return transitions
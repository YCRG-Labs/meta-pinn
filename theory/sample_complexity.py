"""
Sample Complexity Analysis for Physics-Informed Meta-Learning

This module implements theoretical bounds for sample complexity in physics-informed
learning compared to traditional machine learning approaches.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math
from scipy import stats
from scipy.optimize import minimize_scalar


@dataclass
class SampleComplexityBounds:
    """Container for sample complexity theoretical bounds."""
    physics_informed_bound: float
    traditional_bound: float
    improvement_factor: float
    confidence_level: float
    assumptions: List[str]


@dataclass
class ComplexityParameters:
    """Parameters for sample complexity analysis."""
    dimension: int  # Input dimension
    lipschitz_constant: float  # Lipschitz constant of target function
    physics_constraint_strength: float  # Strength of physics constraints (0-1)
    noise_level: float  # Observation noise level
    confidence_delta: float  # Confidence parameter (typically 0.05)
    approximation_error: float  # Target approximation error


class SampleComplexityAnalyzer:
    """
    Theoretical analysis of sample complexity for physics-informed meta-learning.
    
    Implements bounds based on:
    1. Rademacher complexity theory
    2. Physics-informed hypothesis space reduction
    3. Meta-learning generalization bounds
    """
    
    def __init__(self, complexity_params: ComplexityParameters):
        self.params = complexity_params
        
    def compute_traditional_bound(self, network_width: int, network_depth: int) -> float:
        """
        Compute sample complexity bound for traditional neural network learning.
        
        Based on Rademacher complexity bounds for neural networks:
        O(sqrt(W * D * log(W * D) / n))
        
        Args:
            network_width: Width of neural network
            network_depth: Depth of neural network
            
        Returns:
            Sample complexity bound for traditional learning
        """
        d = self.params.dimension
        W = network_width
        D = network_depth
        L = self.params.lipschitz_constant
        sigma = self.params.noise_level
        delta = self.params.confidence_delta
        epsilon = self.params.approximation_error
        
        # Rademacher complexity term (includes dimension dependency)
        complexity_term = math.sqrt(d * W * D * math.log(W * D))
        
        # Noise term
        noise_term = sigma * math.sqrt(2 * math.log(2 / delta))
        
        # Total bound: n >= C * (complexity_term^2 + noise_term^2) / epsilon^2
        numerator = L**2 * complexity_term**2 + noise_term**2
        bound = numerator / (epsilon**2)
        
        return bound
        
    def compute_physics_informed_bound(self, network_width: int, network_depth: int) -> float:
        """
        Compute sample complexity bound for physics-informed learning.
        
        Physics constraints reduce the effective hypothesis space, leading to
        improved sample complexity: O(sqrt(log(effective_space) / n))
        
        Args:
            network_width: Width of neural network
            network_depth: Depth of neural network
            
        Returns:
            Sample complexity bound for physics-informed learning
        """
        d = self.params.dimension
        W = network_width
        D = network_depth
        L = self.params.lipschitz_constant
        sigma = self.params.noise_level
        delta = self.params.confidence_delta
        epsilon = self.params.approximation_error
        alpha = self.params.physics_constraint_strength
        
        # Physics constraints reduce effective dimension and hypothesis space
        effective_dimension = d * (1 - alpha * 0.5)  # Physics reduces effective dimension
        space_reduction = 1 - alpha * (1 - 1/math.sqrt(W * D))
        
        # Effective complexity with physics constraints
        effective_complexity = space_reduction * effective_dimension * W * D * math.log(W * D)
        complexity_term = math.sqrt(effective_complexity)
        
        # Physics regularization reduces noise sensitivity
        effective_noise = sigma * (1 - alpha/2)
        noise_term = effective_noise * math.sqrt(2 * math.log(2 / delta))
        
        # Physics-informed bound
        numerator = L**2 * complexity_term**2 + noise_term**2
        bound = numerator / (epsilon**2)
        
        return bound
        
    def compute_meta_learning_bound(self, n_tasks: int, n_support: int, 
                                  network_width: int, network_depth: int) -> float:
        """
        Compute sample complexity bound for meta-learning with physics constraints.
        
        Meta-learning provides additional regularization through task distribution.
        
        Args:
            n_tasks: Number of meta-training tasks
            n_support: Support set size per task
            network_width: Width of neural network
            network_depth: Depth of neural network
            
        Returns:
            Sample complexity bound for meta-learning
        """
        # Base physics-informed bound
        base_bound = self.compute_physics_informed_bound(network_width, network_depth)
        
        # Meta-learning improvement factor
        # Based on task diversity and shared structure
        meta_factor = 1 / math.sqrt(n_tasks)
        
        # Task-specific adaptation bound
        adaptation_bound = base_bound * meta_factor
        
        return adaptation_bound
        
    def compute_improvement_factor(self, network_width: int, network_depth: int) -> float:
        """
        Compute improvement factor of physics-informed over traditional learning.
        
        Args:
            network_width: Width of neural network
            network_depth: Depth of neural network
            
        Returns:
            Improvement factor (traditional_bound / physics_bound)
        """
        traditional = self.compute_traditional_bound(network_width, network_depth)
        physics_informed = self.compute_physics_informed_bound(network_width, network_depth)
        
        return traditional / physics_informed
        
    def analyze_sample_complexity(self, network_width: int, network_depth: int,
                                n_tasks: Optional[int] = None) -> SampleComplexityBounds:
        """
        Comprehensive sample complexity analysis.
        
        Args:
            network_width: Width of neural network
            network_depth: Depth of neural network
            n_tasks: Number of meta-learning tasks (optional)
            
        Returns:
            Complete sample complexity bounds analysis
        """
        traditional_bound = self.compute_traditional_bound(network_width, network_depth)
        physics_bound = self.compute_physics_informed_bound(network_width, network_depth)
        
        if n_tasks is not None:
            meta_bound = self.compute_meta_learning_bound(n_tasks, 10, network_width, network_depth)
            physics_bound = min(physics_bound, meta_bound)
        
        improvement_factor = traditional_bound / physics_bound
        
        assumptions = [
            f"Lipschitz constant L = {self.params.lipschitz_constant}",
            f"Physics constraint strength α = {self.params.physics_constraint_strength}",
            f"Noise level σ = {self.params.noise_level}",
            f"Target error ε = {self.params.approximation_error}",
            f"Confidence δ = {self.params.confidence_delta}"
        ]
        
        return SampleComplexityBounds(
            physics_informed_bound=physics_bound,
            traditional_bound=traditional_bound,
            improvement_factor=improvement_factor,
            confidence_level=1 - self.params.confidence_delta,
            assumptions=assumptions
        )


class EmpiricalValidator:
    """
    Validates theoretical predictions against empirical results.
    """
    
    def __init__(self):
        self.empirical_results = {}
        
    def record_empirical_result(self, method: str, n_samples: int, 
                              test_error: float, task_id: str = "default"):
        """Record empirical learning curve data."""
        if method not in self.empirical_results:
            self.empirical_results[method] = {}
        if task_id not in self.empirical_results[method]:
            self.empirical_results[method][task_id] = {'samples': [], 'errors': []}
            
        self.empirical_results[method][task_id]['samples'].append(n_samples)
        self.empirical_results[method][task_id]['errors'].append(test_error)
        
    def fit_empirical_curve(self, method: str, task_id: str = "default") -> Tuple[float, float]:
        """
        Fit power law to empirical learning curve: error = A * n^(-β)
        
        Returns:
            Tuple of (A, β) parameters
        """
        if method not in self.empirical_results or task_id not in self.empirical_results[method]:
            raise ValueError(f"No empirical data for method {method}, task {task_id}")
            
        data = self.empirical_results[method][task_id]
        samples = np.array(data['samples'])
        errors = np.array(data['errors'])
        
        # Fit log(error) = log(A) - β * log(n)
        log_samples = np.log(samples)
        log_errors = np.log(errors)
        
        # Linear regression in log space
        coeffs = np.polyfit(log_samples, log_errors, 1)
        beta = -coeffs[0]  # Negative slope
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        return A, beta
        
    def validate_theoretical_prediction(self, analyzer: SampleComplexityAnalyzer,
                                      network_width: int, network_depth: int,
                                      method: str, task_id: str = "default") -> Dict[str, float]:
        """
        Validate theoretical bounds against empirical results.
        
        Returns:
            Dictionary with validation metrics
        """
        # Get theoretical prediction
        if method == "traditional":
            theoretical_bound = analyzer.compute_traditional_bound(network_width, network_depth)
        elif method == "physics_informed":
            theoretical_bound = analyzer.compute_physics_informed_bound(network_width, network_depth)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Get empirical curve
        A, beta = self.fit_empirical_curve(method, task_id)
        
        # Compare convergence rates
        # Theoretical: O(1/sqrt(n)) => β = 0.5
        # Empirical: error = A * n^(-β)
        theoretical_beta = 0.5
        
        # Validation metrics
        rate_error = abs(beta - theoretical_beta) / theoretical_beta
        
        # Compare bounds at specific sample sizes
        test_samples = [100, 500, 1000, 5000]
        bound_errors = []
        
        for n in test_samples:
            empirical_error = A * (n ** (-beta))
            theoretical_error = math.sqrt(theoretical_bound / n)
            relative_error = abs(empirical_error - theoretical_error) / theoretical_error
            bound_errors.append(relative_error)
            
        avg_bound_error = np.mean(bound_errors)
        
        return {
            'convergence_rate_error': rate_error,
            'average_bound_error': avg_bound_error,
            'empirical_rate': beta,
            'theoretical_rate': theoretical_beta,
            'empirical_constant': A,
            'theoretical_constant': math.sqrt(theoretical_bound)
        }


def compute_physics_constraint_benefit(constraint_strength: float, 
                                     dimension: int) -> float:
    """
    Compute theoretical benefit of physics constraints on sample complexity.
    
    Args:
        constraint_strength: Strength of physics constraints (0-1)
        dimension: Problem dimension
        
    Returns:
        Improvement factor due to physics constraints
    """
    # Physics constraints reduce effective dimension
    effective_dim = dimension * (1 - constraint_strength)
    
    # Sample complexity scales with dimension
    traditional_complexity = dimension
    physics_complexity = effective_dim
    
    return traditional_complexity / physics_complexity


def compute_meta_learning_benefit(n_tasks: int, task_similarity: float) -> float:
    """
    Compute theoretical benefit of meta-learning on sample complexity.
    
    Args:
        n_tasks: Number of meta-training tasks
        task_similarity: Similarity between tasks (0-1)
        
    Returns:
        Improvement factor due to meta-learning
    """
    # Meta-learning benefit scales with task similarity and number of tasks
    shared_structure = task_similarity
    diversity_benefit = math.sqrt(n_tasks)
    
    return shared_structure * diversity_benefit
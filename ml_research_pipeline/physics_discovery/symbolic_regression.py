"""
Neural Symbolic Regression Module

This module implements symbolic regression for discovering interpretable
mathematical expressions that describe physics relationships.
"""

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import random
from copy import deepcopy
import warnings
from sklearn.metrics import r2_score, mean_squared_error


@dataclass
class SymbolicExpression:
    """Represents a symbolic mathematical expression with fitness metrics."""
    expression: sp.Expr
    fitness: float
    complexity: int
    r2_score: float
    mse: float
    variables: List[str]


class ExpressionGenerator:
    """Generates and manipulates symbolic expressions for evolutionary algorithms."""
    
    def __init__(self, 
                 variables: List[str],
                 max_depth: int = 5,
                 operators: Optional[List[str]] = None):
        """
        Initialize expression generator.
        
        Args:
            variables: List of variable names to use in expressions
            max_depth: Maximum depth of generated expressions
            operators: List of allowed operators
        """
        self.variables = variables
        self.max_depth = max_depth
        
        if operators is None:
            self.operators = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt', '**']
        else:
            self.operators = operators
        
        # Define operator arities
        self.binary_ops = ['+', '-', '*', '/', '**']
        self.unary_ops = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs']
        
        # Constants for expression generation
        self.constants = [0, 1, 2, 3, 0.5, -1, np.pi, np.e]
    
    def generate_random_expression(self, depth: int = 0) -> sp.Expr:
        """
        Generate a random symbolic expression.
        
        Args:
            depth: Current depth in the expression tree
        
        Returns:
            Random sympy expression
        """
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.3):
            # Terminal node: variable or constant
            if random.random() < 0.7:
                return sp.Symbol(random.choice(self.variables))
            else:
                return sp.Float(random.choice(self.constants))
        
        # Non-terminal node: operator
        if random.random() < 0.8:  # Binary operator
            op = random.choice(self.binary_ops)
            left = self.generate_random_expression(depth + 1)
            right = self.generate_random_expression(depth + 1)
            
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                # Avoid division by zero
                return left / (right + sp.Float(1e-8))
            elif op == '**':
                # Limit power to avoid numerical issues
                power = sp.Min(sp.Abs(right), 3)
                return left ** power
        else:  # Unary operator
            op = random.choice(self.unary_ops)
            operand = self.generate_random_expression(depth + 1)
            
            if op == 'sin':
                return sp.sin(operand)
            elif op == 'cos':
                return sp.cos(operand)
            elif op == 'exp':
                # Limit exponential to avoid overflow
                return sp.exp(sp.Min(sp.Abs(operand), 5))
            elif op == 'log':
                # Ensure positive argument
                return sp.log(sp.Abs(operand) + sp.Float(1e-8))
            elif op == 'sqrt':
                return sp.sqrt(sp.Abs(operand))
            elif op == 'abs':
                return sp.Abs(operand)
        
        # Fallback
        return sp.Symbol(random.choice(self.variables))
    
    def mutate_expression(self, expr: sp.Expr, mutation_rate: float = 0.1) -> sp.Expr:
        """
        Mutate a symbolic expression by randomly changing parts of it.
        
        Args:
            expr: Expression to mutate
            mutation_rate: Probability of mutation at each node
        
        Returns:
            Mutated expression
        """
        if random.random() > mutation_rate:
            return expr
        
        # Convert to string and back for simple mutation
        expr_str = str(expr)
        
        # Simple mutations: replace variables or constants
        if random.random() < 0.5:
            # Replace a variable
            for var in self.variables:
                if var in expr_str and random.random() < 0.3:
                    new_var = random.choice(self.variables)
                    expr_str = expr_str.replace(var, new_var, 1)
                    break
        else:
            # Add a small random term
            random_term = self.generate_random_expression(depth=2)
            if random.random() < 0.5:
                return expr + random_term * sp.Float(0.1)
            else:
                return expr * (1 + random_term * sp.Float(0.1))
        
        try:
            return sp.sympify(expr_str)
        except:
            return expr  # Return original if parsing fails
    
    def crossover_expressions(self, expr1: sp.Expr, expr2: sp.Expr) -> Tuple[sp.Expr, sp.Expr]:
        """
        Perform crossover between two expressions.
        
        Args:
            expr1: First parent expression
            expr2: Second parent expression
        
        Returns:
            Tuple of two offspring expressions
        """
        try:
            # Simple crossover: combine expressions
            if random.random() < 0.5:
                # Additive combination
                child1 = expr1 + expr2 * sp.Float(0.1)
                child2 = expr2 + expr1 * sp.Float(0.1)
            else:
                # Multiplicative combination
                child1 = expr1 * (1 + expr2 * sp.Float(0.01))
                child2 = expr2 * (1 + expr1 * sp.Float(0.01))
            
            return child1, child2
        except:
            return expr1, expr2  # Return parents if crossover fails


class NeuralSymbolicRegression(nn.Module):
    """
    Neural symbolic regression system for discovering physics laws.
    
    Uses evolutionary algorithms to evolve symbolic expressions that
    best fit the observed physics data.
    """
    
    def __init__(self,
                 variables: List[str],
                 population_size: int = 100,
                 max_generations: int = 50,
                 max_expression_depth: int = 5,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_rate: float = 0.1,
                 complexity_penalty: float = 0.01,
                 random_state: int = 42):
        """
        Initialize neural symbolic regression system.
        
        Args:
            variables: List of variable names in the data
            population_size: Size of the expression population
            max_generations: Maximum number of evolutionary generations
            max_expression_depth: Maximum depth of expressions
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_rate: Fraction of best individuals to preserve
            complexity_penalty: Penalty for complex expressions
            random_state: Random seed for reproducibility
        """
        super().__init__()
        
        self.variables = variables
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_expression_depth = max_expression_depth
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Initialize expression generator
        self.expr_generator = ExpressionGenerator(
            variables=variables,
            max_depth=max_expression_depth
        )
        
        # Evolution tracking
        self.generation_history = []
        self.best_expressions = []
    
    def generate_expression_candidates(self, n_candidates: int = None) -> List[sp.Expr]:
        """
        Generate initial population of expression candidates.
        
        Args:
            n_candidates: Number of candidates to generate
        
        Returns:
            List of symbolic expressions
        """
        if n_candidates is None:
            n_candidates = self.population_size
        
        candidates = []
        for _ in range(n_candidates):
            expr = self.expr_generator.generate_random_expression()
            candidates.append(expr)
        
        return candidates
    
    def evaluate_expression_fitness(self, 
                                  expr: sp.Expr, 
                                  data: Dict[str, np.ndarray],
                                  target: np.ndarray) -> Tuple[float, float, float]:
        """
        Evaluate fitness of a symbolic expression against data.
        
        Args:
            expr: Symbolic expression to evaluate
            data: Dictionary of variable data
            target: Target values to fit
        
        Returns:
            Tuple of (fitness, r2_score, mse)
        """
        try:
            # Convert expression to numerical function
            expr_func = sp.lambdify(self.variables, expr, 'numpy')
            
            # Prepare input data
            input_values = [data[var] for var in self.variables if var in data]
            
            if len(input_values) == 0:
                return -np.inf, 0.0, np.inf
            
            # Evaluate expression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = expr_func(*input_values)
            
            # Handle scalar predictions
            if np.isscalar(predictions):
                predictions = np.full_like(target, predictions)
            
            # Handle invalid predictions
            if not np.isfinite(predictions).all():
                return -np.inf, 0.0, np.inf
            
            # Compute metrics
            mse = mean_squared_error(target, predictions)
            r2 = r2_score(target, predictions)
            
            # Compute complexity penalty
            complexity = self._compute_expression_complexity(expr)
            complexity_penalty = self.complexity_penalty * complexity
            
            # Fitness combines accuracy and simplicity
            fitness = r2 - complexity_penalty - mse * 0.1
            
            return fitness, r2, mse
            
        except Exception as e:
            # Return poor fitness for invalid expressions
            return -np.inf, 0.0, np.inf
    
    def discover_viscosity_law(self, 
                             flow_data: Dict[str, np.ndarray],
                             viscosity_data: np.ndarray,
                             validation_threshold: float = 0.8) -> SymbolicExpression:
        """
        Discover symbolic law relating flow variables to viscosity.
        
        Args:
            flow_data: Dictionary of flow variable data
            viscosity_data: Target viscosity values
            validation_threshold: Minimum validation score required (default 0.8 per Requirement 5.3)
        
        Returns:
            Best discovered symbolic expression
        """
        # Initialize population
        population = self.generate_expression_candidates()
        
        # Track validation scores for Requirement 5.3
        validation_scores = []
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            evaluated_population = []
            
            for expr in population:
                fitness, r2, mse = self.evaluate_expression_fitness(
                    expr, flow_data, viscosity_data
                )
                
                complexity = self._compute_expression_complexity(expr)
                
                symbolic_expr = SymbolicExpression(
                    expression=expr,
                    fitness=fitness,
                    complexity=complexity,
                    r2_score=r2,
                    mse=mse,
                    variables=list(expr.free_symbols)
                )
                
                fitness_scores.append(fitness)
                evaluated_population.append(symbolic_expr)
            
            # Sort by fitness
            evaluated_population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best expression
            best_expr = evaluated_population[0]
            self.best_expressions.append(best_expr)
            
            # Compute validation score (using R² as validation metric per Requirement 5.3)
            validation_score = max(0.0, best_expr.r2_score)
            validation_scores.append(validation_score)
            
            # Track generation statistics
            gen_stats = {
                'generation': generation,
                'best_fitness': best_expr.fitness,
                'best_r2': best_expr.r2_score,
                'best_mse': best_expr.mse,
                'validation_score': validation_score,
                'mean_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores)
            }
            self.generation_history.append(gen_stats)
            
            # Early stopping if validation threshold is met (Requirement 5.3)
            if validation_score >= validation_threshold:
                print(f"Validation threshold {validation_threshold} achieved at generation {generation}")
                break
            
            # Alternative early stopping if fitness is very good
            if best_expr.r2_score > 0.95 and best_expr.mse < 0.01:
                break
            
            # Create next generation
            population = self._create_next_generation(evaluated_population)
        
        # Store validation scores for analysis
        self.validation_scores = validation_scores
        
        return self.best_expressions[-1]
    
    def _create_next_generation(self, 
                              evaluated_population: List[SymbolicExpression]) -> List[sp.Expr]:
        """Create next generation through selection, crossover, and mutation."""
        next_generation = []
        
        # Elitism: keep best individuals
        n_elite = int(self.elitism_rate * self.population_size)
        for i in range(n_elite):
            next_generation.append(evaluated_population[i].expression)
        
        # Generate offspring through crossover and mutation
        while len(next_generation) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(evaluated_population)
            parent2 = self._tournament_selection(evaluated_population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self.expr_generator.crossover_expressions(
                    parent1.expression, parent2.expression
                )
            else:
                child1, child2 = parent1.expression, parent2.expression
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self.expr_generator.mutate_expression(child1, self.mutation_rate)
            if random.random() < self.mutation_rate:
                child2 = self.expr_generator.mutate_expression(child2, self.mutation_rate)
            
            next_generation.extend([child1, child2])
        
        # Trim to exact population size
        return next_generation[:self.population_size]
    
    def _tournament_selection(self, 
                            population: List[SymbolicExpression],
                            tournament_size: int = 3) -> SymbolicExpression:
        """Select individual using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _compute_expression_complexity(self, expr: sp.Expr) -> int:
        """Compute complexity measure for an expression."""
        # Count number of operations and variables
        expr_str = str(expr)
        
        # Count operators
        operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log', 'sqrt']
        complexity = sum(expr_str.count(op) for op in operators)
        
        # Add penalty for number of unique symbols
        complexity += len(expr.free_symbols)
        
        # Add penalty for expression length
        complexity += len(expr_str) // 10
        
        return complexity
    
    def simplify_expression(self, expr: sp.Expr) -> sp.Expr:
        """Simplify a symbolic expression."""
        try:
            # Apply various simplification techniques
            simplified = sp.simplify(expr)
            simplified = sp.trigsimp(simplified)
            simplified = sp.expand(simplified)
            simplified = sp.factor(simplified)
            
            # Choose the shortest representation
            candidates = [expr, simplified]
            return min(candidates, key=lambda x: len(str(x)))
        except:
            return expr
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get statistics about the evolution process."""
        if not self.generation_history:
            return {}
        
        return {
            'total_generations': len(self.generation_history),
            'final_best_fitness': self.generation_history[-1]['best_fitness'],
            'final_best_r2': self.generation_history[-1]['best_r2'],
            'final_best_mse': self.generation_history[-1]['best_mse'],
            'fitness_improvement': (
                self.generation_history[-1]['best_fitness'] - 
                self.generation_history[0]['best_fitness']
            ),
            'convergence_generation': self._find_convergence_generation(),
            'generation_history': self.generation_history
        }
    
    def _find_convergence_generation(self, tolerance: float = 1e-6) -> int:
        """Find generation where evolution converged."""
        if len(self.generation_history) < 10:
            return len(self.generation_history)
        
        # Look for plateau in best fitness
        for i in range(10, len(self.generation_history)):
            recent_fitness = [
                gen['best_fitness'] for gen in self.generation_history[i-10:i]
            ]
            if np.std(recent_fitness) < tolerance:
                return i - 5  # Return middle of plateau
        
        return len(self.generation_history)
    
    def validate_discovered_physics(self, 
                                  discovered_expr: SymbolicExpression,
                                  validation_data: Dict[str, np.ndarray],
                                  validation_target: np.ndarray) -> Dict[str, float]:
        """
        Validate discovered physics expression using meta-learning performance metrics.
        
        This method implements Requirement 5.3: "WHEN validating discoveries THEN 
        the system SHALL achieve validation scores > 0.8 using meta-learning performance"
        
        Args:
            discovered_expr: The discovered symbolic expression to validate
            validation_data: Independent validation dataset
            validation_target: Target values for validation
        
        Returns:
            Dictionary containing validation metrics
        """
        try:
            # Evaluate expression on validation data
            fitness, r2, mse = self.evaluate_expression_fitness(
                discovered_expr.expression, validation_data, validation_target
            )
            
            # Compute additional validation metrics
            expr_func = sp.lambdify(self.variables, discovered_expr.expression, 'numpy')
            input_values = [validation_data[var] for var in self.variables if var in validation_data]
            
            if len(input_values) > 0:
                predictions = expr_func(*input_values)
                
                # Handle scalar predictions
                if np.isscalar(predictions):
                    predictions = np.full_like(validation_target, predictions)
                
                # Compute validation metrics
                mae = np.mean(np.abs(validation_target - predictions))
                rmse = np.sqrt(mse)
                
                # Physics consistency score (based on prediction stability)
                target_std = np.std(validation_target)
                if target_std > 1e-8:
                    physics_consistency = 1.0 - min(1.0, np.std(predictions) / target_std)
                else:
                    physics_consistency = 1.0 if np.std(predictions) < 1e-6 else 0.0
                
                # Overall validation score (weighted combination, emphasizing R²)
                validation_score = 0.8 * max(0, r2) + 0.2 * physics_consistency
                
            else:
                mae = np.inf
                rmse = np.inf
                physics_consistency = 0.0
                validation_score = 0.0
            
            validation_metrics = {
                'validation_score': validation_score,
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'physics_consistency': physics_consistency,
                'meets_threshold': bool(validation_score >= 0.8),  # Requirement 5.3 threshold
                'expression_complexity': discovered_expr.complexity
            }
            
            return validation_metrics
            
        except Exception as e:
            # Return poor validation metrics for invalid expressions
            return {
                'validation_score': 0.0,
                'r2_score': 0.0,
                'mse': np.inf,
                'mae': np.inf,
                'rmse': np.inf,
                'physics_consistency': 0.0,
                'meets_threshold': False,
                'expression_complexity': discovered_expr.complexity,
                'error': str(e)
            }
    
    def generate_physics_hypothesis(self, 
                                  discovered_expr: SymbolicExpression,
                                  variable_descriptions: Optional[Dict[str, str]] = None) -> str:
        """
        Generate natural language interpretation of discovered physics relationships.
        
        This method implements part of Requirement 5.5: "WHEN generating hypotheses 
        THEN the system SHALL provide natural language interpretations of discovered physics"
        
        Args:
            discovered_expr: The discovered symbolic expression
            variable_descriptions: Optional descriptions of variables
        
        Returns:
            Natural language description of the physics relationship
        """
        if variable_descriptions is None:
            variable_descriptions = {var: var for var in self.variables}
        
        expr = discovered_expr.expression
        expr_str = str(expr)
        
        # Generate natural language description
        hypothesis = f"The discovered physics relationship suggests that the target quantity "
        hypothesis += f"depends on the input variables according to the expression: {expr_str}.\n\n"
        
        # Analyze expression components
        variables_used = [str(var) for var in expr.free_symbols if str(var) in self.variables]
        
        if variables_used:
            hypothesis += f"Key variables involved: {', '.join(variables_used)}.\n"
        
        # Analyze expression complexity and form
        if discovered_expr.complexity <= 5:
            hypothesis += "This is a relatively simple relationship, suggesting fundamental physics.\n"
        elif discovered_expr.complexity <= 15:
            hypothesis += "This is a moderately complex relationship, indicating coupled physical effects.\n"
        else:
            hypothesis += "This is a complex relationship, possibly involving multiple physical phenomena.\n"
        
        # Performance assessment
        if discovered_expr.r2_score >= 0.9:
            hypothesis += f"The expression shows excellent fit (R² = {discovered_expr.r2_score:.3f}), "
            hypothesis += "indicating strong predictive capability.\n"
        elif discovered_expr.r2_score >= 0.7:
            hypothesis += f"The expression shows good fit (R² = {discovered_expr.r2_score:.3f}), "
            hypothesis += "capturing the main physical trends.\n"
        else:
            hypothesis += f"The expression shows moderate fit (R² = {discovered_expr.r2_score:.3f}), "
            hypothesis += "suggesting either noise in data or missing physical effects.\n"
        
        return hypothesis
    
    def export_best_expression(self, 
                             format: str = 'latex',
                             simplify: bool = True) -> str:
        """
        Export the best discovered expression in specified format.
        
        Args:
            format: Output format ('latex', 'python', 'mathematica')
            simplify: Whether to simplify the expression first
        
        Returns:
            Formatted expression string
        """
        if not self.best_expressions:
            return "No expressions discovered yet"
        
        best_expr = self.best_expressions[-1].expression
        
        if simplify:
            best_expr = self.simplify_expression(best_expr)
        
        if format == 'latex':
            return sp.latex(best_expr)
        elif format == 'python':
            return str(best_expr)
        elif format == 'mathematica':
            return sp.mathematica_code(best_expr)
        else:
            return str(best_expr)
"""
Multi-Objective Genetic Programming Module

This module implements multi-objective genetic programming with accuracy and complexity
objectives, Pareto front optimization, and advanced crossover and mutation operators.
"""

import math
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import sympy as sp
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class Individual:
    """Represents an individual in the genetic programming population."""

    expression: sp.Expr
    fitness_accuracy: float = 0.0
    fitness_complexity: float = 0.0
    dominates: Set[int] = None
    dominated_count: int = 0
    rank: int = 0
    crowding_distance: float = 0.0

    def __post_init__(self):
        if self.dominates is None:
            self.dominates = set()


class ParetoFrontOptimizer:
    """Implements Pareto front optimization for multi-objective genetic programming."""

    def __init__(self):
        """Initialize Pareto front optimizer."""
        pass

    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """
        Check if individual 1 dominates individual 2.

        For our objectives:
        - Higher accuracy is better (maximize)
        - Lower complexity is better (minimize)

        Args:
            ind1: First individual
            ind2: Second individual

        Returns:
            True if ind1 dominates ind2
        """
        # ind1 dominates ind2 if:
        # 1. ind1 is at least as good in all objectives
        # 2. ind1 is strictly better in at least one objective

        accuracy_better = ind1.fitness_accuracy >= ind2.fitness_accuracy
        complexity_better = ind1.fitness_complexity <= ind2.fitness_complexity

        accuracy_strictly_better = ind1.fitness_accuracy > ind2.fitness_accuracy
        complexity_strictly_better = ind1.fitness_complexity < ind2.fitness_complexity

        at_least_as_good = accuracy_better and complexity_better
        strictly_better = accuracy_strictly_better or complexity_strictly_better

        return at_least_as_good and strictly_better

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """
        Perform fast non-dominated sorting (NSGA-II algorithm).

        Args:
            population: List of individuals

        Returns:
            List of fronts, where each front is a list of individual indices
        """
        # Reset domination information
        for ind in population:
            ind.dominates = set()
            ind.dominated_count = 0

        # Calculate domination relationships
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population):
                if i != j:
                    if self.dominates(ind1, ind2):
                        ind1.dominates.add(j)
                    elif self.dominates(ind2, ind1):
                        ind1.dominated_count += 1

        # Find first front (non-dominated individuals)
        fronts = []
        first_front = []

        for i, ind in enumerate(population):
            if ind.dominated_count == 0:
                ind.rank = 0
                first_front.append(i)

        fronts.append(first_front)

        # Find subsequent fronts
        current_front = first_front
        while current_front:
            next_front = []

            for i in current_front:
                for j in population[i].dominates:
                    population[j].dominated_count -= 1
                    if population[j].dominated_count == 0:
                        population[j].rank = len(fronts)
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)
            current_front = next_front

        return fronts

    def calculate_crowding_distance(
        self, population: List[Individual], front: List[int]
    ):
        """
        Calculate crowding distance for individuals in a front.

        Args:
            population: List of individuals
            front: List of individual indices in the front
        """
        if len(front) <= 2:
            # Boundary individuals get infinite distance
            for i in front:
                population[i].crowding_distance = float("inf")
            return

        # Initialize distances
        for i in front:
            population[i].crowding_distance = 0.0

        # Calculate for each objective
        objectives = ["fitness_accuracy", "fitness_complexity"]

        for obj in objectives:
            # Sort by objective value
            front_sorted = sorted(front, key=lambda i: getattr(population[i], obj))

            # Boundary individuals get infinite distance
            population[front_sorted[0]].crowding_distance = float("inf")
            population[front_sorted[-1]].crowding_distance = float("inf")

            # Calculate range
            obj_values = [getattr(population[i], obj) for i in front_sorted]
            obj_range = max(obj_values) - min(obj_values)

            if obj_range == 0:
                continue  # All individuals have same objective value

            # Calculate crowding distance for intermediate individuals
            for j in range(1, len(front_sorted) - 1):
                distance = (
                    getattr(population[front_sorted[j + 1]], obj)
                    - getattr(population[front_sorted[j - 1]], obj)
                ) / obj_range
                population[front_sorted[j]].crowding_distance += distance

    def select_parents(
        self, population: List[Individual], num_parents: int
    ) -> List[Individual]:
        """
        Select parents using tournament selection based on Pareto dominance.

        Args:
            population: Population of individuals
            num_parents: Number of parents to select

        Returns:
            Selected parent individuals
        """
        parents = []

        for _ in range(num_parents):
            # Tournament selection
            tournament_size = min(3, len(population))
            tournament = random.sample(population, tournament_size)

            # Select best individual from tournament
            best = tournament[0]
            for ind in tournament[1:]:
                if self._is_better(ind, best):
                    best = ind

            parents.append(deepcopy(best))

        return parents

    def _is_better(self, ind1: Individual, ind2: Individual) -> bool:
        """
        Compare two individuals based on Pareto dominance and crowding distance.

        Args:
            ind1: First individual
            ind2: Second individual

        Returns:
            True if ind1 is better than ind2
        """
        # First compare by rank (lower is better)
        if ind1.rank < ind2.rank:
            return True
        elif ind1.rank > ind2.rank:
            return False

        # Same rank, compare by crowding distance (higher is better)
        return ind1.crowding_distance > ind2.crowding_distance


class AdvancedGeneticOperators:
    """Implements advanced crossover and mutation operators for genetic programming."""

    def __init__(self, variables: List[str]):
        """
        Initialize genetic operators.

        Args:
            variables: List of variable names
        """
        self.variables = variables
        self.operators = ["+", "-", "*", "/", "**"]
        self.functions = ["sin", "cos", "exp", "log", "sqrt", "abs"]
        self.constants = [0, 1, 2, 3, 0.5, -1, np.pi, np.e]

    def subtree_crossover(
        self, parent1: sp.Expr, parent2: sp.Expr
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Perform subtree crossover between two expressions.

        Args:
            parent1: First parent expression
            parent2: Second parent expression

        Returns:
            Tuple of two offspring expressions
        """
        try:
            # Convert to string representation for manipulation
            p1_str = str(parent1)
            p2_str = str(parent2)

            # Simple crossover: combine parts of expressions
            if random.random() < 0.5:
                # Additive crossover
                child1 = parent1 + parent2 * sp.Float(random.uniform(0.01, 0.1))
                child2 = parent2 + parent1 * sp.Float(random.uniform(0.01, 0.1))
            else:
                # Multiplicative crossover
                child1 = parent1 * (1 + parent2 * sp.Float(random.uniform(-0.1, 0.1)))
                child2 = parent2 * (1 + parent1 * sp.Float(random.uniform(-0.1, 0.1)))

            return child1, child2

        except Exception:
            # Return parents if crossover fails
            return parent1, parent2

    def uniform_crossover(
        self, parent1: sp.Expr, parent2: sp.Expr
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Perform uniform crossover by randomly selecting components.

        Args:
            parent1: First parent expression
            parent2: Second parent expression

        Returns:
            Tuple of two offspring expressions
        """
        try:
            # Extract subexpressions
            p1_args = list(parent1.args) if parent1.args else [parent1]
            p2_args = list(parent2.args) if parent2.args else [parent2]

            # Randomly combine arguments
            child1_args = []
            child2_args = []

            max_args = max(len(p1_args), len(p2_args))

            for i in range(max_args):
                if random.random() < 0.5:
                    if i < len(p1_args):
                        child1_args.append(p1_args[i])
                    if i < len(p2_args):
                        child2_args.append(p2_args[i])
                else:
                    if i < len(p2_args):
                        child1_args.append(p2_args[i])
                    if i < len(p1_args):
                        child2_args.append(p1_args[i])

            # Reconstruct expressions
            if child1_args:
                child1 = sum(child1_args) if len(child1_args) > 1 else child1_args[0]
            else:
                child1 = parent1

            if child2_args:
                child2 = sum(child2_args) if len(child2_args) > 1 else child2_args[0]
            else:
                child2 = parent2

            return child1, child2

        except Exception:
            return parent1, parent2

    def point_mutation(
        self, expression: sp.Expr, mutation_rate: float = 0.1
    ) -> sp.Expr:
        """
        Perform point mutation on an expression.

        Args:
            expression: Expression to mutate
            mutation_rate: Probability of mutation

        Returns:
            Mutated expression
        """
        if random.random() > mutation_rate:
            return expression

        try:
            # Different mutation strategies
            mutation_type = random.choice(
                ["add_term", "multiply_factor", "replace_constant", "add_function"]
            )

            if mutation_type == "add_term":
                # Add a small random term
                random_var = sp.Symbol(random.choice(self.variables))
                random_coeff = sp.Float(random.uniform(-0.1, 0.1))
                return expression + random_coeff * random_var

            elif mutation_type == "multiply_factor":
                # Multiply by a factor close to 1
                factor = sp.Float(random.uniform(0.9, 1.1))
                return expression * factor

            elif mutation_type == "replace_constant":
                # Replace constants with new values
                expr_str = str(expression)
                for const in self.constants:
                    if str(const) in expr_str:
                        new_const = random.choice(self.constants)
                        expr_str = expr_str.replace(str(const), str(new_const), 1)
                        break

                try:
                    return sp.sympify(expr_str)
                except:
                    return expression

            elif mutation_type == "add_function":
                # Wrap in a function
                func = random.choice(["sin", "cos", "exp", "log", "sqrt"])
                if func == "sin":
                    return sp.sin(expression * sp.Float(0.1))
                elif func == "cos":
                    return sp.cos(expression * sp.Float(0.1))
                elif func == "exp":
                    return sp.exp(expression * sp.Float(0.01))
                elif func == "log":
                    return sp.log(sp.Abs(expression) + sp.Float(1))
                elif func == "sqrt":
                    return sp.sqrt(sp.Abs(expression))

            return expression

        except Exception:
            return expression

    def subtree_mutation(
        self, expression: sp.Expr, mutation_rate: float = 0.1
    ) -> sp.Expr:
        """
        Perform subtree mutation by replacing a subtree with a new random subtree.

        Args:
            expression: Expression to mutate
            mutation_rate: Probability of mutation

        Returns:
            Mutated expression
        """
        if random.random() > mutation_rate:
            return expression

        try:
            # Generate a small random subtree
            random_subtree = self._generate_random_subtree(depth=2)

            # Simple replacement: add the random subtree
            if random.random() < 0.5:
                return expression + random_subtree * sp.Float(0.1)
            else:
                return expression * (1 + random_subtree * sp.Float(0.01))

        except Exception:
            return expression

    def _generate_random_subtree(self, depth: int = 2) -> sp.Expr:
        """Generate a random subtree of given depth."""
        if depth <= 0 or random.random() < 0.3:
            # Terminal node
            if random.random() < 0.7:
                return sp.Symbol(random.choice(self.variables))
            else:
                return sp.Float(random.choice(self.constants))

        # Non-terminal node
        if random.random() < 0.8:
            # Binary operator
            op = random.choice(self.operators)
            left = self._generate_random_subtree(depth - 1)
            right = self._generate_random_subtree(depth - 1)

            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "/":
                return left / (right + sp.Float(1e-8))
            elif op == "**":
                return left ** sp.Min(sp.Abs(right), 2)
        else:
            # Unary function
            func = random.choice(self.functions)
            operand = self._generate_random_subtree(depth - 1)

            if func == "sin":
                return sp.sin(operand)
            elif func == "cos":
                return sp.cos(operand)
            elif func == "exp":
                return sp.exp(sp.Min(sp.Abs(operand), 3))
            elif func == "log":
                return sp.log(sp.Abs(operand) + sp.Float(1e-8))
            elif func == "sqrt":
                return sp.sqrt(sp.Abs(operand))
            elif func == "abs":
                return sp.Abs(operand)

        # Fallback
        return sp.Symbol(random.choice(self.variables))


class MultiObjectiveGeneticProgramming:
    """
    Multi-objective genetic programming for symbolic regression.

    Optimizes both accuracy and complexity using Pareto front optimization.
    """

    def __init__(
        self,
        variables: List[str],
        population_size: int = 100,
        max_generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        max_expression_depth: int = 6,
        complexity_weight: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize multi-objective genetic programming.

        Args:
            variables: List of variable names
            population_size: Size of the population
            max_generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            max_expression_depth: Maximum depth of expressions
            complexity_weight: Weight for complexity in fitness calculation
            random_state: Random seed
        """
        self.variables = variables
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_expression_depth = max_expression_depth
        self.complexity_weight = complexity_weight
        self.random_state = random_state

        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)

        # Initialize components
        self.pareto_optimizer = ParetoFrontOptimizer()
        self.genetic_operators = AdvancedGeneticOperators(variables)

        # Evolution tracking
        self.generation_history = []
        self.pareto_fronts_history = []
        self.population = []

    def initialize_population(self) -> List[Individual]:
        """Initialize random population of expressions."""
        population = []

        for _ in range(self.population_size):
            expr = self.genetic_operators._generate_random_subtree(
                depth=random.randint(1, self.max_expression_depth)
            )
            individual = Individual(expression=expr)
            population.append(individual)

        return population

    def evaluate_fitness(
        self, individual: Individual, data: Dict[str, np.ndarray], target: np.ndarray
    ) -> Tuple[float, float]:
        """
        Evaluate fitness for accuracy and complexity objectives.

        Args:
            individual: Individual to evaluate
            data: Input data
            target: Target values

        Returns:
            Tuple of (accuracy_fitness, complexity_fitness)
        """
        try:
            # Evaluate accuracy
            expr_func = sp.lambdify(self.variables, individual.expression, "numpy")
            input_values = [data[var] for var in self.variables if var in data]

            if len(input_values) == 0:
                accuracy_fitness = 0.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predictions = expr_func(*input_values)

                # Handle scalar predictions
                if np.isscalar(predictions):
                    predictions = np.full_like(target, predictions)

                # Handle invalid predictions
                if not np.isfinite(predictions).all():
                    accuracy_fitness = 0.0
                else:
                    # Use R² as accuracy measure
                    accuracy_fitness = max(0.0, r2_score(target, predictions))

            # Evaluate complexity
            complexity_fitness = self._compute_complexity(individual.expression)

            return accuracy_fitness, complexity_fitness

        except Exception:
            return 0.0, float(
                "inf"
            )  # Poor accuracy, high complexity for invalid expressions

    def _compute_complexity(self, expression: sp.Expr) -> float:
        """Compute complexity measure for an expression."""
        expr_str = str(expression)

        # Count different complexity factors
        length_complexity = len(expr_str) / 100.0

        # Count operators
        operators = ["+", "-", "*", "/", "**", "sin", "cos", "exp", "log", "sqrt"]
        operator_count = sum(expr_str.count(op) for op in operators)
        operator_complexity = operator_count * 0.5

        # Count variables
        variable_count = len(expression.free_symbols)
        variable_complexity = variable_count * 0.2

        # Count nesting depth
        nesting_complexity = self._compute_nesting_depth(expression) * 0.3

        total_complexity = (
            length_complexity
            + operator_complexity
            + variable_complexity
            + nesting_complexity
        )

        return total_complexity

    def _compute_nesting_depth(self, expr: sp.Expr) -> int:
        """Compute nesting depth of an expression."""
        if expr.is_Atom:
            return 0

        if expr.args:
            return 1 + max(self._compute_nesting_depth(arg) for arg in expr.args)

        return 0

    def evolve(
        self, data: Dict[str, np.ndarray], target: np.ndarray
    ) -> List[Individual]:
        """
        Evolve population using multi-objective genetic programming.

        Args:
            data: Input data
            target: Target values

        Returns:
            Final population
        """
        # Initialize population
        self.population = self.initialize_population()

        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate fitness for all individuals
            for individual in self.population:
                accuracy, complexity = self.evaluate_fitness(individual, data, target)
                individual.fitness_accuracy = accuracy
                individual.fitness_complexity = complexity

            # Perform non-dominated sorting
            fronts = self.pareto_optimizer.fast_non_dominated_sort(self.population)

            # Calculate crowding distances
            for front in fronts:
                self.pareto_optimizer.calculate_crowding_distance(
                    self.population, front
                )

            # Store generation statistics
            gen_stats = self._compute_generation_statistics(generation, fronts)
            self.generation_history.append(gen_stats)

            # Store Pareto front
            if fronts:
                pareto_front = [self.population[i] for i in fronts[0]]
                self.pareto_fronts_history.append(pareto_front)

            # Create next generation
            if generation < self.max_generations - 1:
                self.population = self._create_next_generation(fronts)

        return self.population

    def _compute_generation_statistics(
        self, generation: int, fronts: List[List[int]]
    ) -> Dict[str, Any]:
        """Compute statistics for the current generation."""
        accuracies = [ind.fitness_accuracy for ind in self.population]
        complexities = [ind.fitness_complexity for ind in self.population]

        stats = {
            "generation": generation,
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "max_accuracy": np.max(accuracies),
            "mean_complexity": np.mean(complexities),
            "std_complexity": np.std(complexities),
            "min_complexity": np.min(complexities),
            "num_fronts": len(fronts),
            "pareto_front_size": len(fronts[0]) if fronts else 0,
        }

        return stats

    def _create_next_generation(self, fronts: List[List[int]]) -> List[Individual]:
        """Create next generation using NSGA-II selection."""
        next_population = []

        # Add individuals from fronts until population is full
        for front in fronts:
            if len(next_population) + len(front) <= self.population_size:
                # Add entire front
                for i in front:
                    next_population.append(deepcopy(self.population[i]))
            else:
                # Add part of front based on crowding distance
                remaining_slots = self.population_size - len(next_population)
                front_individuals = [
                    (i, self.population[i].crowding_distance) for i in front
                ]
                front_individuals.sort(
                    key=lambda x: x[1], reverse=True
                )  # Sort by crowding distance

                for i, _ in front_individuals[:remaining_slots]:
                    next_population.append(deepcopy(self.population[i]))
                break

        # Generate offspring through crossover and mutation
        offspring = []
        while len(offspring) < self.population_size:
            # Select parents
            parents = self.pareto_optimizer.select_parents(next_population, 2)

            if len(parents) >= 2:
                parent1, parent2 = parents[0], parents[1]

                # Crossover
                if random.random() < self.crossover_rate:
                    child1_expr, child2_expr = self.genetic_operators.subtree_crossover(
                        parent1.expression, parent2.expression
                    )
                else:
                    child1_expr, child2_expr = parent1.expression, parent2.expression

                # Mutation
                if random.random() < self.mutation_rate:
                    child1_expr = self.genetic_operators.point_mutation(
                        child1_expr, self.mutation_rate
                    )
                if random.random() < self.mutation_rate:
                    child2_expr = self.genetic_operators.point_mutation(
                        child2_expr, self.mutation_rate
                    )

                # Create offspring individuals
                child1 = Individual(expression=child1_expr)
                child2 = Individual(expression=child2_expr)

                offspring.extend([child1, child2])

        # Replace population with offspring
        return offspring[: self.population_size]

    def get_pareto_front(self) -> List[Individual]:
        """Get the current Pareto front."""
        if not self.population:
            return []

        fronts = self.pareto_optimizer.fast_non_dominated_sort(self.population)
        if fronts:
            return [self.population[i] for i in fronts[0]]
        return []

    def get_best_expressions(
        self, num_expressions: int = 5
    ) -> List[Tuple[sp.Expr, float, float]]:
        """
        Get best expressions from Pareto front.

        Args:
            num_expressions: Number of expressions to return

        Returns:
            List of (expression, accuracy, complexity) tuples
        """
        pareto_front = self.get_pareto_front()

        if not pareto_front:
            return []

        # Sort by accuracy (descending) and complexity (ascending)
        pareto_front.sort(
            key=lambda ind: (-ind.fitness_accuracy, ind.fitness_complexity)
        )

        results = []
        for ind in pareto_front[:num_expressions]:
            results.append(
                (ind.expression, ind.fitness_accuracy, ind.fitness_complexity)
            )

        return results

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics."""
        if not self.generation_history:
            return {}

        return {
            "total_generations": len(self.generation_history),
            "final_pareto_front_size": self.generation_history[-1]["pareto_front_size"],
            "max_accuracy_achieved": max(
                gen["max_accuracy"] for gen in self.generation_history
            ),
            "min_complexity_achieved": min(
                gen["min_complexity"] for gen in self.generation_history
            ),
            "convergence_generation": self._find_convergence_generation(),
            "generation_history": self.generation_history,
            "pareto_fronts_history": len(self.pareto_fronts_history),
        }

    def _find_convergence_generation(self, tolerance: float = 1e-6) -> int:
        """Find generation where evolution converged."""
        if len(self.generation_history) < 10:
            return len(self.generation_history)

        # Look for plateau in maximum accuracy
        for i in range(10, len(self.generation_history)):
            recent_accuracies = [
                gen["max_accuracy"] for gen in self.generation_history[i - 10 : i]
            ]
            if np.std(recent_accuracies) < tolerance:
                return i - 5

        return len(self.generation_history)

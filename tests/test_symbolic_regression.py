"""
Unit tests for Neural Symbolic Regression module.
"""

import pytest
import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock
import warnings

from ml_research_pipeline.physics_discovery.symbolic_regression import (
    NeuralSymbolicRegression, SymbolicExpression, ExpressionGenerator
)


class TestExpressionGenerator:
    """Test suite for ExpressionGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create an ExpressionGenerator for testing."""
        return ExpressionGenerator(
            variables=['x', 'y', 'z'],
            max_depth=3,
            operators=['+', '-', '*', '/', 'sin', 'cos']
        )
    
    def test_initialization(self):
        """Test proper initialization of ExpressionGenerator."""
        gen = ExpressionGenerator(
            variables=['a', 'b'],
            max_depth=4,
            operators=['+', '*']
        )
        
        assert gen.variables == ['a', 'b']
        assert gen.max_depth == 4
        assert gen.operators == ['+', '*']
        assert '+' in gen.binary_ops
        assert 'sin' in gen.unary_ops
    
    def test_generate_random_expression(self, generator):
        """Test random expression generation."""
        expr = generator.generate_random_expression()
        
        # Should be a valid sympy expression
        assert isinstance(expr, sp.Basic)
        
        # Should contain only allowed variables
        free_symbols = expr.free_symbols
        symbol_names = {str(sym) for sym in free_symbols}
        assert symbol_names.issubset(set(generator.variables + [str(c) for c in generator.constants]))
    
    def test_generate_expression_depth_limit(self, generator):
        """Test that expression generation respects depth limits."""
        # Generate many expressions and check they don't exceed depth
        for _ in range(10):
            expr = generator.generate_random_expression()
            # This is a heuristic check - very deep expressions tend to be very long
            assert len(str(expr)) < 200  # Reasonable length limit
    
    def test_mutate_expression(self, generator):
        """Test expression mutation."""
        original_expr = sp.Symbol('x') + sp.Symbol('y')
        
        # Test with high mutation rate
        mutated = generator.mutate_expression(original_expr, mutation_rate=1.0)
        assert isinstance(mutated, sp.Basic)
        
        # Test with zero mutation rate
        unmutated = generator.mutate_expression(original_expr, mutation_rate=0.0)
        # Should be similar to original (mutation might still occur due to randomness)
        assert isinstance(unmutated, sp.Basic)
    
    def test_crossover_expressions(self, generator):
        """Test expression crossover."""
        expr1 = sp.Symbol('x') + 1
        expr2 = sp.Symbol('y') * 2
        
        child1, child2 = generator.crossover_expressions(expr1, expr2)
        
        assert isinstance(child1, sp.Basic)
        assert isinstance(child2, sp.Basic)
        
        # Children should be different from parents (in most cases)
        # This is probabilistic, so we just check they're valid expressions


class TestSymbolicExpression:
    """Test suite for SymbolicExpression dataclass."""
    
    def test_symbolic_expression_creation(self):
        """Test creation of SymbolicExpression."""
        expr = sp.Symbol('x') + 1
        symbolic_expr = SymbolicExpression(
            expression=expr,
            fitness=0.8,
            complexity=5,
            r2_score=0.9,
            mse=0.1,
            variables=['x']
        )
        
        assert symbolic_expr.expression == expr
        assert symbolic_expr.fitness == 0.8
        assert symbolic_expr.complexity == 5
        assert symbolic_expr.r2_score == 0.9
        assert symbolic_expr.mse == 0.1
        assert symbolic_expr.variables == ['x']


class TestNeuralSymbolicRegression:
    """Test suite for NeuralSymbolicRegression class."""
    
    @pytest.fixture
    def symbolic_regression(self):
        """Create a NeuralSymbolicRegression instance for testing."""
        return NeuralSymbolicRegression(
            variables=['x', 'y'],
            population_size=20,
            max_generations=5,
            max_expression_depth=3,
            random_state=42
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 50
        
        x = np.random.uniform(-2, 2, n_samples)
        y = np.random.uniform(-2, 2, n_samples)
        
        # Simple known relationship: z = x^2 + y
        z = x**2 + y + np.random.normal(0, 0.1, n_samples)
        
        return {
            'x': x,
            'y': y
        }, z
    
    def test_initialization(self):
        """Test proper initialization of NeuralSymbolicRegression."""
        sr = NeuralSymbolicRegression(
            variables=['a', 'b', 'c'],
            population_size=50,
            max_generations=10,
            complexity_penalty=0.02,
            random_state=123
        )
        
        assert sr.variables == ['a', 'b', 'c']
        assert sr.population_size == 50
        assert sr.max_generations == 10
        assert sr.complexity_penalty == 0.02
        assert sr.random_state == 123
        assert isinstance(sr.expr_generator, ExpressionGenerator)
    
    def test_generate_expression_candidates(self, symbolic_regression):
        """Test generation of expression candidates."""
        candidates = symbolic_regression.generate_expression_candidates(n_candidates=10)
        
        assert len(candidates) == 10
        for candidate in candidates:
            assert isinstance(candidate, sp.Basic)
    
    def test_evaluate_expression_fitness_valid(self, symbolic_regression, sample_data):
        """Test fitness evaluation with valid expression."""
        flow_data, target = sample_data
        
        # Test with a simple expression
        expr = sp.Symbol('x') + sp.Symbol('y')
        fitness, r2, mse = symbolic_regression.evaluate_expression_fitness(
            expr, flow_data, target
        )
        
        assert isinstance(fitness, float)
        assert isinstance(r2, float)
        assert isinstance(mse, float)
        assert np.isfinite(fitness)
        assert np.isfinite(r2)
        assert np.isfinite(mse)
    
    def test_evaluate_expression_fitness_invalid(self, symbolic_regression, sample_data):
        """Test fitness evaluation with invalid expression."""
        flow_data, target = sample_data
        
        # Test with expression that might cause numerical issues
        expr = sp.log(sp.Symbol('x') - 10)  # Will be negative for our data range
        fitness, r2, mse = symbolic_regression.evaluate_expression_fitness(
            expr, flow_data, target
        )
        
        # Should handle gracefully
        assert isinstance(fitness, float)
        assert isinstance(r2, float)
        assert isinstance(mse, float)
    
    def test_compute_expression_complexity(self, symbolic_regression):
        """Test expression complexity computation."""
        # Simple expression
        simple_expr = sp.Symbol('x')
        simple_complexity = symbolic_regression._compute_expression_complexity(simple_expr)
        
        # Complex expression
        complex_expr = sp.sin(sp.Symbol('x')) + sp.cos(sp.Symbol('y')) * sp.exp(sp.Symbol('z'))
        complex_complexity = symbolic_regression._compute_expression_complexity(complex_expr)
        
        assert isinstance(simple_complexity, int)
        assert isinstance(complex_complexity, int)
        assert complex_complexity > simple_complexity
    
    def test_simplify_expression(self, symbolic_regression):
        """Test expression simplification."""
        # Expression that can be simplified
        expr = sp.Symbol('x') + sp.Symbol('x')
        simplified = symbolic_regression.simplify_expression(expr)
        
        assert isinstance(simplified, sp.Basic)
        # Should be simplified to 2*x or equivalent
        assert len(str(simplified)) <= len(str(expr))
    
    def test_tournament_selection(self, symbolic_regression):
        """Test tournament selection mechanism."""
        # Create mock population
        population = [
            SymbolicExpression(sp.Symbol('x'), 0.1, 1, 0.1, 1.0, ['x']),
            SymbolicExpression(sp.Symbol('y'), 0.8, 1, 0.8, 0.2, ['y']),
            SymbolicExpression(sp.Symbol('z'), 0.5, 1, 0.5, 0.5, ['z'])
        ]
        
        # Tournament should tend to select higher fitness individuals
        selected = symbolic_regression._tournament_selection(population, tournament_size=2)
        assert isinstance(selected, SymbolicExpression)
        assert selected in population
    
    def test_create_next_generation(self, symbolic_regression):
        """Test next generation creation."""
        # Create mock evaluated population
        population = [
            SymbolicExpression(sp.Symbol('x'), 0.8, 1, 0.8, 0.2, ['x']),
            SymbolicExpression(sp.Symbol('y'), 0.6, 1, 0.6, 0.4, ['y']),
            SymbolicExpression(sp.Symbol('z'), 0.4, 1, 0.4, 0.6, ['z']),
            SymbolicExpression(sp.Symbol('x') + 1, 0.2, 2, 0.2, 0.8, ['x'])
        ]
        
        next_gen = symbolic_regression._create_next_generation(population)
        
        assert len(next_gen) == symbolic_regression.population_size
        for expr in next_gen:
            assert isinstance(expr, sp.Basic)
    
    def test_discover_viscosity_law_simple(self, symbolic_regression, sample_data):
        """Test viscosity law discovery with simple data."""
        flow_data, target = sample_data
        
        # Use very small population and generations for fast testing
        symbolic_regression.population_size = 10
        symbolic_regression.max_generations = 3
        
        result = symbolic_regression.discover_viscosity_law(flow_data, target)
        
        assert isinstance(result, SymbolicExpression)
        assert isinstance(result.expression, sp.Basic)
        assert isinstance(result.fitness, float)
        assert isinstance(result.r2_score, float)
        assert isinstance(result.mse, float)
        assert len(symbolic_regression.generation_history) > 0
        assert len(symbolic_regression.best_expressions) > 0
    
    def test_get_evolution_statistics(self, symbolic_regression):
        """Test evolution statistics retrieval."""
        # Initially should be empty
        stats = symbolic_regression.get_evolution_statistics()
        assert stats == {}
        
        # Add some mock history
        symbolic_regression.generation_history = [
            {'generation': 0, 'best_fitness': 0.1, 'best_r2': 0.1, 'best_mse': 1.0},
            {'generation': 1, 'best_fitness': 0.5, 'best_r2': 0.5, 'best_mse': 0.5},
            {'generation': 2, 'best_fitness': 0.8, 'best_r2': 0.8, 'best_mse': 0.2}
        ]
        
        stats = symbolic_regression.get_evolution_statistics()
        
        assert 'total_generations' in stats
        assert 'final_best_fitness' in stats
        assert 'fitness_improvement' in stats
        assert stats['total_generations'] == 3
        assert stats['final_best_fitness'] == 0.8
        assert abs(stats['fitness_improvement'] - 0.7) < 1e-10
    
    def test_find_convergence_generation(self, symbolic_regression):
        """Test convergence detection."""
        # Test with no history
        convergence_gen = symbolic_regression._find_convergence_generation()
        assert convergence_gen == 0
        
        # Test with converged history
        symbolic_regression.generation_history = [
            {'best_fitness': 0.1 + i * 0.01} for i in range(20)
        ]  # Gradually improving
        
        convergence_gen = symbolic_regression._find_convergence_generation()
        assert isinstance(convergence_gen, int)
        assert convergence_gen >= 0
    
    def test_export_best_expression(self, symbolic_regression):
        """Test expression export in different formats."""
        # Add a mock best expression
        best_expr = SymbolicExpression(
            expression=sp.Symbol('x') + sp.Symbol('y'),
            fitness=0.8,
            complexity=2,
            r2_score=0.8,
            mse=0.2,
            variables=['x', 'y']
        )
        symbolic_regression.best_expressions = [best_expr]
        
        # Test different formats
        latex_export = symbolic_regression.export_best_expression(format='latex')
        python_export = symbolic_regression.export_best_expression(format='python')
        mathematica_export = symbolic_regression.export_best_expression(format='mathematica')
        
        assert isinstance(latex_export, str)
        assert isinstance(python_export, str)
        assert isinstance(mathematica_export, str)
        assert len(latex_export) > 0
        assert len(python_export) > 0
        assert len(mathematica_export) > 0
    
    def test_export_no_expressions(self, symbolic_regression):
        """Test export when no expressions have been discovered."""
        export = symbolic_regression.export_best_expression()
        assert "No expressions discovered yet" in export
    
    def test_integration_with_known_function(self):
        """Test symbolic regression on a known mathematical function."""
        # Create data from known function: y = 2*x + 1
        np.random.seed(42)
        x_data = np.linspace(-2, 2, 30)
        y_data = 2 * x_data + 1 + np.random.normal(0, 0.1, len(x_data))
        
        flow_data = {'x': x_data}
        
        # Use small parameters for fast testing
        sr = NeuralSymbolicRegression(
            variables=['x'],
            population_size=20,
            max_generations=10,
            max_expression_depth=3,
            random_state=42
        )
        
        result = sr.discover_viscosity_law(flow_data, y_data)
        
        # Should find a reasonable approximation
        assert result.r2_score > 0.5  # Should capture some of the relationship
        assert result.mse < 2.0  # Should have reasonable error
        
        # Check that evolution occurred
        assert len(sr.generation_history) > 0
        assert len(sr.best_expressions) > 0
        
        # Check statistics
        stats = sr.get_evolution_statistics()
        assert stats['total_generations'] > 0
        assert 'fitness_improvement' in stats
    
    def test_error_handling_invalid_data(self, symbolic_regression):
        """Test error handling with invalid data."""
        # Test with mismatched data sizes
        invalid_data = {
            'x': np.array([1, 2, 3]),
            'y': np.array([1, 2])  # Different size
        }
        target = np.array([1, 2, 3])
        
        # Should handle gracefully
        expr = sp.Symbol('x') + sp.Symbol('y')
        fitness, r2, mse = symbolic_regression.evaluate_expression_fitness(
            expr, invalid_data, target
        )
        
        # Should return poor fitness for invalid data
        assert fitness == -np.inf or np.isnan(fitness) or fitness < -1000
    
    def test_expression_with_no_variables(self, symbolic_regression, sample_data):
        """Test handling of expressions with no variables."""
        flow_data, target = sample_data
        
        # Constant expression
        expr = sp.Float(5.0)
        fitness, r2, mse = symbolic_regression.evaluate_expression_fitness(
            expr, flow_data, target
        )
        
        assert isinstance(fitness, float)
        assert isinstance(r2, float)
        assert isinstance(mse, float)
    
    def test_validate_discovered_physics(self, symbolic_regression, sample_data):
        """Test validation of discovered physics expressions (Requirement 5.3)."""
        flow_data, target = sample_data
        
        # Create a good expression for testing
        good_expr = SymbolicExpression(
            expression=sp.Symbol('x')**2 + sp.Symbol('y'),
            fitness=0.9,
            complexity=3,
            r2_score=0.85,
            mse=0.1,
            variables=['x', 'y']
        )
        
        # Test validation
        validation_metrics = symbolic_regression.validate_discovered_physics(
            good_expr, flow_data, target
        )
        
        # Check that all required metrics are present
        required_metrics = [
            'validation_score', 'r2_score', 'mse', 'mae', 'rmse',
            'physics_consistency', 'meets_threshold', 'expression_complexity'
        ]
        for metric in required_metrics:
            assert metric in validation_metrics
        
        # Check metric types
        assert isinstance(validation_metrics['validation_score'], float)
        assert isinstance(validation_metrics['meets_threshold'], bool)
        assert validation_metrics['validation_score'] >= 0.0
        assert validation_metrics['validation_score'] <= 1.0
    
    def test_validation_threshold_requirement(self, symbolic_regression):
        """Test that validation can achieve > 0.8 threshold (Requirement 5.3)."""
        # Create perfect synthetic data
        np.random.seed(42)
        x = np.linspace(-1, 1, 30)
        y = np.linspace(-1, 1, 30)
        # Perfect relationship: z = x + y
        z = x + y
        
        flow_data = {'x': x, 'y': y}
        
        # Create perfect expression
        perfect_expr = SymbolicExpression(
            expression=sp.Symbol('x') + sp.Symbol('y'),
            fitness=1.0,
            complexity=2,
            r2_score=1.0,
            mse=0.0,
            variables=['x', 'y']
        )
        
        # Validate
        validation_metrics = symbolic_regression.validate_discovered_physics(
            perfect_expr, flow_data, z
        )
        
        # Should meet the 0.8 threshold requirement (with perfect R² = 1.0)
        assert validation_metrics['validation_score'] >= 0.8
        assert validation_metrics['meets_threshold'] is True
        assert validation_metrics['r2_score'] == 1.0
    
    def test_discover_viscosity_law_with_validation_threshold(self, symbolic_regression):
        """Test discovery with validation threshold tracking."""
        # Create simple test data
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 1, 1, 1, 1])
        z = x + 1  # Simple linear relationship
        
        flow_data = {'x': x, 'y': y}
        
        # Use small parameters for fast testing
        symbolic_regression.population_size = 10
        symbolic_regression.max_generations = 5
        
        # Test discovery with validation threshold
        result = symbolic_regression.discover_viscosity_law(
            flow_data, z, validation_threshold=0.5
        )
        
        # Check that validation scores were tracked
        assert hasattr(symbolic_regression, 'validation_scores')
        assert len(symbolic_regression.validation_scores) > 0
        
        # Check generation history includes validation scores
        if symbolic_regression.generation_history:
            assert 'validation_score' in symbolic_regression.generation_history[0]
    
    def test_generate_physics_hypothesis(self, symbolic_regression):
        """Test natural language hypothesis generation (Requirement 5.5)."""
        # Create test expression
        test_expr = SymbolicExpression(
            expression=sp.Symbol('x')**2 + sp.Symbol('y'),
            fitness=0.8,
            complexity=3,
            r2_score=0.85,
            mse=0.1,
            variables=['x', 'y']
        )
        
        # Generate hypothesis
        hypothesis = symbolic_regression.generate_physics_hypothesis(test_expr)
        
        # Check that hypothesis is a non-empty string
        assert isinstance(hypothesis, str)
        assert len(hypothesis) > 0
        
        # Check that it contains key information
        assert 'relationship' in hypothesis.lower()
        assert 'x' in hypothesis
        assert 'y' in hypothesis
        
        # Test with variable descriptions
        var_descriptions = {'x': 'velocity', 'y': 'pressure'}
        hypothesis_with_desc = symbolic_regression.generate_physics_hypothesis(
            test_expr, var_descriptions
        )
        
        assert isinstance(hypothesis_with_desc, str)
        assert len(hypothesis_with_desc) > 0
    
    def test_validation_error_handling(self, symbolic_regression):
        """Test validation error handling with invalid data."""
        # Create invalid expression
        invalid_expr = SymbolicExpression(
            expression=sp.log(sp.Symbol('x') - 100),  # Will cause issues
            fitness=0.0,
            complexity=5,
            r2_score=0.0,
            mse=np.inf,
            variables=['x']
        )
        
        # Test with mismatched data
        invalid_data = {'x': np.array([1, 2, 3])}
        target = np.array([1, 2])  # Different size
        
        validation_metrics = symbolic_regression.validate_discovered_physics(
            invalid_expr, invalid_data, target
        )
        
        # Should handle gracefully
        assert validation_metrics['validation_score'] == 0.0
        assert validation_metrics['meets_threshold'] is False
    
    def test_interpretable_expression_discovery(self):
        """Test that system discovers interpretable expressions (Requirement 5.2)."""
        # Create data with known interpretable relationship
        np.random.seed(42)
        x = np.linspace(0, 2, 20)
        y = np.linspace(0, 1, 20)
        # Interpretable relationship: z = 2*x + y + 1
        z = 2*x + y + 1 + np.random.normal(0, 0.05, 20)
        
        flow_data = {'x': x, 'y': y}
        
        # Use focused parameters for better discovery
        sr = NeuralSymbolicRegression(
            variables=['x', 'y'],
            population_size=30,
            max_generations=15,
            max_expression_depth=4,
            complexity_penalty=0.02,  # Favor simpler expressions
            random_state=42
        )
        
        result = sr.discover_viscosity_law(flow_data, z)
        
        # Should discover an interpretable expression
        assert isinstance(result.expression, sp.Basic)
        assert result.complexity < 100  # Should be reasonably bounded
        
        # Should have reasonable performance
        assert result.r2_score > 0.3  # Should capture some relationship
        
        # Expression should be interpretable (contain recognizable operations)
        expr_str = str(result.expression)
        # Should contain variables
        assert any(var in expr_str for var in ['x', 'y'])


if __name__ == '__main__':
    pytest.main([__file__])
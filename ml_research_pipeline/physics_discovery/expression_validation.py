"""
Expression Validation and Regularization Module

This module implements dimensional analysis validation, physics consistency checks,
and regularization terms for symbolic expressions in physics discovery.
"""

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import sympy as sp


class PhysicalDimension(Enum):
    """Enumeration of fundamental physical dimensions."""

    LENGTH = "L"
    MASS = "M"
    TIME = "T"
    ELECTRIC_CURRENT = "I"
    TEMPERATURE = "K"
    AMOUNT_OF_SUBSTANCE = "N"
    LUMINOUS_INTENSITY = "J"
    DIMENSIONLESS = "1"


@dataclass
class DimensionalVector:
    """Represents the dimensional vector of a physical quantity."""

    length: float = 0.0  # L
    mass: float = 0.0  # M
    time: float = 0.0  # T
    current: float = 0.0  # I
    temperature: float = 0.0  # K
    substance: float = 0.0  # N
    luminosity: float = 0.0  # J

    def __add__(self, other: "DimensionalVector") -> "DimensionalVector":
        """Add dimensional vectors (for multiplication of quantities)."""
        return DimensionalVector(
            length=self.length + other.length,
            mass=self.mass + other.mass,
            time=self.time + other.time,
            current=self.current + other.current,
            temperature=self.temperature + other.temperature,
            substance=self.substance + other.substance,
            luminosity=self.luminosity + other.luminosity,
        )

    def __sub__(self, other: "DimensionalVector") -> "DimensionalVector":
        """Subtract dimensional vectors (for division of quantities)."""
        return DimensionalVector(
            length=self.length - other.length,
            mass=self.mass - other.mass,
            time=self.time - other.time,
            current=self.current - other.current,
            temperature=self.temperature - other.temperature,
            substance=self.substance - other.substance,
            luminosity=self.luminosity - other.luminosity,
        )

    def __mul__(self, scalar: float) -> "DimensionalVector":
        """Multiply dimensional vector by scalar (for powers)."""
        return DimensionalVector(
            length=self.length * scalar,
            mass=self.mass * scalar,
            time=self.time * scalar,
            current=self.current * scalar,
            temperature=self.temperature * scalar,
            substance=self.substance * scalar,
            luminosity=self.luminosity * scalar,
        )

    def __eq__(self, other: "DimensionalVector") -> bool:
        """Check dimensional equality."""
        tolerance = 1e-10
        return (
            abs(self.length - other.length) < tolerance
            and abs(self.mass - other.mass) < tolerance
            and abs(self.time - other.time) < tolerance
            and abs(self.current - other.current) < tolerance
            and abs(self.temperature - other.temperature) < tolerance
            and abs(self.substance - other.substance) < tolerance
            and abs(self.luminosity - other.luminosity) < tolerance
        )

    def is_dimensionless(self) -> bool:
        """Check if the quantity is dimensionless."""
        return self == DimensionalVector()

    def to_string(self) -> str:
        """Convert to string representation."""
        components = []
        dims = [
            ("L", self.length),
            ("M", self.mass),
            ("T", self.time),
            ("I", self.current),
            ("K", self.temperature),
            ("N", self.substance),
            ("J", self.luminosity),
        ]

        for symbol, power in dims:
            if abs(power) > 1e-10:
                if abs(power - 1.0) < 1e-10:
                    components.append(symbol)
                else:
                    components.append(f"{symbol}^{power}")

        return " ".join(components) if components else "1"


class DimensionalAnalyzer:
    """Performs dimensional analysis on symbolic expressions."""

    def __init__(self):
        """Initialize dimensional analyzer with common physical quantities."""
        # Define dimensions for common physical quantities
        self.known_dimensions = {
            # Basic quantities
            "length": DimensionalVector(length=1),
            "mass": DimensionalVector(mass=1),
            "time": DimensionalVector(time=1),
            "current": DimensionalVector(current=1),
            "temperature": DimensionalVector(temperature=1),
            # Derived quantities
            "velocity": DimensionalVector(length=1, time=-1),
            "acceleration": DimensionalVector(length=1, time=-2),
            "force": DimensionalVector(mass=1, length=1, time=-2),
            "energy": DimensionalVector(mass=1, length=2, time=-2),
            "power": DimensionalVector(mass=1, length=2, time=-3),
            "pressure": DimensionalVector(mass=1, length=-1, time=-2),
            "density": DimensionalVector(mass=1, length=-3),
            "viscosity": DimensionalVector(mass=1, length=-1, time=-1),
            "frequency": DimensionalVector(time=-1),
            "angular_velocity": DimensionalVector(time=-1),
            # Common variables
            "x": DimensionalVector(length=1),
            "y": DimensionalVector(length=1),
            "z": DimensionalVector(length=1),
            "t": DimensionalVector(time=1),
            "v": DimensionalVector(length=1, time=-1),
            "a": DimensionalVector(length=1, time=-2),
            "F": DimensionalVector(mass=1, length=1, time=-2),
            "E": DimensionalVector(mass=1, length=2, time=-2),
            "P": DimensionalVector(mass=1, length=2, time=-3),
            "rho": DimensionalVector(mass=1, length=-3),
            "mu": DimensionalVector(mass=1, length=-1, time=-1),
            "omega": DimensionalVector(time=-1),
            "f": DimensionalVector(time=-1),
            # Dimensionless constants
            "pi": DimensionalVector(),
            "e": DimensionalVector(),
        }

        # Variable dimension registry
        self.variable_dimensions = {}

    def set_variable_dimension(self, variable: str, dimension: DimensionalVector):
        """Set the dimension of a variable."""
        self.variable_dimensions[variable] = dimension

    def get_variable_dimension(self, variable: str) -> DimensionalVector:
        """Get the dimension of a variable."""
        if variable in self.variable_dimensions:
            return self.variable_dimensions[variable]
        elif variable in self.known_dimensions:
            return self.known_dimensions[variable]
        else:
            # Default to dimensionless for unknown variables
            return DimensionalVector()

    def analyze_expression_dimensions(self, expr: sp.Expr) -> DimensionalVector:
        """
        Analyze the dimensions of a symbolic expression.

        Args:
            expr: Sympy expression to analyze

        Returns:
            Dimensional vector of the expression
        """
        try:
            return self._analyze_expr_recursive(expr)
        except Exception as e:
            warnings.warn(f"Dimensional analysis failed for {expr}: {e}")
            return DimensionalVector()  # Return dimensionless on failure

    def _analyze_expr_recursive(self, expr: sp.Expr) -> DimensionalVector:
        """Recursively analyze expression dimensions."""
        if expr.is_Symbol:
            return self.get_variable_dimension(str(expr))

        elif expr.is_Number:
            return DimensionalVector()  # Numbers are dimensionless

        elif expr.is_Add:
            # All terms in addition must have same dimensions
            dimensions = [self._analyze_expr_recursive(arg) for arg in expr.args]
            if dimensions:
                first_dim = dimensions[0]
                for dim in dimensions[1:]:
                    if not (first_dim == dim):
                        raise ValueError(
                            f"Dimensional inconsistency in addition: {first_dim.to_string()} + {dim.to_string()}"
                        )
                return first_dim
            return DimensionalVector()

        elif expr.is_Mul:
            # Multiply dimensions
            result = DimensionalVector()
            for arg in expr.args:
                arg_dim = self._analyze_expr_recursive(arg)
                result = result + arg_dim
            return result

        elif expr.is_Pow:
            base_dim = self._analyze_expr_recursive(expr.base)
            exponent = expr.exp

            # Exponent must be dimensionless
            if not exponent.is_Number:
                exp_dim = self._analyze_expr_recursive(exponent)
                if not exp_dim.is_dimensionless():
                    raise ValueError(
                        f"Non-dimensionless exponent: {exp_dim.to_string()}"
                    )

            # Power operation
            if exponent.is_Number:
                return base_dim * float(exponent)
            else:
                # For non-numeric exponents, assume dimensionless result
                return DimensionalVector()

        elif expr.func in [sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh]:
            # Trigonometric functions require dimensionless arguments
            arg_dim = self._analyze_expr_recursive(expr.args[0])
            if not arg_dim.is_dimensionless():
                raise ValueError(
                    f"Non-dimensionless argument to {expr.func}: {arg_dim.to_string()}"
                )
            return DimensionalVector()  # Trig functions return dimensionless

        elif expr.func in [sp.exp, sp.log, sp.sqrt]:
            if expr.func == sp.exp:
                # Exponential requires dimensionless argument
                arg_dim = self._analyze_expr_recursive(expr.args[0])
                if not arg_dim.is_dimensionless():
                    raise ValueError(
                        f"Non-dimensionless argument to exp: {arg_dim.to_string()}"
                    )
                return DimensionalVector()

            elif expr.func == sp.log:
                # Logarithm requires dimensionless argument
                arg_dim = self._analyze_expr_recursive(expr.args[0])
                if not arg_dim.is_dimensionless():
                    raise ValueError(
                        f"Non-dimensionless argument to log: {arg_dim.to_string()}"
                    )
                return DimensionalVector()

            elif expr.func == sp.sqrt:
                # Square root
                arg_dim = self._analyze_expr_recursive(expr.args[0])
                return arg_dim * 0.5

        elif expr.func == sp.Abs:
            # Absolute value preserves dimensions
            return self._analyze_expr_recursive(expr.args[0])

        else:
            # Unknown function - assume dimensionless
            return DimensionalVector()

    def validate_dimensional_consistency(
        self, expr: sp.Expr, target_dimension: DimensionalVector
    ) -> bool:
        """
        Validate that an expression has the expected dimensions.

        Args:
            expr: Expression to validate
            target_dimension: Expected dimensional vector

        Returns:
            True if dimensions are consistent
        """
        try:
            expr_dimension = self.analyze_expression_dimensions(expr)
            return expr_dimension == target_dimension
        except Exception:
            return False


class PhysicsConsistencyChecker:
    """Checks physics consistency of symbolic expressions."""

    def __init__(self):
        """Initialize physics consistency checker."""
        self.conservation_laws = [
            self._check_energy_conservation,
            self._check_momentum_conservation,
            self._check_mass_conservation,
        ]

        self.symmetry_checks = [
            self._check_translational_symmetry,
            self._check_rotational_symmetry,
            self._check_time_reversal_symmetry,
        ]

    def check_conservation_laws(
        self, expr: sp.Expr, variables: List[str]
    ) -> Dict[str, bool]:
        """
        Check if expression respects fundamental conservation laws.

        Args:
            expr: Expression to check
            variables: List of variable names

        Returns:
            Dictionary of conservation law compliance
        """
        results = {}

        try:
            # Energy conservation (for energy-related expressions)
            results["energy_conservation"] = self._check_energy_conservation(
                expr, variables
            )

            # Momentum conservation (for momentum-related expressions)
            results["momentum_conservation"] = self._check_momentum_conservation(
                expr, variables
            )

            # Mass conservation (for mass-related expressions)
            results["mass_conservation"] = self._check_mass_conservation(
                expr, variables
            )

        except Exception as e:
            warnings.warn(f"Conservation law check failed: {e}")
            results = {
                law: False
                for law in [
                    "energy_conservation",
                    "momentum_conservation",
                    "mass_conservation",
                ]
            }

        return results

    def check_symmetries(self, expr: sp.Expr, variables: List[str]) -> Dict[str, bool]:
        """
        Check if expression respects fundamental symmetries.

        Args:
            expr: Expression to check
            variables: List of variable names

        Returns:
            Dictionary of symmetry compliance
        """
        results = {}

        try:
            # Translational symmetry
            results["translational_symmetry"] = self._check_translational_symmetry(
                expr, variables
            )

            # Rotational symmetry
            results["rotational_symmetry"] = self._check_rotational_symmetry(
                expr, variables
            )

            # Time reversal symmetry
            results["time_reversal_symmetry"] = self._check_time_reversal_symmetry(
                expr, variables
            )

        except Exception as e:
            warnings.warn(f"Symmetry check failed: {e}")
            results = {
                sym: False
                for sym in [
                    "translational_symmetry",
                    "rotational_symmetry",
                    "time_reversal_symmetry",
                ]
            }

        return results

    def _check_energy_conservation(self, expr: sp.Expr, variables: List[str]) -> bool:
        """Check energy conservation principles."""
        # Simple heuristic: energy-related expressions should not create energy from nothing
        # Look for terms that could represent energy sources/sinks

        # Check if expression contains only conservative operations
        expr_str = str(expr)

        # Energy should be conserved in isolated systems
        # This is a simplified check - in practice, would need more sophisticated analysis

        # Check for problematic patterns that might violate energy conservation
        problematic_patterns = [
            r"exp\([^)]*t[^)]*\)",  # Exponential growth in time
            r"\*\*[^0-9\s\+\-\*/\(\)]*t",  # Time-dependent exponents
        ]

        for pattern in problematic_patterns:
            if re.search(pattern, expr_str):
                return False

        return True

    def _check_momentum_conservation(self, expr: sp.Expr, variables: List[str]) -> bool:
        """Check momentum conservation principles."""
        # Momentum should be conserved in isolated systems
        # Check for velocity-dependent terms that respect Newton's laws

        expr_str = str(expr)

        # Look for velocity variables
        velocity_vars = [
            var for var in variables if var in ["v", "vx", "vy", "vz", "velocity"]
        ]

        if velocity_vars:
            # Check if momentum-related expressions follow expected patterns
            # This is a simplified heuristic check

            # Momentum should be linear in velocity for simple cases
            for var in velocity_vars:
                # Check if velocity appears in non-linear ways that might violate conservation
                if f"{var}**" in expr_str and "**2" not in expr_str:
                    # Non-quadratic velocity terms might be problematic
                    return False

        return True

    def _check_mass_conservation(self, expr: sp.Expr, variables: List[str]) -> bool:
        """Check mass conservation principles."""
        # Mass should be conserved in non-relativistic mechanics

        expr_str = str(expr)

        # Look for mass variables
        mass_vars = [var for var in variables if var in ["m", "mass", "rho", "density"]]

        if mass_vars:
            # Check for problematic mass creation/destruction patterns
            problematic_patterns = [
                r"exp\([^)]*m[^)]*\)",  # Exponential mass terms
                r"m\*\*[^12\s]",  # Non-linear mass terms (except m^2 for energy)
            ]

            for pattern in problematic_patterns:
                if re.search(pattern, expr_str):
                    return False

        return True

    def _check_translational_symmetry(
        self, expr: sp.Expr, variables: List[str]
    ) -> bool:
        """Check translational symmetry."""
        # Physics laws should be invariant under spatial translations

        # Look for spatial coordinates
        spatial_vars = [var for var in variables if var in ["x", "y", "z", "r"]]

        if not spatial_vars:
            return True  # No spatial dependence

        # Check if expression depends only on relative positions or derivatives
        # This is a simplified check

        expr_str = str(expr)

        # Absolute position dependence might violate translational symmetry
        # unless it's in a derivative or relative context
        for var in spatial_vars:
            # Look for bare position variables (not in derivatives)
            if var in expr_str:
                # This is a very simplified check
                # In practice, would need to analyze the mathematical structure more carefully
                pass

        return True  # Default to true for this simplified implementation

    def _check_rotational_symmetry(self, expr: sp.Expr, variables: List[str]) -> bool:
        """Check rotational symmetry."""
        # Physics laws should be invariant under rotations (in isotropic systems)

        # Look for directional variables
        directional_vars = [
            var for var in variables if var in ["x", "y", "z", "theta", "phi"]
        ]

        if len(directional_vars) <= 1:
            return True  # No rotational structure to check

        # Check if expression treats all directions equally
        # This is a simplified heuristic

        expr_str = str(expr)

        # Look for expressions that might break rotational symmetry
        # In a rotationally symmetric system, x, y, z should appear symmetrically

        coord_counts = {}
        for var in ["x", "y", "z"]:
            if var in variables:
                coord_counts[var] = expr_str.count(var)

        if len(coord_counts) > 1:
            # Check if coordinates appear with similar frequency
            counts = list(coord_counts.values())
            if max(counts) - min(counts) > 2:  # Allow some asymmetry
                return False

        return True

    def _check_time_reversal_symmetry(
        self, expr: sp.Expr, variables: List[str]
    ) -> bool:
        """Check time reversal symmetry."""
        # Many physics laws are invariant under time reversal

        # Look for time variable
        if "t" not in variables and "time" not in variables:
            return True  # No time dependence

        # Check if expression is even in time derivatives
        # This is a simplified check

        expr_str = str(expr)

        # Look for odd powers of time that might break time reversal symmetry
        # (except for dissipative systems)

        # First-order time derivatives are okay (velocity)
        # Third-order and higher odd derivatives might be problematic

        if "t**3" in expr_str or "t**5" in expr_str:
            return False

        return True


class ExpressionRegularizer:
    """Applies regularization to symbolic expressions."""

    def __init__(
        self,
        complexity_weight: float = 1.0,
        dimensional_weight: float = 2.0,
        physics_weight: float = 1.5,
    ):
        """
        Initialize expression regularizer.

        Args:
            complexity_weight: Weight for complexity penalty
            dimensional_weight: Weight for dimensional consistency penalty
            physics_weight: Weight for physics consistency penalty
        """
        self.complexity_weight = complexity_weight
        self.dimensional_weight = dimensional_weight
        self.physics_weight = physics_weight

        self.dimensional_analyzer = DimensionalAnalyzer()
        self.physics_checker = PhysicsConsistencyChecker()

    def compute_complexity_penalty(self, expr: sp.Expr) -> float:
        """
        Compute complexity penalty for an expression.

        Args:
            expr: Expression to analyze

        Returns:
            Complexity penalty score
        """
        # Count different types of complexity
        expr_str = str(expr)

        # Basic complexity measures
        length_penalty = len(expr_str) / 100.0  # Normalize by typical length

        # Count operators
        operators = ["+", "-", "*", "/", "**", "sin", "cos", "exp", "log", "sqrt"]
        operator_count = sum(expr_str.count(op) for op in operators)
        operator_penalty = operator_count * 0.1

        # Count variables
        variables = len(expr.free_symbols)
        variable_penalty = variables * 0.05

        # Count nested functions
        nesting_level = self._compute_nesting_level(expr)
        nesting_penalty = nesting_level * 0.2

        # Total complexity penalty
        complexity = (
            length_penalty + operator_penalty + variable_penalty + nesting_penalty
        )

        return complexity * self.complexity_weight

    def compute_dimensional_penalty(
        self,
        expr: sp.Expr,
        target_dimension: DimensionalVector,
        variable_dimensions: Dict[str, DimensionalVector],
    ) -> float:
        """
        Compute dimensional consistency penalty.

        Args:
            expr: Expression to analyze
            target_dimension: Expected dimension
            variable_dimensions: Dimensions of variables

        Returns:
            Dimensional penalty score
        """
        # Set variable dimensions
        for var, dim in variable_dimensions.items():
            self.dimensional_analyzer.set_variable_dimension(var, dim)

        try:
            # Analyze expression dimensions
            expr_dimension = self.dimensional_analyzer.analyze_expression_dimensions(
                expr
            )

            # Compute dimensional mismatch
            mismatch = 0.0

            # Compare each dimension component
            target_components = [
                target_dimension.length,
                target_dimension.mass,
                target_dimension.time,
                target_dimension.current,
                target_dimension.temperature,
                target_dimension.substance,
                target_dimension.luminosity,
            ]

            expr_components = [
                expr_dimension.length,
                expr_dimension.mass,
                expr_dimension.time,
                expr_dimension.current,
                expr_dimension.temperature,
                expr_dimension.substance,
                expr_dimension.luminosity,
            ]

            for target_comp, expr_comp in zip(target_components, expr_components):
                mismatch += abs(target_comp - expr_comp)

            return mismatch * self.dimensional_weight

        except Exception:
            # Large penalty for expressions that can't be analyzed
            return 10.0 * self.dimensional_weight

    def compute_physics_penalty(self, expr: sp.Expr, variables: List[str]) -> float:
        """
        Compute physics consistency penalty.

        Args:
            expr: Expression to analyze
            variables: List of variable names

        Returns:
            Physics penalty score
        """
        penalty = 0.0

        # Check conservation laws
        conservation_results = self.physics_checker.check_conservation_laws(
            expr, variables
        )
        for law, passed in conservation_results.items():
            if not passed:
                penalty += 1.0

        # Check symmetries
        symmetry_results = self.physics_checker.check_symmetries(expr, variables)
        for symmetry, passed in symmetry_results.items():
            if not passed:
                penalty += 0.5  # Symmetry violations are less severe

        return penalty * self.physics_weight

    def compute_total_regularization(
        self,
        expr: sp.Expr,
        target_dimension: DimensionalVector,
        variable_dimensions: Dict[str, DimensionalVector],
        variables: List[str],
    ) -> Dict[str, float]:
        """
        Compute total regularization penalty for an expression.

        Args:
            expr: Expression to analyze
            target_dimension: Expected dimension
            variable_dimensions: Dimensions of variables
            variables: List of variable names

        Returns:
            Dictionary of penalty components and total
        """
        # Compute individual penalties
        complexity_penalty = self.compute_complexity_penalty(expr)
        dimensional_penalty = self.compute_dimensional_penalty(
            expr, target_dimension, variable_dimensions
        )
        physics_penalty = self.compute_physics_penalty(expr, variables)

        # Total penalty
        total_penalty = complexity_penalty + dimensional_penalty + physics_penalty

        return {
            "complexity_penalty": complexity_penalty,
            "dimensional_penalty": dimensional_penalty,
            "physics_penalty": physics_penalty,
            "total_penalty": total_penalty,
        }

    def _compute_nesting_level(self, expr: sp.Expr) -> int:
        """Compute the maximum nesting level of functions in an expression."""
        if expr.is_Atom:
            return 0

        if expr.args:
            return 1 + max(self._compute_nesting_level(arg) for arg in expr.args)

        return 0


class ExpressionValidator:
    """Main class for validating and regularizing symbolic expressions."""

    def __init__(
        self,
        complexity_weight: float = 1.0,
        dimensional_weight: float = 2.0,
        physics_weight: float = 1.5,
    ):
        """
        Initialize expression validator.

        Args:
            complexity_weight: Weight for complexity penalty
            dimensional_weight: Weight for dimensional consistency penalty
            physics_weight: Weight for physics consistency penalty
        """
        self.regularizer = ExpressionRegularizer(
            complexity_weight, dimensional_weight, physics_weight
        )
        self.dimensional_analyzer = DimensionalAnalyzer()
        self.physics_checker = PhysicsConsistencyChecker()

    def validate_expression(
        self,
        expr: sp.Expr,
        target_dimension: DimensionalVector,
        variable_dimensions: Dict[str, DimensionalVector],
        variables: List[str],
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of a symbolic expression.

        Args:
            expr: Expression to validate
            target_dimension: Expected dimension
            variable_dimensions: Dimensions of variables
            variables: List of variable names

        Returns:
            Validation results dictionary
        """
        results = {}

        # Dimensional analysis
        try:
            for var, dim in variable_dimensions.items():
                self.dimensional_analyzer.set_variable_dimension(var, dim)

            expr_dimension = self.dimensional_analyzer.analyze_expression_dimensions(
                expr
            )
            dimensional_consistent = expr_dimension == target_dimension

            results["dimensional_analysis"] = {
                "expression_dimension": expr_dimension.to_string(),
                "target_dimension": target_dimension.to_string(),
                "consistent": dimensional_consistent,
            }
        except Exception as e:
            results["dimensional_analysis"] = {"error": str(e), "consistent": False}

        # Physics consistency
        conservation_results = self.physics_checker.check_conservation_laws(
            expr, variables
        )
        symmetry_results = self.physics_checker.check_symmetries(expr, variables)

        results["physics_consistency"] = {
            "conservation_laws": conservation_results,
            "symmetries": symmetry_results,
        }

        # Regularization penalties
        regularization = self.regularizer.compute_total_regularization(
            expr, target_dimension, variable_dimensions, variables
        )
        results["regularization"] = regularization

        # Overall validation score
        dimensional_score = (
            1.0 if results["dimensional_analysis"]["consistent"] else 0.0
        )
        conservation_score = sum(conservation_results.values()) / len(
            conservation_results
        )
        symmetry_score = sum(symmetry_results.values()) / len(symmetry_results)

        # Combine scores (higher is better)
        overall_score = (
            0.4 * dimensional_score
            + 0.3 * conservation_score
            + 0.2 * symmetry_score
            + 0.1 * max(0, 1.0 - regularization["total_penalty"] / 10.0)
        )

        results["overall_validation_score"] = overall_score
        results["expression_string"] = str(expr)

        return results

    def rank_expressions_by_validity(
        self,
        expressions: List[sp.Expr],
        target_dimension: DimensionalVector,
        variable_dimensions: Dict[str, DimensionalVector],
        variables: List[str],
    ) -> List[Tuple[sp.Expr, Dict[str, Any]]]:
        """
        Rank expressions by their validation scores.

        Args:
            expressions: List of expressions to rank
            target_dimension: Expected dimension
            variable_dimensions: Dimensions of variables
            variables: List of variable names

        Returns:
            List of (expression, validation_results) tuples, sorted by score
        """
        results = []

        for expr in expressions:
            validation = self.validate_expression(
                expr, target_dimension, variable_dimensions, variables
            )
            results.append((expr, validation))

        # Sort by overall validation score (descending)
        results.sort(key=lambda x: x[1]["overall_validation_score"], reverse=True)

        return results

"""
Comprehensive evaluation framework for PINN methods.

This module provides a complete evaluation framework including:
- Benchmark suite with multiple problem types (cavity, channel, cylinder, thermal)
- Evaluation metrics and statistical analysis
- Automated method comparison system
"""

from .advanced_validation_framework import (
    AdvancedValidationFramework,
    ValidationConfig,
    ValidationReport,
    ValidationScore,
)
from .benchmark_suite import (
    BenchmarkConfig,
    BenchmarkResult,
    CavityFlowBenchmark,
    ChannelFlowBenchmark,
    CylinderFlowBenchmark,
    PINNBenchmarkSuite,
    ThermalConvectionBenchmark,
)
from .cross_validation import CrossValidationFramework, CVResult, NestedCVResult
from .method_comparison import (
    ComparisonResult,
    MethodComparison,
    MethodComparisonConfig,
)
from .metrics import (
    EvaluationMetrics,
    MetricResult,
    StatisticalAnalysis,
    StatisticalTest,
)
from .physics_consistency import (
    ConservationLaw,
    PhysicsConsistencyChecker,
    PhysicsConsistencyResult,
    SymmetryType,
)
from .statistical_validator import (
    BootstrapResult,
    StatisticalResult,
    StatisticalValidator,
)
from .uncertainty_quantifier import (
    BayesianUncertaintyQuantifier,
    MonteCarloDropout,
    UncertaintyQuantifier,
)

__all__ = [
    # Benchmark suite
    "PINNBenchmarkSuite",
    "BenchmarkConfig",
    "BenchmarkResult",
    "CavityFlowBenchmark",
    "ChannelFlowBenchmark",
    "CylinderFlowBenchmark",
    "ThermalConvectionBenchmark",
    # Metrics and statistics
    "EvaluationMetrics",
    "StatisticalAnalysis",
    "MetricResult",
    "StatisticalTest",
    # Method comparison
    "MethodComparison",
    "MethodComparisonConfig",
    "ComparisonResult",
    # Statistical validation
    "StatisticalValidator",
    "StatisticalResult",
    "BootstrapResult",
    # Cross-validation
    "CrossValidationFramework",
    "CVResult",
    "NestedCVResult",
    # Physics consistency
    "PhysicsConsistencyChecker",
    "PhysicsConsistencyResult",
    "ConservationLaw",
    "SymmetryType",
    # Uncertainty quantification
    "UncertaintyQuantifier",
    "BayesianUncertaintyQuantifier",
    "MonteCarloDropout",
    # Advanced validation framework
    "AdvancedValidationFramework",
    "ValidationConfig",
    "ValidationScore",
    "ValidationReport",
]

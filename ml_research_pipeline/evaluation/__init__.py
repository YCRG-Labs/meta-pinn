"""
Comprehensive evaluation framework for PINN methods.

This module provides a complete evaluation framework including:
- Benchmark suite with multiple problem types (cavity, channel, cylinder, thermal)
- Evaluation metrics and statistical analysis
- Automated method comparison system
"""

from .benchmark_suite import (
    PINNBenchmarkSuite,
    BenchmarkConfig,
    BenchmarkResult,
    CavityFlowBenchmark,
    ChannelFlowBenchmark,
    CylinderFlowBenchmark,
    ThermalConvectionBenchmark
)

from .metrics import (
    EvaluationMetrics,
    StatisticalAnalysis,
    MetricResult,
    StatisticalTest
)

from .method_comparison import (
    MethodComparison,
    MethodComparisonConfig,
    ComparisonResult
)

__all__ = [
    # Benchmark suite
    'PINNBenchmarkSuite',
    'BenchmarkConfig',
    'BenchmarkResult',
    'CavityFlowBenchmark',
    'ChannelFlowBenchmark',
    'CylinderFlowBenchmark',
    'ThermalConvectionBenchmark',
    
    # Metrics and statistics
    'EvaluationMetrics',
    'StatisticalAnalysis',
    'MetricResult',
    'StatisticalTest',
    
    # Method comparison
    'MethodComparison',
    'MethodComparisonConfig',
    'ComparisonResult'
]
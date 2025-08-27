"""
Theory module for theoretical analysis and mathematical foundations.

This module contains theoretical analysis components including sample complexity,
convergence rate analysis, and formal mathematical proofs for meta-learning PINNs.
"""

__version__ = "0.1.0"
__author__ = "ML Research Team"

# Import theoretical analysis components
from .sample_complexity import SampleComplexityAnalyzer
from .convergence_analysis import ConvergenceAnalyzer
from .proofs.mathematical_proofs import TheoremGenerator

__all__ = [
    "SampleComplexityAnalyzer",
    "ConvergenceAnalyzer",
    "TheoremGenerator",
]
"""
Bayesian uncertainty quantification components for meta-learning PINNs.

This module provides Bayesian extensions to the meta-learning PINN framework,
including variational inference, uncertainty quantification, and calibration.
"""

from .bayesian_meta_pinn import BayesianMetaPINN
from .uncertainty_calibrator import CalibrationEvaluator, UncertaintyCalibrator

__all__ = ["BayesianMetaPINN", "UncertaintyCalibrator", "CalibrationEvaluator"]

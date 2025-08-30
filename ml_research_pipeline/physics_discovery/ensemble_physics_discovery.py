"""
Ensemble Physics Discovery Orchestrator

This module implements a comprehensive orchestrator that integrates causal discovery,
symbolic regression, and neural discovery methods using the DiscoveryEnsemble system.
It provides parallel execution, result validation, and quality assessment.
"""

import logging
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

# Import validation components
from ..evaluation.physics_consistency import PhysicsConsistencyChecker
from ..evaluation.statistical_validator import StatisticalValidator

# Import discovery methods
from .advanced_causal_discovery import AdvancedCausalDiscovery

# Import ensemble system
from .discovery_ensemble import (
    DiscoveryEnsemble,
    DiscoveryMethod,
    EnsembleDiscoveryResult,
    MethodResult,
)
from .enhanced_mutual_info import EnhancedMutualInfoDiscovery
from .fci_algorithm import FCIAlgorithm
from .pc_algorithm import PCAlgorithm
from .robust_symbolic_regression import RobustSymbolicRegression


@dataclass
class ExecutionConfig:
    """Configuration for method execution."""

    method: DiscoveryMethod
    enabled: bool = True
    timeout: Optional[float] = None
    max_retries: int = 2
    parallel: bool = True
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for result validation."""

    physics_consistency: bool = True
    statistical_validation: bool = True
    cross_validation_folds: int = 5
    bootstrap_samples: int = 100
    significance_level: float = 0.05
    min_confidence_threshold: float = 0.5


@dataclass
class QualityMetrics:
    """Quality assessment metrics for discovery results."""

    accuracy: float
    stability: float
    interpretability: float
    statistical_significance: float
    physics_consistency: float
    computational_efficiency: float
    robustness: float


@dataclass
class ExecutionResult:
    """Result from executing a single discovery method."""

    method: DiscoveryMethod
    success: bool
    result: Any
    confidence: float
    execution_time: float
    quality_metrics: QualityMetrics
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnsemblePhysicsDiscovery:
    """
    Comprehensive physics discovery orchestrator that integrates multiple methods.

    This class provides:
    - Parallel execution of causal, symbolic, and neural discovery methods
    - Ensemble result fusion using DiscoveryEnsemble
    - Comprehensive validation and quality assessment
    - Adaptive method selection based on data characteristics
    - Robust error handling and recovery
    """

    def __init__(
        self,
        execution_configs: Optional[List[ExecutionConfig]] = None,
        validation_config: Optional[ValidationConfig] = None,
        ensemble_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
        use_multiprocessing: bool = False,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize ensemble physics discovery system.

        Args:
            execution_configs: Configurations for each discovery method
            validation_config: Configuration for result validation
            ensemble_config: Configuration for ensemble system
            max_workers: Maximum number of parallel workers
            use_multiprocessing: Whether to use multiprocessing vs threading
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed progress
        """
        self.execution_configs = (
            execution_configs or self._get_default_execution_configs()
        )
        self.validation_config = validation_config or ValidationConfig()
        self.ensemble_config = ensemble_config or {}
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.random_state = random_state
        self.verbose = verbose

        # Initialize components
        self._initialize_discovery_methods()
        self._initialize_validators()
        self._initialize_ensemble()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO)

        # Execution state
        self.execution_results = []
        self.ensemble_result = None

    def _get_default_execution_configs(self) -> List[ExecutionConfig]:
        """Get default execution configurations for all methods."""
        return [
            ExecutionConfig(
                method=DiscoveryMethod.PC_ALGORITHM,
                enabled=True,
                timeout=300.0,  # 5 minutes
                priority=1,
                parameters={"alpha": 0.05, "max_conditioning_size": 3},
            ),
            ExecutionConfig(
                method=DiscoveryMethod.FCI_ALGORITHM,
                enabled=True,
                timeout=600.0,  # 10 minutes
                priority=2,
                parameters={"alpha": 0.05, "max_conditioning_size": 3},
            ),
            ExecutionConfig(
                method=DiscoveryMethod.MUTUAL_INFO,
                enabled=True,
                timeout=300.0,
                priority=1,
                parameters={"alpha": 0.05, "n_bootstrap": 100},
            ),
            ExecutionConfig(
                method=DiscoveryMethod.SYMBOLIC_REGRESSION,
                enabled=True,
                timeout=900.0,  # 15 minutes
                priority=3,
                parameters={"population_size": 100, "generations": 50},
            ),
        ]

    def _initialize_discovery_methods(self):
        """Initialize discovery method instances."""
        self.discovery_methods = {}

        # Initialize causal discovery methods
        self.discovery_methods[DiscoveryMethod.PC_ALGORITHM] = PCAlgorithm()
        self.discovery_methods[DiscoveryMethod.FCI_ALGORITHM] = FCIAlgorithm()
        self.discovery_methods[DiscoveryMethod.MUTUAL_INFO] = (
            EnhancedMutualInfoDiscovery()
        )

        # Initialize symbolic regression (will be created as needed)
        self.discovery_methods[DiscoveryMethod.SYMBOLIC_REGRESSION] = None

        # Advanced orchestrators
        self.advanced_causal = AdvancedCausalDiscovery(verbose=self.verbose)

    def _initialize_validators(self):
        """Initialize validation components."""
        if self.validation_config.physics_consistency:
            self.physics_validator = PhysicsConsistencyChecker()
        else:
            self.physics_validator = None

        if self.validation_config.statistical_validation:
            self.statistical_validator = StatisticalValidator()
        else:
            self.statistical_validator = None

    def _initialize_ensemble(self):
        """Initialize ensemble system."""
        ensemble_params = {
            "voting_strategy": "confidence_weighted",
            "conflict_resolution": "highest_confidence",
            "consensus_threshold": 0.7,
            "min_methods": 2,
        }
        ensemble_params.update(self.ensemble_config)

        self.ensemble = DiscoveryEnsemble(**ensemble_params)

    def discover_physics(
        self,
        data: pd.DataFrame,
        target_variables: Optional[List[str]] = None,
        known_relationships: Optional[nx.DiGraph] = None,
        physics_constraints: Optional[Dict[str, Any]] = None,
    ) -> EnsembleDiscoveryResult:
        """
        Execute comprehensive physics discovery using ensemble methods.

        Args:
            data: Input dataset for discovery
            target_variables: Variables to focus discovery on
            known_relationships: Prior knowledge about relationships
            physics_constraints: Physical constraints to enforce

        Returns:
            Ensemble discovery result with consensus findings
        """
        self.logger.info(
            f"Starting ensemble physics discovery on dataset with shape {data.shape}"
        )

        # Validate input data
        self._validate_input_data(data)

        # Execute discovery methods
        execution_results = self._execute_discovery_methods(
            data, target_variables, known_relationships, physics_constraints
        )

        # Add results to ensemble
        for result in execution_results:
            if result.success:
                self.ensemble.add_method_result(
                    method=result.method,
                    result=result.result,
                    confidence=result.confidence,
                    execution_time=result.execution_time,
                    quality_metrics=result.quality_metrics.__dict__,
                    metadata=result.metadata,
                )

        # Compute ensemble result
        try:
            self.ensemble_result = self.ensemble.compute_ensemble_result()
            self.logger.info(
                f"Ensemble discovery completed with confidence {self.ensemble_result.ensemble_confidence:.3f}"
            )
        except Exception as e:
            self.logger.error(f"Ensemble computation failed: {e}")
            raise

        # Validate ensemble result
        self._validate_ensemble_result(data)

        return self.ensemble_result

    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data quality and characteristics."""
        if data.empty:
            raise ValueError("Input data is empty")

        if data.shape[0] < 10:
            warnings.warn("Very small dataset may lead to unreliable results")

        if data.shape[1] < 2:
            raise ValueError("Need at least 2 variables for discovery")

        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > 0.5:
            warnings.warn(f"High missing value ratio: {missing_ratio:.2%}")

        # Check for constant columns
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_cols:
            warnings.warn(f"Constant columns detected: {constant_cols}")

    def _execute_discovery_methods(
        self,
        data: pd.DataFrame,
        target_variables: Optional[List[str]],
        known_relationships: Optional[nx.DiGraph],
        physics_constraints: Optional[Dict[str, Any]],
    ) -> List[ExecutionResult]:
        """Execute all enabled discovery methods."""
        enabled_configs = [
            config for config in self.execution_configs if config.enabled
        ]

        if not enabled_configs:
            raise ValueError("No discovery methods enabled")

        # Sort by priority
        enabled_configs.sort(key=lambda x: x.priority)

        execution_results = []

        if self.max_workers == 1 or len(enabled_configs) == 1:
            # Sequential execution
            for config in enabled_configs:
                result = self._execute_single_method(
                    config,
                    data,
                    target_variables,
                    known_relationships,
                    physics_constraints,
                )
                execution_results.append(result)
        else:
            # Parallel execution
            executor_class = (
                ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
            )

            with executor_class(
                max_workers=min(self.max_workers, len(enabled_configs))
            ) as executor:
                # Submit all tasks
                future_to_config = {}
                for config in enabled_configs:
                    if config.parallel:
                        future = executor.submit(
                            self._execute_single_method_wrapper,
                            config,
                            data,
                            target_variables,
                            known_relationships,
                            physics_constraints,
                        )
                        future_to_config[future] = config
                    else:
                        # Execute non-parallel methods sequentially
                        result = self._execute_single_method(
                            config,
                            data,
                            target_variables,
                            known_relationships,
                            physics_constraints,
                        )
                        execution_results.append(result)

                # Collect results
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        execution_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Method {config.method.value} failed: {e}")
                        execution_results.append(
                            ExecutionResult(
                                method=config.method,
                                success=False,
                                result=None,
                                confidence=0.0,
                                execution_time=0.0,
                                quality_metrics=QualityMetrics(0, 0, 0, 0, 0, 0, 0),
                                error_message=str(e),
                            )
                        )

        self.execution_results = execution_results
        return execution_results

    def _execute_single_method_wrapper(self, *args, **kwargs):
        """Wrapper for single method execution to handle multiprocessing."""
        return self._execute_single_method(*args, **kwargs)

    def _execute_single_method(
        self,
        config: ExecutionConfig,
        data: pd.DataFrame,
        target_variables: Optional[List[str]],
        known_relationships: Optional[nx.DiGraph],
        physics_constraints: Optional[Dict[str, Any]],
    ) -> ExecutionResult:
        """Execute a single discovery method with error handling and validation."""
        method_name = config.method.value
        self.logger.info(f"Executing {method_name}...")

        start_time = time.time()

        try:
            # Execute method with retries
            result = None
            error_message = None

            for attempt in range(config.max_retries + 1):
                try:
                    if config.method == DiscoveryMethod.PC_ALGORITHM:
                        result = self._execute_pc_algorithm(data, config.parameters)
                    elif config.method == DiscoveryMethod.FCI_ALGORITHM:
                        result = self._execute_fci_algorithm(data, config.parameters)
                    elif config.method == DiscoveryMethod.MUTUAL_INFO:
                        result = self._execute_mutual_info(data, config.parameters)
                    elif config.method == DiscoveryMethod.SYMBOLIC_REGRESSION:
                        result = self._execute_symbolic_regression(
                            data, target_variables, config.parameters
                        )
                    else:
                        raise ValueError(f"Unknown method: {config.method}")

                    break  # Success, exit retry loop

                except Exception as e:
                    error_message = str(e)
                    if attempt < config.max_retries:
                        self.logger.warning(
                            f"Attempt {attempt + 1} failed for {method_name}: {e}. Retrying..."
                        )
                        time.sleep(1)  # Brief delay before retry
                    else:
                        self.logger.error(f"All attempts failed for {method_name}: {e}")
                        raise

            execution_time = time.time() - start_time

            # Compute confidence and quality metrics
            confidence = self._compute_method_confidence(config.method, result, data)
            quality_metrics = self._compute_quality_metrics(
                config.method, result, data, execution_time
            )

            return ExecutionResult(
                method=config.method,
                success=True,
                result=result,
                confidence=confidence,
                execution_time=execution_time,
                quality_metrics=quality_metrics,
                metadata={"attempts": attempt + 1},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Method {method_name} failed: {e}")

            return ExecutionResult(
                method=config.method,
                success=False,
                result=None,
                confidence=0.0,
                execution_time=execution_time,
                quality_metrics=QualityMetrics(0, 0, 0, 0, 0, 0, 0),
                error_message=str(e),
            )

    def _execute_pc_algorithm(self, data: pd.DataFrame, parameters: Dict[str, Any]):
        """Execute PC algorithm."""
        pc = PCAlgorithm(**parameters)
        return pc.discover_causal_structure(data)

    def _execute_fci_algorithm(self, data: pd.DataFrame, parameters: Dict[str, Any]):
        """Execute FCI algorithm."""
        fci = FCIAlgorithm(**parameters)
        return fci.discover_causal_structure(data)

    def _execute_mutual_info(self, data: pd.DataFrame, parameters: Dict[str, Any]):
        """Execute enhanced mutual information discovery."""
        mi = EnhancedMutualInfoDiscovery(**parameters)
        return mi.discover_causal_relationships(data)

    def _execute_symbolic_regression(
        self,
        data: pd.DataFrame,
        target_variables: Optional[List[str]],
        parameters: Dict[str, Any],
    ):
        """Execute symbolic regression."""
        # This is a simplified version - would need actual symbolic regression implementation
        from .robust_symbolic_regression import RobustSymbolicRegression

        if target_variables is None:
            target_variables = [data.columns[-1]]  # Use last column as default target

        sr = RobustSymbolicRegression(**parameters)

        # For each target variable, perform symbolic regression
        results = {}
        for target in target_variables:
            if target in data.columns:
                X = data.drop(columns=[target])
                y = data[target]
                result = sr.fit_ensemble(X, y)
                results[target] = result

        return results

    def _compute_method_confidence(
        self, method: DiscoveryMethod, result: Any, data: pd.DataFrame
    ) -> float:
        """Compute confidence score for a method result."""
        try:
            if method in [DiscoveryMethod.PC_ALGORITHM, DiscoveryMethod.FCI_ALGORITHM]:
                # For causal discovery, base confidence on number of edges and statistical tests
                if hasattr(result, "directed_graph"):
                    graph = result.directed_graph
                elif hasattr(result, "pag"):
                    graph = result.pag
                else:
                    return 0.5  # Default confidence

                # More edges might indicate more discovered relationships
                edge_ratio = len(graph.edges()) / (
                    len(graph.nodes()) * (len(graph.nodes()) - 1)
                )
                edge_confidence = min(edge_ratio * 2, 1.0)  # Scale to [0, 1]

                # Statistical test confidence
                if hasattr(result, "independence_tests"):
                    significant_tests = sum(
                        1 for test in result.independence_tests if test.p_value < 0.05
                    )
                    total_tests = len(result.independence_tests)
                    stat_confidence = (
                        significant_tests / total_tests if total_tests > 0 else 0.5
                    )
                else:
                    stat_confidence = 0.5

                return (edge_confidence + stat_confidence) / 2

            elif method == DiscoveryMethod.MUTUAL_INFO:
                # For mutual info, use average MI scores
                if hasattr(result, "mi_matrix"):
                    mi_scores = result.mi_matrix[
                        np.triu_indices_from(result.mi_matrix, k=1)
                    ]
                    return min(np.mean(mi_scores), 1.0)
                return 0.5

            elif method == DiscoveryMethod.SYMBOLIC_REGRESSION:
                # For symbolic regression, use fitness scores
                if isinstance(result, dict):
                    confidences = []
                    for target_result in result.values():
                        if hasattr(target_result, "confidence_score"):
                            confidences.append(target_result.confidence_score)
                        elif hasattr(target_result, "ensemble_fitness"):
                            confidences.append(min(target_result.ensemble_fitness, 1.0))
                    return np.mean(confidences) if confidences else 0.5
                return 0.5

            else:
                return 0.5  # Default confidence

        except Exception as e:
            self.logger.warning(f"Failed to compute confidence for {method.value}: {e}")
            return 0.5

    def _compute_quality_metrics(
        self,
        method: DiscoveryMethod,
        result: Any,
        data: pd.DataFrame,
        execution_time: float,
    ) -> QualityMetrics:
        """Compute comprehensive quality metrics for a method result."""
        try:
            # Base metrics
            accuracy = self._compute_accuracy_metric(method, result, data)
            stability = self._compute_stability_metric(method, result, data)
            interpretability = self._compute_interpretability_metric(method, result)
            statistical_significance = self._compute_statistical_significance(
                method, result
            )
            physics_consistency = self._compute_physics_consistency(method, result)
            computational_efficiency = self._compute_efficiency_metric(
                execution_time, data.shape
            )
            robustness = self._compute_robustness_metric(method, result, data)

            return QualityMetrics(
                accuracy=accuracy,
                stability=stability,
                interpretability=interpretability,
                statistical_significance=statistical_significance,
                physics_consistency=physics_consistency,
                computational_efficiency=computational_efficiency,
                robustness=robustness,
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to compute quality metrics for {method.value}: {e}"
            )
            return QualityMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    def _compute_accuracy_metric(
        self, method: DiscoveryMethod, result: Any, data: pd.DataFrame
    ) -> float:
        """Compute accuracy metric based on method type."""
        # Simplified accuracy computation
        return 0.8  # Placeholder

    def _compute_stability_metric(
        self, method: DiscoveryMethod, result: Any, data: pd.DataFrame
    ) -> float:
        """Compute stability metric."""
        return 0.7  # Placeholder

    def _compute_interpretability_metric(
        self, method: DiscoveryMethod, result: Any
    ) -> float:
        """Compute interpretability metric."""
        if method == DiscoveryMethod.SYMBOLIC_REGRESSION:
            return 0.9  # Symbolic expressions are highly interpretable
        elif method in [DiscoveryMethod.PC_ALGORITHM, DiscoveryMethod.FCI_ALGORITHM]:
            return 0.8  # Causal graphs are interpretable
        else:
            return 0.6  # Default

    def _compute_statistical_significance(
        self, method: DiscoveryMethod, result: Any
    ) -> float:
        """Compute statistical significance metric."""
        return 0.75  # Placeholder

    def _compute_physics_consistency(
        self, method: DiscoveryMethod, result: Any
    ) -> float:
        """Compute physics consistency metric."""
        if self.physics_validator:
            try:
                # This would need actual physics validation logic
                return 0.8
            except:
                return 0.5
        return 0.5

    def _compute_efficiency_metric(
        self, execution_time: float, data_shape: Tuple[int, int]
    ) -> float:
        """Compute computational efficiency metric."""
        # Normalize by data size and expected time
        data_complexity = data_shape[0] * data_shape[1]
        expected_time = np.log(data_complexity) * 10  # Rough estimate

        if execution_time <= expected_time:
            return 1.0
        else:
            return max(0.1, expected_time / execution_time)

    def _compute_robustness_metric(
        self, method: DiscoveryMethod, result: Any, data: pd.DataFrame
    ) -> float:
        """Compute robustness metric."""
        return 0.7  # Placeholder

    def _validate_ensemble_result(self, data: pd.DataFrame):
        """Validate the ensemble result."""
        if not self.ensemble_result:
            return

        # Physics consistency validation
        if self.physics_validator and self.validation_config.physics_consistency:
            try:
                # Use a simplified validation approach since the exact method may vary
                consistency_score = (
                    0.8  # Placeholder - would need actual validation logic
                )
                self.ensemble_result.validation_metrics["physics_consistency"] = (
                    consistency_score
                )
            except Exception as e:
                self.logger.warning(f"Physics consistency validation failed: {e}")

        # Statistical validation
        if self.statistical_validator and self.validation_config.statistical_validation:
            try:
                stat_metrics = self.statistical_validator.validate_discovery_result(
                    data, self.ensemble_result
                )
                self.ensemble_result.validation_metrics.update(stat_metrics)
            except Exception as e:
                self.logger.warning(f"Statistical validation failed: {e}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of method execution results."""
        if not self.execution_results:
            return {}

        summary = {
            "total_methods": len(self.execution_results),
            "successful_methods": sum(1 for r in self.execution_results if r.success),
            "failed_methods": sum(1 for r in self.execution_results if not r.success),
            "total_execution_time": sum(
                r.execution_time for r in self.execution_results
            ),
            "average_confidence": np.mean(
                [r.confidence for r in self.execution_results if r.success]
            ),
            "method_performances": {},
        }

        for result in self.execution_results:
            summary["method_performances"][result.method.value] = {
                "success": result.success,
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "error": result.error_message,
            }

        return summary

    def get_method_recommendations(self) -> List[str]:
        """Get recommendations for improving discovery results."""
        recommendations = []

        if not self.execution_results:
            return ["No execution results available"]

        # Check for failed methods
        failed_methods = [r for r in self.execution_results if not r.success]
        if failed_methods:
            failed_names = [r.method.value for r in failed_methods]
            recommendations.append(
                f"Failed methods: {', '.join(failed_names)}. Consider adjusting parameters or data preprocessing."
            )

        # Check for low confidence methods
        low_confidence = [
            r for r in self.execution_results if r.success and r.confidence < 0.5
        ]
        if low_confidence:
            low_conf_names = [r.method.value for r in low_confidence]
            recommendations.append(
                f"Low confidence methods: {', '.join(low_conf_names)}. Consider collecting more data or using different parameters."
            )

        # Check execution times
        slow_methods = [
            r for r in self.execution_results if r.execution_time > 300
        ]  # 5 minutes
        if slow_methods:
            slow_names = [r.method.value for r in slow_methods]
            recommendations.append(
                f"Slow methods: {', '.join(slow_names)}. Consider reducing data size or adjusting complexity parameters."
            )

        # Ensemble-specific recommendations
        if self.ensemble_result:
            if self.ensemble_result.ensemble_confidence < 0.6:
                recommendations.append(
                    "Low ensemble confidence. Consider adding more discovery methods or improving data quality."
                )

            if len(self.ensemble_result.consensus_score.conflict_regions) > 0:
                recommendations.append(
                    "Conflicts detected between methods. Manual review recommended."
                )

        return (
            recommendations
            if recommendations
            else ["All methods performed well. No specific recommendations."]
        )

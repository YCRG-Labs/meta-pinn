"""
Discovery Ensemble System

This module implements a comprehensive ensemble system that combines results from
multiple physics discovery methods including causal discovery, symbolic regression,
and neural physics discovery. It provides Bayesian model averaging, weighted voting,
and consensus scoring mechanisms.
"""

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

from .advanced_causal_discovery import EnsembleResult as CausalEnsembleResult
from .enhanced_mutual_info import MutualInfoResult
from .fci_algorithm import FCIResult

# Import discovery method results
from .pc_algorithm import PCResult
from .robust_symbolic_regression import EnsembleResult as SymbolicEnsembleResult


class DiscoveryMethod(Enum):
    """Enumeration of available discovery methods."""

    PC_ALGORITHM = "pc_algorithm"
    FCI_ALGORITHM = "fci_algorithm"
    MUTUAL_INFO = "mutual_info"
    SYMBOLIC_REGRESSION = "symbolic_regression"
    NEURAL_DISCOVERY = "neural_discovery"


@dataclass
class MethodResult:
    """Standardized result from any discovery method."""

    method: DiscoveryMethod
    confidence: float
    execution_time: float
    raw_result: Any
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusScore:
    """Consensus scoring for ensemble results."""

    overall_consensus: float
    method_agreement: Dict[str, float]
    conflict_regions: List[str]
    confidence_weighted_score: float
    statistical_significance: float


@dataclass
class EnsembleDiscoveryResult:
    """Final result from ensemble discovery system."""

    consensus_causal_graph: nx.DiGraph
    consensus_expressions: Dict[str, Any]
    consensus_score: ConsensusScore
    method_contributions: Dict[DiscoveryMethod, float]
    individual_results: List[MethodResult]
    ensemble_confidence: float
    validation_metrics: Dict[str, float]
    recommendations: List[str]


class BayesianModelAveraging:
    """Implements Bayesian model averaging for discovery results."""

    def __init__(
        self,
        prior_weights: Optional[Dict[DiscoveryMethod, float]] = None,
        evidence_threshold: float = 0.1,
    ):
        """
        Initialize Bayesian model averaging.

        Args:
            prior_weights: Prior weights for each method
            evidence_threshold: Minimum evidence threshold for inclusion
        """
        self.prior_weights = prior_weights or {}
        self.evidence_threshold = evidence_threshold
        self.posterior_weights = {}

    def compute_model_evidence(self, result: MethodResult) -> float:
        """
        Compute model evidence (marginal likelihood) for a discovery result.

        Args:
            result: Method result to evaluate

        Returns:
            Model evidence score
        """
        # Base evidence from confidence and quality metrics
        base_evidence = result.confidence

        # Adjust based on quality metrics
        quality_adjustment = 1.0
        if result.quality_metrics:
            # Weight different quality aspects
            weights = {
                "accuracy": 0.3,
                "stability": 0.2,
                "interpretability": 0.2,
                "statistical_significance": 0.3,
            }

            for metric, weight in weights.items():
                if metric in result.quality_metrics:
                    quality_adjustment *= 1 + weight * result.quality_metrics[metric]

        # Penalize long execution times (efficiency consideration)
        time_penalty = np.exp(
            -result.execution_time / 100.0
        )  # Normalize by 100 seconds

        evidence = base_evidence * quality_adjustment * time_penalty
        return max(evidence, self.evidence_threshold)

    def update_posterior_weights(
        self, results: List[MethodResult]
    ) -> Dict[DiscoveryMethod, float]:
        """
        Update posterior weights using Bayesian updating.

        Args:
            results: List of method results

        Returns:
            Posterior weights for each method
        """
        log_weights = {}

        for result in results:
            method = result.method

            # Prior (log space)
            log_prior = np.log(self.prior_weights.get(method, 1.0))

            # Likelihood (evidence)
            evidence = self.compute_model_evidence(result)
            log_likelihood = np.log(evidence)

            # Posterior (unnormalized)
            log_weights[method] = log_prior + log_likelihood

        # Normalize weights
        if log_weights:
            log_normalizer = logsumexp(list(log_weights.values()))
            self.posterior_weights = {
                method: np.exp(log_weight - log_normalizer)
                for method, log_weight in log_weights.items()
            }

        return self.posterior_weights


class WeightedVoting:
    """Implements weighted voting mechanisms for ensemble decisions."""

    def __init__(
        self,
        voting_strategy: str = "confidence_weighted",
        conflict_resolution: str = "highest_confidence",
    ):
        """
        Initialize weighted voting system.

        Args:
            voting_strategy: Strategy for weighting votes
            conflict_resolution: How to resolve conflicts between methods
        """
        self.voting_strategy = voting_strategy
        self.conflict_resolution = conflict_resolution

    def compute_vote_weights(
        self, results: List[MethodResult]
    ) -> Dict[DiscoveryMethod, float]:
        """
        Compute voting weights for each method.

        Args:
            results: List of method results

        Returns:
            Voting weights for each method
        """
        weights = {}

        if self.voting_strategy == "confidence_weighted":
            # Weight by confidence scores
            total_confidence = sum(r.confidence for r in results)
            if total_confidence > 0:
                for result in results:
                    weights[result.method] = result.confidence / total_confidence

        elif self.voting_strategy == "quality_weighted":
            # Weight by overall quality metrics
            quality_scores = []
            for result in results:
                if result.quality_metrics:
                    avg_quality = np.mean(list(result.quality_metrics.values()))
                    quality_scores.append(avg_quality)
                else:
                    quality_scores.append(result.confidence)

            total_quality = sum(quality_scores)
            if total_quality > 0:
                for i, result in enumerate(results):
                    weights[result.method] = quality_scores[i] / total_quality

        elif self.voting_strategy == "uniform":
            # Equal weights for all methods
            weight = 1.0 / len(results) if results else 0.0
            for result in results:
                weights[result.method] = weight

        return weights

    def resolve_conflicts(
        self,
        conflicting_results: List[MethodResult],
        weights: Dict[DiscoveryMethod, float],
    ) -> MethodResult:
        """
        Resolve conflicts between different method results.

        Args:
            conflicting_results: Results that conflict with each other
            weights: Voting weights for each method

        Returns:
            Resolved result
        """
        if self.conflict_resolution == "highest_confidence":
            return max(conflicting_results, key=lambda r: r.confidence)

        elif self.conflict_resolution == "weighted_average":
            # Implement weighted averaging logic based on result type
            # This is a simplified version - real implementation would depend on result types
            total_weight = sum(weights.get(r.method, 0.0) for r in conflicting_results)
            if total_weight > 0:
                # Return the result with highest weighted score
                weighted_scores = [
                    r.confidence * weights.get(r.method, 0.0) / total_weight
                    for r in conflicting_results
                ]
                best_idx = np.argmax(weighted_scores)
                return conflicting_results[best_idx]

        # Default: return first result
        return conflicting_results[0] if conflicting_results else None


class ConsensusScoring:
    """Implements consensus scoring and conflict detection."""

    def __init__(
        self, agreement_threshold: float = 0.7, significance_level: float = 0.05
    ):
        """
        Initialize consensus scoring.

        Args:
            agreement_threshold: Minimum agreement for consensus
            significance_level: Statistical significance level
        """
        self.agreement_threshold = agreement_threshold
        self.significance_level = significance_level

    def compute_pairwise_agreement(
        self, result1: MethodResult, result2: MethodResult
    ) -> float:
        """
        Compute agreement between two method results.

        Args:
            result1: First method result
            result2: Second method result

        Returns:
            Agreement score between 0 and 1
        """
        # This is a simplified version - real implementation would depend on result types
        # For now, use confidence correlation as a proxy
        conf_diff = abs(result1.confidence - result2.confidence)
        agreement = 1.0 - conf_diff

        # Adjust based on method compatibility
        method_compatibility = self._get_method_compatibility(
            result1.method, result2.method
        )

        return agreement * method_compatibility

    def _get_method_compatibility(
        self, method1: DiscoveryMethod, method2: DiscoveryMethod
    ) -> float:
        """Get compatibility score between two methods."""
        # Define method compatibility matrix
        compatibility = {
            (DiscoveryMethod.PC_ALGORITHM, DiscoveryMethod.FCI_ALGORITHM): 0.9,
            (DiscoveryMethod.PC_ALGORITHM, DiscoveryMethod.MUTUAL_INFO): 0.8,
            (DiscoveryMethod.FCI_ALGORITHM, DiscoveryMethod.MUTUAL_INFO): 0.8,
            (
                DiscoveryMethod.SYMBOLIC_REGRESSION,
                DiscoveryMethod.NEURAL_DISCOVERY,
            ): 0.7,
        }

        # Check both directions
        key1 = (method1, method2)
        key2 = (method2, method1)

        if key1 in compatibility:
            return compatibility[key1]
        elif key2 in compatibility:
            return compatibility[key2]
        elif method1 == method2:
            return 1.0
        else:
            return 0.5  # Default compatibility

    def compute_consensus_score(self, results: List[MethodResult]) -> ConsensusScore:
        """
        Compute overall consensus score for ensemble results.

        Args:
            results: List of method results

        Returns:
            Consensus score object
        """
        if len(results) < 2:
            return ConsensusScore(
                overall_consensus=1.0 if results else 0.0,
                method_agreement={},
                conflict_regions=[],
                confidence_weighted_score=results[0].confidence if results else 0.0,
                statistical_significance=1.0,
            )

        # Compute pairwise agreements
        agreements = []
        method_agreement = {}

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                agreement = self.compute_pairwise_agreement(results[i], results[j])
                agreements.append(agreement)

                key = f"{results[i].method.value}_{results[j].method.value}"
                method_agreement[key] = agreement

        # Overall consensus
        overall_consensus = np.mean(agreements) if agreements else 0.0

        # Identify conflicts
        conflict_regions = []
        for i, agreement in enumerate(agreements):
            if agreement < self.agreement_threshold:
                conflict_regions.append(f"conflict_{i}")

        # Confidence-weighted score
        total_confidence = sum(r.confidence for r in results)
        if total_confidence > 0:
            confidence_weighted_score = sum(
                r.confidence * r.confidence / total_confidence for r in results
            )
        else:
            confidence_weighted_score = 0.0

        # Statistical significance (simplified)
        confidences = [r.confidence for r in results]
        if len(confidences) > 1:
            _, p_value = stats.ttest_1samp(confidences, 0.5)
            statistical_significance = 1.0 - p_value
        else:
            statistical_significance = 1.0

        return ConsensusScore(
            overall_consensus=overall_consensus,
            method_agreement=method_agreement,
            conflict_regions=conflict_regions,
            confidence_weighted_score=confidence_weighted_score,
            statistical_significance=statistical_significance,
        )


class DiscoveryEnsemble:
    """
    Main ensemble class that combines multiple physics discovery methods.

    This class implements:
    - Bayesian model averaging for discovery results
    - Weighted voting based on method confidence
    - Result consensus scoring and conflict resolution
    """

    def __init__(
        self,
        prior_weights: Optional[Dict[DiscoveryMethod, float]] = None,
        voting_strategy: str = "confidence_weighted",
        conflict_resolution: str = "highest_confidence",
        consensus_threshold: float = 0.7,
        min_methods: int = 2,
    ):
        """
        Initialize discovery ensemble.

        Args:
            prior_weights: Prior weights for each discovery method
            voting_strategy: Strategy for weighted voting
            conflict_resolution: Method for resolving conflicts
            consensus_threshold: Minimum consensus score for acceptance
            min_methods: Minimum number of methods required
        """
        self.prior_weights = prior_weights or {}
        self.voting_strategy = voting_strategy
        self.conflict_resolution = conflict_resolution
        self.consensus_threshold = consensus_threshold
        self.min_methods = min_methods

        # Initialize components
        self.bayesian_averaging = BayesianModelAveraging(prior_weights)
        self.weighted_voting = WeightedVoting(voting_strategy, conflict_resolution)
        self.consensus_scoring = ConsensusScoring(consensus_threshold)

        # Results storage
        self.method_results = []
        self.ensemble_result = None

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def add_method_result(
        self,
        method: DiscoveryMethod,
        result: Any,
        confidence: float,
        execution_time: float,
        quality_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a result from a discovery method to the ensemble.

        Args:
            method: Discovery method type
            result: Raw result from the method
            confidence: Confidence score for the result
            execution_time: Time taken to compute the result
            quality_metrics: Quality metrics for the result
            metadata: Additional metadata
        """
        method_result = MethodResult(
            method=method,
            confidence=confidence,
            execution_time=execution_time,
            raw_result=result,
            quality_metrics=quality_metrics or {},
            metadata=metadata or {},
        )

        self.method_results.append(method_result)
        self.logger.info(
            f"Added result from {method.value} with confidence {confidence:.3f}"
        )

    def compute_ensemble_result(self) -> EnsembleDiscoveryResult:
        """
        Compute the final ensemble result from all method results.

        Returns:
            Ensemble discovery result
        """
        if len(self.method_results) < self.min_methods:
            raise ValueError(
                f"Need at least {self.min_methods} method results, got {len(self.method_results)}"
            )

        # Compute Bayesian posterior weights
        posterior_weights = self.bayesian_averaging.update_posterior_weights(
            self.method_results
        )

        # Compute voting weights
        voting_weights = self.weighted_voting.compute_vote_weights(self.method_results)

        # Compute consensus score
        consensus_score = self.consensus_scoring.compute_consensus_score(
            self.method_results
        )

        # Combine weights (average of Bayesian and voting weights)
        combined_weights = {}
        for method in set(posterior_weights.keys()) | set(voting_weights.keys()):
            bayes_weight = posterior_weights.get(method, 0.0)
            vote_weight = voting_weights.get(method, 0.0)
            combined_weights[method] = (bayes_weight + vote_weight) / 2.0

        # Create consensus results (simplified - would need method-specific logic)
        consensus_causal_graph = self._create_consensus_causal_graph()
        consensus_expressions = self._create_consensus_expressions()

        # Compute ensemble confidence
        ensemble_confidence = self._compute_ensemble_confidence(
            combined_weights, consensus_score
        )

        # Generate validation metrics
        validation_metrics = self._compute_validation_metrics()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            consensus_score, combined_weights
        )

        self.ensemble_result = EnsembleDiscoveryResult(
            consensus_causal_graph=consensus_causal_graph,
            consensus_expressions=consensus_expressions,
            consensus_score=consensus_score,
            method_contributions=combined_weights,
            individual_results=self.method_results.copy(),
            ensemble_confidence=ensemble_confidence,
            validation_metrics=validation_metrics,
            recommendations=recommendations,
        )

        return self.ensemble_result

    def _create_consensus_causal_graph(self) -> nx.DiGraph:
        """Create consensus causal graph from method results."""
        # Simplified implementation - would need method-specific logic
        consensus_graph = nx.DiGraph()

        # Extract causal relationships from different method types
        for result in self.method_results:
            if result.method in [
                DiscoveryMethod.PC_ALGORITHM,
                DiscoveryMethod.FCI_ALGORITHM,
                DiscoveryMethod.MUTUAL_INFO,
            ]:
                # Extract graph from causal discovery results
                if hasattr(result.raw_result, "directed_graph"):
                    graph = result.raw_result.directed_graph
                elif hasattr(result.raw_result, "pag"):
                    graph = result.raw_result.pag
                elif hasattr(result.raw_result, "causal_graph"):
                    graph = result.raw_result.causal_graph
                else:
                    continue

                # Add edges with weights based on method confidence
                for edge in graph.edges():
                    if consensus_graph.has_edge(*edge):
                        # Increase weight if edge already exists
                        consensus_graph[edge[0]][edge[1]]["weight"] += result.confidence
                    else:
                        # Add new edge
                        consensus_graph.add_edge(
                            edge[0], edge[1], weight=result.confidence
                        )

        return consensus_graph

    def _create_consensus_expressions(self) -> Dict[str, Any]:
        """Create consensus expressions from symbolic regression results."""
        consensus_expressions = {}

        # Extract expressions from symbolic regression results
        for result in self.method_results:
            if result.method == DiscoveryMethod.SYMBOLIC_REGRESSION:
                if hasattr(result.raw_result, "best_expression"):
                    expr_key = f"expression_{len(consensus_expressions)}"
                    consensus_expressions[expr_key] = {
                        "expression": result.raw_result.best_expression,
                        "confidence": result.confidence,
                        "method": result.method.value,
                    }

        return consensus_expressions

    def _compute_ensemble_confidence(
        self, weights: Dict[DiscoveryMethod, float], consensus_score: ConsensusScore
    ) -> float:
        """Compute overall ensemble confidence."""
        # Weighted average of individual confidences
        weighted_confidence = sum(
            result.confidence * weights.get(result.method, 0.0)
            for result in self.method_results
        )

        # Adjust by consensus score
        ensemble_confidence = weighted_confidence * consensus_score.overall_consensus

        return min(ensemble_confidence, 1.0)

    def _compute_validation_metrics(self) -> Dict[str, float]:
        """Compute validation metrics for the ensemble."""
        metrics = {}

        # Average quality metrics across methods
        all_quality_metrics = {}
        for result in self.method_results:
            for metric, value in result.quality_metrics.items():
                if metric not in all_quality_metrics:
                    all_quality_metrics[metric] = []
                all_quality_metrics[metric].append(value)

        for metric, values in all_quality_metrics.items():
            metrics[f"avg_{metric}"] = np.mean(values)
            metrics[f"std_{metric}"] = np.std(values)

        # Ensemble-specific metrics
        metrics["method_diversity"] = len(set(r.method for r in self.method_results))
        metrics["total_execution_time"] = sum(
            r.execution_time for r in self.method_results
        )

        return metrics

    def _generate_recommendations(
        self, consensus_score: ConsensusScore, weights: Dict[DiscoveryMethod, float]
    ) -> List[str]:
        """Generate recommendations based on ensemble results."""
        recommendations = []

        # Check consensus quality
        if consensus_score.overall_consensus < self.consensus_threshold:
            recommendations.append(
                f"Low consensus score ({consensus_score.overall_consensus:.3f}). "
                "Consider collecting more data or adjusting method parameters."
            )

        # Check for conflicts
        if consensus_score.conflict_regions:
            recommendations.append(
                f"Found {len(consensus_score.conflict_regions)} conflict regions. "
                "Manual review recommended."
            )

        # Check method contributions
        max_weight = max(weights.values()) if weights else 0.0
        if max_weight > 0.8:
            dominant_method = max(weights.keys(), key=lambda k: weights[k])
            recommendations.append(
                f"Results dominated by {dominant_method.value}. "
                "Consider improving other methods or adjusting weights."
            )

        # Check statistical significance
        if consensus_score.statistical_significance < 0.95:
            recommendations.append(
                "Low statistical significance. Consider increasing sample size or "
                "using more robust methods."
            )

        return recommendations

    def get_method_rankings(self) -> List[Tuple[DiscoveryMethod, float]]:
        """Get methods ranked by their contribution to the ensemble."""
        if not self.ensemble_result:
            raise ValueError("Must compute ensemble result first")

        contributions = self.ensemble_result.method_contributions
        return sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    def get_confidence_intervals(
        self, confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Get confidence intervals for ensemble metrics."""
        if not self.method_results:
            return {}

        intervals = {}
        alpha = 1 - confidence_level

        # Confidence interval for ensemble confidence
        confidences = [r.confidence for r in self.method_results]
        if len(confidences) > 1:
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences, ddof=1)
            margin = (
                stats.t.ppf(1 - alpha / 2, len(confidences) - 1)
                * std_conf
                / np.sqrt(len(confidences))
            )
            intervals["ensemble_confidence"] = (mean_conf - margin, mean_conf + margin)

        return intervals

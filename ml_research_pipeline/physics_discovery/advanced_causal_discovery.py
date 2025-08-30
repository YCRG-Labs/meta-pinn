"""
Advanced Causal Discovery Orchestrator

This module implements an advanced causal discovery system that integrates
multiple causal discovery methods with ensemble voting, method selection
based on data characteristics, and comprehensive validation.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .enhanced_mutual_info import EnhancedMutualInfoDiscovery, MutualInfoResult
from .fci_algorithm import FCIAlgorithm, FCIResult
from .pc_algorithm import PCAlgorithm, PCResult


@dataclass
class DataCharacteristics:
    """Characteristics of the input data for method selection."""

    n_samples: int
    n_variables: int
    has_missing_values: bool
    has_mixed_types: bool
    estimated_latent_confounders: bool
    temporal_structure: bool
    nonlinearity_score: float
    noise_level: float
    sample_complexity: str  # 'low', 'medium', 'high'


@dataclass
class MethodPerformance:
    """Performance metrics for a causal discovery method."""

    method_name: str
    execution_time: float
    convergence_score: float
    stability_score: float
    edge_count: int
    confidence_score: float
    validation_score: float


@dataclass
class EnsembleResult:
    """Result from ensemble causal discovery."""

    consensus_graph: nx.DiGraph
    method_results: Dict[str, Any]
    ensemble_weights: Dict[str, float]
    confidence_matrix: np.ndarray
    validation_metrics: Dict[str, float]
    data_characteristics: DataCharacteristics
    method_performances: List[MethodPerformance]


class AdvancedCausalDiscovery:
    """
    Advanced causal discovery orchestrator that integrates multiple methods.

    This class combines PC algorithm, FCI algorithm, and enhanced mutual information
    methods with intelligent method selection, ensemble voting, and comprehensive
    validation to provide robust causal discovery for physics applications.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_conditioning_size: int = 3,
        ensemble_strategy: str = "weighted_voting",
        validation_method: str = "bootstrap",
        n_bootstrap: int = 100,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize advanced causal discovery system.

        Args:
            alpha: Significance level for statistical tests
            max_conditioning_size: Maximum conditioning set size
            ensemble_strategy: Strategy for combining methods ('weighted_voting', 'majority_vote', 'confidence_weighted')
            validation_method: Method for validation ('bootstrap', 'cross_validation', 'permutation')
            n_bootstrap: Number of bootstrap samples for validation
            random_state: Random seed for reproducibility
            verbose: Whether to print detailed progress
        """
        self.alpha = alpha
        self.max_conditioning_size = max_conditioning_size
        self.ensemble_strategy = ensemble_strategy
        self.validation_method = validation_method
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.verbose = verbose

        # Initialize individual methods
        self.pc_algorithm = PCAlgorithm(
            alpha=alpha,
            max_conditioning_size=max_conditioning_size,
            independence_test="partial_correlation",
            verbose=verbose,
        )

        self.fci_algorithm = FCIAlgorithm(
            alpha=alpha,
            max_conditioning_size=max_conditioning_size,
            independence_test="partial_correlation",
            verbose=verbose,
        )

        self.mi_discovery = EnhancedMutualInfoDiscovery(
            alpha=alpha,
            mi_threshold=0.1,
            n_bootstrap=min(100, n_bootstrap),  # Limit for efficiency
            n_permutations=min(100, n_bootstrap),
            random_state=random_state,
        )

        # Method selection weights based on data characteristics
        self.method_weights = {"pc": 1.0, "fci": 1.0, "mutual_info": 1.0}

        self.scaler = StandardScaler()
        np.random.seed(random_state)

        # Setup logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def discover_causal_structure(self, data: Dict[str, np.ndarray]) -> EnsembleResult:
        """
        Discover causal structure using ensemble of methods.

        Args:
            data: Dictionary with variable names as keys and data arrays as values

        Returns:
            EnsembleResult containing consensus graph and detailed results
        """
        if self.verbose:
            self.logger.info("Starting advanced causal discovery...")

        # Analyze data characteristics
        data_chars = self._analyze_data_characteristics(data)

        if self.verbose:
            self.logger.info(
                f"Data characteristics: {data_chars.n_samples} samples, "
                f"{data_chars.n_variables} variables, "
                f"complexity: {data_chars.sample_complexity}"
            )

        # Select and weight methods based on data characteristics
        selected_methods = self._select_methods(data_chars)
        method_weights = self._compute_method_weights(data_chars, selected_methods)

        # Run individual methods
        method_results = {}
        method_performances = []

        if "pc" in selected_methods:
            pc_result, pc_performance = self._run_pc_method(data)
            method_results["pc"] = pc_result
            method_performances.append(pc_performance)

        if "fci" in selected_methods:
            fci_result, fci_performance = self._run_fci_method(data)
            method_results["fci"] = fci_result
            method_performances.append(fci_performance)

        if "mutual_info" in selected_methods:
            mi_result, mi_performance = self._run_mi_method(data)
            method_results["mutual_info"] = mi_result
            method_performances.append(mi_performance)

        # Create ensemble result
        consensus_graph = self._create_consensus_graph(
            method_results, method_weights, list(data.keys())
        )

        # Compute confidence matrix
        confidence_matrix = self._compute_confidence_matrix(
            method_results, method_weights, list(data.keys())
        )

        # Validate results
        validation_metrics = self._validate_ensemble_result(
            consensus_graph, method_results, data
        )

        if self.verbose:
            self.logger.info(
                f"Ensemble discovery complete. "
                f"Consensus graph has {consensus_graph.number_of_edges()} edges. "
                f"Validation score: {validation_metrics.get('overall_score', 0.0):.3f}"
            )

        return EnsembleResult(
            consensus_graph=consensus_graph,
            method_results=method_results,
            ensemble_weights=method_weights,
            confidence_matrix=confidence_matrix,
            validation_metrics=validation_metrics,
            data_characteristics=data_chars,
            method_performances=method_performances,
        )

    def _analyze_data_characteristics(
        self, data: Dict[str, np.ndarray]
    ) -> DataCharacteristics:
        """
        Analyze characteristics of input data for method selection.

        Args:
            data: Input data dictionary

        Returns:
            DataCharacteristics object
        """
        variables = list(data.keys())
        n_variables = len(variables)

        # Get sample size (assume all variables have same length)
        sample_sizes = [len(arr) for arr in data.values()]
        n_samples = min(sample_sizes) if sample_sizes else 0

        # Check for missing values
        has_missing = any(np.any(np.isnan(arr)) for arr in data.values())

        # Check for mixed types (simplified - assume all continuous for now)
        has_mixed_types = False

        # Estimate nonlinearity using correlation vs mutual information
        nonlinearity_score = self._estimate_nonlinearity(data)

        # Estimate noise level
        noise_level = self._estimate_noise_level(data)

        # Estimate presence of latent confounders using tetrad test approximation
        estimated_latent_confounders = self._estimate_latent_confounders(data)

        # Check for temporal structure (simplified)
        temporal_structure = any(
            "time" in var.lower() or "t_" in var.lower() for var in variables
        )

        # Determine sample complexity
        if n_samples < 100:
            sample_complexity = "low"
        elif n_samples < 1000:
            sample_complexity = "medium"
        else:
            sample_complexity = "high"

        return DataCharacteristics(
            n_samples=n_samples,
            n_variables=n_variables,
            has_missing_values=has_missing,
            has_mixed_types=has_mixed_types,
            estimated_latent_confounders=estimated_latent_confounders,
            temporal_structure=temporal_structure,
            nonlinearity_score=nonlinearity_score,
            noise_level=noise_level,
            sample_complexity=sample_complexity,
        )

    def _estimate_nonlinearity(self, data: Dict[str, np.ndarray]) -> float:
        """Estimate nonlinearity in the data using correlation vs MI comparison."""
        try:
            variables = list(data.keys())
            if len(variables) < 2:
                return 0.0

            nonlinearity_scores = []

            # Compare a few variable pairs
            for i in range(min(5, len(variables))):
                for j in range(i + 1, min(i + 3, len(variables))):
                    var1, var2 = variables[i], variables[j]

                    # Get data
                    x = data[var1].flatten() if data[var1].ndim > 1 else data[var1]
                    y = data[var2].flatten() if data[var2].ndim > 1 else data[var2]

                    # Ensure same length
                    min_len = min(len(x), len(y))
                    x, y = x[:min_len], y[:min_len]

                    if len(x) < 10:
                        continue

                    # Compute correlation
                    corr = abs(np.corrcoef(x, y)[0, 1])
                    if np.isnan(corr):
                        continue

                    # Compute mutual information (simplified)
                    try:
                        from sklearn.feature_selection import mutual_info_regression

                        mi = mutual_info_regression(
                            x.reshape(-1, 1), y, random_state=self.random_state
                        )[0]

                        # Nonlinearity indicator: MI much higher than correlation^2
                        if corr > 0.01:  # Avoid division by very small numbers
                            nonlinearity = max(0, (mi - corr**2) / (corr**2 + 0.01))
                            nonlinearity_scores.append(nonlinearity)
                    except:
                        continue

            return np.mean(nonlinearity_scores) if nonlinearity_scores else 0.0

        except Exception as e:
            warnings.warn(f"Nonlinearity estimation failed: {e}")
            return 0.0

    def _estimate_noise_level(self, data: Dict[str, np.ndarray]) -> float:
        """Estimate noise level in the data."""
        try:
            noise_estimates = []

            for var_name, var_data in data.items():
                # Flatten data
                flat_data = var_data.flatten() if var_data.ndim > 1 else var_data

                if len(flat_data) < 10:
                    continue

                # Estimate noise using signal-to-noise ratio approximation
                # Use coefficient of variation as a simple noise indicator
                if np.std(flat_data) > 0:
                    cv = np.std(flat_data) / (abs(np.mean(flat_data)) + 1e-10)
                    noise_estimates.append(min(cv, 2.0))  # Cap at reasonable value

            return np.mean(noise_estimates) if noise_estimates else 0.5

        except Exception as e:
            warnings.warn(f"Noise level estimation failed: {e}")
            return 0.5

    def _estimate_latent_confounders(self, data: Dict[str, np.ndarray]) -> bool:
        """
        Estimate presence of latent confounders using simplified tetrad test.

        This is a simplified version that looks for patterns suggesting hidden variables.
        """
        try:
            variables = list(data.keys())
            if len(variables) < 4:
                return False

            # Look for tetrad patterns in correlation matrix
            # Simplified: check if there are many high correlations suggesting common causes

            correlations = []
            for i in range(len(variables)):
                for j in range(i + 1, len(variables)):
                    var1, var2 = variables[i], variables[j]

                    x = data[var1].flatten() if data[var1].ndim > 1 else data[var1]
                    y = data[var2].flatten() if data[var2].ndim > 1 else data[var2]

                    min_len = min(len(x), len(y))
                    x, y = x[:min_len], y[:min_len]

                    if len(x) >= 10:
                        corr = abs(np.corrcoef(x, y)[0, 1])
                        if not np.isnan(corr):
                            correlations.append(corr)

            if not correlations:
                return False

            # If many variables are highly correlated, suggest latent confounders
            high_corr_fraction = np.mean(np.array(correlations) > 0.5)
            return high_corr_fraction > 0.3

        except Exception as e:
            warnings.warn(f"Latent confounder estimation failed: {e}")
            return False

    def _select_methods(self, data_chars: DataCharacteristics) -> List[str]:
        """
        Select appropriate methods based on data characteristics.

        Args:
            data_chars: Data characteristics

        Returns:
            List of method names to use
        """
        selected_methods = []

        # Always include PC algorithm as baseline
        selected_methods.append("pc")

        # Include FCI if latent confounders are suspected
        if data_chars.estimated_latent_confounders or data_chars.n_variables > 10:
            selected_methods.append("fci")

        # Include mutual information for nonlinear relationships or small samples
        if (
            data_chars.nonlinearity_score > 0.2
            or data_chars.sample_complexity == "low"
            or data_chars.noise_level > 0.8
        ):
            selected_methods.append("mutual_info")

        # Ensure at least two methods for ensemble
        if len(selected_methods) == 1:
            selected_methods.append("mutual_info")

        return selected_methods

    def _compute_method_weights(
        self, data_chars: DataCharacteristics, selected_methods: List[str]
    ) -> Dict[str, float]:
        """
        Compute weights for each method based on data characteristics.

        Args:
            data_chars: Data characteristics
            selected_methods: Selected methods

        Returns:
            Dictionary of method weights
        """
        weights = {}

        for method in selected_methods:
            if method == "pc":
                # PC works well with linear relationships and sufficient samples
                weight = 1.0
                if data_chars.sample_complexity == "low":
                    weight *= 0.7
                if data_chars.nonlinearity_score > 0.5:
                    weight *= 0.8
                weights["pc"] = weight

            elif method == "fci":
                # FCI is better for latent confounders but needs more samples
                weight = 1.0
                if data_chars.estimated_latent_confounders:
                    weight *= 1.3
                if data_chars.sample_complexity == "low":
                    weight *= 0.6
                weights["fci"] = weight

            elif method == "mutual_info":
                # MI is good for nonlinear relationships and robust to noise
                weight = 1.0
                if data_chars.nonlinearity_score > 0.3:
                    weight *= 1.2
                if data_chars.noise_level > 0.7:
                    weight *= 1.1
                if data_chars.sample_complexity == "low":
                    weight *= 1.1
                weights["mutual_info"] = weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _run_pc_method(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[PCResult, MethodPerformance]:
        """Run PC algorithm and measure performance."""
        import time

        start_time = time.time()

        try:
            result = self.pc_algorithm.discover_causal_structure(data)
            execution_time = time.time() - start_time

            # Compute performance metrics
            convergence_score = self._compute_convergence_score(
                result.independence_tests
            )
            stability_score = self._compute_stability_score_pc(result, data)
            confidence_score = self._compute_confidence_score_pc(result)

            performance = MethodPerformance(
                method_name="pc",
                execution_time=execution_time,
                convergence_score=convergence_score,
                stability_score=stability_score,
                edge_count=result.skeleton.number_of_edges(),
                confidence_score=confidence_score,
                validation_score=0.0,  # Will be computed later
            )

            return result, performance

        except Exception as e:
            self.logger.warning(f"PC algorithm failed: {e}")
            # Return empty result
            empty_graph = nx.Graph()
            empty_digraph = nx.DiGraph()
            empty_result = PCResult(empty_graph, empty_digraph, [], [], [])

            performance = MethodPerformance(
                method_name="pc",
                execution_time=time.time() - start_time,
                convergence_score=0.0,
                stability_score=0.0,
                edge_count=0,
                confidence_score=0.0,
                validation_score=0.0,
            )

            return empty_result, performance

    def _run_fci_method(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[FCIResult, MethodPerformance]:
        """Run FCI algorithm and measure performance."""
        import time

        start_time = time.time()

        try:
            result = self.fci_algorithm.discover_causal_structure(data)
            execution_time = time.time() - start_time

            # Compute performance metrics
            convergence_score = self._compute_convergence_score(
                result.independence_tests
            )
            stability_score = self._compute_stability_score_fci(result, data)
            confidence_score = self._compute_confidence_score_fci(result)

            performance = MethodPerformance(
                method_name="fci",
                execution_time=execution_time,
                convergence_score=convergence_score,
                stability_score=stability_score,
                edge_count=result.pag.number_of_edges(),
                confidence_score=confidence_score,
                validation_score=0.0,
            )

            return result, performance

        except Exception as e:
            self.logger.warning(f"FCI algorithm failed: {e}")
            # Return empty result
            empty_graph = nx.Graph()
            empty_digraph = nx.DiGraph()
            empty_result = FCIResult(empty_digraph, empty_graph, [], [], [], {})

            performance = MethodPerformance(
                method_name="fci",
                execution_time=time.time() - start_time,
                convergence_score=0.0,
                stability_score=0.0,
                edge_count=0,
                confidence_score=0.0,
                validation_score=0.0,
            )

            return empty_result, performance

    def _run_mi_method(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[MutualInfoResult, MethodPerformance]:
        """Run mutual information method and measure performance."""
        import time

        start_time = time.time()

        try:
            result = self.mi_discovery.discover_causal_structure(data)
            execution_time = time.time() - start_time

            # Compute performance metrics
            convergence_score = 1.0  # MI method doesn't have iterative convergence
            stability_score = self._compute_stability_score_mi(result, data)
            confidence_score = self._compute_confidence_score_mi(result)

            performance = MethodPerformance(
                method_name="mutual_info",
                execution_time=execution_time,
                convergence_score=convergence_score,
                stability_score=stability_score,
                edge_count=result.causal_graph.number_of_edges(),
                confidence_score=confidence_score,
                validation_score=0.0,
            )

            return result, performance

        except Exception as e:
            self.logger.warning(f"Mutual information method failed: {e}")
            # Return empty result
            empty_graph = nx.DiGraph()
            empty_matrix = np.zeros((len(data), len(data)))
            empty_result = MutualInfoResult(
                empty_graph, empty_matrix, empty_matrix, {}, list(data.keys()), {}
            )

            performance = MethodPerformance(
                method_name="mutual_info",
                execution_time=time.time() - start_time,
                convergence_score=0.0,
                stability_score=0.0,
                edge_count=0,
                confidence_score=0.0,
                validation_score=0.0,
            )

            return empty_result, performance

    def _compute_convergence_score(self, independence_tests: List) -> float:
        """Compute convergence score based on independence test consistency."""
        if not independence_tests:
            return 0.0

        # Measure consistency of p-values (lower variance indicates better convergence)
        p_values = [
            test.p_value for test in independence_tests if hasattr(test, "p_value")
        ]

        if not p_values:
            return 0.0

        # Convergence score based on p-value distribution
        p_array = np.array(p_values)

        # Good convergence: clear separation between significant and non-significant
        significant = p_array < self.alpha
        if len(significant) == 0:
            return 0.0

        sig_fraction = np.mean(significant)

        # Optimal convergence when we have clear decisions (not too many borderline cases)
        borderline_fraction = np.mean(
            (p_array > self.alpha / 2) & (p_array < self.alpha * 2)
        )
        convergence_score = 1.0 - borderline_fraction

        return max(0.0, min(1.0, convergence_score))

    def _compute_stability_score_pc(
        self, result: PCResult, data: Dict[str, np.ndarray]
    ) -> float:
        """Compute stability score for PC algorithm using bootstrap."""
        if len(data) == 0:
            return 0.0

        try:
            # Simplified stability assessment
            n_samples = len(next(iter(data.values())))
            if n_samples < 20:
                return 0.5  # Default for small samples

            # Run a few bootstrap samples to check stability
            n_bootstrap = min(5, self.n_bootstrap // 20)  # Much smaller for efficiency
            edge_agreements = []

            original_edges = set(result.skeleton.edges())

            for _ in range(n_bootstrap):
                try:
                    # Bootstrap sample
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    bootstrap_data = {
                        var: values[indices] for var, values in data.items()
                    }

                    # Run PC on bootstrap sample
                    bootstrap_result = self.pc_algorithm.discover_causal_structure(
                        bootstrap_data
                    )
                    bootstrap_edges = set(bootstrap_result.skeleton.edges())

                    # Compute edge agreement
                    if len(original_edges) == 0 and len(bootstrap_edges) == 0:
                        agreement = 1.0
                    elif len(original_edges) == 0 or len(bootstrap_edges) == 0:
                        agreement = 0.0
                    else:
                        intersection = len(original_edges.intersection(bootstrap_edges))
                        union = len(original_edges.union(bootstrap_edges))
                        agreement = intersection / union if union > 0 else 0.0

                    edge_agreements.append(agreement)

                except Exception:
                    continue

            return np.mean(edge_agreements) if edge_agreements else 0.5

        except Exception as e:
            warnings.warn(f"PC stability computation failed: {e}")
            return 0.5

    def _compute_stability_score_fci(
        self, result: FCIResult, data: Dict[str, np.ndarray]
    ) -> float:
        """Compute stability score for FCI algorithm."""
        # Similar to PC but for PAG edges
        if len(data) == 0:
            return 0.0

        try:
            n_samples = len(next(iter(data.values())))
            if n_samples < 20:
                return 0.5

            # Simplified stability check
            n_bootstrap = min(3, self.n_bootstrap // 30)  # Even smaller for FCI
            edge_agreements = []

            original_edges = set(result.pag.edges())

            for _ in range(n_bootstrap):
                try:
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    bootstrap_data = {
                        var: values[indices] for var, values in data.items()
                    }

                    bootstrap_result = self.fci_algorithm.discover_causal_structure(
                        bootstrap_data
                    )
                    bootstrap_edges = set(bootstrap_result.pag.edges())

                    if len(original_edges) == 0 and len(bootstrap_edges) == 0:
                        agreement = 1.0
                    elif len(original_edges) == 0 or len(bootstrap_edges) == 0:
                        agreement = 0.0
                    else:
                        intersection = len(original_edges.intersection(bootstrap_edges))
                        union = len(original_edges.union(bootstrap_edges))
                        agreement = intersection / union if union > 0 else 0.0

                    edge_agreements.append(agreement)

                except Exception:
                    continue

            return np.mean(edge_agreements) if edge_agreements else 0.5

        except Exception as e:
            warnings.warn(f"FCI stability computation failed: {e}")
            return 0.5

    def _compute_stability_score_mi(
        self, result: MutualInfoResult, data: Dict[str, np.ndarray]
    ) -> float:
        """Compute stability score for mutual information method."""
        # Use bootstrap results if available
        if "edge_stability" in result.bootstrap_results:
            stability_scores = list(result.bootstrap_results["edge_stability"].values())
            return np.mean(stability_scores) if stability_scores else 0.5

        return 0.5  # Default if no bootstrap results

    def _compute_confidence_score_pc(self, result: PCResult) -> float:
        """Compute confidence score for PC algorithm results."""
        if not result.independence_tests:
            return 0.0

        # Average confidence based on p-values
        p_values = [test.p_value for test in result.independence_tests]

        # Convert p-values to confidence scores
        confidence_scores = []
        for p_val in p_values:
            if p_val < self.alpha:
                # Significant result - confidence increases as p-value decreases
                confidence = 1.0 - (p_val / self.alpha)
            else:
                # Non-significant result - confidence based on how far from threshold
                confidence = min(1.0, (p_val - self.alpha) / (1.0 - self.alpha))

            confidence_scores.append(confidence)

        return np.mean(confidence_scores)

    def _compute_confidence_score_fci(self, result: FCIResult) -> float:
        """Compute confidence score for FCI algorithm results."""
        return self._compute_confidence_score_pc(result)  # Same logic

    def _compute_confidence_score_mi(self, result: MutualInfoResult) -> float:
        """Compute confidence score for mutual information results."""
        if result.causal_graph.number_of_edges() == 0:
            return 0.0

        # Use p-values from edges
        p_values = []
        for u, v, data in result.causal_graph.edges(data=True):
            if "p_value" in data:
                p_values.append(data["p_value"])

        if not p_values:
            return 0.5

        # Convert to confidence scores
        confidence_scores = []
        for p_val in p_values:
            if p_val < self.alpha:
                confidence = 1.0 - (p_val / self.alpha)
            else:
                confidence = min(1.0, (p_val - self.alpha) / (1.0 - self.alpha))
            confidence_scores.append(confidence)

        return np.mean(confidence_scores)

    def _create_consensus_graph(
        self,
        method_results: Dict[str, Any],
        method_weights: Dict[str, float],
        variables: List[str],
    ) -> nx.DiGraph:
        """
        Create consensus graph from multiple method results.

        Args:
            method_results: Results from individual methods
            method_weights: Weights for each method
            variables: List of variable names

        Returns:
            Consensus directed graph
        """
        consensus_graph = nx.DiGraph()
        consensus_graph.add_nodes_from(variables)

        # Collect edges from all methods with weights
        edge_votes = {}  # (source, target) -> [(weight, strength, method)]

        # Process PC results
        if "pc" in method_results and "pc" in method_weights:
            pc_result = method_results["pc"]
            method_weight = method_weights["pc"]

            # Add edges from skeleton
            for u, v in pc_result.skeleton.edges():
                if u in variables and v in variables:
                    # Get strength from independence tests
                    strength = self._get_edge_strength_pc(pc_result, u, v)

                    # Add both directions for undirected edges
                    for source, target in [(u, v), (v, u)]:
                        if (source, target) not in edge_votes:
                            edge_votes[(source, target)] = []
                        edge_votes[(source, target)].append(
                            (method_weight, strength, "pc")
                        )

        # Process FCI results
        if "fci" in method_results and "fci" in method_weights:
            fci_result = method_results["fci"]
            method_weight = method_weights["fci"]

            # Add edges from PAG
            for u, v in fci_result.pag.edges():
                if u in variables and v in variables:
                    strength = self._get_edge_strength_fci(fci_result, u, v)

                    if (u, v) not in edge_votes:
                        edge_votes[(u, v)] = []
                    edge_votes[(u, v)].append((method_weight, strength, "fci"))

        # Process MI results
        if "mutual_info" in method_results and "mutual_info" in method_weights:
            mi_result = method_results["mutual_info"]
            method_weight = method_weights["mutual_info"]

            # Add edges from causal graph
            for u, v, data in mi_result.causal_graph.edges(data=True):
                if u in variables and v in variables:
                    strength = data.get("weight", 0.0)

                    if (u, v) not in edge_votes:
                        edge_votes[(u, v)] = []
                    edge_votes[(u, v)].append((method_weight, strength, "mutual_info"))

        # Create consensus based on ensemble strategy
        if self.ensemble_strategy == "weighted_voting":
            consensus_graph = self._weighted_voting_consensus(edge_votes, variables)
        elif self.ensemble_strategy == "majority_vote":
            consensus_graph = self._majority_vote_consensus(edge_votes, variables)
        elif self.ensemble_strategy == "confidence_weighted":
            consensus_graph = self._confidence_weighted_consensus(edge_votes, variables)
        else:
            # Default to weighted voting
            consensus_graph = self._weighted_voting_consensus(edge_votes, variables)

        return consensus_graph

    def _get_edge_strength_pc(self, pc_result: PCResult, u: str, v: str) -> float:
        """Get edge strength from PC result."""
        max_strength = 0.0

        for test in pc_result.independence_tests:
            if (test.variables[0] == u and test.variables[1] == v) or (
                test.variables[0] == v and test.variables[1] == u
            ):
                if not test.is_independent:
                    max_strength = max(max_strength, test.test_statistic)

        return max_strength

    def _get_edge_strength_fci(self, fci_result: FCIResult, u: str, v: str) -> float:
        """Get edge strength from FCI result."""
        max_strength = 0.0

        for test in fci_result.independence_tests:
            if (test.variables[0] == u and test.variables[1] == v) or (
                test.variables[0] == v and test.variables[1] == u
            ):
                if not test.is_independent:
                    max_strength = max(max_strength, test.test_statistic)

        return max_strength

    def _weighted_voting_consensus(
        self,
        edge_votes: Dict[Tuple[str, str], List[Tuple[float, float, str]]],
        variables: List[str],
    ) -> nx.DiGraph:
        """Create consensus using weighted voting."""
        consensus_graph = nx.DiGraph()
        consensus_graph.add_nodes_from(variables)

        for (source, target), votes in edge_votes.items():
            # Compute weighted score
            total_weight = 0.0
            weighted_strength = 0.0

            for method_weight, strength, method_name in votes:
                total_weight += method_weight
                weighted_strength += method_weight * strength

            if total_weight > 0:
                avg_strength = weighted_strength / total_weight

                # Add edge if weighted strength exceeds threshold
                if avg_strength > 0.05 and total_weight > 0.3:  # More lenient consensus
                    consensus_graph.add_edge(
                        source,
                        target,
                        weight=avg_strength,
                        consensus_score=total_weight,
                        supporting_methods=[method for _, _, method in votes],
                    )

        return consensus_graph

    def _majority_vote_consensus(
        self,
        edge_votes: Dict[Tuple[str, str], List[Tuple[float, float, str]]],
        variables: List[str],
    ) -> nx.DiGraph:
        """Create consensus using majority voting."""
        consensus_graph = nx.DiGraph()
        consensus_graph.add_nodes_from(variables)

        total_methods = len(self.method_weights)

        for (source, target), votes in edge_votes.items():
            # Count votes (each method gets one vote)
            vote_count = len(set(method for _, _, method in votes))

            if vote_count > total_methods / 2:  # Majority
                # Use average strength
                avg_strength = np.mean([strength for _, strength, _ in votes])

                consensus_graph.add_edge(
                    source,
                    target,
                    weight=avg_strength,
                    vote_count=vote_count,
                    supporting_methods=[method for _, _, method in votes],
                )

        return consensus_graph

    def _confidence_weighted_consensus(
        self,
        edge_votes: Dict[Tuple[str, str], List[Tuple[float, float, str]]],
        variables: List[str],
    ) -> nx.DiGraph:
        """Create consensus using confidence-weighted voting."""
        # Similar to weighted voting but with confidence adjustment
        return self._weighted_voting_consensus(edge_votes, variables)

    def _compute_confidence_matrix(
        self,
        method_results: Dict[str, Any],
        method_weights: Dict[str, float],
        variables: List[str],
    ) -> np.ndarray:
        """
        Compute confidence matrix for all variable pairs.

        Args:
            method_results: Results from individual methods
            method_weights: Method weights
            variables: Variable names

        Returns:
            Confidence matrix
        """
        n_vars = len(variables)
        confidence_matrix = np.zeros((n_vars, n_vars))
        var_to_idx = {var: i for i, var in enumerate(variables)}

        # Aggregate confidence scores from all methods
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    total_confidence = 0.0
                    total_weight = 0.0

                    # PC method confidence
                    if "pc" in method_results and "pc" in method_weights:
                        pc_confidence = self._get_pair_confidence_pc(
                            method_results["pc"], var1, var2
                        )
                        weight = method_weights["pc"]
                        total_confidence += weight * pc_confidence
                        total_weight += weight

                    # FCI method confidence
                    if "fci" in method_results and "fci" in method_weights:
                        fci_confidence = self._get_pair_confidence_fci(
                            method_results["fci"], var1, var2
                        )
                        weight = method_weights["fci"]
                        total_confidence += weight * fci_confidence
                        total_weight += weight

                    # MI method confidence
                    if (
                        "mutual_info" in method_results
                        and "mutual_info" in method_weights
                    ):
                        mi_confidence = self._get_pair_confidence_mi(
                            method_results["mutual_info"], var1, var2
                        )
                        weight = method_weights["mutual_info"]
                        total_confidence += weight * mi_confidence
                        total_weight += weight

                    if total_weight > 0:
                        confidence_matrix[i, j] = total_confidence / total_weight

        return confidence_matrix

    def _get_pair_confidence_pc(
        self, pc_result: PCResult, var1: str, var2: str
    ) -> float:
        """Get confidence for a variable pair from PC results."""
        for test in pc_result.independence_tests:
            if (test.variables[0] == var1 and test.variables[1] == var2) or (
                test.variables[0] == var2 and test.variables[1] == var1
            ):

                if test.p_value < self.alpha:
                    return 1.0 - (test.p_value / self.alpha)
                else:
                    return min(1.0, (test.p_value - self.alpha) / (1.0 - self.alpha))

        return 0.0

    def _get_pair_confidence_fci(
        self, fci_result: FCIResult, var1: str, var2: str
    ) -> float:
        """Get confidence for a variable pair from FCI results."""
        return self._get_pair_confidence_pc(fci_result, var1, var2)

    def _get_pair_confidence_mi(
        self, mi_result: MutualInfoResult, var1: str, var2: str
    ) -> float:
        """Get confidence for a variable pair from MI results."""
        if mi_result.causal_graph.has_edge(var1, var2):
            edge_data = mi_result.causal_graph[var1][var2]
            p_value = edge_data.get("p_value", 1.0)

            if p_value < self.alpha:
                return 1.0 - (p_value / self.alpha)
            else:
                return min(1.0, (p_value - self.alpha) / (1.0 - self.alpha))

        return 0.0

    def _validate_ensemble_result(
        self,
        consensus_graph: nx.DiGraph,
        method_results: Dict[str, Any],
        data: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Validate ensemble results using multiple validation strategies.

        Args:
            consensus_graph: Consensus causal graph
            method_results: Individual method results
            data: Original data

        Returns:
            Dictionary of validation metrics
        """
        validation_metrics = {}

        # 1. Graph consistency validation
        validation_metrics["graph_consistency"] = self._validate_graph_consistency(
            consensus_graph
        )

        # 2. Cross-method agreement
        validation_metrics["method_agreement"] = self._compute_method_agreement(
            method_results
        )

        # 3. Statistical validation
        validation_metrics["statistical_significance"] = (
            self._validate_statistical_significance(consensus_graph, data)
        )

        # 4. Physics consistency (if applicable)
        validation_metrics["physics_consistency"] = self._validate_physics_consistency(
            consensus_graph
        )

        # 5. Overall validation score
        validation_metrics["overall_score"] = np.mean(
            [
                validation_metrics["graph_consistency"],
                validation_metrics["method_agreement"],
                validation_metrics["statistical_significance"],
                validation_metrics["physics_consistency"],
            ]
        )

        return validation_metrics

    def _validate_graph_consistency(self, graph: nx.DiGraph) -> float:
        """Validate graph consistency (no cycles, reasonable structure)."""
        try:
            # Check for cycles
            if nx.is_directed_acyclic_graph(graph):
                acyclic_score = 1.0
            else:
                # Penalize based on number of cycles
                try:
                    cycles = list(nx.simple_cycles(graph))
                    acyclic_score = max(
                        0.0, 1.0 - len(cycles) / max(1, graph.number_of_nodes())
                    )
                except:
                    acyclic_score = 0.5

            # Check connectivity (not too sparse, not too dense)
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()

            if n_nodes <= 1:
                connectivity_score = 0.0
            else:
                max_edges = n_nodes * (n_nodes - 1)
                density = n_edges / max_edges if max_edges > 0 else 0.0

                # Optimal density is around 0.1-0.3 for causal graphs
                if 0.05 <= density <= 0.4:
                    connectivity_score = 1.0
                elif density < 0.05:
                    connectivity_score = density / 0.05
                else:
                    connectivity_score = max(0.0, 1.0 - (density - 0.4) / 0.6)

            return (acyclic_score + connectivity_score) / 2.0

        except Exception as e:
            warnings.warn(f"Graph consistency validation failed: {e}")
            return 0.5

    def _compute_method_agreement(self, method_results: Dict[str, Any]) -> float:
        """Compute agreement between different methods."""
        if len(method_results) < 2:
            return 1.0  # Perfect agreement if only one method

        try:
            # Compare edge sets between methods
            edge_sets = []

            for method_name, result in method_results.items():
                if method_name == "pc":
                    edges = set(result.skeleton.edges())
                elif method_name == "fci":
                    edges = set(result.pag.edges())
                elif method_name == "mutual_info":
                    edges = set(result.causal_graph.edges())
                else:
                    continue

                edge_sets.append(edges)

            if len(edge_sets) < 2:
                return 1.0

            # Compute pairwise Jaccard similarities
            similarities = []
            for i in range(len(edge_sets)):
                for j in range(i + 1, len(edge_sets)):
                    set1, set2 = edge_sets[i], edge_sets[j]

                    if len(set1) == 0 and len(set2) == 0:
                        similarity = 1.0
                    elif len(set1) == 0 or len(set2) == 0:
                        similarity = 0.0
                    else:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        similarity = intersection / union if union > 0 else 0.0

                    similarities.append(similarity)

            return np.mean(similarities) if similarities else 0.0

        except Exception as e:
            warnings.warn(f"Method agreement computation failed: {e}")
            return 0.5

    def _validate_statistical_significance(
        self, graph: nx.DiGraph, data: Dict[str, np.ndarray]
    ) -> float:
        """Validate statistical significance of discovered edges."""
        if graph.number_of_edges() == 0:
            return 0.0

        try:
            significant_edges = 0
            total_edges = 0

            for u, v, edge_data in graph.edges(data=True):
                if u in data and v in data:
                    total_edges += 1

                    # Check if edge has significance information
                    if "p_value" in edge_data:
                        if edge_data["p_value"] < self.alpha:
                            significant_edges += 1
                    else:
                        # Compute significance using simple correlation test
                        x = data[u].flatten() if data[u].ndim > 1 else data[u]
                        y = data[v].flatten() if data[v].ndim > 1 else data[v]

                        min_len = min(len(x), len(y))
                        x, y = x[:min_len], y[:min_len]

                        if len(x) >= 10:
                            try:
                                from scipy.stats import pearsonr

                                _, p_value = pearsonr(x, y)
                                if p_value < self.alpha:
                                    significant_edges += 1
                            except:
                                pass

            return significant_edges / total_edges if total_edges > 0 else 0.0

        except Exception as e:
            warnings.warn(f"Statistical significance validation failed: {e}")
            return 0.5

    def _validate_physics_consistency(self, graph: nx.DiGraph) -> float:
        """Validate physics consistency of discovered relationships."""
        try:
            # Basic physics consistency checks
            consistency_score = 1.0

            # Check for known physics relationships
            physics_patterns = [
                ("reynolds_number", "viscosity"),
                ("shear_rate", "viscosity"),
                ("temperature", "viscosity"),
                ("velocity", "pressure"),
                ("density", "pressure"),
            ]

            expected_relationships = 0
            found_relationships = 0

            for source_pattern, target_pattern in physics_patterns:
                # Look for variables matching patterns
                source_vars = [v for v in graph.nodes() if source_pattern in v.lower()]
                target_vars = [v for v in graph.nodes() if target_pattern in v.lower()]

                if source_vars and target_vars:
                    expected_relationships += 1

                    # Check if any relationship exists
                    relationship_found = False
                    for s_var in source_vars:
                        for t_var in target_vars:
                            if graph.has_edge(s_var, t_var) or graph.has_edge(
                                t_var, s_var
                            ):
                                relationship_found = True
                                break
                        if relationship_found:
                            break

                    if relationship_found:
                        found_relationships += 1

            if expected_relationships > 0:
                physics_score = found_relationships / expected_relationships
            else:
                physics_score = 1.0  # No expected relationships to validate

            return physics_score

        except Exception as e:
            warnings.warn(f"Physics consistency validation failed: {e}")
            return 0.5

    def get_ensemble_summary(self, result: EnsembleResult) -> Dict[str, Any]:
        """
        Get comprehensive summary of ensemble causal discovery results.

        Args:
            result: EnsembleResult from ensemble discovery

        Returns:
            Dictionary containing summary information
        """
        summary = {
            "data_characteristics": {
                "n_samples": result.data_characteristics.n_samples,
                "n_variables": result.data_characteristics.n_variables,
                "sample_complexity": result.data_characteristics.sample_complexity,
                "nonlinearity_score": result.data_characteristics.nonlinearity_score,
                "noise_level": result.data_characteristics.noise_level,
                "estimated_latent_confounders": result.data_characteristics.estimated_latent_confounders,
            },
            "consensus_graph": {
                "n_nodes": result.consensus_graph.number_of_nodes(),
                "n_edges": result.consensus_graph.number_of_edges(),
                "density": (
                    result.consensus_graph.number_of_edges()
                    / max(
                        1,
                        result.consensus_graph.number_of_nodes()
                        * (result.consensus_graph.number_of_nodes() - 1),
                    )
                ),
            },
            "method_performances": {
                perf.method_name: {
                    "execution_time": perf.execution_time,
                    "stability_score": perf.stability_score,
                    "confidence_score": perf.confidence_score,
                    "edge_count": perf.edge_count,
                }
                for perf in result.method_performances
            },
            "ensemble_weights": result.ensemble_weights,
            "validation_metrics": result.validation_metrics,
            "strongest_relationships": self._get_strongest_relationships(
                result.consensus_graph, top_k=5
            ),
        }

        return summary

    def _get_strongest_relationships(
        self, graph: nx.DiGraph, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get the strongest causal relationships from the graph."""
        relationships = []

        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 0.0)
            consensus_score = data.get("consensus_score", 0.0)
            supporting_methods = data.get("supporting_methods", [])

            relationships.append(
                {
                    "source": u,
                    "target": v,
                    "strength": weight,
                    "consensus_score": consensus_score,
                    "supporting_methods": supporting_methods,
                }
            )

        # Sort by strength and return top k
        relationships.sort(key=lambda x: x["strength"], reverse=True)
        return relationships[:top_k]

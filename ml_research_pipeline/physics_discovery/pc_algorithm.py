"""
PC Algorithm Implementation for Causal Discovery

This module implements the PC (Peter-Clark) algorithm for causal discovery
with conditional independence testing and statistical significance assessment.
"""

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


@dataclass
class ConditionalIndependenceTest:
    """Result of a conditional independence test."""

    variables: Tuple[str, str]
    conditioning_set: Set[str]
    test_statistic: float
    p_value: float
    is_independent: bool
    test_method: str


@dataclass
class PCResult:
    """Result from PC algorithm execution."""

    skeleton: nx.Graph
    directed_graph: nx.DiGraph
    independence_tests: List[ConditionalIndependenceTest]
    removed_edges: List[Tuple[str, str, Set[str]]]
    orientation_rules: List[str]


class PCAlgorithm:
    """
    Implementation of the PC algorithm for causal discovery.

    The PC algorithm discovers causal relationships by:
    1. Starting with a complete undirected graph
    2. Removing edges based on conditional independence tests
    3. Orienting edges using causal orientation rules
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_conditioning_size: int = 3,
        independence_test: str = "partial_correlation",
        verbose: bool = False,
    ):
        """
        Initialize PC algorithm.

        Args:
            alpha: Significance level for independence tests
            max_conditioning_size: Maximum size of conditioning sets
            independence_test: Type of independence test ('partial_correlation', 'mutual_info')
            verbose: Whether to print detailed progress
        """
        self.alpha = alpha
        self.max_conditioning_size = max_conditioning_size
        self.independence_test = independence_test
        self.verbose = verbose
        self.scaler = StandardScaler()

        # Store test results for analysis
        self.independence_tests = []
        self.removed_edges = []
        self.orientation_rules = []

    def discover_causal_structure(self, data: Dict[str, np.ndarray]) -> PCResult:
        """
        Discover causal structure using PC algorithm.

        Args:
            data: Dictionary with variable names as keys and data arrays as values

        Returns:
            PCResult containing skeleton, directed graph, and test results
        """
        if self.verbose:
            print("Starting PC algorithm for causal discovery...")

        # Prepare data
        prepared_data = self._prepare_data(data)
        variables = list(prepared_data.keys())

        # Phase 1: Learn skeleton (undirected graph)
        skeleton = self._learn_skeleton(prepared_data, variables)

        if self.verbose:
            print(f"Skeleton learned with {skeleton.number_of_edges()} edges")

        # Phase 2: Orient edges
        directed_graph = self._orient_edges(skeleton, prepared_data)

        if self.verbose:
            print(
                f"Directed graph created with {directed_graph.number_of_edges()} directed edges"
            )

        return PCResult(
            skeleton=skeleton,
            directed_graph=directed_graph,
            independence_tests=self.independence_tests.copy(),
            removed_edges=self.removed_edges.copy(),
            orientation_rules=self.orientation_rules.copy(),
        )

    def _prepare_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prepare and standardize data for analysis."""
        prepared_data = {}

        for var_name, var_data in data.items():
            # Flatten if multidimensional
            if var_data.ndim > 1:
                if var_data.shape[1] == 1:
                    var_data = var_data.flatten()
                else:
                    # Take mean across features for multidimensional data
                    var_data = np.mean(var_data, axis=1)

            # Standardize
            var_data = self.scaler.fit_transform(var_data.reshape(-1, 1)).flatten()
            prepared_data[var_name] = var_data

        return prepared_data

    def _learn_skeleton(
        self, data: Dict[str, np.ndarray], variables: List[str]
    ) -> nx.Graph:
        """
        Learn the skeleton (undirected graph) by testing conditional independence.

        Args:
            data: Prepared data dictionary
            variables: List of variable names

        Returns:
            Undirected graph representing the skeleton
        """
        # Start with complete graph
        skeleton = nx.complete_graph(variables)

        # Test conditional independence for increasing conditioning set sizes
        for conditioning_size in range(self.max_conditioning_size + 1):
            if self.verbose:
                print(f"Testing with conditioning set size: {conditioning_size}")

            # Get all edges to test (copy to avoid modification during iteration)
            edges_to_test = list(skeleton.edges())

            for var1, var2 in edges_to_test:
                if not skeleton.has_edge(var1, var2):
                    continue  # Edge already removed

                # Get potential conditioning variables (neighbors of var1 and var2)
                neighbors_var1 = set(skeleton.neighbors(var1)) - {var2}
                neighbors_var2 = set(skeleton.neighbors(var2)) - {var1}
                potential_conditioning = neighbors_var1.union(neighbors_var2)

                # Test all conditioning sets of current size
                if len(potential_conditioning) >= conditioning_size:
                    for conditioning_set in combinations(
                        potential_conditioning, conditioning_size
                    ):
                        conditioning_set = set(conditioning_set)

                        # Perform conditional independence test
                        test_result = self._test_conditional_independence(
                            var1, var2, conditioning_set, data
                        )

                        self.independence_tests.append(test_result)

                        if test_result.is_independent:
                            # Remove edge and record the conditioning set
                            skeleton.remove_edge(var1, var2)
                            self.removed_edges.append((var1, var2, conditioning_set))

                            if self.verbose:
                                print(
                                    f"Removed edge {var1} - {var2} | {conditioning_set}"
                                )
                            break  # Move to next edge

        return skeleton

    def _test_conditional_independence(
        self,
        var1: str,
        var2: str,
        conditioning_set: Set[str],
        data: Dict[str, np.ndarray],
    ) -> ConditionalIndependenceTest:
        """
        Test conditional independence between two variables given a conditioning set.

        Args:
            var1: First variable
            var2: Second variable
            conditioning_set: Set of conditioning variables
            data: Data dictionary

        Returns:
            ConditionalIndependenceTest result
        """
        if self.independence_test == "partial_correlation":
            return self._partial_correlation_test(var1, var2, conditioning_set, data)
        elif self.independence_test == "mutual_info":
            return self._mutual_info_test(var1, var2, conditioning_set, data)
        else:
            raise ValueError(f"Unknown independence test: {self.independence_test}")

    def _partial_correlation_test(
        self,
        var1: str,
        var2: str,
        conditioning_set: Set[str],
        data: Dict[str, np.ndarray],
    ) -> ConditionalIndependenceTest:
        """
        Test conditional independence using partial correlation.

        Args:
            var1: First variable
            var2: Second variable
            conditioning_set: Set of conditioning variables
            data: Data dictionary

        Returns:
            ConditionalIndependenceTest result
        """
        n_samples = len(data[var1])

        if len(conditioning_set) == 0:
            # Simple correlation test
            try:
                corr, p_value = pearsonr(data[var1], data[var2])
                test_statistic = abs(corr)

                # Handle NaN values
                if np.isnan(corr) or np.isnan(p_value):
                    test_statistic = 0.0
                    p_value = 1.0

            except Exception as e:
                warnings.warn(f"Error in correlation test: {e}")
                test_statistic = 0.0
                p_value = 1.0
        else:
            # Partial correlation test
            try:
                # Create data matrix
                all_vars = [var1, var2] + list(conditioning_set)
                data_matrix = np.column_stack([data[var] for var in all_vars])

                # Check for sufficient samples
                if n_samples <= len(all_vars) + 3:
                    test_statistic = 0.0
                    p_value = 1.0
                else:
                    # Compute correlation matrix
                    corr_matrix = np.corrcoef(data_matrix.T)

                    # Handle edge cases
                    if corr_matrix.shape[0] < 2 or np.any(np.isnan(corr_matrix)):
                        test_statistic = 0.0
                        p_value = 1.0
                    else:
                        # Compute partial correlation using matrix inversion
                        try:
                            # Add small regularization to avoid singularity
                            reg_matrix = corr_matrix + 1e-6 * np.eye(
                                corr_matrix.shape[0]
                            )
                            precision_matrix = np.linalg.inv(reg_matrix)

                            # Partial correlation coefficient
                            partial_corr = -precision_matrix[0, 1] / np.sqrt(
                                precision_matrix[0, 0] * precision_matrix[1, 1]
                            )

                            # Ensure partial correlation is in valid range
                            partial_corr = np.clip(partial_corr, -0.999, 0.999)
                            test_statistic = abs(partial_corr)

                            # Compute p-value using Fisher's z-transformation
                            if abs(partial_corr) >= 0.999:
                                p_value = 0.0
                            else:
                                z_score = 0.5 * np.log(
                                    (1 + partial_corr) / (1 - partial_corr)
                                )
                                z_score *= np.sqrt(
                                    max(1, n_samples - len(conditioning_set) - 3)
                                )
                                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                            # Singular matrix or other numerical issues
                            test_statistic = 0.0
                            p_value = 1.0

            except Exception as e:
                warnings.warn(f"Error in partial correlation test: {e}")
                test_statistic = 0.0
                p_value = 1.0

        is_independent = p_value > self.alpha

        return ConditionalIndependenceTest(
            variables=(var1, var2),
            conditioning_set=conditioning_set,
            test_statistic=test_statistic,
            p_value=p_value,
            is_independent=is_independent,
            test_method="partial_correlation",
        )

    def _mutual_info_test(
        self,
        var1: str,
        var2: str,
        conditioning_set: Set[str],
        data: Dict[str, np.ndarray],
    ) -> ConditionalIndependenceTest:
        """
        Test conditional independence using conditional mutual information.

        Args:
            var1: First variable
            var2: Second variable
            conditioning_set: Set of conditioning variables
            data: Data dictionary

        Returns:
            ConditionalIndependenceTest result
        """
        if len(conditioning_set) == 0:
            # Simple mutual information
            mi_score = mutual_info_regression(
                data[var1].reshape(-1, 1), data[var2], random_state=42
            )[0]
        else:
            # Conditional mutual information: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
            try:
                # I(X;Y,Z)
                conditioning_data = np.column_stack(
                    [data[var] for var in conditioning_set]
                )
                combined_data = np.column_stack([data[var2], conditioning_data])
                mi_xyz = mutual_info_regression(
                    data[var1].reshape(-1, 1), combined_data, random_state=42
                )[0]

                # I(X;Z)
                mi_xz = mutual_info_regression(
                    data[var1].reshape(-1, 1), conditioning_data, random_state=42
                )[0]

                mi_score = mi_xyz - mi_xz
                mi_score = max(0, mi_score)  # Ensure non-negative

            except Exception as e:
                warnings.warn(f"Error in conditional MI test: {e}")
                mi_score = 0.0

        # Bootstrap for p-value estimation
        n_bootstrap = 100
        bootstrap_scores = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            # Permute var2 to break relationship
            var2_perm = np.random.permutation(data[var2])

            if len(conditioning_set) == 0:
                bootstrap_mi = mutual_info_regression(
                    data[var1].reshape(-1, 1), var2_perm, random_state=42
                )[0]
            else:
                conditioning_data = np.column_stack(
                    [data[var] for var in conditioning_set]
                )
                combined_data = np.column_stack([var2_perm, conditioning_data])

                mi_xyz_boot = mutual_info_regression(
                    data[var1].reshape(-1, 1), combined_data, random_state=42
                )[0]

                mi_xz_boot = mutual_info_regression(
                    data[var1].reshape(-1, 1), conditioning_data, random_state=42
                )[0]

                bootstrap_mi = max(0, mi_xyz_boot - mi_xz_boot)

            bootstrap_scores.append(bootstrap_mi)

        # Compute p-value
        p_value = np.mean(np.array(bootstrap_scores) >= mi_score)
        is_independent = p_value > self.alpha

        return ConditionalIndependenceTest(
            variables=(var1, var2),
            conditioning_set=conditioning_set,
            test_statistic=mi_score,
            p_value=p_value,
            is_independent=is_independent,
            test_method="mutual_info",
        )

    def _orient_edges(
        self, skeleton: nx.Graph, data: Dict[str, np.ndarray]
    ) -> nx.DiGraph:
        """
        Orient edges in the skeleton using causal orientation rules.

        Args:
            skeleton: Undirected skeleton graph
            data: Data dictionary

        Returns:
            Directed graph with oriented edges
        """
        # Convert skeleton to directed graph (initially with no edge directions)
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(skeleton.nodes())

        # Apply orientation rules
        self._apply_v_structures(skeleton, directed_graph)
        self._apply_orientation_rules(directed_graph)

        return directed_graph

    def _apply_v_structures(self, skeleton: nx.Graph, directed_graph: nx.DiGraph):
        """
        Apply v-structure orientation rule: if X-Y-Z and X,Z not adjacent,
        then orient as X->Y<-Z.
        """
        for node in skeleton.nodes():
            neighbors = list(skeleton.neighbors(node))

            # Check all pairs of neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    neighbor1, neighbor2 = neighbors[i], neighbors[j]

                    # Check if neighbor1 and neighbor2 are not adjacent
                    if not skeleton.has_edge(neighbor1, neighbor2):
                        # Check if this forms a v-structure
                        # Look for conditioning set that separated neighbor1 and neighbor2
                        # but did not include the current node

                        separating_set = None
                        for var1, var2, cond_set in self.removed_edges:
                            if (var1 == neighbor1 and var2 == neighbor2) or (
                                var1 == neighbor2 and var2 == neighbor1
                            ):
                                if node not in cond_set:
                                    separating_set = cond_set
                                    break

                        if separating_set is not None:
                            # Orient as neighbor1 -> node <- neighbor2
                            directed_graph.add_edge(neighbor1, node)
                            directed_graph.add_edge(neighbor2, node)

                            rule_desc = (
                                f"V-structure: {neighbor1} -> {node} <- {neighbor2}"
                            )
                            self.orientation_rules.append(rule_desc)

                            if self.verbose:
                                print(f"Applied v-structure: {rule_desc}")

    def _apply_orientation_rules(self, directed_graph: nx.DiGraph):
        """
        Apply additional orientation rules to avoid cycles and conflicts.
        """
        changed = True
        iteration = 0
        max_iterations = 10

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # Rule 1: If X -> Y and Y - Z, and X and Z not adjacent, then Y -> Z
            for node in list(directed_graph.nodes()):
                predecessors = list(directed_graph.predecessors(node))

                for pred in predecessors:
                    # Find undirected neighbors of node
                    all_neighbors = set()
                    for n in directed_graph.nodes():
                        if directed_graph.has_edge(node, n) and directed_graph.has_edge(
                            n, node
                        ):
                            all_neighbors.add(n)

                    for neighbor in all_neighbors:
                        if (
                            neighbor != pred
                            and not directed_graph.has_edge(pred, neighbor)
                            and not directed_graph.has_edge(neighbor, pred)
                        ):

                            # Orient node -> neighbor
                            if directed_graph.has_edge(neighbor, node):
                                directed_graph.remove_edge(neighbor, node)
                                changed = True

                                rule_desc = f"Rule 1: {pred} -> {node} -> {neighbor}"
                                self.orientation_rules.append(rule_desc)

                                if self.verbose:
                                    print(f"Applied orientation rule: {rule_desc}")

            # Rule 2: If X -> Y -> Z and X - Z, then X -> Z
            for node in list(directed_graph.nodes()):
                successors = list(directed_graph.successors(node))

                for succ in successors:
                    succ_successors = list(directed_graph.successors(succ))

                    for succ_succ in succ_successors:
                        if (
                            succ_succ != node
                            and directed_graph.has_edge(succ_succ, node)
                            and directed_graph.has_edge(node, succ_succ)
                        ):

                            # Orient node -> succ_succ
                            directed_graph.remove_edge(succ_succ, node)
                            changed = True

                            rule_desc = f"Rule 2: {node} -> {succ} -> {succ_succ}, so {node} -> {succ_succ}"
                            self.orientation_rules.append(rule_desc)

                            if self.verbose:
                                print(f"Applied orientation rule: {rule_desc}")

    def get_causal_strength_matrix(
        self, result: PCResult, variables: List[str]
    ) -> np.ndarray:
        """
        Create a causal strength matrix from PC algorithm results.

        Args:
            result: PCResult from PC algorithm
            variables: List of variable names

        Returns:
            Matrix where entry (i,j) represents causal strength from var i to var j
        """
        n_vars = len(variables)
        strength_matrix = np.zeros((n_vars, n_vars))
        var_to_idx = {var: i for i, var in enumerate(variables)}

        # Fill matrix based on skeleton edges (undirected) and independence tests
        for var1, var2 in result.skeleton.edges():
            if var1 in var_to_idx and var2 in var_to_idx:
                i, j = var_to_idx[var1], var_to_idx[var2]

                # Find the strongest test statistic for this edge
                max_strength = 0.0
                for test in result.independence_tests:
                    if (test.variables[0] == var1 and test.variables[1] == var2) or (
                        test.variables[0] == var2 and test.variables[1] == var1
                    ):
                        if (
                            not test.is_independent
                        ):  # Only consider dependent relationships
                            max_strength = max(max_strength, test.test_statistic)

                # For undirected edges, put strength in both directions
                # For directed edges, determine direction
                if result.directed_graph.has_edge(var1, var2):
                    strength_matrix[i, j] = max_strength
                elif result.directed_graph.has_edge(var2, var1):
                    strength_matrix[j, i] = max_strength
                else:
                    # Undirected edge - put strength in both directions
                    strength_matrix[i, j] = max_strength
                    strength_matrix[j, i] = max_strength

        return strength_matrix

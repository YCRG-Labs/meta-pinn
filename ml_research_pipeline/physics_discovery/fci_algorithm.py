"""
FCI Algorithm Implementation for Causal Discovery with Latent Confounders

This module implements the Fast Causal Inference (FCI) algorithm for causal discovery
in the presence of latent (hidden) confounding variables.
"""

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from .pc_algorithm import ConditionalIndependenceTest, PCAlgorithm


@dataclass
class FCIResult:
    """Result from FCI algorithm execution."""

    pag: nx.DiGraph  # Partial Ancestral Graph
    skeleton: nx.Graph
    independence_tests: List[ConditionalIndependenceTest]
    removed_edges: List[Tuple[str, str, Set[str]]]
    orientation_rules: List[str]
    possible_dsep_sets: Dict[Tuple[str, str], List[Set[str]]]


class FCIAlgorithm:
    """
    Implementation of the Fast Causal Inference (FCI) algorithm.

    The FCI algorithm discovers causal relationships in the presence of latent confounders by:
    1. Learning the skeleton using conditional independence tests
    2. Orienting edges using FCI-specific orientation rules
    3. Creating a Partial Ancestral Graph (PAG) that represents equivalence classes
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_conditioning_size: int = 3,
        independence_test: str = "partial_correlation",
        verbose: bool = False,
    ):
        """
        Initialize FCI algorithm.

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

        # Use PC algorithm for skeleton learning
        self.pc_algorithm = PCAlgorithm(
            alpha=alpha,
            max_conditioning_size=max_conditioning_size,
            independence_test=independence_test,
            verbose=verbose,
        )

        # Store test results and d-separation sets
        self.independence_tests = []
        self.removed_edges = []
        self.orientation_rules = []
        self.possible_dsep_sets = {}

    def discover_causal_structure(self, data: Dict[str, np.ndarray]) -> FCIResult:
        """
        Discover causal structure using FCI algorithm.

        Args:
            data: Dictionary with variable names as keys and data arrays as values

        Returns:
            FCIResult containing PAG, skeleton, and test results
        """
        if self.verbose:
            print(
                "Starting FCI algorithm for causal discovery with latent confounders..."
            )

        # Phase 1: Learn skeleton using PC algorithm
        pc_result = self.pc_algorithm.discover_causal_structure(data)
        skeleton = pc_result.skeleton.copy()

        # Store PC results
        self.independence_tests = pc_result.independence_tests.copy()
        self.removed_edges = pc_result.removed_edges.copy()

        # Store possible d-separating sets for each removed edge
        for var1, var2, dsep_set in self.removed_edges:
            edge_key = tuple(sorted([var1, var2]))
            if edge_key not in self.possible_dsep_sets:
                self.possible_dsep_sets[edge_key] = []
            self.possible_dsep_sets[edge_key].append(dsep_set)

        if self.verbose:
            print(f"Skeleton learned with {skeleton.number_of_edges()} edges")

        # Phase 2: Create initial PAG from skeleton
        pag = self._create_initial_pag(skeleton)

        # Phase 3: Apply FCI orientation rules
        self._apply_fci_orientation_rules(pag, data)

        if self.verbose:
            print(f"PAG created with {pag.number_of_edges()} edges")
            print(f"Applied {len(self.orientation_rules)} orientation rules")

        return FCIResult(
            pag=pag,
            skeleton=skeleton,
            independence_tests=self.independence_tests.copy(),
            removed_edges=self.removed_edges.copy(),
            orientation_rules=self.orientation_rules.copy(),
            possible_dsep_sets=self.possible_dsep_sets.copy(),
        )

    def _create_initial_pag(self, skeleton: nx.Graph) -> nx.DiGraph:
        """
        Create initial Partial Ancestral Graph (PAG) from skeleton.

        In FCI, edges can have different types:
        - o-o: circle-circle (uncertain)
        - o->: circle-arrow (partially oriented)
        - <->: bidirected (confounded)
        - -->: directed (causal)

        We represent these using edge attributes.
        """
        pag = nx.DiGraph()
        pag.add_nodes_from(skeleton.nodes())

        # Initialize all edges as uncertain (o-o)
        for var1, var2 in skeleton.edges():
            # Add both directions with circle endpoints initially
            pag.add_edge(var1, var2, endpoint="circle")
            pag.add_edge(var2, var1, endpoint="circle")

        return pag

    def _apply_fci_orientation_rules(
        self, pag: nx.DiGraph, data: Dict[str, np.ndarray]
    ):
        """
        Apply FCI-specific orientation rules to create the final PAG.

        Args:
            pag: Partial Ancestral Graph to orient
            data: Original data for additional tests
        """
        if self.verbose:
            print("Applying FCI orientation rules...")

        # Rule 1: Orient v-structures
        self._orient_v_structures(pag)

        # Rule 2: Apply discriminating path rule
        self._apply_discriminating_path_rule(pag, data)

        # Rule 3: Apply additional FCI rules
        self._apply_additional_fci_rules(pag)

        # Rule 4: Final cleanup and bidirected edge detection
        self._detect_bidirected_edges(pag)

    def _orient_v_structures(self, pag: nx.DiGraph):
        """
        Orient v-structures in the PAG.

        If X-Y-Z and X,Z are not adjacent, and Y was not in the d-separating
        set of X and Z, then orient as X->Y<-Z.
        """
        nodes = list(pag.nodes())

        for y in nodes:
            # Get all neighbors of Y
            neighbors = []
            for x in nodes:
                if x != y and pag.has_edge(x, y) and pag.has_edge(y, x):
                    neighbors.append(x)

            # Check all pairs of neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    x, z = neighbors[i], neighbors[j]

                    # Check if X and Z are not adjacent
                    if not (pag.has_edge(x, z) and pag.has_edge(z, x)):
                        # Check if Y was not in any d-separating set of X and Z
                        edge_key = tuple(sorted([x, z]))
                        y_not_in_dsep = True

                        if edge_key in self.possible_dsep_sets:
                            for dsep_set in self.possible_dsep_sets[edge_key]:
                                if y in dsep_set:
                                    y_not_in_dsep = False
                                    break

                        if y_not_in_dsep:
                            # Orient as X->Y<-Z
                            if pag.has_edge(x, y):
                                pag[x][y]["endpoint"] = "arrow"
                            if pag.has_edge(z, y):
                                pag[z][y]["endpoint"] = "arrow"
                            if pag.has_edge(y, x):
                                pag[y][x]["endpoint"] = "tail"
                            if pag.has_edge(y, z):
                                pag[y][z]["endpoint"] = "tail"

                            rule_desc = f"V-structure: {x} -> {y} <- {z}"
                            self.orientation_rules.append(rule_desc)

                            if self.verbose:
                                print(f"Applied v-structure: {rule_desc}")

    def _apply_discriminating_path_rule(
        self, pag: nx.DiGraph, data: Dict[str, np.ndarray]
    ):
        """
        Apply the discriminating path rule for FCI.

        This rule helps distinguish between different types of confounding.
        """
        nodes = list(pag.nodes())

        # Look for discriminating paths of length 3: A-B-C-D
        for a in nodes:
            for b in nodes:
                if a == b or not self._has_edge(pag, a, b):
                    continue

                for c in nodes:
                    if c in [a, b] or not self._has_edge(pag, b, c):
                        continue

                    for d in nodes:
                        if d in [a, b, c] or not self._has_edge(pag, c, d):
                            continue

                        # Check if this forms a discriminating path
                        if self._is_discriminating_path(pag, [a, b, c, d]):
                            # Apply discriminating path rule
                            self._apply_discriminating_rule(pag, [a, b, c, d], data)

    def _is_discriminating_path(self, pag: nx.DiGraph, path: List[str]) -> bool:
        """
        Check if a path is discriminating for FCI.

        A path A-B-C-D is discriminating for B if:
        1. A and D are not adjacent
        2. B is adjacent to A and C
        3. C is adjacent to B and D
        4. Every node between A and D (except B) is a collider
        """
        if len(path) != 4:
            return False

        a, b, c, d = path

        # Check if A and D are not adjacent
        if self._has_edge(pag, a, d):
            return False

        # Check adjacencies
        if not (
            self._has_edge(pag, a, b)
            and self._has_edge(pag, b, c)
            and self._has_edge(pag, c, d)
        ):
            return False

        # For a 4-node path, C should be a collider (B->C<-D)
        if (
            self._get_endpoint(pag, b, c) == "arrow"
            and self._get_endpoint(pag, d, c) == "arrow"
        ):
            return True

        return False

    def _apply_discriminating_rule(
        self, pag: nx.DiGraph, path: List[str], data: Dict[str, np.ndarray]
    ):
        """
        Apply the discriminating path rule to orient edges.
        """
        a, b, c, d = path

        # Test if B and D are conditionally independent given {A, C}
        conditioning_set = {a, c}
        test_result = self.pc_algorithm._test_conditional_independence(
            b, d, conditioning_set, data
        )

        if test_result.is_independent:
            # B and D are independent given {A, C}, so no confounding
            # Orient as A->B->C<-D
            if self._has_edge(pag, a, b):
                pag[a][b]["endpoint"] = "arrow"
                pag[b][a]["endpoint"] = "tail"
            if self._has_edge(pag, b, c):
                pag[b][c]["endpoint"] = "arrow"
                pag[c][b]["endpoint"] = "tail"
        else:
            # B and D are dependent given {A, C}, suggesting confounding
            # Orient as A->B<->D<-C (bidirected B<->D)
            if self._has_edge(pag, b, d):
                pag[b][d]["endpoint"] = "arrow"
                pag[d][b]["endpoint"] = "arrow"

        rule_desc = f"Discriminating path rule applied to {'-'.join(path)}"
        self.orientation_rules.append(rule_desc)

        if self.verbose:
            print(f"Applied discriminating path rule: {rule_desc}")

    def _apply_additional_fci_rules(self, pag: nx.DiGraph):
        """
        Apply additional FCI orientation rules.
        """
        changed = True
        iteration = 0
        max_iterations = 10

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            # Rule R1: If A o-* B *-> C and A and C are not adjacent, then A -> B
            for a in pag.nodes():
                for b in pag.nodes():
                    if a == b:
                        continue

                    # Check if A o-* B (A has circle endpoint to B)
                    if (
                        self._has_edge(pag, a, b)
                        and self._get_endpoint(pag, a, b) == "circle"
                    ):

                        for c in pag.nodes():
                            if c in [a, b]:
                                continue

                            # Check if B *-> C (B has arrow to C)
                            if (
                                self._has_edge(pag, b, c)
                                and self._get_endpoint(pag, b, c) == "arrow"
                            ):

                                # Check if A and C are not adjacent
                                if not self._has_edge(pag, a, c):
                                    # Orient A -> B
                                    pag[a][b]["endpoint"] = "arrow"
                                    pag[b][a]["endpoint"] = "tail"
                                    changed = True

                                    rule_desc = (
                                        f"FCI R1: {a} -> {b} (due to {b} -> {c})"
                                    )
                                    self.orientation_rules.append(rule_desc)

                                    if self.verbose:
                                        print(f"Applied FCI R1: {rule_desc}")

            # Rule R2: If A -> B o-> C and A *-o C, then B -> C
            for a in pag.nodes():
                for b in pag.nodes():
                    if a == b:
                        continue

                    # Check if A -> B
                    if (
                        self._has_edge(pag, a, b)
                        and self._get_endpoint(pag, a, b) == "arrow"
                        and self._get_endpoint(pag, b, a) == "tail"
                    ):

                        for c in pag.nodes():
                            if c in [a, b]:
                                continue

                            # Check if B o-> C
                            if (
                                self._has_edge(pag, b, c)
                                and self._get_endpoint(pag, b, c) == "arrow"
                                and self._get_endpoint(pag, c, b) == "circle"
                            ):

                                # Check if A *-o C
                                if (
                                    self._has_edge(pag, a, c)
                                    and self._get_endpoint(pag, c, a) == "circle"
                                ):

                                    # Orient B -> C
                                    pag[c][b]["endpoint"] = "tail"
                                    changed = True

                                    rule_desc = (
                                        f"FCI R2: {b} -> {c} (due to {a} -> {b})"
                                    )
                                    self.orientation_rules.append(rule_desc)

                                    if self.verbose:
                                        print(f"Applied FCI R2: {rule_desc}")

    def _detect_bidirected_edges(self, pag: nx.DiGraph):
        """
        Detect and mark bidirected edges (confounded relationships).
        """
        nodes = list(pag.nodes())

        for a in nodes:
            for b in nodes:
                if a >= b:  # Avoid duplicate checking
                    continue

                # Check if both A->B and B->A have arrow endpoints
                if (
                    self._has_edge(pag, a, b)
                    and self._has_edge(pag, b, a)
                    and self._get_endpoint(pag, a, b) == "arrow"
                    and self._get_endpoint(pag, b, a) == "arrow"
                ):

                    # Mark as bidirected edge (confounded)
                    pag[a][b]["edge_type"] = "bidirected"
                    pag[b][a]["edge_type"] = "bidirected"

                    rule_desc = f"Bidirected edge: {a} <-> {b} (confounded)"
                    self.orientation_rules.append(rule_desc)

                    if self.verbose:
                        print(f"Detected bidirected edge: {rule_desc}")

    def _has_edge(self, pag: nx.DiGraph, u: str, v: str) -> bool:
        """Check if there is an edge from u to v in the PAG."""
        return pag.has_edge(u, v)

    def _get_endpoint(self, pag: nx.DiGraph, u: str, v: str) -> str:
        """Get the endpoint type of edge u->v."""
        if pag.has_edge(u, v):
            return pag[u][v].get("endpoint", "circle")
        return "none"

    def get_causal_relationships(self, result: FCIResult) -> List[Dict[str, Any]]:
        """
        Extract causal relationships from FCI result.

        Args:
            result: FCIResult from FCI algorithm

        Returns:
            List of dictionaries describing causal relationships
        """
        relationships = []

        for u, v, data in result.pag.edges(data=True):
            endpoint_uv = data.get("endpoint", "circle")
            edge_type = data.get("edge_type", "unknown")

            # Get reverse edge info
            endpoint_vu = "none"
            if result.pag.has_edge(v, u):
                endpoint_vu = result.pag[v][u].get("endpoint", "circle")

            # Determine relationship type
            if endpoint_uv == "arrow" and endpoint_vu == "tail":
                rel_type = "causal"
                description = f"{u} -> {v}"
            elif endpoint_uv == "arrow" and endpoint_vu == "arrow":
                rel_type = "confounded"
                description = f"{u} <-> {v}"
            elif endpoint_uv == "circle" and endpoint_vu == "circle":
                rel_type = "uncertain"
                description = f"{u} o-o {v}"
            elif endpoint_uv == "arrow" and endpoint_vu == "circle":
                rel_type = "partially_oriented"
                description = f"{u} o-> {v}"
            else:
                rel_type = "unknown"
                description = f"{u} ? {v}"

            # Find strength from independence tests
            strength = 0.0
            p_value = 1.0
            for test in result.independence_tests:
                if (test.variables[0] == u and test.variables[1] == v) or (
                    test.variables[0] == v and test.variables[1] == u
                ):
                    if not test.is_independent:
                        strength = max(strength, test.test_statistic)
                        p_value = min(p_value, test.p_value)

            relationships.append(
                {
                    "source": u,
                    "target": v,
                    "type": rel_type,
                    "description": description,
                    "strength": strength,
                    "p_value": p_value,
                    "edge_type": edge_type,
                }
            )

        return relationships

    def get_confounded_pairs(self, result: FCIResult) -> List[Tuple[str, str]]:
        """
        Get pairs of variables that are confounded (have bidirected edges).

        Args:
            result: FCIResult from FCI algorithm

        Returns:
            List of tuples representing confounded variable pairs
        """
        confounded_pairs = []

        for u, v, data in result.pag.edges(data=True):
            if data.get("edge_type") == "bidirected":
                # Only add each pair once
                pair = tuple(sorted([u, v]))
                if pair not in confounded_pairs:
                    confounded_pairs.append(pair)

        return confounded_pairs

    def estimate_confidence_intervals(
        self, result: FCIResult, data: Dict[str, np.ndarray], n_bootstrap: int = 100
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        Estimate confidence intervals for causal strengths using bootstrap.

        Args:
            result: FCIResult from FCI algorithm
            data: Original data
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary mapping edge pairs to confidence intervals
        """
        confidence_intervals = {}
        n_samples = len(next(iter(data.values())))

        # Get all edges from PAG
        edges = [(u, v) for u, v in result.pag.edges()]

        np.random.seed(42)
        bootstrap_strengths = {edge: [] for edge in edges}

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = {var: values[indices] for var, values in data.items()}

            # Run FCI on bootstrap sample
            try:
                bootstrap_result = self.discover_causal_structure(bootstrap_data)

                # Extract strengths for each edge
                for edge in edges:
                    u, v = edge
                    strength = 0.0

                    for test in bootstrap_result.independence_tests:
                        if (test.variables[0] == u and test.variables[1] == v) or (
                            test.variables[0] == v and test.variables[1] == u
                        ):
                            if not test.is_independent:
                                strength = max(strength, test.test_statistic)

                    bootstrap_strengths[edge].append(strength)

            except Exception as e:
                # Skip failed bootstrap samples
                warnings.warn(f"Bootstrap sample failed: {e}")
                continue

        # Compute confidence intervals
        for edge, strengths in bootstrap_strengths.items():
            if strengths:
                ci_lower = np.percentile(strengths, 2.5)
                ci_upper = np.percentile(strengths, 97.5)
                confidence_intervals[edge] = (ci_lower, ci_upper)
            else:
                confidence_intervals[edge] = (0.0, 0.0)

        return confidence_intervals

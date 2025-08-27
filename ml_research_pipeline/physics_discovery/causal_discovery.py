"""
Physics Causal Discovery Module

This module implements causal discovery methods for identifying relationships
between physical variables and viscosity in fluid dynamics systems.
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class CausalRelationship:
    """Represents a causal relationship between variables."""
    source: str
    target: str
    strength: float
    p_value: float
    confidence_interval: Tuple[float, float]


class PhysicsCausalDiscovery:
    """
    Causal discovery system for identifying physics relationships in fluid dynamics.
    
    This class implements mutual information-based causal discovery to identify
    relationships between flow variables and viscosity parameters.
    """
    
    def __init__(self, 
                 significance_threshold: float = 0.05,
                 min_mutual_info: float = 0.1,
                 random_state: int = 42):
        """
        Initialize the causal discovery system.
        
        Args:
            significance_threshold: P-value threshold for statistical significance
            min_mutual_info: Minimum mutual information for considering relationships
            random_state: Random seed for reproducibility
        """
        self.significance_threshold = significance_threshold
        self.min_mutual_info = min_mutual_info
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Physical variables of interest
        self.flow_variables = [
            'velocity_x', 'velocity_y', 'pressure', 'reynolds_number',
            'shear_rate', 'temperature', 'density'
        ]
        self.viscosity_variables = [
            'viscosity', 'viscosity_gradient_x', 'viscosity_gradient_y',
            'viscosity_type', 'viscosity_params'
        ]
    
    def discover_viscosity_dependencies(self, 
                                     flow_data: Dict[str, np.ndarray]) -> List[CausalRelationship]:
        """
        Discover causal relationships between flow variables and viscosity.
        
        Args:
            flow_data: Dictionary containing flow field data with keys as variable names
                      and values as numpy arrays of shape (n_samples, n_features)
        
        Returns:
            List of CausalRelationship objects sorted by strength
        """
        # Validate input data
        self._validate_flow_data(flow_data)
        
        # Prepare data for analysis
        prepared_data = self._prepare_data(flow_data)
        
        # Compute mutual information between all variable pairs
        relationships = []
        
        for flow_var in self.flow_variables:
            if flow_var not in prepared_data:
                continue
                
            for visc_var in self.viscosity_variables:
                if visc_var not in prepared_data:
                    continue
                
                # Compute mutual information
                mi_score, p_value, ci = self._compute_mutual_information(
                    prepared_data[flow_var], 
                    prepared_data[visc_var]
                )
                
                # Check significance and minimum threshold
                if p_value < self.significance_threshold and mi_score > self.min_mutual_info:
                    relationship = CausalRelationship(
                        source=flow_var,
                        target=visc_var,
                        strength=mi_score,
                        p_value=p_value,
                        confidence_interval=ci
                    )
                    relationships.append(relationship)
        
        # Sort by strength (descending)
        relationships.sort(key=lambda x: x.strength, reverse=True)
        
        return relationships
    
    def build_causal_graph(self, relationships: List[CausalRelationship]) -> nx.DiGraph:
        """
        Build a directed causal graph from discovered relationships.
        
        Args:
            relationships: List of causal relationships
        
        Returns:
            NetworkX directed graph representing causal structure
        """
        graph = nx.DiGraph()
        
        # Add nodes for all variables
        all_variables = set()
        for rel in relationships:
            all_variables.add(rel.source)
            all_variables.add(rel.target)
        
        graph.add_nodes_from(all_variables)
        
        # Add edges with weights representing causal strength
        for rel in relationships:
            graph.add_edge(
                rel.source, 
                rel.target,
                weight=rel.strength,
                p_value=rel.p_value,
                confidence_interval=rel.confidence_interval
            )
        
        return graph
    
    def generate_physics_hypothesis(self, 
                                  relationships: List[CausalRelationship],
                                  graph: nx.DiGraph) -> str:
        """
        Generate natural language hypothesis from discovered relationships.
        
        Args:
            relationships: List of causal relationships
            graph: Causal graph structure
        
        Returns:
            Natural language description of physics hypothesis
        """
        if not relationships:
            return "No significant causal relationships discovered."
        
        hypothesis_parts = []
        
        # Identify strongest relationships
        strong_relationships = [r for r in relationships if r.strength > 0.3]
        
        if strong_relationships:
            hypothesis_parts.append("Strong causal relationships discovered:")
            
            for rel in strong_relationships[:5]:  # Top 5 strongest
                strength_desc = self._describe_strength(rel.strength)
                hypothesis_parts.append(
                    f"- {rel.source} shows {strength_desc} influence on {rel.target} "
                    f"(MI = {rel.strength:.3f}, p = {rel.p_value:.3e})"
                )
        
        # Identify hub variables (high connectivity)
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        
        influential_vars = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        influenced_vars = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if influential_vars and influential_vars[0][1] > 0:
            hypothesis_parts.append(f"\nMost influential variable: {influential_vars[0][0]}")
        
        if influenced_vars and influenced_vars[0][1] > 0:
            hypothesis_parts.append(f"Most influenced variable: {influenced_vars[0][0]}")
        
        # Generate physics interpretation
        physics_interpretation = self._generate_physics_interpretation(relationships)
        if physics_interpretation:
            hypothesis_parts.append(f"\nPhysics interpretation:\n{physics_interpretation}")
        
        return "\n".join(hypothesis_parts)
    
    def visualize_causal_graph(self, 
                             graph: nx.DiGraph, 
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize the causal graph with edge weights representing strength.
        
        Args:
            graph: NetworkX directed graph
            save_path: Optional path to save the figure
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)
        
        # Create layout
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Draw nodes
        node_colors = []
        for node in graph.nodes():
            if any(var in node for var in self.viscosity_variables):
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=1000, alpha=0.8)
        
        # Draw edges with varying thickness based on strength
        edges = graph.edges(data=True)
        edge_weights = [data['weight'] for _, _, data in edges]
        
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [5 * (weight / max_weight) for weight in edge_weights]
            
            nx.draw_networkx_edges(graph, pos, width=edge_widths, 
                                 alpha=0.6, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
        
        # Add edge labels for strongest relationships
        strong_edges = {(u, v): f"{data['weight']:.2f}" 
                       for u, v, data in edges if data['weight'] > 0.2}
        
        if strong_edges:
            nx.draw_networkx_edge_labels(graph, pos, strong_edges, font_size=6)
        
        plt.title("Causal Relationships in Physics Variables", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _validate_flow_data(self, flow_data: Dict[str, np.ndarray]) -> None:
        """Validate input flow data format and content."""
        if not isinstance(flow_data, dict):
            raise ValueError("flow_data must be a dictionary")
        
        if not flow_data:
            raise ValueError("flow_data cannot be empty")
        
        # Check that all arrays have the same number of samples
        sample_counts = [data.shape[0] for data in flow_data.values()]
        if len(set(sample_counts)) > 1:
            raise ValueError("All variables must have the same number of samples")
        
        # Check minimum sample size
        min_samples = min(sample_counts)
        if min_samples < 10:
            raise ValueError("Need at least 10 samples for reliable causal discovery")
    
    def _prepare_data(self, flow_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Prepare and normalize data for causal analysis."""
        prepared_data = {}
        
        for var_name, data in flow_data.items():
            # Handle different data shapes
            if data.ndim == 1:
                prepared_data[var_name] = data.reshape(-1, 1)
            elif data.ndim == 2:
                prepared_data[var_name] = data
            else:
                # Flatten higher dimensional data
                prepared_data[var_name] = data.reshape(data.shape[0], -1)
            
            # Normalize data
            prepared_data[var_name] = self.scaler.fit_transform(prepared_data[var_name])
        
        return prepared_data
    
    def _compute_mutual_information(self, 
                                  x: np.ndarray, 
                                  y: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """
        Compute mutual information between two variables with statistical testing.
        
        Args:
            x: Source variable data
            y: Target variable data
        
        Returns:
            Tuple of (mutual_info_score, p_value, confidence_interval)
        """
        # Handle multidimensional data by taking mean across features
        if x.ndim > 1 and x.shape[1] > 1:
            x = np.mean(x, axis=1)
        else:
            x = x.flatten()
        
        if y.ndim > 1 and y.shape[1] > 1:
            y = np.mean(y, axis=1)
        else:
            y = y.flatten()
        
        # Compute mutual information
        mi_score = mutual_info_regression(x.reshape(-1, 1), y, random_state=self.random_state)[0]
        
        # Bootstrap for confidence interval and p-value
        n_bootstrap = 1000
        bootstrap_scores = []
        
        np.random.seed(self.random_state)
        for _ in range(n_bootstrap):
            # Permute y to break any real relationship
            y_perm = np.random.permutation(y)
            bootstrap_mi = mutual_info_regression(
                x.reshape(-1, 1), y_perm, random_state=self.random_state
            )[0]
            bootstrap_scores.append(bootstrap_mi)
        
        # Compute p-value (proportion of bootstrap scores >= observed score)
        p_value = np.mean(np.array(bootstrap_scores) >= mi_score)
        
        # Compute confidence interval
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        return mi_score, p_value, (ci_lower, ci_upper)
    
    def _describe_strength(self, strength: float) -> str:
        """Convert numerical strength to descriptive text."""
        if strength >= 0.5:
            return "very strong"
        elif strength >= 0.3:
            return "strong"
        elif strength >= 0.2:
            return "moderate"
        elif strength >= 0.1:
            return "weak"
        else:
            return "very weak"
    
    def _generate_physics_interpretation(self, 
                                       relationships: List[CausalRelationship]) -> str:
        """Generate physics-based interpretation of discovered relationships."""
        interpretations = []
        
        # Look for specific physics patterns
        for rel in relationships:
            if rel.source == 'reynolds_number' and 'viscosity' in rel.target:
                interpretations.append(
                    "Reynolds number influences viscosity behavior, suggesting "
                    "flow regime-dependent viscosity characteristics."
                )
            
            elif rel.source == 'shear_rate' and 'viscosity' in rel.target:
                interpretations.append(
                    "Shear rate affects viscosity, indicating non-Newtonian "
                    "fluid behavior with shear-dependent viscosity."
                )
            
            elif rel.source == 'temperature' and 'viscosity' in rel.target:
                interpretations.append(
                    "Temperature influences viscosity, consistent with "
                    "thermodynamic effects on fluid properties."
                )
            
            elif 'velocity' in rel.source and 'viscosity' in rel.target:
                interpretations.append(
                    "Velocity field affects viscosity distribution, suggesting "
                    "flow-dependent viscosity variations."
                )
        
        return "\n".join(interpretations) if interpretations else ""
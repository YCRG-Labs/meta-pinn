"""
Unit tests for Physics Causal Discovery module.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from ml_research_pipeline.physics_discovery.causal_discovery import (
    PhysicsCausalDiscovery, CausalRelationship
)


class TestPhysicsCausalDiscovery:
    """Test suite for PhysicsCausalDiscovery class."""
    
    @pytest.fixture
    def causal_discovery(self):
        """Create a PhysicsCausalDiscovery instance for testing."""
        return PhysicsCausalDiscovery(
            significance_threshold=0.05,
            min_mutual_info=0.1,
            random_state=42
        )
    
    @pytest.fixture
    def sample_flow_data(self):
        """Create sample flow data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create correlated data to simulate causal relationships
        reynolds = np.random.uniform(100, 1000, n_samples)
        velocity_x = 0.5 * reynolds + np.random.normal(0, 10, n_samples)
        velocity_y = 0.3 * reynolds + np.random.normal(0, 5, n_samples)
        pressure = 0.8 * velocity_x + np.random.normal(0, 20, n_samples)
        
        # Viscosity depends on Reynolds number (inverse relationship)
        viscosity = 1000 / reynolds + np.random.normal(0, 0.1, n_samples)
        
        return {
            'reynolds_number': reynolds,
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'pressure': pressure,
            'viscosity': viscosity,
            'temperature': np.random.uniform(280, 320, n_samples),
            'shear_rate': np.random.uniform(0.1, 10, n_samples)
        }
    
    def test_initialization(self):
        """Test proper initialization of PhysicsCausalDiscovery."""
        discovery = PhysicsCausalDiscovery(
            significance_threshold=0.01,
            min_mutual_info=0.2,
            random_state=123
        )
        
        assert discovery.significance_threshold == 0.01
        assert discovery.min_mutual_info == 0.2
        assert discovery.random_state == 123
        assert len(discovery.flow_variables) > 0
        assert len(discovery.viscosity_variables) > 0
    
    def test_validate_flow_data_valid(self, causal_discovery, sample_flow_data):
        """Test validation with valid flow data."""
        # Should not raise any exception
        causal_discovery._validate_flow_data(sample_flow_data)
    
    def test_validate_flow_data_invalid_type(self, causal_discovery):
        """Test validation with invalid data type."""
        with pytest.raises(ValueError, match="flow_data must be a dictionary"):
            causal_discovery._validate_flow_data("invalid_data")
    
    def test_validate_flow_data_empty(self, causal_discovery):
        """Test validation with empty data."""
        with pytest.raises(ValueError, match="flow_data cannot be empty"):
            causal_discovery._validate_flow_data({})
    
    def test_validate_flow_data_mismatched_samples(self, causal_discovery):
        """Test validation with mismatched sample counts."""
        invalid_data = {
            'var1': np.random.rand(100),
            'var2': np.random.rand(50)  # Different sample count
        }
        
        with pytest.raises(ValueError, match="same number of samples"):
            causal_discovery._validate_flow_data(invalid_data)
    
    def test_validate_flow_data_insufficient_samples(self, causal_discovery):
        """Test validation with insufficient samples."""
        insufficient_data = {
            'var1': np.random.rand(5),
            'var2': np.random.rand(5)
        }
        
        with pytest.raises(ValueError, match="at least 10 samples"):
            causal_discovery._validate_flow_data(insufficient_data)
    
    def test_prepare_data(self, causal_discovery, sample_flow_data):
        """Test data preparation and normalization."""
        prepared_data = causal_discovery._prepare_data(sample_flow_data)
        
        # Check that all variables are present
        assert set(prepared_data.keys()) == set(sample_flow_data.keys())
        
        # Check normalization (mean should be close to 0, std close to 1)
        for var_name, data in prepared_data.items():
            assert data.ndim == 2  # Should be 2D
            assert abs(np.mean(data)) < 0.1  # Normalized mean
            assert abs(np.std(data) - 1.0) < 0.1  # Normalized std
    
    def test_compute_mutual_information(self, causal_discovery):
        """Test mutual information computation."""
        np.random.seed(42)
        
        # Create strongly correlated variables
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.1  # Strong correlation
        
        mi_score, p_value, ci = causal_discovery._compute_mutual_information(
            x.reshape(-1, 1), y.reshape(-1, 1)
        )
        
        # Should detect strong relationship
        assert mi_score > 0.1
        assert p_value < 0.05  # Should be significant
        assert ci[0] < ci[1]  # Valid confidence interval
    
    def test_compute_mutual_information_independent(self, causal_discovery):
        """Test mutual information with independent variables."""
        np.random.seed(42)
        
        # Create independent variables
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        mi_score, p_value, ci = causal_discovery._compute_mutual_information(
            x.reshape(-1, 1), y.reshape(-1, 1)
        )
        
        # Should detect weak/no relationship
        assert mi_score < 0.3  # Weak relationship
        assert p_value > 0.01  # Not strongly significant
    
    def test_discover_viscosity_dependencies(self, causal_discovery, sample_flow_data):
        """Test discovery of viscosity dependencies."""
        relationships = causal_discovery.discover_viscosity_dependencies(sample_flow_data)
        
        # Should find some relationships
        assert isinstance(relationships, list)
        
        # Check relationship structure
        for rel in relationships:
            assert isinstance(rel, CausalRelationship)
            assert rel.source in causal_discovery.flow_variables
            assert rel.target in causal_discovery.viscosity_variables
            assert rel.strength >= causal_discovery.min_mutual_info
            assert rel.p_value < causal_discovery.significance_threshold
            assert len(rel.confidence_interval) == 2
        
        # Should be sorted by strength (descending)
        if len(relationships) > 1:
            for i in range(len(relationships) - 1):
                assert relationships[i].strength >= relationships[i + 1].strength
    
    def test_build_causal_graph(self, causal_discovery):
        """Test causal graph construction."""
        # Create sample relationships
        relationships = [
            CausalRelationship(
                source='reynolds_number',
                target='viscosity',
                strength=0.8,
                p_value=0.001,
                confidence_interval=(0.1, 0.3)
            ),
            CausalRelationship(
                source='velocity_x',
                target='viscosity',
                strength=0.6,
                p_value=0.01,
                confidence_interval=(0.05, 0.25)
            )
        ]
        
        graph = causal_discovery.build_causal_graph(relationships)
        
        # Check graph structure
        assert isinstance(graph, nx.DiGraph)
        assert 'reynolds_number' in graph.nodes()
        assert 'velocity_x' in graph.nodes()
        assert 'viscosity' in graph.nodes()
        
        # Check edges
        assert graph.has_edge('reynolds_number', 'viscosity')
        assert graph.has_edge('velocity_x', 'viscosity')
        
        # Check edge attributes
        edge_data = graph['reynolds_number']['viscosity']
        assert edge_data['weight'] == 0.8
        assert edge_data['p_value'] == 0.001
    
    def test_generate_physics_hypothesis_empty(self, causal_discovery):
        """Test hypothesis generation with no relationships."""
        hypothesis = causal_discovery.generate_physics_hypothesis([], nx.DiGraph())
        assert "No significant causal relationships discovered" in hypothesis
    
    def test_generate_physics_hypothesis_with_relationships(self, causal_discovery):
        """Test hypothesis generation with relationships."""
        relationships = [
            CausalRelationship(
                source='reynolds_number',
                target='viscosity',
                strength=0.8,
                p_value=0.001,
                confidence_interval=(0.1, 0.3)
            ),
            CausalRelationship(
                source='shear_rate',
                target='viscosity',
                strength=0.6,
                p_value=0.01,
                confidence_interval=(0.05, 0.25)
            )
        ]
        
        graph = causal_discovery.build_causal_graph(relationships)
        hypothesis = causal_discovery.generate_physics_hypothesis(relationships, graph)
        
        # Should contain relationship descriptions
        assert "Strong causal relationships discovered" in hypothesis
        assert "reynolds_number" in hypothesis
        assert "shear_rate" in hypothesis
        assert "viscosity" in hypothesis
        
        # Should contain physics interpretation
        assert "Reynolds number influences viscosity" in hypothesis
        assert "Shear rate affects viscosity" in hypothesis
    
    def test_describe_strength(self, causal_discovery):
        """Test strength description mapping."""
        assert causal_discovery._describe_strength(0.6) == "very strong"
        assert causal_discovery._describe_strength(0.4) == "strong"
        assert causal_discovery._describe_strength(0.25) == "moderate"
        assert causal_discovery._describe_strength(0.15) == "weak"
        assert causal_discovery._describe_strength(0.05) == "very weak"
    
    def test_generate_physics_interpretation(self, causal_discovery):
        """Test physics interpretation generation."""
        relationships = [
            CausalRelationship(
                source='reynolds_number',
                target='viscosity',
                strength=0.8,
                p_value=0.001,
                confidence_interval=(0.1, 0.3)
            ),
            CausalRelationship(
                source='temperature',
                target='viscosity',
                strength=0.6,
                p_value=0.01,
                confidence_interval=(0.05, 0.25)
            )
        ]
        
        interpretation = causal_discovery._generate_physics_interpretation(relationships)
        
        assert "Reynolds number influences viscosity" in interpretation
        assert "Temperature influences viscosity" in interpretation
        assert "thermodynamic effects" in interpretation
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_causal_graph(self, mock_savefig, mock_show, causal_discovery):
        """Test causal graph visualization."""
        # Create sample graph
        relationships = [
            CausalRelationship(
                source='reynolds_number',
                target='viscosity',
                strength=0.8,
                p_value=0.001,
                confidence_interval=(0.1, 0.3)
            )
        ]
        
        graph = causal_discovery.build_causal_graph(relationships)
        
        # Test visualization without saving
        causal_discovery.visualize_causal_graph(graph)
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()
        
        # Test visualization with saving
        mock_show.reset_mock()
        causal_discovery.visualize_causal_graph(graph, save_path='test_graph.png')
        mock_show.assert_called_once()
        mock_savefig.assert_called_once_with('test_graph.png', dpi=300, bbox_inches='tight')
    
    def test_causal_relationship_dataclass(self):
        """Test CausalRelationship dataclass."""
        rel = CausalRelationship(
            source='test_source',
            target='test_target',
            strength=0.5,
            p_value=0.02,
            confidence_interval=(0.1, 0.4)
        )
        
        assert rel.source == 'test_source'
        assert rel.target == 'test_target'
        assert rel.strength == 0.5
        assert rel.p_value == 0.02
        assert rel.confidence_interval == (0.1, 0.4)
    
    def test_integration_with_realistic_data(self, causal_discovery):
        """Test integration with realistic physics data."""
        np.random.seed(42)
        n_samples = 200
        
        # Create realistic physics relationships
        reynolds = np.random.uniform(100, 2000, n_samples)
        temperature = np.random.uniform(280, 350, n_samples)
        
        # Viscosity inversely related to temperature and Reynolds number
        viscosity = (1000 / reynolds) * np.exp(-0.01 * (temperature - 300)) + \
                   np.random.normal(0, 0.05, n_samples)
        
        # Velocity related to Reynolds number
        velocity_x = reynolds * 0.001 + np.random.normal(0, 0.1, n_samples)
        
        flow_data = {
            'reynolds_number': reynolds,
            'temperature': temperature,
            'velocity_x': velocity_x,
            'viscosity': viscosity
        }
        
        # Discover relationships
        relationships = causal_discovery.discover_viscosity_dependencies(flow_data)
        
        # Should find temperature and Reynolds number relationships
        sources = [rel.source for rel in relationships]
        assert 'reynolds_number' in sources or 'temperature' in sources
        
        # Build and analyze graph
        if relationships:
            graph = causal_discovery.build_causal_graph(relationships)
            hypothesis = causal_discovery.generate_physics_hypothesis(relationships, graph)
            
            assert isinstance(graph, nx.DiGraph)
            assert len(hypothesis) > 0


if __name__ == '__main__':
    pytest.main([__file__])
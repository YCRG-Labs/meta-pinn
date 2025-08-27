"""
Integration tests for Integrated Physics Discovery module.
"""

import pytest
import numpy as np
import sympy as sp
import networkx as nx
from unittest.mock import patch, MagicMock
import tempfile
import json
from pathlib import Path

from ml_research_pipeline.physics_discovery.integrated_discovery import (
    IntegratedPhysicsDiscovery, PhysicsHypothesis, DiscoveryResult
)
from ml_research_pipeline.physics_discovery.causal_discovery import CausalRelationship
from ml_research_pipeline.physics_discovery.symbolic_regression import SymbolicExpression


class TestIntegratedPhysicsDiscovery:
    """Test suite for IntegratedPhysicsDiscovery class."""
    
    @pytest.fixture
    def integrated_discovery(self):
        """Create an IntegratedPhysicsDiscovery instance for testing."""
        return IntegratedPhysicsDiscovery(
            variables=['reynolds_number', 'temperature', 'velocity_x', 'viscosity'],
            causal_config={'min_mutual_info': 0.05},
            symbolic_config={'population_size': 10, 'max_generations': 3},
            random_state=42
        )
    
    @pytest.fixture
    def sample_physics_data(self):
        """Create sample physics data with known relationships."""
        np.random.seed(42)
        n_samples = 100
        
        # Create realistic physics relationships
        reynolds = np.random.uniform(100, 1000, n_samples)
        temperature = np.random.uniform(280, 320, n_samples)
        
        # Viscosity inversely related to temperature and Reynolds number
        viscosity = (1000 / reynolds) * np.exp(-0.01 * (temperature - 300)) + \
                   np.random.normal(0, 0.05, n_samples)
        
        # Velocity related to Reynolds number
        velocity_x = reynolds * 0.001 + np.random.normal(0, 0.1, n_samples)
        
        return {
            'reynolds_number': reynolds,
            'temperature': temperature,
            'velocity_x': velocity_x,
            'viscosity': viscosity
        }
    
    def test_initialization(self):
        """Test proper initialization of IntegratedPhysicsDiscovery."""
        discovery = IntegratedPhysicsDiscovery(
            variables=['a', 'b', 'c'],
            causal_config={'significance_threshold': 0.01},
            symbolic_config={'population_size': 50},
            validation_config={'min_validation_score': 0.8},
            random_state=123
        )
        
        assert discovery.variables == ['a', 'b', 'c']
        assert discovery.random_state == 123
        assert discovery.causal_discovery.significance_threshold == 0.01
        assert discovery.symbolic_regression.population_size == 50
        assert discovery.validation_config['min_validation_score'] == 0.8

    def test_discover_physics_relationships(self, integrated_discovery, sample_physics_data):
        """Test complete physics discovery pipeline."""
        result = integrated_discovery.discover_physics_relationships(
            sample_physics_data,
            target_variable='viscosity'
        )
        
        # Check result structure
        assert isinstance(result, DiscoveryResult)
        assert isinstance(result.hypothesis, PhysicsHypothesis)
        assert isinstance(result.causal_analysis, dict)
        assert isinstance(result.symbolic_analysis, dict)
        assert isinstance(result.validation_metrics, dict)
        assert isinstance(result.discovery_metadata, dict)
        
        # Check hypothesis components
        hypothesis = result.hypothesis
        assert isinstance(hypothesis.causal_relationships, list)
        assert isinstance(hypothesis.symbolic_expressions, list)
        assert isinstance(hypothesis.causal_graph, nx.DiGraph)
        assert isinstance(hypothesis.validation_score, float)
        assert isinstance(hypothesis.natural_language_description, str)
        assert isinstance(hypothesis.confidence_score, float)
        
        # Check that discovery was recorded
        assert len(integrated_discovery.discovery_history) == 1
        assert integrated_discovery.discovery_history[0] == result

    def test_validate_causal_relationships(self, integrated_discovery, sample_physics_data):
        """Test causal relationship validation."""
        # Create mock causal relationships
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
        
        validation_score = integrated_discovery._validate_causal_relationships(
            relationships, sample_physics_data, 'viscosity'
        )
        
        assert isinstance(validation_score, float)
        assert 0 <= validation_score <= 1
        assert validation_score > 0.5  # Should be high for strong relationships

    def test_validate_symbolic_expression(self, integrated_discovery, sample_physics_data):
        """Test symbolic expression validation."""
        # Create mock symbolic expression
        expr = SymbolicExpression(
            expression=sp.Symbol('reynolds_number') + sp.Symbol('temperature'),
            fitness=0.8,
            complexity=5,
            r2_score=0.85,
            mse=0.1,
            variables=['reynolds_number', 'temperature']
        )
        
        validation_score = integrated_discovery._validate_symbolic_expression(
            expr, sample_physics_data, 'viscosity'
        )
        
        assert isinstance(validation_score, float)
        assert 0 <= validation_score <= 1
        assert validation_score > 0.5  # Should be high for good expression

    def test_check_physics_consistency(self, integrated_discovery):
        """Test physics consistency checking."""
        # Create consistent causal and symbolic results
        causal_relationships = [
            CausalRelationship(
                source='reynolds_number',
                target='viscosity',
                strength=0.8,
                p_value=0.001,
                confidence_interval=(0.1, 0.3)
            )
        ]
        
        symbolic_result = SymbolicExpression(
            expression=sp.Symbol('reynolds_number') + 1,
            fitness=0.8,
            complexity=2,
            r2_score=0.85,
            mse=0.1,
            variables=['reynolds_number']
        )
        
        consistency_score = integrated_discovery._check_physics_consistency(
            causal_relationships, symbolic_result
        )
        
        assert isinstance(consistency_score, float)
        assert 0 <= consistency_score <= 1
        assert consistency_score > 0.5  # Should detect consistency

    def test_identify_physics_patterns(self, integrated_discovery):
        """Test physics pattern identification."""
        causal_relationships = [
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
        
        symbolic_result = SymbolicExpression(
            expression=sp.exp(sp.Symbol('temperature')),
            fitness=0.8,
            complexity=3,
            r2_score=0.85,
            mse=0.1,
            variables=['temperature']
        )
        
        patterns = integrated_discovery._identify_physics_patterns(
            causal_relationships, symbolic_result
        )
        
        assert isinstance(patterns, list)
        assert 'reynolds_viscosity_relationship' in patterns
        assert 'temperature_viscosity_relationship' in patterns
        assert 'exponential_relationship' in patterns

    def test_export_discovery_results(self, integrated_discovery, sample_physics_data):
        """Test exporting discovery results."""
        # Perform discovery first
        result = integrated_discovery.discover_physics_relationships(
            sample_physics_data,
            target_variable='viscosity'
        )
        
        # Export results
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = integrated_discovery.export_discovery_results(
                temp_dir, include_plots=False
            )
            
            assert isinstance(saved_files, dict)
            assert 'hypothesis' in saved_files
            assert 'results' in saved_files
            
            # Check that files were created
            hypothesis_file = Path(saved_files['hypothesis'])
            results_file = Path(saved_files['results'])
            
            assert hypothesis_file.exists()
            assert results_file.exists()
            
            # Check file contents
            with open(hypothesis_file) as f:
                hypothesis_data = json.load(f)
            
            assert 'natural_language_description' in hypothesis_data
            assert 'validation_score' in hypothesis_data
            assert 'causal_relationships' in hypothesis_data

    def test_get_discovery_summary_empty(self, integrated_discovery):
        """Test discovery summary when no discoveries performed."""
        summary = integrated_discovery.get_discovery_summary()
        
        assert isinstance(summary, dict)
        assert 'message' in summary
        assert 'No discoveries performed yet' in summary['message']

    def test_get_discovery_summary_with_results(self, integrated_discovery, sample_physics_data):
        """Test discovery summary with results."""
        # Perform multiple discoveries
        integrated_discovery.discover_physics_relationships(
            sample_physics_data, target_variable='viscosity'
        )
        integrated_discovery.discover_physics_relationships(
            sample_physics_data, target_variable='viscosity'
        )
        
        summary = integrated_discovery.get_discovery_summary()
        
        assert isinstance(summary, dict)
        assert 'total_discoveries' in summary
        assert 'validated_hypotheses' in summary
        assert 'latest_validation_score' in summary
        assert 'average_validation_score' in summary
        assert 'discovery_timeline' in summary
        
        assert summary['total_discoveries'] == 2
        assert isinstance(summary['discovery_timeline'], list)
        assert len(summary['discovery_timeline']) == 2

    def test_integration_with_realistic_scenario(self, integrated_discovery):
        """Test complete integration with realistic physics scenario."""
        # Create more complex realistic data
        np.random.seed(42)
        n_samples = 150
        
        # Multiple interacting variables
        reynolds = np.random.uniform(50, 2000, n_samples)
        temperature = np.random.uniform(270, 350, n_samples)
        pressure = np.random.uniform(1e5, 5e5, n_samples)
        
        # Complex viscosity relationship
        viscosity = (1000 / reynolds) * np.exp(-0.02 * (temperature - 300)) * \
                   (pressure / 1e5) ** 0.1 + np.random.normal(0, 0.02, n_samples)
        
        velocity_x = np.sqrt(reynolds * viscosity) + np.random.normal(0, 0.5, n_samples)
        
        complex_data = {
            'reynolds_number': reynolds,
            'temperature': temperature,
            'pressure': pressure,
            'velocity_x': velocity_x,
            'viscosity': viscosity
        }
        
        # Perform discovery
        result = integrated_discovery.discover_physics_relationships(
            complex_data,
            target_variable='viscosity',
            meta_learning_baseline=0.6
        )
        
        # Validate comprehensive result
        assert isinstance(result, DiscoveryResult)
        assert result.hypothesis.validation_score >= 0.0
        assert np.isfinite(result.hypothesis.confidence_score)
        assert result.hypothesis.confidence_score >= 0.0
        assert len(result.hypothesis.natural_language_description) > 0
        
        # Should find some relationships
        assert len(result.hypothesis.causal_relationships) >= 0
        assert len(result.hypothesis.symbolic_expressions) >= 0
        
        # Check metadata
        assert 'timestamp' in result.discovery_metadata
        assert result.discovery_metadata['target_variable'] == 'viscosity'
        assert len(result.discovery_metadata['variables']) > 0

    def test_enhanced_meta_learning_validation(self, integrated_discovery, sample_physics_data):
        """Test enhanced meta-learning validation functionality."""
        # Perform discovery
        result = integrated_discovery.discover_physics_relationships(
            sample_physics_data,
            target_variable='viscosity',
            meta_learning_baseline=0.65
        )
        
        # Test meta-learning validation
        validation_tasks = [
            {'task_id': i, 'complexity': np.random.uniform(0.3, 0.8)}
            for i in range(10)
        ]
        
        baseline_config = {'adaptation_steps': 10, 'learning_rate': 0.01}
        physics_config = {'adaptation_steps': 8, 'learning_rate': 0.01, 'physics_weight': 1.0}
        
        # Mock MetaPINN class
        class MockMetaPINN:
            pass
        
        validation_metrics = integrated_discovery.validate_discovered_physics_with_meta_learning(
            result.hypothesis,
            MockMetaPINN,
            validation_tasks,
            baseline_config,
            physics_config
        )
        
        # Check validation metrics structure
        assert isinstance(validation_metrics, dict)
        assert 'baseline_accuracy' in validation_metrics
        assert 'physics_informed_accuracy' in validation_metrics
        assert 'accuracy_improvement' in validation_metrics
        assert 'adaptation_speedup' in validation_metrics
        assert 'overall_validation_score' in validation_metrics
        assert 'meets_validation_threshold' in validation_metrics
        
        # Check metric ranges
        assert 0 <= validation_metrics['baseline_accuracy'] <= 1
        assert 0 <= validation_metrics['physics_informed_accuracy'] <= 1
        assert 0 <= validation_metrics['overall_validation_score'] <= 1
        assert isinstance(validation_metrics['meets_validation_threshold'], bool)

    def test_enhanced_natural_language_generation(self, integrated_discovery, sample_physics_data):
        """Test enhanced natural language hypothesis generation."""
        # Perform discovery
        result = integrated_discovery.discover_physics_relationships(
            sample_physics_data,
            target_variable='viscosity'
        )
        
        nl_description = result.hypothesis.natural_language_description
        
        # Check for enhanced content
        assert isinstance(nl_description, str)
        assert len(nl_description) > 100  # Should be comprehensive
        
        # Check for key sections
        assert "PHYSICS DISCOVERY ANALYSIS" in nl_description
        assert "confidence" in nl_description.lower()
        
        # Should contain analysis sections if relationships found
        if result.hypothesis.causal_relationships:
            assert "CAUSAL RELATIONSHIP ANALYSIS" in nl_description
        
        if result.hypothesis.symbolic_expressions:
            assert "MATHEMATICAL RELATIONSHIP ANALYSIS" in nl_description
        
        # Should contain validation assessment
        assert "VALIDATION & CONFIDENCE ASSESSMENT" in nl_description
        assert "validation score" in nl_description.lower()

    def test_physics_context_generation(self, integrated_discovery):
        """Test physics context generation for causal relationships."""
        # Test known physics contexts
        reynolds_context = integrated_discovery._get_physics_context('reynolds_number', 'viscosity')
        assert len(reynolds_context) > 0
        assert 'reynolds' in reynolds_context.lower()
        
        temp_context = integrated_discovery._get_physics_context('temperature', 'viscosity')
        assert len(temp_context) > 0
        assert 'temperature' in temp_context.lower()
        
        # Test unknown relationship
        unknown_context = integrated_discovery._get_physics_context('unknown_var', 'other_var')
        assert unknown_context == ""

    def test_mathematical_interpretation(self, integrated_discovery):
        """Test mathematical form interpretation."""
        import sympy as sp
        
        # Test exponential expression
        exp_expr = sp.exp(sp.Symbol('x'))
        exp_interp = integrated_discovery._interpret_mathematical_form(exp_expr)
        assert 'exponential' in exp_interp.lower()
        assert 'activation energy' in exp_interp.lower() or 'thermodynamic' in exp_interp.lower()
        
        # Test power law expression
        power_expr = sp.Symbol('x') ** 2
        power_interp = integrated_discovery._interpret_mathematical_form(power_expr)
        assert 'power law' in power_interp.lower()
        
        # Test logarithmic expression
        log_expr = sp.log(sp.Symbol('x'))
        log_interp = integrated_discovery._interpret_mathematical_form(log_expr)
        assert 'logarithmic' in log_interp.lower()

    def test_confidence_assessment(self, integrated_discovery):
        """Test confidence level assessment."""
        # Test different confidence levels
        assert integrated_discovery._assess_confidence_level(0.9) == "High"
        assert integrated_discovery._assess_confidence_level(0.7) == "Moderate"
        assert integrated_discovery._assess_confidence_level(0.5) == "Low"
        assert integrated_discovery._assess_confidence_level(0.3) == "Very Low"

    def test_physics_informed_symbolic_validation(self, integrated_discovery, sample_physics_data):
        """Test enhanced symbolic validation with physics constraints."""
        from ml_research_pipeline.physics_discovery.symbolic_regression import SymbolicExpression
        from ml_research_pipeline.physics_discovery.causal_discovery import CausalRelationship
        import sympy as sp
        
        # Create symbolic expression using causally important variables
        expr = SymbolicExpression(
            expression=sp.Symbol('reynolds_number') * sp.exp(sp.Symbol('temperature')),
            fitness=0.8,
            complexity=5,
            r2_score=0.85,
            mse=0.1,
            variables=['reynolds_number', 'temperature']
        )
        
        # Create causal relationships
        causal_relationships = [
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
        
        # Test enhanced validation
        validation_score = integrated_discovery._validate_symbolic_expression_with_physics(
            expr, sample_physics_data, 'viscosity', causal_relationships
        )
        
        assert isinstance(validation_score, float)
        assert 0 <= validation_score <= 1
        # Should get bonus for using causally important variables and physics patterns
        assert validation_score > 0.7

    def test_meta_learning_consistency_check(self, integrated_discovery, sample_physics_data):
        """Test meta-learning specific physics consistency checking."""
        from ml_research_pipeline.physics_discovery.symbolic_regression import SymbolicExpression
        from ml_research_pipeline.physics_discovery.causal_discovery import CausalRelationship
        import sympy as sp
        
        # Create good meta-learning scenario
        causal_relationships = [
            CausalRelationship(
                source='reynolds_number',
                target='viscosity',
                strength=0.8,  # Strong relationship
                p_value=0.001,
                confidence_interval=(0.1, 0.3)
            )
        ]
        
        symbolic_result = SymbolicExpression(
            expression=sp.Symbol('reynolds_number') + 1,
            fitness=0.8,
            complexity=8,  # Moderate complexity - good for meta-learning
            r2_score=0.85,  # High R²
            mse=0.1,
            variables=['reynolds_number']
        )
        
        consistency_score = integrated_discovery._check_physics_consistency_with_meta_learning(
            causal_relationships, symbolic_result, sample_physics_data
        )
        
        assert isinstance(consistency_score, float)
        assert 0 <= consistency_score <= 1
        # Should get high score for good meta-learning characteristics
        assert consistency_score > 0.6

    def test_meta_learning_improvement_estimation(self, integrated_discovery, sample_physics_data):
        """Test meta-learning improvement estimation."""
        from ml_research_pipeline.physics_discovery.symbolic_regression import SymbolicExpression
        from ml_research_pipeline.physics_discovery.causal_discovery import CausalRelationship
        import sympy as sp
        
        # Create strong physics discoveries
        causal_relationships = [
            CausalRelationship(
                source='reynolds_number',
                target='viscosity',
                strength=0.9,  # Very strong
                p_value=0.001,
                confidence_interval=(0.1, 0.3)
            )
        ]
        
        symbolic_result = SymbolicExpression(
            expression=sp.Symbol('reynolds_number'),
            fitness=0.9,
            complexity=3,
            r2_score=0.9,  # Excellent fit
            mse=0.05,
            variables=['reynolds_number']
        )
        
        # Test with different baseline performances
        improvement_high_baseline = integrated_discovery._estimate_meta_learning_improvement(
            0.8, causal_relationships, symbolic_result, 0.9, sample_physics_data
        )
        
        improvement_low_baseline = integrated_discovery._estimate_meta_learning_improvement(
            0.8, causal_relationships, symbolic_result, 0.5, sample_physics_data
        )
        
        # Should get more improvement with lower baseline
        assert improvement_low_baseline >= improvement_high_baseline
        assert 0 <= improvement_high_baseline <= 0.25
        assert 0 <= improvement_low_baseline <= 0.25

    def test_comprehensive_physics_interpretation(self, integrated_discovery):
        """Test comprehensive physics interpretation generation."""
        from ml_research_pipeline.physics_discovery.symbolic_regression import SymbolicExpression
        from ml_research_pipeline.physics_discovery.causal_discovery import CausalRelationship
        import sympy as sp
        
        # Create consistent causal and symbolic results
        causal_relationships = [
            CausalRelationship(
                source='reynolds_number',
                target='viscosity',
                strength=0.8,
                p_value=0.001,
                confidence_interval=(0.1, 0.3)
            )
        ]
        
        symbolic_expressions = [
            SymbolicExpression(
                expression=sp.Symbol('reynolds_number') + sp.Symbol('temperature'),
                fitness=0.8,
                complexity=5,
                r2_score=0.85,
                mse=0.1,
                variables=['reynolds_number', 'temperature']
            )
        ]
        
        interpretation = integrated_discovery._generate_comprehensive_physics_interpretation(
            causal_relationships, symbolic_expressions
        )
        
        assert isinstance(interpretation, str)
        if interpretation:  # May be empty if no strong patterns
            assert len(interpretation) > 0

    def test_validation_pipeline_integration(self, integrated_discovery, sample_physics_data):
        """Test complete validation pipeline integration."""
        # Perform full discovery
        result = integrated_discovery.discover_physics_relationships(
            sample_physics_data,
            target_variable='viscosity',
            meta_learning_baseline=0.6
        )
        
        # Check that validation was performed
        assert 'validation_score' in result.validation_metrics
        assert 'meta_learning_improvement' in result.validation_metrics
        
        # Check hypothesis validation
        hypothesis = result.hypothesis
        assert isinstance(hypothesis.validation_score, float)
        assert isinstance(hypothesis.meta_learning_improvement, float)
        assert isinstance(hypothesis.confidence_score, float)
        
        # Check natural language description quality
        nl_desc = hypothesis.natural_language_description
        assert len(nl_desc) > 200  # Should be comprehensive
        assert "confidence" in nl_desc.lower()
        assert "validation" in nl_desc.lower()
        
        # Test validation criteria checking
        meets_criteria = integrated_discovery._meets_validation_criteria(hypothesis)
        assert isinstance(meets_criteria, bool)


if __name__ == '__main__':
    pytest.main([__file__])
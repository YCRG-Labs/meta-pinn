"""
Integration tests for comprehensive report generation pipeline.

Tests the complete end-to-end report generation workflow including
natural language generation, plot integration, table integration,
and multi-format output generation.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

from ml_research_pipeline.papers.report_generator import ReportGenerator
from ml_research_pipeline.papers.plot_generator import PaperPlotGenerator
from ml_research_pipeline.papers.table_generator import LaTeXTableGenerator


class TestReportGenerationIntegration:
    """Integration test suite for complete report generation pipeline."""
    
    @pytest.fixture
    def comprehensive_experimental_results(self):
        """Comprehensive experimental results for integration testing."""
        return {
            'experiment_type': 'Meta-Learning Physics-Informed Neural Networks',
            'num_tasks': 500,
            'metrics': ['parameter_accuracy', 'adaptation_speed', 'physics_consistency', 'computational_efficiency'],
            'method_results': {
                'Meta-PINN': {
                    'parameter_accuracy_mean': 0.94,
                    'parameter_accuracy_std': 0.03,
                    'adaptation_speed_mean': 8.2,
                    'adaptation_speed_std': 1.1,
                    'physics_consistency_mean': 0.98,
                    'physics_consistency_std': 0.01,
                    'computational_efficiency_mean': 0.85,
                    'computational_efficiency_std': 0.05
                },
                'Standard PINN': {
                    'parameter_accuracy_mean': 0.82,
                    'parameter_accuracy_std': 0.06,
                    'adaptation_speed_mean': 45.7,
                    'adaptation_speed_std': 8.3,
                    'physics_consistency_mean': 0.91,
                    'physics_consistency_std': 0.04,
                    'computational_efficiency_mean': 0.45,
                    'computational_efficiency_std': 0.08
                },
                'Transfer Learning PINN': {
                    'parameter_accuracy_mean': 0.88,
                    'parameter_accuracy_std': 0.04,         
           'adaptation_speed_mean': 22.1,
                    'adaptation_speed_std': 3.5,
                    'physics_consistency_mean': 0.94,
                    'physics_consistency_std': 0.02,
                    'computational_efficiency_mean': 0.68,
                    'computational_efficiency_std': 0.06
                },
                'Fourier Neural Operator': {
                    'parameter_accuracy_mean': 0.79,
                    'parameter_accuracy_std': 0.07,
                    'adaptation_speed_mean': 12.4,
                    'adaptation_speed_std': 2.1,
                    'physics_consistency_mean': 0.76,
                    'physics_consistency_std': 0.08,
                    'computational_efficiency_mean': 0.92,
                    'computational_efficiency_std': 0.03
                },
                'DeepONet': {
                    'parameter_accuracy_mean': 0.81,
                    'parameter_accuracy_std': 0.05,
                    'adaptation_speed_mean': 15.6,
                    'adaptation_speed_std': 2.8,
                    'physics_consistency_mean': 0.83,
                    'physics_consistency_std': 0.05,
                    'computational_efficiency_mean': 0.78,
                    'computational_efficiency_std': 0.04
                }
            },
            'statistical_tests': {
                'Meta-PINN': {
                    'parameter_accuracy': 0.001,
                    'adaptation_speed': 0.001,
                    'physics_consistency': 0.001,
                    'computational_efficiency': 0.01
                },
                'Standard PINN': {
                    'parameter_accuracy': 0.15,
                    'adaptation_speed': 0.2,
                    'physics_consistency': 0.08,
                    'computational_efficiency': 0.3
                },
                'Transfer Learning PINN': {
                    'parameter_accuracy': 0.02,
                    'adaptation_speed': 0.01,
                    'physics_consistency': 0.03,
                    'computational_efficiency': 0.04
                },
                'Fourier Neural Operator': {
                    'parameter_accuracy': 0.25,
                    'adaptation_speed': 0.05,
                    'physics_consistency': 0.4,
                    'computational_efficiency': 0.001
                },
                'DeepONet': {
                    'parameter_accuracy': 0.12,
                    'adaptation_speed': 0.03,
                    'physics_consistency': 0.18,
                    'computational_efficiency': 0.02
                }
            },  
          'convergence_data': {
                'Meta-PINN': [1.0, 0.3, 0.08, 0.02, 0.005, 0.001],
                'Standard PINN': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                'Transfer Learning PINN': [1.0, 0.6, 0.25, 0.08, 0.03, 0.01],
                'Fourier Neural Operator': [1.0, 0.4, 0.15, 0.06, 0.02],
                'DeepONet': [1.0, 0.5, 0.2, 0.08, 0.03, 0.01]
            },
            'efficiency_data': {
                'training_time': {
                    'Meta-PINN': 120.5,
                    'Standard PINN': 450.2,
                    'Transfer Learning PINN': 280.1,
                    'Fourier Neural Operator': 95.3,
                    'DeepONet': 180.7
                },
                'memory_usage': {
                    'Meta-PINN': 2.1,
                    'Standard PINN': 1.8,
                    'Transfer Learning PINN': 2.3,
                    'Fourier Neural Operator': 3.2,
                    'DeepONet': 2.8
                }
            },
            'robustness_data': {
                'variance_across_tasks': {
                    'Meta-PINN': 0.012,
                    'Standard PINN': 0.045,
                    'Transfer Learning PINN': 0.028,
                    'Fourier Neural Operator': 0.067,
                    'DeepONet': 0.038
                }
            },
            'uncertainty_data': {
                'x': np.linspace(0, 1, 100),
                'y_mean': np.sin(np.linspace(0, 2*np.pi, 100)),
                'y_std': 0.1 * np.ones(100),
                'ground_truth': np.sin(np.linspace(0, 2*np.pi, 100))
            },
            'hyperparameters': {
                'Meta-PINN': {
                    'meta_learning_rate': 0.001,
                    'adaptation_learning_rate': 0.01,
                    'adaptation_steps': 5,
                    'hidden_layers': [64, 128, 128, 64],
                    'batch_size': 32
                },
                'Standard PINN': {
                    'learning_rate': 0.001,
                    'hidden_layers': [32, 64, 64, 32],
                    'batch_size': 64,
                    'epochs': 1000
                },
                'Transfer Learning PINN': {
                    'pretrain_learning_rate': 0.001,
                    'finetune_learning_rate': 0.0001,
                    'hidden_layers': [64, 128, 64],
                    'batch_size': 32,
                    'pretrain_epochs': 500,
                    'finetune_epochs': 100
                }
            },     
       'physics_discovery': {
                'discovered_relationships': [
                    {
                        'relationship': 'viscosity ~ temperature^(-1.5)',
                        'confidence': 0.92,
                        'validation_score': 0.88,
                        'description': 'Inverse power law relationship between viscosity and temperature'
                    },
                    {
                        'relationship': 'viscosity ~ pressure * exp(-activation_energy/RT)',
                        'confidence': 0.87,
                        'validation_score': 0.84,
                        'description': 'Arrhenius-type temperature dependence with pressure scaling'
                    }
                ],
                'causal_strengths': {
                    'temperature': 0.85,
                    'pressure': 0.72,
                    'shear_rate': 0.43,
                    'concentration': 0.38
                }
            },
            'effect_sizes': {
                'Meta-PINN vs Standard PINN': {
                    'parameter_accuracy': 1.8,  # Large effect
                    'adaptation_speed': 2.3,    # Very large effect
                    'physics_consistency': 1.2  # Medium effect
                },
                'Meta-PINN vs Transfer Learning': {
                    'parameter_accuracy': 0.7,  # Medium effect
                    'adaptation_speed': 1.1,    # Medium effect
                    'physics_consistency': 0.9  # Medium effect
                }
            },
            'confidence_intervals': {
                'Meta-PINN': {
                    'parameter_accuracy': [0.91, 0.97],
                    'adaptation_speed': [7.1, 9.3],
                    'physics_consistency': [0.97, 0.99]
                },
                'Standard PINN': {
                    'parameter_accuracy': [0.76, 0.88],
                    'adaptation_speed': [37.4, 54.0],
                    'physics_consistency': [0.87, 0.95]
                }
            }
        }
    
    def test_end_to_end_report_generation_all_formats(self, comprehensive_experimental_results):
        """Test complete end-to-end report generation in all formats."""
        generator = ReportGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Test all three formats
            formats = ['markdown', 'latex', 'html']
            
            for format_type in formats:
                format_output_dir = output_dir / format_type
                format_output_dir.mkdir(exist_ok=True)
                
                saved_files = generator.generate_comprehensive_report(
                    experimental_results=comprehensive_experimental_results,
                    output_dir=format_output_dir,
                    report_title=f"Meta-Learning PINN Research Report ({format_type.upper()})",
                    author="ML Research Pipeline",
                    include_plots=True,
                    include_tables=True,
                    format_type=format_type
                )
                
                # Verify core files are generated
                assert 'report' in saved_files
                assert 'metadata' in saved_files
                
                # Verify report file exists and has content
                report_path = saved_files['report']
                assert report_path.exists()
                assert report_path.stat().st_size > 1000  # Should be substantial content
                
                # Verify metadata
                metadata_path = saved_files['metadata']
                assert metadata_path.exists()
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                assert metadata['experiment_info']['num_methods'] == 5
                assert len(metadata['experiment_info']['metrics']) == 4
                
                # Format-specific checks
                if format_type == 'markdown':
                    assert report_path.suffix == '.md'
                    with open(report_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    assert '# Meta-Learning PINN Research Report (MARKDOWN)' in content
                    assert '## Executive Summary' in content
                    assert '## Table of Contents' in content
                
                elif format_type == 'latex':
                    assert report_path.suffix == '.tex'
                    with open(report_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    assert '\\documentclass{article}' in content
                    assert '\\begin{document}' in content
                    assert '\\end{document}' in content
                
                elif format_type == 'html':
                    assert report_path.suffix == '.html'
                    with open(report_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    assert '<!DOCTYPE html>' in content
                    assert '<html>' in content
                    assert '</html>' in content 
   
    def test_natural_language_generation_quality(self, comprehensive_experimental_results):
        """Test the quality and accuracy of natural language generation."""
        generator = ReportGenerator()
        
        # Test executive summary generation
        summary = generator._generate_executive_summary(comprehensive_experimental_results)
        
        # Should mention key findings
        assert 'Meta-Learning Physics-Informed Neural Networks' in summary
        assert '5 different methods' in summary
        assert '500 tasks' in summary
        assert 'Meta-PINN' in summary  # Best performing method
        assert 'parameter_accuracy' in summary
        
        # Should include statistical information
        assert 'statistically significant' in summary.lower()
        
        # Test method comparison analysis
        comparison = generator._generate_method_comparison_analysis(comprehensive_experimental_results)
        
        # Should rank methods correctly
        assert 'Meta-PINN' in comparison
        assert 'Standard PINN' in comparison
        assert 'Transfer Learning' in comparison
        assert 'Fourier Neural Operator' in comparison
        assert 'DeepONet' in comparison
        
        # Test performance analysis with physics discovery
        performance = generator._generate_performance_analysis(comprehensive_experimental_results)
        
        # Should mention convergence
        assert 'convergence' in performance.lower() or 'converged' in performance.lower()
        
        # Should mention efficiency
        assert 'efficiency' in performance.lower() or 'efficient' in performance.lower()
    
    def test_physics_discovery_integration(self, comprehensive_experimental_results):
        """Test integration of physics discovery results into natural language descriptions."""
        generator = ReportGenerator()
        
        # Add physics discovery analysis method
        physics_analysis = generator._analyze_physics_discoveries(
            comprehensive_experimental_results.get('physics_discovery', {})
        )
        
        assert isinstance(physics_analysis, str)
        assert len(physics_analysis) > 0
        
        # Should mention discovered relationships
        if 'discovered_relationships' in comprehensive_experimental_results.get('physics_discovery', {}):
            assert 'viscosity' in physics_analysis.lower()
            assert 'temperature' in physics_analysis.lower()
    
    def test_statistical_analysis_integration(self, comprehensive_experimental_results):
        """Test integration of statistical analysis into report generation."""
        generator = ReportGenerator()
        
        # Test statistical analysis generation
        stats_analysis = generator._generate_statistical_analysis(comprehensive_experimental_results)
        
        # Should include effect sizes
        assert 'effect' in stats_analysis.lower()
        
        # Should include confidence intervals
        assert 'confidence' in stats_analysis.lower()
        
        # Should mention significance testing
        assert 'significant' in stats_analysis.lower()
    
    def test_comprehensive_content_validation(self, comprehensive_experimental_results):
        """Test that comprehensive reports contain all expected content sections."""
        generator = ReportGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = generator.generate_comprehensive_report(
                experimental_results=comprehensive_experimental_results,
                output_dir=output_dir,
                report_title="Comprehensive Content Validation Test",
                format_type='markdown'
            )
            
            report_path = saved_files['report']
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check all required sections are present
            required_sections = [
                'Executive Summary',
                'Method Comparison',
                'Performance Analysis',
                'Statistical Analysis',
                'Detailed Results'
            ]
            
            for section in required_sections:
                assert section in content, f"Missing required section: {section}"
            
            # Check that numerical results are included
            assert '0.94' in content  # Meta-PINN accuracy
            assert '8.2' in content   # Meta-PINN adaptation speed
            
            # Check that method names are mentioned
            methods = ['Meta-PINN', 'Standard PINN', 'Transfer Learning', 'Fourier Neural Operator', 'DeepONet']
            for method in methods:
                assert method in content, f"Missing method: {method}"
            
            # Check that metrics are discussed
            metrics = ['parameter_accuracy', 'adaptation_speed', 'physics_consistency', 'computational_efficiency']
            for metric in metrics:
                # Check for metric name or readable version
                readable_metric = metric.replace('_', ' ')
                assert metric in content or readable_metric in content, f"Missing metric: {metric}"
    
    def test_error_resilience_and_graceful_degradation(self):
        """Test that report generation handles missing or incomplete data gracefully."""
        generator = ReportGenerator()
        
        # Test with minimal data
        minimal_results = {
            'method_results': {
                'Method A': {'accuracy_mean': 0.8}
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Should not crash with minimal data
            saved_files = generator.generate_comprehensive_report(
                experimental_results=minimal_results,
                output_dir=output_dir,
                include_plots=False,  # Avoid plot generation errors
                include_tables=False  # Avoid table generation errors
            )
            
            assert 'report' in saved_files
            assert saved_files['report'].exists()
            
            # Report should have some content even with minimal data
            with open(saved_files['report'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert len(content) > 100  # Should have some meaningful content
            assert 'Method A' in content
    
    def test_large_scale_report_generation(self, comprehensive_experimental_results):
        """Test report generation performance with large-scale data."""
        generator = ReportGenerator()
        
        # Expand the dataset to simulate large-scale experiments
        large_scale_results = comprehensive_experimental_results.copy()
        large_scale_results['num_tasks'] = 5000
        
        # Add more methods
        for i in range(10, 20):
            method_name = f'Method_{i}'
            large_scale_results['method_results'][method_name] = {
                'parameter_accuracy_mean': 0.7 + 0.2 * np.random.random(),
                'parameter_accuracy_std': 0.02 + 0.05 * np.random.random(),
                'adaptation_speed_mean': 10 + 30 * np.random.random(),
                'adaptation_speed_std': 1 + 5 * np.random.random()
            }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Should handle large datasets efficiently
            import time
            start_time = time.time()
            
            saved_files = generator.generate_comprehensive_report(
                experimental_results=large_scale_results,
                output_dir=output_dir,
                include_plots=False,  # Skip plots for performance test
                include_tables=False  # Skip tables for performance test
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Should complete in reasonable time (less than 30 seconds)
            assert generation_time < 30, f"Report generation took too long: {generation_time:.2f} seconds"
            
            # Verify all methods are included
            report_path = saved_files['report']
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Should mention the expanded number of methods
            assert '15 different methods' in content or '15 methods' in content


if __name__ == '__main__':
    pytest.main([__file__])
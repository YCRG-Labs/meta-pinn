"""
Integration tests for the complete theoretical analysis system.
"""

import pytest
import tempfile
import os
from theory.sample_complexity import SampleComplexityAnalyzer, ComplexityParameters
from theory.convergence_analysis import ConvergenceAnalyzer, ConvergenceParameters
from theory.proofs.mathematical_proofs import TheoremGenerator


class TestTheoreticalAnalysisIntegration:
    """Test integration of all theoretical analysis components."""
    
    def test_complete_theoretical_pipeline(self):
        """Test the complete theoretical analysis pipeline."""
        
        # 1. Sample Complexity Analysis
        complexity_params = ComplexityParameters(
            dimension=3,
            lipschitz_constant=5.0,
            physics_constraint_strength=0.6,
            noise_level=0.05,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        
        sample_analyzer = SampleComplexityAnalyzer(complexity_params)
        sample_bounds = sample_analyzer.analyze_sample_complexity(
            network_width=64, network_depth=4, n_tasks=200
        )
        
        # Verify sample complexity results
        assert sample_bounds.improvement_factor > 1.0
        assert sample_bounds.physics_informed_bound < sample_bounds.traditional_bound
        
        # 2. Convergence Rate Analysis
        convergence_params = ConvergenceParameters(
            lipschitz_constant=5.0,
            strong_convexity=0.5,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=10,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        
        convergence_analyzer = ConvergenceAnalyzer(convergence_params)
        convergence_analysis = convergence_analyzer.compute_comprehensive_analysis(
            n_tasks=200, n_support=20, physics_regularization=0.6
        )
        
        # Verify convergence results
        assert 0 < convergence_analysis.task_level_rate < 1.0
        assert convergence_analysis.meta_level_rate > 0
        assert len(convergence_analysis.error_bounds) > 0
        
        # 3. Formal Proofs and Documentation
        theorem_generator = TheoremGenerator()
        
        # Generate all theorems
        sample_theorem = theorem_generator.generate_sample_complexity_theorem()
        convergence_theorem = theorem_generator.generate_convergence_rate_theorem()
        physics_theorem = theorem_generator.generate_physics_benefit_theorem()
        
        # Generate all proofs
        proofs = theorem_generator.generate_all_proofs()
        
        # Verify theorem generation
        assert len(theorem_generator.theorems) == 3
        assert len(proofs) == 4  # Including meta-generalization proof
        
        # 4. Export Complete Documentation
        with tempfile.TemporaryDirectory() as temp_dir:
            latex_path = os.path.join(temp_dir, "theoretical_analysis.tex")
            theorem_generator.export_latex_document(latex_path)
            
            # Verify document was created
            assert os.path.exists(latex_path)
            
            # Read and verify content
            with open(latex_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check that all components are integrated
            assert "Sample Complexity" in content
            assert "Convergence Rate" in content
            assert "Physics Constraint" in content
            assert "\\begin{proof}" in content
            assert "\\section{References}" in content
            
        # 5. Verify Theoretical Consistency
        # The improvement factors should be consistent across analyses
        sample_improvement = sample_bounds.improvement_factor
        
        # Physics regularization should improve both sample complexity and convergence
        no_physics_bounds = sample_analyzer.analyze_sample_complexity(
            network_width=64, network_depth=4, n_tasks=200
        )
        
        # With physics constraints, we should get better bounds
        physics_bounds = sample_analyzer.compute_physics_informed_bound(64, 4)
        traditional_bounds = sample_analyzer.compute_traditional_bound(64, 4)
        
        assert physics_bounds < traditional_bounds
        
    def test_theoretical_predictions_validation(self):
        """Test validation of theoretical predictions."""
        
        # Create analyzers with consistent parameters
        complexity_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=10.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        
        convergence_params = ConvergenceParameters(
            lipschitz_constant=10.0,  # Same as complexity analysis
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        
        sample_analyzer = SampleComplexityAnalyzer(complexity_params)
        convergence_analyzer = ConvergenceAnalyzer(convergence_params)
        
        # Analyze with different physics regularization strengths
        physics_strengths = [0.0, 0.3, 0.6, 0.9]
        
        sample_bounds = []
        convergence_rates = []
        
        for strength in physics_strengths:
            # Update physics constraint strength
            complexity_params.physics_constraint_strength = strength
            sample_analyzer.params = complexity_params
            
            # Get sample complexity bounds
            bounds = sample_analyzer.analyze_sample_complexity(64, 3, 100)
            sample_bounds.append(bounds.improvement_factor)
            
            # Get convergence rates
            rate = convergence_analyzer.compute_task_level_convergence_rate(strength)
            convergence_rates.append(rate)
            
        # Verify monotonic improvement with physics constraints
        # Sample complexity improvement should increase
        for i in range(1, len(sample_bounds)):
            assert sample_bounds[i] >= sample_bounds[i-1]
            
        # Convergence rates should improve (decrease) with physics constraints
        for i in range(1, len(convergence_rates)):
            assert convergence_rates[i] <= convergence_rates[i-1]
            
    def test_scalability_analysis(self):
        """Test theoretical analysis scalability with problem size."""
        
        complexity_params = ComplexityParameters(
            dimension=2,
            lipschitz_constant=10.0,
            physics_constraint_strength=0.5,
            noise_level=0.1,
            confidence_delta=0.05,
            approximation_error=0.01
        )
        
        sample_analyzer = SampleComplexityAnalyzer(complexity_params)
        
        # Test scaling with network size
        network_sizes = [(32, 2), (64, 3), (128, 4), (256, 5)]
        bounds = []
        
        for width, depth in network_sizes:
            bound = sample_analyzer.compute_traditional_bound(width, depth)
            bounds.append(bound)
            
        # Bounds should increase with network complexity
        for i in range(1, len(bounds)):
            assert bounds[i] > bounds[i-1]
            
        # Test scaling with number of tasks
        task_counts = [10, 50, 100, 500, 1000]
        meta_rates = []
        
        convergence_params = ConvergenceParameters(
            lipschitz_constant=10.0,
            strong_convexity=1.0,
            gradient_noise_variance=0.01,
            task_similarity=0.7,
            adaptation_steps=5,
            meta_learning_rate=0.001,
            task_learning_rate=0.01
        )
        
        convergence_analyzer = ConvergenceAnalyzer(convergence_params)
        
        for n_tasks in task_counts:
            rate = convergence_analyzer.compute_meta_level_convergence_rate(n_tasks)
            meta_rates.append(rate)
            
        # Meta-learning rates should improve (decrease) with more tasks
        for i in range(1, len(meta_rates)):
            assert meta_rates[i] < meta_rates[i-1]
            
    def test_documentation_completeness(self):
        """Test completeness of generated documentation."""
        
        theorem_generator = TheoremGenerator()
        
        # Generate all components
        theorem_generator.generate_sample_complexity_theorem()
        theorem_generator.generate_convergence_rate_theorem()
        theorem_generator.generate_physics_benefit_theorem()
        proofs = theorem_generator.generate_all_proofs()
        
        # Check all theorems have required components
        for theorem_name, theorem in theorem_generator.theorems.items():
            assert len(theorem.statement) > 50  # Substantial statement
            assert len(theorem.assumptions) >= 3  # Multiple assumptions
            assert len(theorem.proof_sketch) > 50  # Substantial sketch
            assert "\\begin{theorem}" in theorem.latex_statement
            assert len(theorem.references) >= 2  # Multiple references
            
        # Check all proofs have multiple steps
        for proof_name, proof in proofs.items():
            assert len(proof.steps) >= 4  # Multi-step proofs
            
            # Each step should have content
            for step in proof.steps:
                assert len(step.statement) > 10
                assert len(step.justification) > 5
                
        # Test LaTeX export completeness
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "complete_analysis.tex")
            theorem_generator.export_latex_document(output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check document structure
            sections = ["\\section{Main Theorems}", "\\section{Proofs}", "\\section{References}"]
            for section in sections:
                assert section in content
                
            # Check mathematical content
            math_elements = ["\\begin{theorem}", "\\begin{proof}", "\\end{proof}", "$$"]
            for element in math_elements:
                assert element in content
                
            # Check references
            assert "\\bibitem" in content
            assert "\\end{thebibliography}" in content


if __name__ == "__main__":
    pytest.main([__file__])
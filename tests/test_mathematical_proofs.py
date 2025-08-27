"""
Unit tests for mathematical proofs and theorem generation.
"""

import pytest
import os
import tempfile
from theory.proofs.mathematical_proofs import (
    TheoremGenerator,
    SampleComplexityProof,
    ConvergenceRateProof,
    PhysicsConstraintBenefitProof,
    MetaLearningGeneralizationProof,
    verify_mathematical_notation,
    TheoremStatement,
    ProofStep
)


class TestTheoremGenerator:
    """Test theorem generation functionality."""
    
    @pytest.fixture
    def generator(self):
        """Theorem generator instance."""
        return TheoremGenerator()
    
    def test_sample_complexity_theorem_generation(self, generator):
        """Test sample complexity theorem generation."""
        theorem = generator.generate_sample_complexity_theorem()
        
        # Check theorem structure
        assert isinstance(theorem, TheoremStatement)
        assert theorem.name == "Sample Complexity Bound for Physics-Informed Meta-Learning"
        assert len(theorem.statement) > 0
        assert len(theorem.assumptions) > 0
        assert len(theorem.proof_sketch) > 0
        assert len(theorem.latex_statement) > 0
        assert len(theorem.references) > 0
        
        # Check key concepts are mentioned
        assert "sample complexity" in theorem.statement.lower()
        assert "physics" in theorem.statement.lower()
        assert ("meta-learning" in theorem.statement.lower() or "meta-training" in theorem.statement.lower())
        
    def test_convergence_rate_theorem_generation(self, generator):
        """Test convergence rate theorem generation."""
        theorem = generator.generate_convergence_rate_theorem()
        
        # Check theorem structure
        assert isinstance(theorem, TheoremStatement)
        assert theorem.name == "Convergence Rate for Physics-Informed Meta-Learning"
        assert len(theorem.statement) > 0
        assert len(theorem.assumptions) > 0
        
        # Check key concepts
        assert ("convergence" in theorem.statement.lower() or "converge" in theorem.statement.lower())
        assert "gradient descent" in theorem.statement.lower()
        assert "condition number" in theorem.statement.lower()
        
    def test_physics_benefit_theorem_generation(self, generator):
        """Test physics constraint benefit theorem generation."""
        theorem = generator.generate_physics_benefit_theorem()
        
        # Check theorem structure
        assert isinstance(theorem, TheoremStatement)
        assert theorem.name == "Physics Constraint Benefit"
        assert len(theorem.statement) > 0
        
        # Check key concepts
        assert "physics constraints" in theorem.statement.lower()
        assert "sample complexity" in theorem.statement.lower()
        assert "improvement" in theorem.statement.lower()
        
    def test_all_theorems_generation(self, generator):
        """Test generation of all theorems."""
        # Generate all theorems
        sample_theorem = generator.generate_sample_complexity_theorem()
        convergence_theorem = generator.generate_convergence_rate_theorem()
        physics_theorem = generator.generate_physics_benefit_theorem()
        
        # Check they're stored in the generator
        assert "sample_complexity" in generator.theorems
        assert "convergence_rate" in generator.theorems
        assert "physics_benefit" in generator.theorems
        
        # Check they're different theorems
        assert sample_theorem.name != convergence_theorem.name
        assert convergence_theorem.name != physics_theorem.name
        
    def test_latex_document_export(self, generator):
        """Test LaTeX document export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "theorems.tex")
            
            # Export document
            generator.export_latex_document(output_path)
            
            # Check file was created
            assert os.path.exists(output_path)
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check LaTeX structure
            assert "\\documentclass" in content
            assert "\\begin{document}" in content
            assert "\\end{document}" in content
            assert "\\section{Main Theorems}" in content
            assert "\\section{Proofs}" in content
            
            # Check theorems are included
            assert "Sample Complexity" in content
            assert "Convergence Rate" in content
            assert "Physics Constraint" in content


class TestFormalProofs:
    """Test formal proof construction."""
    
    def test_sample_complexity_proof(self):
        """Test sample complexity proof construction."""
        proof = SampleComplexityProof()
        steps = proof.construct_proof()
        
        # Check proof structure
        assert len(steps) > 0
        assert all(isinstance(step, ProofStep) for step in steps)
        
        # Check step numbering
        for i, step in enumerate(steps):
            assert step.step_number == i + 1
            
        # Check content
        step_statements = [step.statement for step in steps]
        combined_text = " ".join(step_statements).lower()
        
        assert "hypothesis class" in combined_text or "physics constraints" in combined_text
        assert "rademacher" in combined_text or "complexity" in combined_text
        assert "bound" in combined_text
        
    def test_convergence_rate_proof(self):
        """Test convergence rate proof construction."""
        proof = ConvergenceRateProof()
        steps = proof.construct_proof()
        
        # Check proof structure
        assert len(steps) > 0
        assert all(isinstance(step, ProofStep) for step in steps)
        
        # Check content
        step_statements = [step.statement for step in steps]
        combined_text = " ".join(step_statements).lower()
        
        assert "optimization" in combined_text or "gradient" in combined_text
        assert "convergence" in combined_text or "convexity" in combined_text
        assert "condition number" in combined_text or "lipschitz" in combined_text
        
    def test_physics_constraint_benefit_proof(self):
        """Test physics constraint benefit proof construction."""
        proof = PhysicsConstraintBenefitProof()
        steps = proof.construct_proof()
        
        # Check proof structure
        assert len(steps) > 0
        assert all(isinstance(step, ProofStep) for step in steps)
        
        # Check content
        step_statements = [step.statement for step in steps]
        combined_text = " ".join(step_statements).lower()
        
        assert "hypothesis space" in combined_text or "constraint" in combined_text
        assert "dimension" in combined_text or "complexity" in combined_text
        
    def test_meta_learning_generalization_proof(self):
        """Test meta-learning generalization proof construction."""
        proof = MetaLearningGeneralizationProof()
        steps = proof.construct_proof()
        
        # Check proof structure
        assert len(steps) > 0
        assert all(isinstance(step, ProofStep) for step in steps)
        
        # Check content
        step_statements = [step.statement for step in steps]
        combined_text = " ".join(step_statements).lower()
        
        assert "meta-learning" in combined_text or "task" in combined_text
        assert "generalization" in combined_text or "adaptation" in combined_text
        
    def test_proof_latex_generation(self):
        """Test LaTeX generation for proofs."""
        proof = SampleComplexityProof()
        proof.construct_proof()
        
        latex_output = proof.generate_latex()
        
        # Check LaTeX structure
        assert "\\begin{proof}" in latex_output
        assert "\\end{proof}" in latex_output
        
        # Check steps are included
        for step in proof.steps:
            # Statement should be in the output
            assert step.statement in latex_output or step.statement.replace("F", "$F$") in latex_output
            
    def test_proof_step_addition(self):
        """Test adding steps to proofs."""
        proof = SampleComplexityProof()
        
        # Add a custom step
        proof.add_step(
            "This is a test statement.",
            "Test justification",
            "x = y + z"
        )
        
        # Check step was added
        assert len(proof.steps) == 1
        step = proof.steps[0]
        assert step.step_number == 1
        assert step.statement == "This is a test statement."
        assert step.justification == "Test justification"
        assert step.latex_expression == "x = y + z"


class TestMathematicalNotationVerification:
    """Test mathematical notation verification."""
    
    def test_consistent_notation(self):
        """Test verification of consistent notation."""
        consistent_content = """
        Let $\\theta$ be the parameter vector and $\\epsilon$ be the error.
        We have $\\theta^* = \\arg\\min L(\\theta)$ with error $\\epsilon < 0.01$.
        """
        
        issues = verify_mathematical_notation(consistent_content)
        
        # Should have no issues with consistent notation
        assert len(issues) == 0
        
    def test_inconsistent_theta_notation(self):
        """Test detection of inconsistent theta notation."""
        inconsistent_content = """
        Let $\\theta$ be the parameter and $\\Theta$ be the space.
        """
        
        issues = verify_mathematical_notation(inconsistent_content)
        
        # Should detect mixed theta notation
        theta_issues = [issue for issue in issues if "theta" in issue.lower()]
        assert len(theta_issues) > 0
        
    def test_inconsistent_epsilon_notation(self):
        """Test detection of inconsistent epsilon notation."""
        inconsistent_content = """
        The error $\\epsilon$ is bounded by $\\varepsilon$.
        """
        
        issues = verify_mathematical_notation(inconsistent_content)
        
        # Should detect mixed epsilon notation
        epsilon_issues = [issue for issue in issues if "epsilon" in issue.lower()]
        assert len(epsilon_issues) > 0
        
    def test_unmatched_delimiters(self):
        """Test detection of unmatched delimiters."""
        unmatched_content = """
        We have $\\left( x + y \\right)$ and $\\left[ a + b$.
        """
        
        issues = verify_mathematical_notation(unmatched_content)
        
        # Should detect unmatched brackets
        bracket_issues = [issue for issue in issues if "bracket" in issue.lower()]
        assert len(bracket_issues) > 0
        
    def test_unmatched_align_environments(self):
        """Test detection of unmatched align environments."""
        unmatched_content = """
        \\begin{align}
        x &= y + z
        """
        
        issues = verify_mathematical_notation(unmatched_content)
        
        # Should detect unmatched align environment
        align_issues = [issue for issue in issues if "align" in issue.lower()]
        assert len(align_issues) > 0
        
    def test_complex_document_verification(self):
        """Test verification of complex document."""
        complex_content = """
        \\documentclass{article}
        \\begin{document}
        
        Let $\\theta \\in \\mathbb{R}^d$ be the parameter vector.
        We consider the optimization problem:
        \\begin{align}
        \\min_{\\theta} L(\\theta) &= \\frac{1}{n} \\sum_{i=1}^n \\ell(f(x_i; \\theta), y_i) \\\\
        &\\quad + \\lambda \\|\\mathcal{L}[f](x; \\theta)\\|^2
        \\end{align}
        
        The convergence rate is $O\\left(\\left(\\frac{\\kappa-1}{\\kappa+1}\\right)^k\\right)$.
        
        \\end{document}
        """
        
        issues = verify_mathematical_notation(complex_content)
        
        # Well-formed document should have minimal issues
        assert len(issues) <= 1  # Allow for minor formatting variations


class TestProofIntegration:
    """Test integration of proofs with theorem generator."""
    
    def test_proof_generation_integration(self):
        """Test that all proofs can be generated and integrated."""
        generator = TheoremGenerator()
        
        # Generate all proofs
        proofs = generator.generate_all_proofs()
        
        # Check all expected proofs are present
        expected_proofs = ["sample_complexity", "convergence_rate", "physics_benefit", "meta_generalization"]
        for proof_name in expected_proofs:
            assert proof_name in proofs
            assert len(proofs[proof_name].steps) > 0
            
    def test_complete_document_generation(self):
        """Test complete document generation with theorems and proofs."""
        generator = TheoremGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "complete_document.tex")
            
            # Generate complete document
            generator.export_latex_document(output_path)
            
            # Read and verify content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Verify mathematical notation
            issues = verify_mathematical_notation(content)
            
            # Should have minimal notation issues
            assert len(issues) <= 3  # Allow for some minor inconsistencies
            
            # Check document completeness
            assert "\\section{Main Theorems}" in content
            assert "\\section{Proofs}" in content
            assert "\\section{References}" in content
            
            # Check all theorems are present
            assert "Sample Complexity" in content
            assert "Convergence Rate" in content
            assert "Physics Constraint" in content
            
    def test_latex_compilation_readiness(self):
        """Test that generated LaTeX is ready for compilation."""
        generator = TheoremGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "compilation_test.tex")
            
            # Generate document
            generator.export_latex_document(output_path)
            
            # Read content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check essential LaTeX elements
            assert "\\documentclass" in content
            assert ("\\usepackage{amsmath}" in content or "amsmath" in content)
            assert ("\\usepackage{amsthm}" in content or "amsthm" in content)
            assert "\\begin{document}" in content
            assert "\\maketitle" in content
            assert "\\end{document}" in content
            
            # Check theorem environments
            assert "\\newtheorem{theorem}" in content
            assert "\\begin{theorem}" in content or "Theorem" in content
            
            # Verify no obvious LaTeX errors
            issues = verify_mathematical_notation(content)
            serious_issues = [issue for issue in issues if "unmatched" in issue.lower()]
            assert len(serious_issues) == 0  # No unmatched delimiters


if __name__ == "__main__":
    pytest.main([__file__])
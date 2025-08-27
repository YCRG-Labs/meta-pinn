"""
Reproducibility validation system for ensuring experiment reproducibility.
"""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch

from .random_utils import set_random_seeds, get_random_state, set_random_state
from .logging_utils import LoggerMixin


@dataclass
class ReproducibilityTest:
    """Configuration for a reproducibility test."""
    
    name: str
    description: str
    test_function: str  # Name of test function
    tolerance: float = 1e-6
    num_runs: int = 3
    check_gradients: bool = True
    check_random_state: bool = True


class ReproducibilityValidator(LoggerMixin):
    """Validator for ensuring experiment reproducibility."""
    
    def __init__(self, tolerance: float = 1e-6):
        """Initialize reproducibility validator.
        
        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        self.test_results = {}
    
    def validate_deterministic_forward_pass(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        seed: int = 42,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Validate that forward passes are deterministic.
        
        Args:
            model: Model to test
            input_data: Input tensor
            seed: Random seed
            num_runs: Number of runs to compare
            
        Returns:
            Validation results
        """
        self.log_info("Validating deterministic forward pass")
        
        outputs = []
        
        for run in range(num_runs):
            # Set deterministic seed
            set_random_seeds(seed, deterministic=True)
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(input_data)
            
            outputs.append(output.clone())
        
        # Check if all outputs are identical
        all_identical = True
        max_diff = 0.0
        
        for i in range(1, len(outputs)):
            diff = torch.abs(outputs[i] - outputs[0]).max().item()
            max_diff = max(max_diff, diff)
            
            if diff > self.tolerance:
                all_identical = False
        
        result = {
            'test_name': 'deterministic_forward_pass',
            'passed': all_identical,
            'max_difference': max_diff,
            'tolerance': self.tolerance,
            'num_runs': num_runs,
            'details': f"Maximum difference: {max_diff:.2e}"
        }
        
        if all_identical:
            self.log_info("✓ Forward pass is deterministic")
        else:
            self.log_warning(f"✗ Forward pass is not deterministic (max diff: {max_diff:.2e})")
        
        return result
    
    def validate_deterministic_training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        seed: int = 42,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Validate that training steps are deterministic.
        
        Args:
            model: Model to test
            optimizer: Optimizer
            loss_fn: Loss function
            input_data: Input tensor
            target_data: Target tensor
            seed: Random seed
            num_runs: Number of runs to compare
            
        Returns:
            Validation results
        """
        self.log_info("Validating deterministic training step")
        
        # Save initial model state
        initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        losses = []
        final_states = []
        
        for run in range(num_runs):
            # Restore initial state
            model.load_state_dict(initial_state)
            optimizer.zero_grad()
            
            # Set deterministic seed
            set_random_seeds(seed, deterministic=True)
            
            # Training step
            model.train()
            output = model(input_data)
            loss = loss_fn(output, target_data)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            final_states.append({k: v.clone() for k, v in model.state_dict().items()})
        
        # Check if all losses are identical
        loss_identical = all(abs(loss - losses[0]) < self.tolerance for loss in losses)
        
        # Check if all final states are identical
        state_identical = True
        max_param_diff = 0.0
        
        for i in range(1, len(final_states)):
            for key in final_states[0].keys():
                diff = torch.abs(final_states[i][key] - final_states[0][key]).max().item()
                max_param_diff = max(max_param_diff, diff)
                
                if diff > self.tolerance:
                    state_identical = False
        
        result = {
            'test_name': 'deterministic_training_step',
            'passed': loss_identical and state_identical,
            'loss_identical': loss_identical,
            'state_identical': state_identical,
            'max_loss_difference': max(abs(loss - losses[0]) for loss in losses),
            'max_parameter_difference': max_param_diff,
            'tolerance': self.tolerance,
            'num_runs': num_runs,
            'losses': losses
        }
        
        if result['passed']:
            self.log_info("✓ Training step is deterministic")
        else:
            self.log_warning("✗ Training step is not deterministic")
        
        return result
    
    def validate_random_state_consistency(
        self,
        seed: int = 42,
        num_operations: int = 100
    ) -> Dict[str, Any]:
        """Validate that random state can be saved and restored consistently.
        
        Args:
            seed: Random seed
            num_operations: Number of random operations to perform
            
        Returns:
            Validation results
        """
        self.log_info("Validating random state consistency")
        
        # Set initial seed
        set_random_seeds(seed, deterministic=True)
        
        # Save initial state
        initial_state = get_random_state()
        
        # Perform random operations and collect results
        results1 = []
        for _ in range(num_operations):
            results1.append(torch.randn(10).numpy())
        
        # Restore state and repeat operations
        set_random_state(initial_state)
        
        results2 = []
        for _ in range(num_operations):
            results2.append(torch.randn(10).numpy())
        
        # Compare results
        all_identical = True
        max_diff = 0.0
        
        for r1, r2 in zip(results1, results2):
            diff = np.abs(r1 - r2).max()
            max_diff = max(max_diff, diff)
            
            if diff > self.tolerance:
                all_identical = False
        
        result = {
            'test_name': 'random_state_consistency',
            'passed': all_identical,
            'max_difference': max_diff,
            'tolerance': self.tolerance,
            'num_operations': num_operations,
            'details': f"Random state save/restore: {'✓' if all_identical else '✗'}"
        }
        
        if all_identical:
            self.log_info("✓ Random state consistency validated")
        else:
            self.log_warning(f"✗ Random state inconsistency detected (max diff: {max_diff:.2e})")
        
        return result
    
    def validate_cross_platform_reproducibility(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        reference_output: Optional[torch.Tensor] = None,
        reference_file: Optional[Path] = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Validate reproducibility across different platforms.
        
        Args:
            model: Model to test
            input_data: Input tensor
            reference_output: Reference output tensor
            reference_file: File containing reference output
            seed: Random seed
            
        Returns:
            Validation results
        """
        self.log_info("Validating cross-platform reproducibility")
        
        # Set deterministic seed
        set_random_seeds(seed, deterministic=True)
        
        # Generate current output
        model.eval()
        with torch.no_grad():
            current_output = model(input_data)
        
        # Load reference output if file exists
        if reference_output is None and reference_file is not None and reference_file.exists():
            reference_output = torch.load(reference_file, map_location='cpu')
        
        if reference_output is None:
            # Save current output as reference
            if reference_file is not None:
                reference_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(current_output, reference_file)
                self.log_info(f"Saved reference output to {reference_file}")
            
            result = {
                'test_name': 'cross_platform_reproducibility',
                'passed': True,
                'is_reference': True,
                'details': "Generated reference output"
            }
        else:
            # Compare with reference
            diff = torch.abs(current_output - reference_output).max().item()
            passed = diff <= self.tolerance
            
            result = {
                'test_name': 'cross_platform_reproducibility',
                'passed': passed,
                'is_reference': False,
                'max_difference': diff,
                'tolerance': self.tolerance,
                'details': f"Difference from reference: {diff:.2e}"
            }
            
            if passed:
                self.log_info("✓ Cross-platform reproducibility validated")
            else:
                self.log_warning(f"✗ Cross-platform reproducibility failed (diff: {diff:.2e})")
        
        return result
    
    def validate_gradient_determinism(
        self,
        model: torch.nn.Module,
        loss_fn: callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        seed: int = 42,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Validate that gradients are computed deterministically.
        
        Args:
            model: Model to test
            loss_fn: Loss function
            input_data: Input tensor
            target_data: Target tensor
            seed: Random seed
            num_runs: Number of runs to compare
            
        Returns:
            Validation results
        """
        self.log_info("Validating gradient determinism")
        
        gradients = []
        
        for run in range(num_runs):
            # Set deterministic seed
            set_random_seeds(seed, deterministic=True)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward and backward pass
            model.train()
            output = model(input_data)
            loss = loss_fn(output, target_data)
            loss.backward()
            
            # Collect gradients
            run_gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    run_gradients[name] = param.grad.clone()
            
            gradients.append(run_gradients)
        
        # Compare gradients across runs
        all_identical = True
        max_diff = 0.0
        
        for i in range(1, len(gradients)):
            for name in gradients[0].keys():
                if name in gradients[i]:
                    diff = torch.abs(gradients[i][name] - gradients[0][name]).max().item()
                    max_diff = max(max_diff, diff)
                    
                    if diff > self.tolerance:
                        all_identical = False
        
        result = {
            'test_name': 'gradient_determinism',
            'passed': all_identical,
            'max_difference': max_diff,
            'tolerance': self.tolerance,
            'num_runs': num_runs,
            'details': f"Gradient determinism: {'✓' if all_identical else '✗'}"
        }
        
        if all_identical:
            self.log_info("✓ Gradient computation is deterministic")
        else:
            self.log_warning(f"✗ Gradient computation is not deterministic (max diff: {max_diff:.2e})")
        
        return result
    
    def run_comprehensive_validation(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        seed: int = 42,
        reference_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run comprehensive reproducibility validation.
        
        Args:
            model: Model to test
            optimizer: Optimizer
            loss_fn: Loss function
            input_data: Input tensor
            target_data: Target tensor
            seed: Random seed
            reference_dir: Directory for reference outputs
            
        Returns:
            Comprehensive validation results
        """
        self.log_info("Running comprehensive reproducibility validation")
        
        results = {}
        
        # Test 1: Deterministic forward pass
        results['forward_pass'] = self.validate_deterministic_forward_pass(
            model, input_data, seed
        )
        
        # Test 2: Deterministic training step
        results['training_step'] = self.validate_deterministic_training_step(
            model, optimizer, loss_fn, input_data, target_data, seed
        )
        
        # Test 3: Random state consistency
        results['random_state'] = self.validate_random_state_consistency(seed)
        
        # Test 4: Gradient determinism
        results['gradients'] = self.validate_gradient_determinism(
            model, loss_fn, input_data, target_data, seed
        )
        
        # Test 5: Cross-platform reproducibility (if reference directory provided)
        if reference_dir is not None:
            reference_dir = Path(reference_dir)
            reference_dir.mkdir(parents=True, exist_ok=True)
            
            reference_file = reference_dir / "reference_output.pth"
            results['cross_platform'] = self.validate_cross_platform_reproducibility(
                model, input_data, reference_file=reference_file, seed=seed
            )
        
        # Overall summary
        all_passed = all(result['passed'] for result in results.values())
        
        summary = {
            'overall_passed': all_passed,
            'total_tests': len(results),
            'passed_tests': sum(1 for result in results.values() if result['passed']),
            'failed_tests': sum(1 for result in results.values() if not result['passed']),
            'individual_results': results
        }
        
        if all_passed:
            self.log_info("✓ All reproducibility tests passed")
        else:
            failed_tests = [name for name, result in results.items() if not result['passed']]
            self.log_warning(f"✗ Some reproducibility tests failed: {failed_tests}")
        
        return summary
    
    def generate_reproducibility_report(
        self,
        validation_results: Dict[str, Any],
        output_file: Path
    ):
        """Generate a detailed reproducibility report.
        
        Args:
            validation_results: Results from comprehensive validation
            output_file: Output file path
        """
        report = {
            'timestamp': torch.utils.data.get_worker_info(),
            'summary': {
                'overall_status': 'PASSED' if validation_results['overall_passed'] else 'FAILED',
                'total_tests': validation_results['total_tests'],
                'passed_tests': validation_results['passed_tests'],
                'failed_tests': validation_results['failed_tests']
            },
            'detailed_results': validation_results['individual_results'],
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.log_info(f"Reproducibility report saved to {output_file}")
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results.
        
        Args:
            validation_results: Validation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not validation_results['overall_passed']:
            recommendations.append("Some reproducibility tests failed. Review individual test results.")
        
        for test_name, result in validation_results['individual_results'].items():
            if not result['passed']:
                if test_name == 'forward_pass':
                    recommendations.append(
                        "Forward pass is not deterministic. Ensure torch.backends.cudnn.deterministic=True "
                        "and torch.use_deterministic_algorithms(True)."
                    )
                elif test_name == 'training_step':
                    recommendations.append(
                        "Training step is not deterministic. Check optimizer settings and ensure "
                        "deterministic algorithms are enabled."
                    )
                elif test_name == 'random_state':
                    recommendations.append(
                        "Random state save/restore is inconsistent. Verify random seed management."
                    )
                elif test_name == 'gradients':
                    recommendations.append(
                        "Gradient computation is not deterministic. Check for non-deterministic operations "
                        "in the model or loss function."
                    )
                elif test_name == 'cross_platform':
                    recommendations.append(
                        "Cross-platform reproducibility failed. This may be due to hardware differences "
                        "or PyTorch version differences."
                    )
        
        if validation_results['overall_passed']:
            recommendations.append("All reproducibility tests passed. Experiment should be fully reproducible.")
        
        return recommendations


def create_reproducibility_test_suite() -> List[ReproducibilityTest]:
    """Create a standard suite of reproducibility tests.
    
    Returns:
        List of reproducibility tests
    """
    return [
        ReproducibilityTest(
            name="forward_pass_determinism",
            description="Test that forward passes produce identical outputs with same seed",
            test_function="validate_deterministic_forward_pass",
            tolerance=1e-6,
            num_runs=5
        ),
        ReproducibilityTest(
            name="training_step_determinism", 
            description="Test that training steps are deterministic",
            test_function="validate_deterministic_training_step",
            tolerance=1e-6,
            num_runs=3
        ),
        ReproducibilityTest(
            name="random_state_consistency",
            description="Test that random states can be saved and restored",
            test_function="validate_random_state_consistency",
            tolerance=1e-10,
            num_runs=1
        ),
        ReproducibilityTest(
            name="gradient_determinism",
            description="Test that gradient computation is deterministic",
            test_function="validate_gradient_determinism", 
            tolerance=1e-6,
            num_runs=3
        ),
        ReproducibilityTest(
            name="cross_platform_reproducibility",
            description="Test reproducibility across different platforms",
            test_function="validate_cross_platform_reproducibility",
            tolerance=1e-5,
            num_runs=1
        )
    ]
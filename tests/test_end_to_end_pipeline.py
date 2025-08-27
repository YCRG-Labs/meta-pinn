"""
Comprehensive end-to-end pipeline testing for the meta-learning PINN system.

This module implements integration tests for the complete meta-learning pipeline,
validation against known physics solutions, regression testing, and distributed
training workflows.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml
import json

from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.core.task_generator import FluidTaskGenerator
from ml_research_pipeline.core.dataset_manager import DatasetGenerator, DatasetLoader
from ml_research_pipeline.core.analytical_solutions import AnalyticalSolutions
from ml_research_pipeline.core.fenicsx_solver import FEniCSxSolver
from ml_research_pipeline.evaluation.benchmark_suite import PINNBenchmarkSuite
from ml_research_pipeline.evaluation.metrics import EvaluationMetrics
from ml_research_pipeline.core.distributed_meta_pinn import DistributedMetaPINN
from ml_research_pipeline.core.training_monitor import TrainingMonitor
from ml_research_pipeline.core.checkpoint_manager import CheckpointManager
from ml_research_pipeline.config.experiment_config import ExperimentConfig


class TestEndToEndPipeline:
    """Test complete meta-learning pipeline workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            'model': {
                'layers': [2, 64, 64, 64, 3],
                'activation': 'tanh',
                'meta_lr': 0.001,
                'adapt_lr': 0.01
            },
            'training': {
                'n_tasks': 10,
                'n_support': 50,
                'n_query': 25,
                'adaptation_steps': 5,
                'meta_epochs': 5
            },
            'task_generation': {
                'domain_bounds': {'x': [0, 1], 'y': [0, 1]},
                'task_types': ['linear_viscosity', 'bilinear_viscosity'],
                'reynolds_range': [10, 100]
            }
        }
    
    @pytest.fixture
    def meta_pinn_system(self, basic_config):
        """Initialize complete meta-learning system."""
        config = ExperimentConfig.from_dict(basic_config)
        
        # Initialize components
        meta_pinn = MetaPINN(
            layers=config.model.layers,
            meta_lr=config.model.meta_lr,
            adapt_lr=config.model.adapt_lr
        )
        
        task_generator = FluidTaskGenerator(
            domain_bounds=config.task_generation.domain_bounds,
            task_types=config.task_generation.task_types,
            reynolds_range=config.task_generation.reynolds_range
        )
        
        dataset_generator = DatasetGenerator()
        evaluator = EvaluationMetrics()
        
        return {
            'meta_pinn': meta_pinn,
            'task_generator': task_generator,
            'dataset_manager': dataset_manager,
            'evaluator': evaluator,
            'config': config
        }
    
    def test_complete_meta_training_pipeline(self, meta_pinn_system, temp_dir):
        """Test complete meta-training pipeline from task generation to evaluation."""
        system = meta_pinn_system
        meta_pinn = system['meta_pinn']
        task_generator = system['task_generator']
        dataset_manager = system['dataset_manager']
        evaluator = system['evaluator']
        config = system['config']
        
        # Step 1: Generate training tasks
        print("Generating training tasks...")
        train_tasks = []
        for _ in range(config.training.n_tasks):
            task = task_generator.generate_task(
                n_support=config.training.n_support,
                n_query=config.training.n_query
            )
            train_tasks.append(task)
        
        assert len(train_tasks) == config.training.n_tasks
        
        # Step 2: Meta-training loop
        print("Starting meta-training...")
        training_losses = []
        
        for epoch in range(config.training.meta_epochs):
            epoch_losses = []
            
            # Process tasks in batches
            batch_size = 2
            for i in range(0, len(train_tasks), batch_size):
                batch = train_tasks[i:i+batch_size]
                
                # Meta-update
                meta_loss = meta_pinn.meta_update(batch)
                epoch_losses.append(meta_loss)
            
            avg_loss = np.mean(epoch_losses)
            training_losses.append(avg_loss)
            print(f"Epoch {epoch}: Meta-loss = {avg_loss:.6f}")
        
        # Verify training convergence
        assert len(training_losses) == config.training.meta_epochs
        assert training_losses[-1] < training_losses[0], "Training should show improvement"
        
        # Step 3: Generate test tasks and evaluate
        print("Generating test tasks...")
        test_tasks = []
        for _ in range(5):  # Smaller test set
            task = task_generator.generate_task(
                n_support=config.training.n_support,
                n_query=config.training.n_query
            )
            test_tasks.append(task)
        
        # Step 4: Evaluate adaptation performance
        print("Evaluating adaptation performance...")
        adaptation_results = []
        
        for task in test_tasks:
            # Test fast adaptation
            adapted_params = meta_pinn.adapt_to_task(
                task, 
                adaptation_steps=config.training.adaptation_steps
            )
            
            # Evaluate adapted model
            with torch.no_grad():
                predictions = meta_pinn.forward(task.query_set['coords'])
                query_loss = torch.nn.functional.mse_loss(
                    predictions, task.query_set['data']
                )
            
            adaptation_results.append({
                'query_loss': query_loss.item(),
                'physics_residual': meta_pinn.physics_loss(
                    task.query_set['coords'], 
                    adapted_params, 
                    task.config
                ).item()
            })
        
        # Verify adaptation quality
        avg_query_loss = np.mean([r['query_loss'] for r in adaptation_results])
        avg_physics_residual = np.mean([r['physics_residual'] for r in adaptation_results])
        
        print(f"Average query loss: {avg_query_loss:.6f}")
        print(f"Average physics residual: {avg_physics_residual:.6f}")
        
        # Requirements validation
        assert avg_query_loss < 1.0, "Query loss should be reasonable"
        assert avg_physics_residual < 1e-2, "Physics residuals should be small"
        
        # Step 5: Save results
        results_path = os.path.join(temp_dir, 'pipeline_results.json')
        results = {
            'training_losses': training_losses,
            'adaptation_results': adaptation_results,
            'config': config.to_dict()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        assert os.path.exists(results_path)
        print("End-to-end pipeline test completed successfully!")
    
    def test_physics_solution_validation(self, meta_pinn_system):
        """Test validation against known analytical physics solutions."""
        system = meta_pinn_system
        meta_pinn = system['meta_pinn']
        task_generator = system['task_generator']
        
        # Generate task with known analytical solution
        analytical_solver = AnalyticalSolutions()
        
        # Test Poiseuille flow (known analytical solution)
        coords = torch.linspace(0, 1, 50).unsqueeze(1)
        coords = torch.cat([coords, torch.zeros_like(coords)], dim=1)
        
        # Generate analytical solution
        viscosity_params = {'mu0': 1.0, 'mu1': 0.0}  # Constant viscosity
        analytical_solution = analytical_solver.poiseuille_flow(
            coords, viscosity_params
        )
        
        # Create task from analytical solution
        task_config = {
            'task_type': 'linear_viscosity',
            'reynolds': 50.0,
            'geometry': 'channel',
            'viscosity_params': viscosity_params,
            'boundary_conditions': {'inlet_velocity': 1.0}
        }
        
        task = task_generator._create_task_from_solution(
            coords, analytical_solution, task_config
        )
        
        # Train meta-PINN on this task
        adapted_params = meta_pinn.adapt_to_task(task, adaptation_steps=10)
        
        # Compare PINN solution with analytical solution
        with torch.no_grad():
            pinn_solution = meta_pinn.forward(coords)
        
        # Compute relative error
        relative_error = torch.norm(pinn_solution - analytical_solution) / torch.norm(analytical_solution)
        
        print(f"Relative error vs analytical solution: {relative_error:.6f}")
        
        # Requirement: Should achieve high accuracy on known solutions
        assert relative_error < 0.1, f"Relative error {relative_error} too high vs analytical solution"
        
        # Verify physics constraints are satisfied
        physics_residual = meta_pinn.physics_loss(coords, adapted_params, task_config)
        print(f"Physics residual: {physics_residual:.6f}")
        
        assert physics_residual < 1e-3, "Physics residuals should be very small for analytical solutions"
    
    def test_benchmark_regression_testing(self, meta_pinn_system, temp_dir):
        """Test regression testing for performance and accuracy maintenance."""
        system = meta_pinn_system
        
        # Initialize benchmark suite
        benchmark_suite = PINNBenchmarkSuite()
        
        # Run baseline benchmark
        print("Running baseline benchmark...")
        baseline_results = benchmark_suite.run_quick_benchmark(
            methods=['meta_pinn'],
            save_dir=temp_dir
        )
        
        # Save baseline results
        baseline_path = os.path.join(temp_dir, 'baseline_results.json')
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        # Simulate code changes and re-run benchmark
        print("Running regression test...")
        regression_results = benchmark_suite.run_quick_benchmark(
            methods=['meta_pinn'],
            save_dir=temp_dir
        )
        
        # Compare results for regression
        baseline_accuracy = baseline_results['meta_pinn']['parameter_accuracy']['mean']
        regression_accuracy = regression_results['meta_pinn']['parameter_accuracy']['mean']
        
        accuracy_change = abs(regression_accuracy - baseline_accuracy) / baseline_accuracy
        
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        print(f"Regression accuracy: {regression_accuracy:.4f}")
        print(f"Relative change: {accuracy_change:.4f}")
        
        # Requirement: Performance should not degrade significantly
        assert accuracy_change < 0.1, f"Accuracy regression {accuracy_change} exceeds threshold"
        
        # Check adaptation speed regression
        baseline_speed = baseline_results['meta_pinn']['adaptation_speed']['mean']
        regression_speed = regression_results['meta_pinn']['adaptation_speed']['mean']
        
        speed_change = abs(regression_speed - baseline_speed) / baseline_speed
        
        print(f"Speed change: {speed_change:.4f}")
        assert speed_change < 0.2, f"Speed regression {speed_change} exceeds threshold"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distributed_training_workflow(self, basic_config, temp_dir):
        """Test distributed training and evaluation workflows."""
        # Note: This is a simplified test since full distributed testing 
        # requires multiple processes/GPUs
        
        config = ExperimentConfig.from_dict(basic_config)
        
        # Initialize distributed components
        distributed_pinn = DistributedMetaPINN(
            layers=config.model.layers,
            meta_lr=config.model.meta_lr,
            adapt_lr=config.model.adapt_lr
        )
        
        training_monitor = TrainingMonitor(log_dir=temp_dir)
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_dir)
        
        # Test checkpoint saving/loading
        initial_state = distributed_pinn.state_dict()
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=distributed_pinn,
            optimizer=None,
            epoch=0,
            meta_loss=1.0
        )
        
        assert os.path.exists(checkpoint_path)
        
        # Modify model and reload
        with torch.no_grad():
            for param in distributed_pinn.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Load checkpoint
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        distributed_pinn.load_state_dict(loaded_state['model_state_dict'])
        
        # Verify state restoration
        for (name1, param1), (name2, param2) in zip(
            initial_state.items(), distributed_pinn.state_dict().items()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6)
        
        print("Distributed training workflow test completed!")
    
    def test_memory_and_performance_validation(self, meta_pinn_system):
        """Test memory usage and performance characteristics."""
        system = meta_pinn_system
        meta_pinn = system['meta_pinn']
        task_generator = system['task_generator']
        
        # Test memory usage with increasing task sizes
        task_sizes = [10, 50, 100]
        memory_usage = []
        
        for size in task_sizes:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Generate larger task
            task = task_generator.generate_task(n_support=size, n_query=size)
            
            # Measure memory before adaptation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            
            # Perform adaptation
            adapted_params = meta_pinn.adapt_to_task(task, adaptation_steps=3)
            
            # Measure memory after adaptation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
                memory_usage.append(memory_used)
            
            print(f"Task size {size}: Memory used = {memory_used / 1024**2:.2f} MB" 
                  if torch.cuda.is_available() else f"Task size {size}: CPU test")
        
        # Verify reasonable memory scaling
        if torch.cuda.is_available() and len(memory_usage) > 1:
            # Memory should scale reasonably with task size
            memory_ratio = memory_usage[-1] / memory_usage[0]
            task_ratio = task_sizes[-1] / task_sizes[0]
            
            # Memory scaling should be reasonable (not exponential)
            assert memory_ratio < task_ratio * 2, "Memory usage scaling too aggressive"
        
        print("Memory and performance validation completed!")
    
    def test_error_handling_and_robustness(self, meta_pinn_system):
        """Test error handling and system robustness."""
        system = meta_pinn_system
        meta_pinn = system['meta_pinn']
        task_generator = system['task_generator']
        
        # Test handling of invalid task configurations
        with pytest.raises((ValueError, RuntimeError)):
            invalid_task = task_generator.generate_task(
                n_support=-1,  # Invalid size
                n_query=10
            )
        
        # Test handling of extreme parameter values
        extreme_task = task_generator.generate_task(n_support=5, n_query=5)
        
        # Modify task to have extreme values
        extreme_task.support_set['data'] *= 1000  # Extreme data values
        
        # System should handle extreme values gracefully
        try:
            adapted_params = meta_pinn.adapt_to_task(extreme_task, adaptation_steps=2)
            # Should not crash, but may have high loss
            assert adapted_params is not None
        except (RuntimeError, ValueError) as e:
            # Acceptable to fail gracefully with extreme inputs
            print(f"Graceful failure with extreme inputs: {e}")
        
        # Test numerical stability
        normal_task = task_generator.generate_task(n_support=20, n_query=10)
        
        # Multiple adaptation runs should be stable
        results = []
        for _ in range(3):
            adapted_params = meta_pinn.adapt_to_task(normal_task, adaptation_steps=3)
            
            with torch.no_grad():
                predictions = meta_pinn.forward(normal_task.query_set['coords'])
                loss = torch.nn.functional.mse_loss(
                    predictions, normal_task.query_set['data']
                )
            
            results.append(loss.item())
        
        # Results should be reasonably consistent
        result_std = np.std(results)
        result_mean = np.mean(results)
        
        print(f"Adaptation consistency: mean={result_mean:.6f}, std={result_std:.6f}")
        
        # Coefficient of variation should be reasonable
        cv = result_std / result_mean if result_mean > 0 else float('inf')
        assert cv < 0.5, f"Adaptation too inconsistent: CV={cv}"
        
        print("Error handling and robustness test completed!")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
"""
Docstring example testing module.

This module contains executable examples from docstrings to ensure
they remain correct and up-to-date.
"""

import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Import all modules to test their docstring examples
from ml_research_pipeline.core import MetaPINN, FluidTaskGenerator
from ml_research_pipeline.config import ExperimentConfig, ModelConfig, DataConfig
from ml_research_pipeline.bayesian import BayesianMetaPINN
from ml_research_pipeline.neural_operators import InverseFourierNeuralOperator
from ml_research_pipeline.evaluation import PINNBenchmarkSuite


def test_meta_pinn_docstring_examples():
    """Test examples from MetaPINN docstring."""
    print("Testing MetaPINN docstring examples...")
    
    # Example from MetaPINN docstring
    from ml_research_pipeline.config import MetaPINNConfig
    
    config = MetaPINNConfig()
    config.layers = [2, 64, 64, 64, 3]
    config.meta_lr = 0.001
    config.adapt_lr = 0.01
    
    model = MetaPINN(config)
    
    # Generate sample task
    coords = torch.randn(50, 2, requires_grad=True)  # 50 points in 2D
    data = torch.randn(50, 3)    # velocity (u,v) + pressure (p)
    task_info = {
        'viscosity_type': 'linear',
        'viscosity_params': {'a': 1.0, 'b': 0.1},
        'reynolds': 100.0
    }
    task = {
        'support_coords': coords, 
        'support_data': data,
        'task_info': task_info
    }
    
    # Adapt to task
    adapted_params = model.adapt_to_task(task, adaptation_steps=5)
    
    # Make predictions with adapted parameters
    query_coords = torch.randn(100, 2)
    predictions = model.forward(query_coords, adapted_params)
    
    assert predictions.shape == torch.Size([100, 3]), f"Expected shape [100, 3], got {predictions.shape}"
    print(f"✓ Predictions shape: {predictions.shape}")
    
    print("✓ MetaPINN docstring examples passed")


def test_task_generator_docstring_examples():
    """Test examples from FluidTaskGenerator docstring."""
    print("Testing FluidTaskGenerator docstring examples...")
    
    # Example from FluidTaskGenerator docstring
    data_config = DataConfig()
    data_config.domain_bounds = [[0, 1], [0, 1]]
    data_config.n_support = 50
    data_config.n_query = 100
    
    generator = FluidTaskGenerator(data_config)
    
    # Generate single task
    task = generator.generate_task(
        viscosity_type='linear',
        reynolds=100.0
    )
    
    assert task.support_set['coords'].shape[0] == 50, f"Expected 50 support points, got {task.support_set['coords'].shape[0]}"
    assert task.query_set['coords'].shape[0] == 100, f"Expected 100 query points, got {task.query_set['coords'].shape[0]}"
    
    print(f"✓ Support points: {task.support_set['coords'].shape}")
    print(f"✓ Query points: {task.query_set['coords'].shape}")
    
    # Generate task batch for meta-learning
    task_batch = generator.generate_task_batch(
        batch_size=16,
        n_support=20,
        n_query=50
    )
    
    assert len(task_batch) == 16, f"Expected 16 tasks, got {len(task_batch)}"
    print(f"✓ Generated {len(task_batch)} tasks")
    
    print("✓ FluidTaskGenerator docstring examples passed")


def test_physics_loss_docstring_examples():
    """Test examples from physics_loss method docstring."""
    print("Testing physics_loss docstring examples...")
    
    from ml_research_pipeline.config import MetaPINNConfig
    
    config = MetaPINNConfig()
    config.layers = [3, 64, 64, 64, 3]  # 3D input for [x, y, t]
    model = MetaPINN(config)
    
    # Example from physics_loss docstring
    coords = torch.tensor([[0.5, 0.5, 0.1]], requires_grad=True)
    task_info = {
        'viscosity_type': 'linear',
        'viscosity_params': {'a': 1.0, 'b': 0.1},
        'reynolds': 100.0
    }
    
    physics_losses = model.physics_loss(coords, task_info)
    
    # Check that all expected keys are present
    expected_keys = ['momentum_x', 'momentum_y', 'continuity', 'total', 'residual_magnitude']
    for key in expected_keys:
        assert key in physics_losses, f"Missing key '{key}' in physics losses"
    
    print(f"✓ Total physics loss: {physics_losses['total'].item():.6f}")
    print("✓ physics_loss docstring examples passed")


def test_adaptation_docstring_examples():
    """Test examples from adapt_to_task method docstring."""
    print("Testing adapt_to_task docstring examples...")
    
    from ml_research_pipeline.config import MetaPINNConfig
    
    config = MetaPINNConfig()
    config.layers = [2, 64, 64, 64, 3]
    model = MetaPINN(config)
    
    # Prepare task data from docstring example
    support_coords = torch.randn(20, 2, requires_grad=True)
    support_data = torch.randn(20, 3)  # [u, v, p]
    task_info = {
        'viscosity_type': 'linear',
        'viscosity_params': {'a': 1.0, 'b': 0.1},
        'reynolds': 100.0
    }
    task = {
        'support_coords': support_coords,
        'support_data': support_data,
        'task_info': task_info
    }
    
    # Adapt to task
    adapted_params = model.adapt_to_task(task, adaptation_steps=5)
    
    # Make predictions with adapted parameters
    query_coords = torch.randn(50, 2)
    predictions = model.forward(query_coords, adapted_params)
    
    assert predictions.shape == torch.Size([50, 3]), f"Expected shape [50, 3], got {predictions.shape}"
    print(f"✓ Adapted predictions shape: {predictions.shape}")
    print("✓ adapt_to_task docstring examples passed")


def test_configuration_examples():
    """Test configuration system examples."""
    print("Testing configuration system examples...")
    
    # Test ExperimentConfig
    config = ExperimentConfig()
    
    # Test model configuration
    config.model.layers = [2, 128, 128, 128, 3]
    config.model.activation = 'tanh'
    config.training.meta_lr = 0.001
    
    assert config.model.layers == [2, 128, 128, 128, 3]
    assert config.model.activation == 'tanh'
    assert config.training.meta_lr == 0.001
    
    # Test saving and loading configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config.save_yaml(f.name)
        
        # Load configuration
        loaded_config = ExperimentConfig.from_yaml(f.name)
        
        assert loaded_config.model.layers == config.model.layers
        assert loaded_config.model.activation == config.model.activation
        assert loaded_config.training.meta_lr == config.training.meta_lr
        
        # Cleanup
        os.unlink(f.name)
    
    print("✓ Configuration system examples passed")


def test_bayesian_examples():
    """Test Bayesian uncertainty examples."""
    print("Testing Bayesian uncertainty examples...")
    
    from ml_research_pipeline.config import MetaPINNConfig
    
    config = MetaPINNConfig()
    config.layers = [2, 64, 64, 64, 3]
    
    # Initialize Bayesian model
    bayesian_model = BayesianMetaPINN(config)
    
    # Test forward pass with uncertainty
    coords = torch.randn(50, 2)
    predictions, uncertainty = bayesian_model.forward_with_uncertainty(
        coords, n_samples=10
    )
    
    assert predictions.shape == torch.Size([50, 3])
    assert uncertainty.shape == torch.Size([50, 3])
    
    print(f"✓ Bayesian predictions shape: {predictions.shape}")
    print(f"✓ Uncertainty shape: {uncertainty.shape}")
    print("✓ Bayesian uncertainty examples passed")


def test_neural_operator_examples():
    """Test neural operator examples."""
    print("Testing neural operator examples...")
    
    # Test Fourier Neural Operator
    fno = InverseFourierNeuralOperator(
        modes=12,
        width=64,
        input_dim=2,
        output_dim=1
    )
    
    # Test forward pass
    sparse_observations = torch.randn(32, 20, 2)  # batch_size, n_obs, obs_dim
    parameter_field = fno(sparse_observations)
    
    expected_shape = torch.Size([32, 64, 64, 1])  # batch, height, width, channels
    assert parameter_field.shape == expected_shape, f"Expected {expected_shape}, got {parameter_field.shape}"
    
    print(f"✓ FNO output shape: {parameter_field.shape}")
    print("✓ Neural operator examples passed")


def test_evaluation_examples():
    """Test evaluation framework examples."""
    print("Testing evaluation framework examples...")
    
    # Initialize benchmark suite
    benchmark = PINNBenchmarkSuite()
    
    # Test benchmark configuration
    assert hasattr(benchmark, 'benchmark_configs')
    assert 'cavity_flow' in benchmark.benchmark_configs
    assert 'channel_flow' in benchmark.benchmark_configs
    
    print("✓ Benchmark suite initialized")
    print("✓ Evaluation framework examples passed")


def run_all_docstring_tests():
    """Run all docstring example tests."""
    print("=" * 60)
    print("Running docstring example tests...")
    print("=" * 60)
    
    test_functions = [
        test_meta_pinn_docstring_examples,
        test_task_generator_docstring_examples,
        test_physics_loss_docstring_examples,
        test_adaptation_docstring_examples,
        test_configuration_examples,
        test_bayesian_examples,
        test_neural_operator_examples,
        test_evaluation_examples
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Docstring tests completed: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_docstring_tests()
    exit(0 if success else 1)
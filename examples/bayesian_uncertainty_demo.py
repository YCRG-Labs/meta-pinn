"""
Demonstration of Bayesian uncertainty quantification in meta-learning PINNs.

This script shows how to use the BayesianMetaPINN for uncertainty quantification
and calibration in physics-informed neural networks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from ml_research_pipeline.bayesian import BayesianMetaPINN, UncertaintyCalibrator
from ml_research_pipeline.config.model_config import MetaPINNConfig


def create_synthetic_task():
    """Create a synthetic fluid dynamics task for demonstration."""
    # Generate coordinates
    x = torch.linspace(-1, 1, 20)
    y = torch.linspace(-1, 1, 20)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Synthetic velocity and pressure data (simplified)
    u = torch.sin(np.pi * coords[:, 0]) * torch.cos(np.pi * coords[:, 1])
    v = -torch.cos(np.pi * coords[:, 0]) * torch.sin(np.pi * coords[:, 1])
    p = torch.sin(2 * np.pi * coords[:, 0]) * torch.sin(2 * np.pi * coords[:, 1])
    
    data = torch.stack([u, v, p], dim=1)
    
    task_info = {
        'viscosity_type': 'constant',
        'viscosity_params': {'mu_0': 0.01}
    }
    
    return {
        'support_coords': coords[:200],  # First 200 points for support
        'support_data': data[:200],
        'query_coords': coords[200:],    # Remaining points for query
        'query_data': data[200:],
        'task_info': task_info
    }


def demonstrate_bayesian_meta_pinn():
    """Demonstrate BayesianMetaPINN functionality."""
    print("=== Bayesian Meta-Learning PINN Demonstration ===\n")
    
    # Create model configuration
    config = MetaPINNConfig()
    config.input_dim = 2  # x, y coordinates
    config.output_dim = 3  # u, v, p (velocity and pressure)
    config.hidden_dims = [32, 32, 32]
    config.activation = "tanh"
    config.meta_lr = 0.001
    config.adapt_lr = 0.01
    config.adaptation_steps = 5
    
    print(f"Model configuration:")
    print(f"  Input dimension: {config.input_dim}")
    print(f"  Output dimension: {config.output_dim}")
    print(f"  Hidden layers: {config.hidden_dims}")
    print(f"  Activation: {config.activation}")
    print()
    
    # Create Bayesian model
    model = BayesianMetaPINN(
        config=config,
        prior_std=1.0,
        kl_weight=1e-4,
        n_mc_samples=50
    )
    
    print(f"Created BayesianMetaPINN with {model.count_parameters()} parameters")
    print(f"KL weight: {model.kl_weight}")
    print(f"Monte Carlo samples: {model.n_mc_samples}")
    print()
    
    # Create synthetic task
    task = create_synthetic_task()
    print(f"Created synthetic task:")
    print(f"  Support set size: {task['support_coords'].shape[0]}")
    print(f"  Query set size: {task['query_coords'].shape[0]}")
    print(f"  Viscosity type: {task['task_info']['viscosity_type']}")
    print()
    
    # Demonstrate uncertainty quantification
    print("=== Uncertainty Quantification ===")
    
    model.eval()
    with torch.no_grad():
        # Forward pass with uncertainty
        mean_pred, uncertainty = model.forward_with_uncertainty(
            task['query_coords'], n_samples=30
        )
        
        print(f"Mean prediction shape: {mean_pred.shape}")
        print(f"Uncertainty shape: {uncertainty.shape}")
        print(f"Mean uncertainty: {torch.mean(uncertainty):.4f}")
        print(f"Max uncertainty: {torch.max(uncertainty):.4f}")
        print()
        
        # Decompose uncertainty
        uncertainty_dict = model.decompose_uncertainty(
            task['query_coords'], 
            task['task_info'],
            n_samples=30
        )
        
        print("Uncertainty decomposition:")
        for key, unc in uncertainty_dict.items():
            print(f"  {key}: mean={torch.mean(unc):.4f}, std={torch.std(unc):.4f}")
        print()
        
        # Confidence intervals
        confidence_dict = model.predict_with_confidence(
            task['query_coords'][:10],  # First 10 points
            task['task_info'],
            confidence_level=0.95
        )
        
        print("95% Confidence intervals (first 10 points):")
        mean = confidence_dict['mean']
        lower = confidence_dict['lower_bound']
        upper = confidence_dict['upper_bound']
        
        for i in range(5):  # Show first 5 points
            print(f"  Point {i}: [{lower[i, 0]:.3f}, {upper[i, 0]:.3f}] (mean: {mean[i, 0]:.3f})")
        print()
    
    # Demonstrate Bayesian adaptation
    print("=== Bayesian Adaptation ===")
    
    adapted_params, uncertainty_info = model.adapt_to_task_bayesian(
        task, adaptation_steps=3, use_uncertainty=True
    )
    
    print(f"Adapted {len(adapted_params)} parameter tensors")
    
    if uncertainty_info['uncertainty_reduction'] is not None:
        print("Uncertainty reduction after adaptation:")
        for key, reduction in uncertainty_info['uncertainty_reduction'].items():
            print(f"  {key}: {reduction:.4f}")
    print()
    
    # Demonstrate variational loss
    print("=== Variational Loss Computation ===")
    
    # Enable gradients for loss computation
    coords = task['support_coords'].clone().requires_grad_(True)
    predictions = model(coords)
    targets = task['support_data']
    
    loss_dict = model.variational_loss(
        predictions, targets, coords, task['task_info']
    )
    
    print("Loss components:")
    for key, loss in loss_dict.items():
        if key != 'physics_components':
            print(f"  {key}: {loss.item():.6f}")
    print()
    
    # Demonstrate weight sampling
    print("=== Weight Sampling ===")
    
    weight_samples = model.sample_weights(n_samples=3)
    print(f"Generated {len(weight_samples)} weight samples")
    
    # Show statistics of first weight tensor
    first_key = list(weight_samples[0].keys())[0]
    sample_weights = [sample[first_key] for sample in weight_samples]
    sample_tensor = torch.stack(sample_weights)
    
    print(f"Sample statistics for '{first_key}':")
    print(f"  Shape: {sample_tensor.shape}")
    print(f"  Mean across samples: {torch.mean(sample_tensor, dim=0).mean():.4f}")
    print(f"  Std across samples: {torch.std(sample_tensor, dim=0).mean():.4f}")
    print()


def demonstrate_uncertainty_calibration():
    """Demonstrate uncertainty calibration functionality."""
    print("=== Uncertainty Calibration Demonstration ===\n")
    
    # Generate synthetic uncertainty and error data
    n_samples = 200
    true_errors = torch.abs(torch.randn(n_samples)) * 0.5 + 0.1
    # Create imperfect uncertainty estimates (correlated but noisy)
    uncertainties = true_errors + 0.2 * torch.randn(n_samples)
    uncertainties = torch.clamp(uncertainties, min=0.01)
    
    print(f"Generated {n_samples} uncertainty-error pairs")
    print(f"Mean true error: {torch.mean(true_errors):.4f}")
    print(f"Mean predicted uncertainty: {torch.mean(uncertainties):.4f}")
    print()
    
    # Create and fit calibrator
    calibrator = UncertaintyCalibrator(n_bins=10, method='isotonic')
    calibrator.fit(uncertainties, true_errors)
    
    print("Fitted isotonic regression calibrator")
    print(f"Calibration error (ECE): {calibrator.calibration_error:.4f}")
    print()
    
    # Apply calibration
    calibrated_uncertainties = calibrator.calibrate(uncertainties)
    
    print("Applied calibration:")
    print(f"  Original uncertainty range: [{torch.min(uncertainties):.3f}, {torch.max(uncertainties):.3f}]")
    print(f"  Calibrated uncertainty range: [{torch.min(calibrated_uncertainties):.3f}, {torch.max(calibrated_uncertainties):.3f}]")
    print()
    
    # Evaluate calibration quality
    metrics_before = calibrator.evaluate_calibration(uncertainties, true_errors)
    metrics_after = calibrator.evaluate_calibration(calibrated_uncertainties, true_errors)
    
    print("Calibration quality:")
    print(f"  Before calibration - ECE: {metrics_before['ece']:.4f}, MCE: {metrics_before['mce']:.4f}")
    print(f"  After calibration - ECE: {metrics_after['ece']:.4f}, MCE: {metrics_after['mce']:.4f}")
    print()
    
    # Get calibration summary
    summary = calibrator.get_calibration_summary()
    print("Calibrator summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()


def demonstrate_physics_informed_calibration():
    """Demonstrate physics-informed uncertainty calibration."""
    print("=== Physics-Informed Calibration ===\n")
    
    # Create simple model and task
    config = MetaPINNConfig()
    config.input_dim = 2
    config.output_dim = 2
    config.hidden_dims = [16, 16]
    
    model = BayesianMetaPINN(config, n_mc_samples=20)
    calibrator = UncertaintyCalibrator()
    
    # Generate test coordinates
    coords = torch.randn(50, 2)
    task_info = {
        'viscosity_type': 'linear',
        'viscosity_params': {'mu_0': 1.0, 'alpha': 0.1, 'beta': 0.0}
    }
    
    print(f"Testing physics-informed calibration with {coords.shape[0]} points")
    print(f"Viscosity type: {task_info['viscosity_type']}")
    print()
    
    # Compute physics-informed uncertainty
    physics_uncertainty = calibrator.physics_informed_calibration(
        model, coords, task_info
    )
    
    print(f"Physics-informed uncertainty:")
    print(f"  Shape: {physics_uncertainty.shape}")
    print(f"  Mean: {torch.mean(physics_uncertainty):.4f}")
    print(f"  Std: {torch.std(physics_uncertainty):.4f}")
    print()
    
    # Get model uncertainty
    model.eval()
    with torch.no_grad():
        _, model_uncertainty = model.forward_with_uncertainty(coords)
        
    print(f"Model uncertainty:")
    print(f"  Shape: {model_uncertainty.shape}")
    print(f"  Mean: {torch.mean(model_uncertainty):.4f}")
    print(f"  Std: {torch.std(model_uncertainty):.4f}")
    print()
    
    # Create synthetic errors for calibration
    synthetic_errors = torch.abs(torch.randn(50, 2)) * 0.3 + 0.1
    
    # Fit calibrator
    calibrator.fit(model_uncertainty.flatten(), synthetic_errors.flatten())
    
    # Apply adaptive calibration
    adaptive_uncertainty = calibrator.adaptive_calibration(
        model_uncertainty.flatten(),
        synthetic_errors.flatten(),
        physics_uncertainty.flatten(),
        physics_weight=0.3
    )
    
    print(f"Adaptive calibration (30% physics weight):")
    print(f"  Shape: {adaptive_uncertainty.shape}")
    print(f"  Mean: {torch.mean(adaptive_uncertainty):.4f}")
    print(f"  Std: {torch.std(adaptive_uncertainty):.4f}")
    print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        demonstrate_bayesian_meta_pinn()
        demonstrate_uncertainty_calibration()
        demonstrate_physics_informed_calibration()
        
        print("=== Demonstration Complete ===")
        print("All Bayesian uncertainty quantification features demonstrated successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from config import cfg
from src.generate_data import generate_collocation_points, generate_boundary_points, generate_sparse_data_points
from src.model.model import PINN
from src.model.train_model import train_model
from src.model.evaluate_model import evaluate_model, analyze_flow_features

def main(args):
    """
    Main function to run the PINN viscosity inference project
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("PINN Viscosity Inference Project")
    print("="*70)
    
    # Update configuration based on command line arguments
    if args.navier_stokes:
        cfg.update_for_navier_stokes()
    
    if args.unsteady:
        cfg.update_for_unsteady_flow()
    
    if args.advanced:
        cfg.enable_all_advanced_features()
    
    # Individual feature toggles
    if args.fourier:
        cfg.USE_FOURIER_FEATURES = True
    
    if args.adaptive_weights:
        cfg.USE_ADAPTIVE_WEIGHTS = True
    
    if args.adaptive_sampling:
        cfg.USE_ADAPTIVE_SAMPLING = True
    
    if args.curriculum:
        cfg.USE_CURRICULUM_LEARNING = True
    
    if args.reinit:
        cfg.USE_REINIT_STRATEGY = True
    
    # Update Reynolds number if specified
    if args.reynolds is not None:
        cfg.REYNOLDS_NUMBER = args.reynolds
    
    # Update epochs if specified
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    
    # Print configuration
    cfg.print_config()
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Generate data
    print("\nGenerating data...")
    collocation_points = generate_collocation_points()
    boundary_points = generate_boundary_points()
    sparse_data = generate_sparse_data_points()
    
    # Create and train model
    if not args.evaluate_only:
        print("\nCreating and training model...")
        model = PINN(cfg)
        model, history = train_model(model, collocation_points, boundary_points, sparse_data, cfg)
    else:
        # Load existing model for evaluation
        print("\nLoading existing model for evaluation...")
        model_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_SAVE_FILENAME)
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Please train a model first.")
            return
        model = PINN.load(model_path, cfg)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, cfg)
    
    # Analyze advanced flow features
    if args.analyze_flow:
        print("\nAnalyzing advanced flow features...")
        analyze_flow_features(model, cfg)
    
    print("\n" + "="*70)
    print("PINN Viscosity Inference Project Complete")
    print("="*70)
    
    # Print final results
    print(f"\nInferred viscosity parameter a: {model.get_inferred_viscosity_param():.6f}")
    print(f"True viscosity parameter a: {cfg.A_TRUE:.6f}")
    print(f"Relative error: {abs(model.get_inferred_viscosity_param() - cfg.A_TRUE) / cfg.A_TRUE * 100:.2f}%")
    
    # Print PDE residuals
    print("\nPDE Residuals (Mean Absolute):")
    for key, value in metrics['pde_residuals'].items():
        print(f"  {key}: {value:.6e}")
    
    print("\nResults saved to:", cfg.OUTPUT_DIR)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PINN Viscosity Inference Project")
    
    # Physics options
    parser.add_argument("--navier-stokes", action="store_true", help="Use Navier-Stokes equations instead of Stokes")
    parser.add_argument("--unsteady", action="store_true", help="Use unsteady flow instead of steady")
    parser.add_argument("--reynolds", type=float, help="Reynolds number for the flow")
    
    # Advanced PINN features
    parser.add_argument("--advanced", action="store_true", help="Enable all advanced PINN features")
    parser.add_argument("--fourier", action="store_true", help="Use Fourier feature embeddings")
    parser.add_argument("--adaptive-weights", action="store_true", help="Use adaptive loss weighting")
    parser.add_argument("--adaptive-sampling", action="store_true", help="Use adaptive collocation sampling")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--reinit", action="store_true", help="Use re-initialization strategy")
    
    # Training options
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    
    # Evaluation options
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and only evaluate existing model")
    parser.add_argument("--analyze-flow", action="store_true", help="Analyze advanced flow features")
    
    args = parser.parse_args()
    
    # Run main function
    main(args)

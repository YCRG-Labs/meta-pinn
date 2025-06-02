import torch
import torch.nn as nn
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Config:
    """
    Configuration class for the PINN viscosity inference project
    with advanced features for Navier-Stokes equations and enhanced PINN architectures
    """
    def __init__(self):
        # Device configuration
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Output directory
        self.OUTPUT_DIR = 'backend/results'
        
        # Domain configuration
        self.X_MIN = 0.0
        self.X_MAX = 2.0
        self.Y_MIN = 0.0
        self.Y_MAX = 1.0
        
        # Physics parameters
        self.NU_BASE_TRUE = 0.01  # Base viscosity
        self.A_TRUE = 0.05        # True parameter for viscosity variation: nu(y) = nu_base + a*y
        self.U_MAX_INLET = 1.0    # Maximum inlet velocity
        
        # Reynolds number (for Navier-Stokes)
        self.REYNOLDS_NUMBER = 100  # Re = U*L/nu
        
        # Unsteady flow configuration
        self.UNSTEADY_FLOW = False  # Set to True for unsteady flow
        self.T_MIN = 0.0            # Start time for unsteady flow
        self.T_MAX = 1.0            # End time for unsteady flow
        
        # Data generation configuration
        self.DATA_SOURCE = 'placeholder'  # 'placeholder', 'load_from_file', or 'generate_with_fenicsx'
        self.DATA_FILE = 'data/cfd_data.npz'  # File to load data from if DATA_SOURCE = 'load_from_file'
        self.DATA_NOISE_LEVEL = 0.0  # Noise level for synthetic data (0.0 = no noise)
        
        # Sparse data configuration
        self.N_DATA_SPARSE = 100  # Number of sparse data points
        self.SPARSE_DATA_SAMPLING_STRATEGY = 'random'  # 'random', 'grid', 'centerline', 'wall_proximal', 'time_slices'
        self.N_TIME_SPARSE = 5    # Number of time slices for sparse data (if UNSTEADY_FLOW)
        
        # Boundary and collocation points
        self.N_BOUNDARY = 50      # Number of points per boundary
        self.N_COLLOCATION = 5000  # Number of collocation points for PDE residual
        self.N_TIME_BOUNDARY = 5  # Number of time slices for boundary conditions (if UNSTEADY_FLOW)
        
        # PINN architecture
        self.PINN_LAYERS = [2, 64, 128, 128, 64, 3]  # [input_dim, hidden_layers..., output_dim]
        self.PINN_ACTIVATION = nn.Tanh()
        
        # Advanced PINN features
        self.USE_FOURIER_FEATURES = False  # Use Fourier feature embeddings
        self.FOURIER_SCALE = 10.0          # Scale for Fourier features
        
        self.USE_ADAPTIVE_WEIGHTS = False  # Use adaptive loss weighting
        
        self.USE_ADAPTIVE_SAMPLING = False  # Use adaptive collocation sampling
        self.ADAPTIVE_SAMPLING_FREQUENCY = 100  # Update collocation points every N epochs
        
        self.USE_CURRICULUM_LEARNING = False  # Use curriculum learning
        
        self.USE_REINIT_STRATEGY = False  # Use re-initialization strategy
        self.REINIT_PATIENCE = 50         # Patience for re-initialization
        self.REINIT_THRESHOLD = 0.001     # Threshold for re-initialization
        
        # Training configuration
        self.LEARNING_RATE = 0.001
        self.EPOCHS = 1000
        self.OPTIMIZER_TYPE = 'Adam'  # 'Adam' or 'LBFGS'
        self.SCHEDULER_STEP_SIZE = 1000
        self.SCHEDULER_GAMMA = 0.5
        self.LOG_FREQUENCY = 100
        
        # Loss weights
        self.WEIGHT_BC = 10.0
        self.WEIGHT_DATA_U = 10.0
        self.WEIGHT_DATA_V = 10.0
        self.WEIGHT_DATA_P = 1.0
        
        # Visualization
        self.SHOW_PLOTS_INTERACTIVE = False
        
        # Model saving
        self.MODEL_SAVE_FILENAME = 'trained_model.pth'
        
        # Update input dimension for unsteady flow
        if self.UNSTEADY_FLOW:
            self.PINN_LAYERS[0] = 3  # Input: (x, y, t)
    
    def update_for_navier_stokes(self):
        """Update configuration for Navier-Stokes equations"""
        # Increase network capacity for more complex physics
        self.PINN_LAYERS = [self.PINN_LAYERS[0], 128, 256, 512, 512, 256, 128, 3]
        
        # Adjust training parameters
        self.EPOCHS = 5000
        self.LEARNING_RATE = 0.0001
        
        # Enable advanced features
        self.USE_FOURIER_FEATURES = True
        self.USE_ADAPTIVE_WEIGHTS = True
        self.USE_CURRICULUM_LEARNING = True
        self.USE_ADAPTIVE_SAMPLING = True
        
        # Increase collocation points for better PDE resolution
        self.N_COLLOCATION = 20000
        
        # Adjust loss weights for better balance
        self.WEIGHT_BC = 20.0
        self.WEIGHT_DATA_U = 20.0
        self.WEIGHT_DATA_V = 20.0
        self.WEIGHT_DATA_P = 5.0
        
        # Use LBFGS optimizer for better convergence
        self.OPTIMIZER_TYPE = 'LBFGS'
        
        print("Configuration updated for Navier-Stokes equations")
    
    def update_for_unsteady_flow(self):
        """Update configuration for unsteady flow"""
        self.UNSTEADY_FLOW = True
        self.PINN_LAYERS[0] = 3  # Input: (x, y, t)
        
        # Increase network capacity for time-dependent flow
        self.PINN_LAYERS = [3, 64, 128, 256, 256, 128, 64, 3]
        
        # Adjust training parameters
        self.EPOCHS = 30000
        self.LEARNING_RATE = 0.0003
        
        # Enable adaptive sampling for better coverage of spacetime domain
        self.USE_ADAPTIVE_SAMPLING = True
        
        print("Configuration updated for unsteady flow")
    
    def enable_all_advanced_features(self):
        """Enable all advanced PINN features"""
        self.USE_FOURIER_FEATURES = True
        self.USE_ADAPTIVE_WEIGHTS = True
        self.USE_ADAPTIVE_SAMPLING = True
        self.USE_CURRICULUM_LEARNING = True
        self.USE_REINIT_STRATEGY = False  # Keep re-initialization disabled
        
        # Increase network capacity with deeper architecture
        self.PINN_LAYERS = [self.PINN_LAYERS[0], 128, 256, 512, 512, 256, 128, 64, 3]
        
        # Adjust training parameters for better convergence
        self.EPOCHS = 5000
        self.LEARNING_RATE = 0.0001
        
        # Increase collocation points for better PDE resolution
        self.N_COLLOCATION = 10000
        
        # Adjust loss weights for better balance
        self.WEIGHT_BC = 50.0
        self.WEIGHT_DATA_U = 50.0
        self.WEIGHT_DATA_V = 50.0
        self.WEIGHT_DATA_P = 10.0
        
        # Use LBFGS optimizer for better convergence
        self.OPTIMIZER_TYPE = 'LBFGS'
        
        # Adjust Fourier feature scale
        self.FOURIER_SCALE = 5.0
        
        # Increase adaptive sampling frequency
        self.ADAPTIVE_SAMPLING_FREQUENCY = 50
        
        print("All advanced PINN features enabled with optimized parameters")
    
    def print_config(self):
        """Print the current configuration"""
        print("\n" + "="*70)
        print("PINN Configuration")
        print("="*70)
        
        print("\nPhysics:")
        print(f"  Equation type: {'Navier-Stokes' if self.REYNOLDS_NUMBER > 0 else 'Stokes'}")
        print(f"  Flow type: {'Unsteady' if self.UNSTEADY_FLOW else 'Steady'}")
        print(f"  Reynolds number: {self.REYNOLDS_NUMBER}")
        print(f"  Base viscosity: {self.NU_BASE_TRUE}")
        print(f"  True viscosity parameter a: {self.A_TRUE}")
        
        print("\nDomain:")
        print(f"  Spatial: x ∈ [{self.X_MIN}, {self.X_MAX}], y ∈ [{self.Y_MIN}, {self.Y_MAX}]")
        if self.UNSTEADY_FLOW:
            print(f"  Temporal: t ∈ [{self.T_MIN}, {self.T_MAX}]")
        
        print("\nData:")
        print(f"  Source: {self.DATA_SOURCE}")
        print(f"  Sparse data points: {self.N_DATA_SPARSE}")
        print(f"  Sampling strategy: {self.SPARSE_DATA_SAMPLING_STRATEGY}")
        print(f"  Noise level: {self.DATA_NOISE_LEVEL}")
        print(f"  Boundary points per edge: {self.N_BOUNDARY}")
        print(f"  Collocation points: {self.N_COLLOCATION}")
        
        print("\nPINN Architecture:")
        print(f"  Layers: {self.PINN_LAYERS}")
        print(f"  Activation: {self.PINN_ACTIVATION.__class__.__name__}")
        print(f"  Device: {self.DEVICE}")
        
        print("\nAdvanced Features:")
        print(f"  Fourier features: {self.USE_FOURIER_FEATURES}")
        print(f"  Adaptive weights: {self.USE_ADAPTIVE_WEIGHTS}")
        print(f"  Adaptive sampling: {self.USE_ADAPTIVE_SAMPLING}")
        print(f"  Curriculum learning: {self.USE_CURRICULUM_LEARNING}")
        print(f"  Re-initialization: {self.USE_REINIT_STRATEGY}")
        
        print("\nTraining:")
        print(f"  Optimizer: {self.OPTIMIZER_TYPE}")
        print(f"  Learning rate: {self.LEARNING_RATE}")
        print(f"  Epochs: {self.EPOCHS}")
        print(f"  Loss weights - BC: {self.WEIGHT_BC}, Data u: {self.WEIGHT_DATA_U}, v: {self.WEIGHT_DATA_V}, p: {self.WEIGHT_DATA_P}")
        
        print("\nOutput:")
        print(f"  Directory: {self.OUTPUT_DIR}")
        print(f"  Model filename: {self.MODEL_SAVE_FILENAME}")
        print(f"  Interactive plots: {self.SHOW_PLOTS_INTERACTIVE}")
        
        print("="*70 + "\n")

# Create global configuration instance
cfg = Config()

if __name__ == "__main__":
    # Print default configuration
    cfg.print_config()
    
    # Example of updating for Navier-Stokes
    cfg.update_for_navier_stokes()
    cfg.print_config()
    
    # Example of enabling all advanced features
    cfg.enable_all_advanced_features()
    cfg.print_config()

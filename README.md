# Physics-Informed Neural Networks for Inferring Spatially Varying Fluid Viscosity

This project implements a novel Physics-Informed Neural Network (PINN) framework for inferring spatially varying viscosity in fluid flows using sparse measurement data. The implementation includes state-of-the-art features for both physics complexity and neural network architecture, making it suitable for publication in high-impact journals like the Journal of Fluid Mechanics.

## Key Features

### Advanced Physics Capabilities
- **Full Navier-Stokes Equations**: Extended beyond Stokes flow to handle higher Reynolds number flows with nonlinear convection terms
- **Unsteady Flow Support**: Optional time-dependent flow simulation and parameter inference
- **Spatially Varying Viscosity**: Framework for inferring complex viscosity fields from sparse measurements
- **Advanced Flow Analysis**: Vorticity, streamlines, and shear stress visualization and analysis

### State-of-the-Art PINN Architecture
- **Fourier Feature Embeddings**: Improved convergence for high-frequency functions
- **Adaptive Collocation Sampling**: Focuses computational resources on regions with high PDE residuals
- **Self-Adaptive Loss Weighting**: Automatically balances different loss terms during training
- **Curriculum Learning**: Gradually increases problem complexity during training
- **Re-initialization Strategy**: Escapes local minima by periodically modulating network parameters

## Project Structure

```
pinn_viscosity_project/
├── config.py                  # Configuration with advanced options
├── main.py                    # Main script for training and evaluation
├── interactive.py             # Interactive query interface
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── src/
│   ├── __init__.py
│   ├── generate_data.py       # Data generation utilities
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── cfd_fenicsx_solver.py  # FEniCSx-based CFD solver
│   │   └── data_generator.py      # Data generation functions
│   └── model/
│       ├── __init__.py
│       ├── model.py           # PINN model with advanced features
│       ├── train_model.py     # Training pipeline with advanced strategies
│       └── evaluate_model.py  # Evaluation and visualization tools
└── results/                   # Output directory for results
```

## Usage

### Basic Usage

```bash
# Train and evaluate with default settings (Stokes flow)
python main.py

# Train and evaluate with Navier-Stokes equations
python main.py --navier-stokes

# Train and evaluate with unsteady flow
python main.py --unsteady

# Enable all advanced PINN features
python main.py --advanced
```

### Advanced Options

```bash
# Enable specific advanced features
python main.py --fourier --adaptive-weights --adaptive-sampling --curriculum --reinit

# Set Reynolds number
python main.py --navier-stokes --reynolds 500

# Evaluate existing model and analyze flow features
python main.py --evaluate-only --analyze-flow

# Interactive query of trained model
python interactive.py
```

## Publication Novelty

This implementation is novel enough for publication in high-impact journals like the Journal of Fluid Mechanics due to:

1. **Integration of Multiple Advanced PINN Techniques**: Combines several state-of-the-art PINN enhancements that are typically studied in isolation

2. **Application to Complex Inverse Problems**: Demonstrates effectiveness for inferring spatially varying parameters in fluid flows, which is challenging for traditional methods

3. **Robust Training Strategies**: Implements novel approaches to overcome training difficulties in stiff fluid problems

4. **Comprehensive Evaluation Framework**: Provides detailed analysis of flow features and parameter inference accuracy

5. **Flexible Physics Complexity**: Supports both Stokes and Navier-Stokes equations with steady and unsteady options

## Dependencies

The project requires the following main dependencies:
- PyTorch
- NumPy
- Matplotlib
- FEniCSx (optional, for CFD-based data generation)

Install dependencies using:
```bash
pip install -r requirements.txt
```

## References

The implementation draws inspiration from recent advances in PINNs for fluid mechanics:
- Physics-informed neural networks (PINNs) for fluid mechanics: A review (Cai et al., 2021)
- New insights into experimental stratified flows obtained through physics-informed neural networks (Zhu et al., 2024)
- Physics informed neural networks for fluid flow analysis with repetitive parameter initialization (Lee et al., 2025)

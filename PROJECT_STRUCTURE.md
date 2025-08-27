# ML Research Pipeline - Project Structure

This document describes the reorganized project structure for the meta-learning PINN research pipeline.

## Directory Structure

```
ml-research-pipeline/
├── ml_research_pipeline/           # Main Python package
│   ├── __init__.py                # Package initialization
│   ├── core/                      # Core meta-learning components
│   │   ├── __init__.py
│   │   ├── meta_pinn.py          # MetaPINN implementation
│   │   ├── task_generator.py     # Task generation system
│   │   ├── physics_models.py     # Physics-informed models
│   │   └── neural_operators.py   # Neural operator implementations
│   ├── bayesian/                  # Bayesian uncertainty quantification
│   │   ├── __init__.py
│   │   ├── bayesian_meta_pinn.py
│   │   ├── uncertainty_calibration.py
│   │   └── variational_layers.py
│   ├── physics_discovery/         # Physics discovery components
│   │   ├── __init__.py
│   │   ├── causal_discovery.py
│   │   ├── symbolic_regression.py
│   │   └── physics_validator.py
│   ├── evaluation/                # Evaluation framework
│   │   ├── __init__.py
│   │   ├── benchmark_suite.py
│   │   ├── metrics.py
│   │   └── statistical_analysis.py
│   ├── config/                    # Configuration management
│   │   ├── __init__.py
│   │   ├── base_config.py
│   │   ├── experiment_config.py
│   │   ├── model_config.py
│   │   ├── training_config.py
│   │   └── data_config.py
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── logging_utils.py
│       ├── random_utils.py
│       ├── io_utils.py
│       └── distributed_utils.py
├── experiments/                   # Experiment configurations and runners
│   ├── __init__.py
│   ├── config.py                 # Experiment configuration
│   ├── runner.py                 # Experiment runner
│   ├── meta_learning/            # Meta-learning experiments
│   │   └── __init__.py
│   └── benchmarks/               # Benchmark experiments
│       └── __init__.py
├── theory/                       # Theoretical analysis
│   ├── __init__.py
│   └── proofs/                   # Mathematical proofs
│       └── __init__.py
├── papers/                       # Publication materials
│   ├── __init__.py
│   ├── figures/                  # Figure generation
│   │   └── __init__.py
│   └── tables/                   # Table generation
│       └── __init__.py
├── configs/                      # Configuration files
│   ├── experiment_default.yaml
│   ├── model_default.yaml
│   ├── training_default.yaml
│   └── data_default.yaml
├── pinn_viscosity/              # Original PINN implementation (preserved)
│   ├── __init__.py
│   ├── generate_data.py
│   ├── data_generation/
│   └── model/
├── results/                     # Experimental results (preserved)
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
├── PROJECT_STRUCTURE.md         # This file
└── README.md                    # Main project README
```

## Key Components

### 1. Main Package (`ml_research_pipeline/`)

The main Python package containing all core functionality:

- **core/**: Meta-learning algorithms, task generation, and physics models
- **bayesian/**: Bayesian uncertainty quantification components
- **physics_discovery/**: Automated physics discovery and symbolic regression
- **evaluation/**: Comprehensive evaluation framework and benchmarking
- **config/**: Configuration management system with YAML/JSON support
- **utils/**: Utility functions for logging, random seeds, I/O, and distributed training

### 2. Experiments (`experiments/`)

Research-focused experiment management:

- **config.py**: Experiment configuration and management utilities
- **runner.py**: Main experiment runner with distributed training support
- **meta_learning/**: Meta-learning specific experiments
- **benchmarks/**: Benchmark experiment configurations

### 3. Theory (`theory/`)

Theoretical analysis and mathematical foundations:

- **proofs/**: Formal mathematical proofs and analysis
- Sample complexity analysis
- Convergence rate analysis

### 4. Papers (`papers/`)

Publication-ready output generation:

- **figures/**: Publication-quality plot generation
- **tables/**: LaTeX table generation
- Automated report generation

### 5. Configuration System

Hierarchical configuration management with:

- **Base configurations**: Extensible configuration classes with serialization
- **Experiment configs**: Complete experiment specifications
- **Model configs**: Architecture and hyperparameter definitions
- **Training configs**: Training procedure specifications
- **Data configs**: Task generation and data handling

## Configuration Management

The configuration system supports:

- **YAML/JSON serialization**: Easy human-readable configuration files
- **Hierarchical merging**: Combine base configs with experiment-specific overrides
- **Type validation**: Dataclass-based configuration with type checking
- **Environment integration**: Support for environment variable overrides
- **Reproducibility**: Deterministic configuration for reproducible experiments

## Integration with Existing Code

The new structure preserves the existing `pinn_viscosity/` package while extending it with:

- **Backward compatibility**: Existing single-task PINN functionality preserved
- **Code reuse**: FEniCSx solver integration maintained
- **Gradual migration**: Existing components can be gradually integrated

## Usage Examples

### Running Experiments

```bash
# Run experiment with default configuration
python experiments/runner.py --config configs/experiment_default.yaml

# Run with custom output directory
python experiments/runner.py --config configs/experiment_default.yaml --output-dir experiments/outputs/custom
```

### Configuration Management

```python
from ml_research_pipeline.config import ExperimentConfig, ModelConfig

# Load configuration
config = ExperimentConfig.from_yaml("configs/experiment_default.yaml")

# Modify configuration
config = config.update(seed=123, epochs=2000)

# Save modified configuration
config.to_yaml("configs/my_experiment.yaml")
```

### Package Installation

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Next Steps

This structure provides the foundation for implementing the meta-learning PINN research pipeline. The subsequent tasks will populate these modules with:

1. **Core meta-learning algorithms** (MetaPINN, MAML implementation)
2. **Task generation system** (Fluid dynamics task creation)
3. **Evaluation framework** (Comprehensive benchmarking)
4. **Advanced features** (Bayesian uncertainty, neural operators, physics discovery)
5. **Publication tools** (Automated figure and table generation)

The modular design ensures that each component can be developed and tested independently while maintaining clear interfaces and dependencies.
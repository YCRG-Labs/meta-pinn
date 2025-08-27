# Meta-Learning Physics-Informed Neural Networks for Fluid Dynamics

A comprehensive research pipeline for meta-learning physics-informed neural networks (PINNs) applied to fluid dynamics problems, with focus on viscosity inference and physics discovery.

## 🚀 Quick Start

### Installation
```bash
# Install the package
pip install -e .[all]

# Optional: Install FEniCSx for high-fidelity solvers (Linux/WSL only)
chmod +x install_fenicsx.sh
./install_fenicsx.sh
```

### Run Complete Pipeline
```bash
# Execute entire research pipeline
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

### Quick Demo
```bash
# Test core functionality
python -m pytest tests/test_meta_pinn.py -v

# Generate publication-ready results
python examples/comprehensive_report_demo.py
python examples/publication_demo.py
```

## 📋 Features

### Core Components
- **Meta-Learning PINNs**: MAML-based meta-learning for physics-informed neural networks
- **Task Generation**: Automated fluid dynamics task creation with analytical solutions
- **Physics Discovery**: Automated discovery of physical relationships using symbolic regression
- **Bayesian Uncertainty**: Uncertainty quantification with calibration
- **Neural Operators**: Fourier Neural Operators and DeepONet implementations

### Advanced Capabilities
- **Reproducibility**: Comprehensive experiment management and versioning
- **Performance Optimization**: Memory monitoring, profiling, and regression testing
- **Publication Tools**: Automated generation of plots, tables, and reports
- **Theoretical Analysis**: Sample complexity and convergence rate analysis
- **High-Fidelity Solvers**: FEniCSx integration for ground truth solutions

### Evaluation Framework
- **Comprehensive Benchmarking**: Multi-method comparison across diverse tasks
- **Statistical Analysis**: Rigorous statistical testing and effect size analysis
- **Performance Metrics**: Accuracy, sample efficiency, adaptation speed
- **Cross-Validation**: Robust evaluation with confidence intervals

## 🏗️ Architecture

```
ml_research_pipeline/
├── core/                    # Meta-learning algorithms and physics models
├── bayesian/               # Bayesian uncertainty quantification
├── neural_operators/       # Neural operator implementations
├── physics_discovery/      # Automated physics discovery
├── evaluation/            # Comprehensive evaluation framework
├── papers/               # Publication-ready output generation
├── config/               # Configuration management
└── utils/                # Utilities and experiment management
```

## 📊 Results

The pipeline generates:
- **Publication-ready figures and tables**
- **Comprehensive research reports** (Markdown, LaTeX, HTML)
- **Performance benchmarks and optimization results**
- **Theoretical analysis documents**
- **Physics discovery results with natural language descriptions**
- **Bayesian uncertainty quantification**
- **Reproducibility validation reports**

## 🧪 Examples

### Basic Usage
```python
from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.core.task_generator import FluidTaskGenerator
from ml_research_pipeline.config.model_config import MetaPINNConfig

# Create model
config = MetaPINNConfig(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])
model = MetaPINN(config)

# Generate tasks
generator = FluidTaskGenerator()
task = generator.generate_task()

# Train with meta-learning
# ... (see examples/ directory for complete workflows)
```

### Advanced Features
```python
# Bayesian uncertainty quantification
from ml_research_pipeline.bayesian.bayesian_meta_pinn import BayesianMetaPINN

# Physics discovery
from ml_research_pipeline.physics_discovery.integrated_discovery import IntegratedPhysicsDiscovery

# Publication tools
from ml_research_pipeline.papers.report_generator import ReportGenerator
```

## 📚 Documentation

- **[Quick Start Commands](QUICK_START_COMMANDS.md)**: Essential commands and usage
- **[Project Structure](PROJECT_STRUCTURE.md)**: Detailed architecture overview
- **[Examples](examples/)**: Complete demonstration scripts
- **[Tests](tests/)**: Comprehensive test suite
- **[Docs](docs/)**: Full documentation (run `make html` in docs/)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_meta_pinn.py -v
python -m pytest tests/test_bayesian_meta_pinn.py -v
python -m pytest tests/test_physics_discovery.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research

This codebase supports research in:
- Meta-learning for scientific computing
- Physics-informed neural networks
- Automated physics discovery
- Bayesian uncertainty quantification in PINNs
- Neural operators for fluid dynamics

## 📞 Contact

For questions and support, please open an issue on GitHub.

---

**Ready for research publication and further experimentation!** 🚀
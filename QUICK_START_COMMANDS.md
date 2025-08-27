# ML Research Pipeline - Quick Start Commands

## 🚀 Essential Commands to Run the Complete Codebase

### 1. **One-Command Complete Execution**
```bash
# Run everything (recommended)
chmod +x run_complete_pipeline.sh
./run_complete_pipeline.sh
```

### 2. **Step-by-Step Execution**

#### **Setup & Installation**
```bash
# Install Python package
pip install -e .

# Install all dependencies
pip install -e .[all]

# Install FEniCSx (optional, for high-fidelity solvers)
chmod +x install_fenicsx.sh
./install_fenicsx.sh
```

#### **Core Testing**
```bash
# Test core functionality
python -m pytest tests/test_meta_pinn.py -v
python -m pytest tests/test_physics_loss.py -v
python -m pytest tests/test_task_generator.py -v

# Run all tests
python -m pytest tests/ -v --tb=short
```

#### **Key Demonstrations**
```bash
# Comprehensive report generation
python examples/comprehensive_report_demo.py

# Publication-ready outputs
python examples/publication_demo.py

# Bayesian uncertainty quantification
python examples/bayesian_uncertainty_demo.py

# Theoretical analysis
python examples/theoretical_analysis_demo.py

# Performance benchmarking
python examples/performance_benchmarking_demo.py

# Physics discovery
python examples/enhanced_integrated_physics_discovery_demo.py

# Large-scale dataset generation
python examples/large_scale_dataset_demo.py

# FEniCSx integration (if installed)
python examples/fenicsx_integration_demo.py
```

#### **Experiment Runner**
```bash
# Run experiments with default config
python experiments/runner.py --config configs/data_default.yaml

# Run with custom settings
python experiments/runner.py --config configs/data_default.yaml --output-dir results/my_experiment
```

#### **Documentation**
```bash
# Generate documentation
cd docs
make html
```

### 3. **Quick Validation Commands**

#### **Test Installation**
```bash
# Test core imports
python -c "import ml_research_pipeline; print('✅ Package imported successfully')"

# Test FEniCSx (if installed)
python -c "import dolfinx; print('✅ FEniCSx available')"

# Test key components
python -c "
from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.core.task_generator import FluidTaskGenerator
from ml_research_pipeline.evaluation.metrics import EvaluationMetrics
print('✅ All core components available')
"
```

#### **Quick Demo**
```bash
# 5-minute demo of key features
python -c "
print('🚀 Quick ML Research Pipeline Demo')
print('=' * 40)

# Test MetaPINN
from ml_research_pipeline.core.meta_pinn import MetaPINN
from ml_research_pipeline.config.model_config import MetaPINNConfig
config = MetaPINNConfig(input_dim=2, output_dim=1, hidden_dims=[32, 32])
model = MetaPINN(config)
print('✅ MetaPINN created')

# Test Task Generator
from ml_research_pipeline.core.task_generator import FluidTaskGenerator
generator = FluidTaskGenerator()
task = generator.generate_task()
print('✅ Fluid task generated')

# Test Evaluation
from ml_research_pipeline.evaluation.metrics import EvaluationMetrics
evaluator = EvaluationMetrics()
print('✅ Evaluation system ready')

print('🎉 Demo completed successfully!')
"
```

### 4. **Publication-Ready Results**

#### **Generate All Publication Materials**
```bash
# Create comprehensive reports
python examples/comprehensive_report_demo.py

# Generate publication plots and tables
python examples/publication_demo.py

# Results will be in: results/comprehensive_report_demo/ and results/publication_outputs/
```

#### **Specific Output Types**
```bash
# LaTeX tables only
python -c "
from ml_research_pipeline.papers.table_generator import LaTeXTableGenerator
generator = LaTeXTableGenerator()
# Use generator to create tables
print('LaTeX table generator ready')
"

# Publication plots only
python -c "
from ml_research_pipeline.papers.plot_generator import PublicationPlotGenerator
generator = PublicationPlotGenerator()
# Use generator to create plots
print('Plot generator ready')
"
```

### 5. **Advanced Features**

#### **Bayesian Uncertainty**
```bash
python examples/bayesian_uncertainty_demo.py
```

#### **Physics Discovery**
```bash
python examples/enhanced_integrated_physics_discovery_demo.py
```

#### **Neural Operators**
```bash
python -m pytest tests/test_fourier_neural_operator.py -v
python -m pytest tests/test_deeponet.py -v
```

#### **Performance Optimization**
```bash
python examples/performance_benchmarking_demo.py
```

### 6. **Troubleshooting Commands**

#### **Check Dependencies**
```bash
# Check Python packages
pip list | grep -E "(torch|numpy|scipy|matplotlib|pandas)"

# Check FEniCSx
python -c "import dolfinx; print(f'FEniCSx version: {dolfinx.__version__}')" || echo "FEniCSx not installed"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### **Clean and Reinstall**
```bash
# Clean installation
pip uninstall ml-research-pipeline -y
pip install -e . --force-reinstall

# Clear cache
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "No CUDA cache to clear"
```

### 7. **Results Locations**

After running the pipeline, check these directories:
```bash
# Main results
ls -la results/

# Comprehensive reports
ls -la results/comprehensive_report_demo/

# Publication outputs
ls -la results/publication_outputs/

# Performance benchmarks
ls -la results/performance_profiling/

# Physics discovery
ls -la results/enhanced_integrated_physics_discovery_demo/
```

## 🎯 **Recommended Execution Order**

1. **First Time Setup:**
   ```bash
   pip install -e .[all]
   ./install_fenicsx.sh  # Optional but recommended
   ```

2. **Quick Validation:**
   ```bash
   python -m pytest tests/test_meta_pinn.py -v
   ```

3. **Full Pipeline:**
   ```bash
   ./run_complete_pipeline.sh
   ```

4. **Review Results:**
   ```bash
   ls -la results/complete_pipeline_run/
   cat results/complete_pipeline_run/PIPELINE_EXECUTION_SUMMARY.md
   ```

## 📊 **Expected Outputs**

After running the complete pipeline, you'll have:
- ✅ **50+ test results** validating all components
- ✅ **Publication-ready plots** (PNG, PDF, SVG formats)
- ✅ **LaTeX tables** for research papers
- ✅ **Comprehensive reports** (Markdown, HTML, LaTeX)
- ✅ **Performance benchmarks** and optimization results
- ✅ **Theoretical analysis** documents
- ✅ **Physics discovery** results with natural language descriptions
- ✅ **Bayesian uncertainty** quantification results
- ✅ **Reproducibility** validation reports

## 🚀 **Ready for Publication!**

The pipeline generates everything needed for research publication:
- Academic paper figures and tables
- Comprehensive experimental results
- Theoretical analysis and proofs
- Performance benchmarking data
- Reproducibility documentation

**Total execution time:** ~10-30 minutes depending on system
**Output size:** ~100-500 MB of results and reports
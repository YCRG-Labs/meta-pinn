Getting Started
===============

Welcome to the ML Research Pipeline! This comprehensive guide will help you get started with meta-learning physics-informed neural networks for fluid dynamics research.

The ML Research Pipeline enables few-shot learning on diverse fluid dynamics tasks through Model-Agnostic Meta-Learning (MAML) combined with physics-informed constraints. Whether you're researching novel meta-learning algorithms, studying fluid dynamics with variable viscosity, or developing physics-informed machine learning methods, this pipeline provides the tools you need.

**Key Features:**

* **Meta-Learning PINNs**: MAML-based adaptation to new fluid dynamics tasks
* **Physics-Informed Learning**: Automatic enforcement of Navier-Stokes equations
* **Diverse Task Generation**: Support for multiple viscosity profiles and geometries
* **Bayesian Uncertainty**: Quantification of epistemic and aleatoric uncertainty
* **Neural Operators**: Integration with Fourier Neural Operators and DeepONet
* **Physics Discovery**: Automated discovery of physical relationships
* **Comprehensive Evaluation**: Statistical analysis and publication-ready outputs
* **Distributed Training**: Multi-GPU support for large-scale experiments

Installation
------------

Prerequisites
~~~~~~~~~~~~~

Before installing the ML Research Pipeline, ensure you have the following prerequisites:

* Python 3.8 or higher
* CUDA-capable GPU (recommended for training)
* Git for version control

System Dependencies
~~~~~~~~~~~~~~~~~~~

For FEniCSx integration, you may need to install system dependencies:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip cmake build-essential

   # macOS (using Homebrew)
   brew install cmake

Python Dependencies
~~~~~~~~~~~~~~~~~~~

Install the package in development mode:

.. code-block:: bash

   git clone https://github.com/example/ml-research-pipeline.git
   cd ml-research-pipeline
   pip install -e .

This will install all required dependencies including PyTorch, NumPy, SciPy, and FEniCSx.

Basic Usage
-----------

Configuration
~~~~~~~~~~~~~

The pipeline uses YAML configuration files for experiment setup:

.. code-block:: python

   from ml_research_pipeline.config import ExperimentConfig
   
   # Load default configuration
   config = ExperimentConfig()
   
   # Or load from file
   config = ExperimentConfig.from_yaml("configs/experiment_default.yaml")
   
   # Modify configuration
   config.model.layers = [2, 128, 128, 128, 3]
   config.training.meta_lr = 0.001

Task Generation
~~~~~~~~~~~~~~~

Generate diverse fluid dynamics tasks for meta-learning:

.. code-block:: python

   from ml_research_pipeline.core import FluidTaskGenerator
   
   # Initialize task generator
   generator = FluidTaskGenerator(
       domain_bounds={"x": [0, 1], "y": [0, 1]},
       task_types=["linear_viscosity", "exponential_viscosity"],
       reynolds_range=[10, 1000]
   )
   
   # Generate a single task
   task = generator.generate_single_task(
       task_type="linear_viscosity",
       n_support=50,
       n_query=100
   )
   
   # Generate task batch for meta-learning
   task_batch = generator.generate_task_batch(
       batch_size=16,
       n_support=50,
       n_query=100
   )

Meta-Learning Training
~~~~~~~~~~~~~~~~~~~~~~

Train a meta-learning PINN model:

.. code-block:: python

   from ml_research_pipeline.core import MetaPINN
   import torch
   
   # Initialize model
   model = MetaPINN(
       layers=[2, 64, 64, 64, 3],
       meta_lr=0.001,
       adapt_lr=0.01
   )
   
   # Meta-training loop
   for epoch in range(1000):
       # Generate task batch
       tasks = generator.generate_task_batch(batch_size=16)
       
       # Meta-update
       meta_loss = model.meta_update(tasks)
       
       if epoch % 100 == 0:
           print(f"Epoch {epoch}, Meta Loss: {meta_loss:.6f}")

Task Adaptation
~~~~~~~~~~~~~~~

Adapt the trained model to new tasks:

.. code-block:: python

   # Generate a new test task
   test_task = generator.generate_single_task("bilinear_viscosity")
   
   # Adapt to the new task
   adapted_params = model.adapt_to_task(
       test_task,
       adaptation_steps=10
   )
   
   # Evaluate on query set
   with torch.no_grad():
       predictions = model.forward(test_task.query_coords, adapted_params)
       accuracy = compute_accuracy(predictions, test_task.query_data)
       print(f"Adaptation accuracy: {accuracy:.4f}")

Evaluation and Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run comprehensive evaluations:

.. code-block:: python

   from ml_research_pipeline.evaluation import PINNBenchmarkSuite
   
   # Initialize benchmark suite
   benchmark = PINNBenchmarkSuite()
   
   # Run full benchmark
   results = benchmark.run_full_benchmark(
       methods=["MetaPINN", "StandardPINN", "TransferLearningPINN"],
       save_dir="results/benchmark"
   )
   
   # Generate publication-ready plots
   benchmark.generate_comparison_plots(results, "results/plots")

Next Steps
----------

* Read the :doc:`user_guide` for detailed usage instructions
* Explore :doc:`tutorials` for step-by-step examples
* Check the :doc:`api_reference` for complete API documentation
* See :doc:`examples` for advanced use cases
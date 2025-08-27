ML Research Pipeline Documentation
===================================

Welcome to the ML Research Pipeline documentation. This package provides a comprehensive framework for meta-learning research on physics-informed neural networks (PINNs).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   api_reference
   tutorials
   examples
   theory
   contributing

Overview
--------

The ML Research Pipeline implements Model-Agnostic Meta-Learning (MAML) for physics-informed neural networks, enabling few-shot adaptation to new fluid dynamics tasks with varying viscosity profiles. The system includes:

* **Meta-Learning Core**: MAML implementation for physics-informed learning
* **Task Generation**: Diverse fluid dynamics scenario creation
* **Neural Operators**: Fourier Neural Operators and DeepONet integration
* **Bayesian Framework**: Uncertainty quantification and calibration
* **Physics Discovery**: Causal discovery and symbolic regression
* **Evaluation Suite**: Comprehensive benchmarking and statistical analysis
* **Publication Tools**: Automated generation of plots, tables, and reports

Quick Start
-----------

.. code-block:: python

   from ml_research_pipeline import MetaPINN, FluidTaskGenerator
   from ml_research_pipeline.config import ExperimentConfig
   
   # Create configuration
   config = ExperimentConfig()
   
   # Initialize meta-learning model
   model = MetaPINN(
       layers=[2, 64, 64, 64, 3],
       meta_lr=0.001,
       adapt_lr=0.01
   )
   
   # Generate tasks
   task_generator = FluidTaskGenerator(
       domain_bounds={"x": [0, 1], "y": [0, 1]},
       task_types=["linear_viscosity", "bilinear_viscosity"]
   )
   
   # Generate a batch of tasks
   tasks = task_generator.generate_task_batch(
       batch_size=16,
       n_support=50,
       n_query=100
   )
   
   # Meta-train the model
   meta_loss = model.meta_update(tasks)

Installation
------------

.. code-block:: bash

   pip install -e .

Requirements
~~~~~~~~~~~~

* Python >= 3.8
* PyTorch >= 1.9.0
* NumPy >= 1.20.0
* SciPy >= 1.7.0
* Matplotlib >= 3.4.0
* FEniCSx >= 0.5.0

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
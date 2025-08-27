User Guide
==========

This comprehensive guide covers all aspects of using the ML Research Pipeline for meta-learning physics-informed neural networks.

Core Concepts
-------------

Meta-Learning for PINNs
~~~~~~~~~~~~~~~~~~~~~~~

Meta-learning enables neural networks to quickly adapt to new tasks with minimal data. In the context of physics-informed neural networks (PINNs), this means:

* **Few-shot adaptation**: Learn new viscosity profiles with <50 examples
* **Physics constraints**: Maintain PDE residuals during adaptation
* **Transfer of physics knowledge**: Leverage learned physics across tasks

The pipeline implements Model-Agnostic Meta-Learning (MAML) specifically adapted for physics-informed learning.

Task Structure
~~~~~~~~~~~~~~

Each fluid dynamics task consists of:

* **Support set**: Training data for adaptation (coordinates + measurements)
* **Query set**: Test data for evaluation
* **Physics parameters**: Viscosity profile, Reynolds number, geometry
* **Boundary conditions**: Domain-specific constraints

Configuration System
---------------------

The pipeline uses hierarchical YAML configuration files:

.. code-block:: yaml

   # experiment_config.yaml
   model:
     layers: [2, 64, 64, 64, 3]
     activation: "tanh"
     meta_lr: 0.001
     adapt_lr: 0.01
   
   training:
     meta_epochs: 1000
     adaptation_steps: 5
     batch_size: 16
   
   data:
     domain_bounds:
       x: [0, 1]
       y: [0, 1]
     task_types:
       - "linear_viscosity"
       - "bilinear_viscosity"
       - "exponential_viscosity"

Loading and modifying configurations:

.. code-block:: python

   from ml_research_pipeline.config import ExperimentConfig
   
   # Load configuration
   config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")
   
   # Modify programmatically
   config.model.layers = [2, 128, 128, 128, 3]
   config.training.meta_lr = 0.0005
   
   # Save modified configuration
   config.save_yaml("configs/modified_experiment.yaml")

Task Generation
---------------

Viscosity Profile Types
~~~~~~~~~~~~~~~~~~~~~~~

The system supports multiple viscosity profile types:

**Linear Viscosity**:

.. math::
   \mu(x, y) = \mu_0 + a \cdot x + b \cdot y

**Bilinear Viscosity**:

.. math::
   \mu(x, y) = \mu_0 + a \cdot x + b \cdot y + c \cdot x \cdot y

**Exponential Viscosity**:

.. math::
   \mu(x, y) = \mu_0 \exp(a \cdot x + b \cdot y)

**Temperature-Dependent**:

.. math::
   \mu(T) = \mu_0 \left(\frac{T}{T_0}\right)^n

Custom Task Generation
~~~~~~~~~~~~~~~~~~~~~~

Create custom task generators:

.. code-block:: python

   from ml_research_pipeline.core import FluidTaskGenerator
   
   class CustomTaskGenerator(FluidTaskGenerator):
       def generate_custom_viscosity(self, params):
           """Generate custom viscosity profile."""
           def viscosity_func(x, y):
               return params['mu0'] * (1 + params['a'] * np.sin(np.pi * x))
           return viscosity_func
       
       def generate_custom_task(self, n_support=50, n_query=100):
           """Generate task with custom viscosity."""
           params = {
               'mu0': np.random.uniform(0.1, 2.0),
               'a': np.random.uniform(-0.5, 0.5)
           }
           
           viscosity_func = self.generate_custom_viscosity(params)
           return self._create_task_from_viscosity(
               viscosity_func, params, n_support, n_query
           )

Large-Scale Dataset Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate and manage large datasets efficiently:

.. code-block:: python

   from ml_research_pipeline.core import DatasetManager
   
   # Initialize dataset manager
   dataset_manager = DatasetManager(
       cache_dir="data/cache",
       max_memory_gb=16
   )
   
   # Generate large dataset
   dataset = dataset_manager.generate_dataset(
       n_tasks=10000,
       task_types=["linear_viscosity", "bilinear_viscosity"],
       n_support=50,
       n_query=100,
       parallel_workers=8
   )
   
   # Save dataset
   dataset_manager.save_dataset(dataset, "data/large_dataset.h5")
   
   # Load dataset with lazy loading
   dataset = dataset_manager.load_dataset(
       "data/large_dataset.h5",
       lazy_loading=True
   )

Meta-Learning Training
----------------------

Basic Training Loop
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ml_research_pipeline.core import MetaPINN
   from ml_research_pipeline.utils import setup_logging, set_random_seeds
   
   # Setup
   setup_logging(level="INFO")
   set_random_seeds(42)
   
   # Initialize model
   model = MetaPINN(
       layers=[2, 64, 64, 64, 3],
       meta_lr=0.001,
       adapt_lr=0.01
   )
   
   # Training loop
   for epoch in range(1000):
       # Generate task batch
       tasks = task_generator.generate_task_batch(batch_size=16)
       
       # Meta-update
       meta_loss = model.meta_update(tasks)
       
       # Logging
       if epoch % 100 == 0:
           print(f"Epoch {epoch}: Meta Loss = {meta_loss:.6f}")
           
           # Validation
           val_tasks = task_generator.generate_task_batch(batch_size=8)
           val_accuracy = model.evaluate_adaptation(val_tasks)
           print(f"Validation Accuracy: {val_accuracy:.4f}")

Advanced Training Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Distributed Training**:

.. code-block:: python

   from ml_research_pipeline.core import DistributedMetaPINN
   import torch.distributed as dist
   
   # Initialize distributed training
   dist.init_process_group(backend='nccl')
   
   # Create distributed model
   model = DistributedMetaPINN(
       layers=[2, 64, 64, 64, 3],
       meta_lr=0.001
   )
   
   # Training with gradient synchronization
   for epoch in range(1000):
       tasks = task_generator.generate_task_batch(batch_size=16)
       meta_loss = model.distributed_meta_update(tasks)

**Checkpointing**:

.. code-block:: python

   from ml_research_pipeline.core import CheckpointManager
   
   # Initialize checkpoint manager
   checkpoint_manager = CheckpointManager(
       save_dir="checkpoints",
       save_frequency=100
   )
   
   # Training with checkpointing
   for epoch in range(1000):
       # Training step
       meta_loss = model.meta_update(tasks)
       
       # Save checkpoint
       if epoch % 100 == 0:
           checkpoint_manager.save_checkpoint(
               model=model,
               optimizer=optimizer,
               epoch=epoch,
               meta_loss=meta_loss
           )
   
   # Resume from checkpoint
   checkpoint = checkpoint_manager.load_latest_checkpoint()
   model.load_state_dict(checkpoint['model_state_dict'])

Bayesian Uncertainty Quantification
-----------------------------------

The pipeline includes comprehensive uncertainty quantification:

.. code-block:: python

   from ml_research_pipeline.bayesian import BayesianMetaPINN
   
   # Initialize Bayesian model
   bayesian_model = BayesianMetaPINN(
       layers=[2, 64, 64, 64, 3],
       prior_std=1.0,
       meta_lr=0.001
   )
   
   # Forward pass with uncertainty
   predictions, uncertainty = bayesian_model.forward_with_uncertainty(
       coords, n_samples=100
   )
   
   # Decompose uncertainty
   epistemic_uncertainty = bayesian_model.compute_epistemic_uncertainty(coords)
   aleatoric_uncertainty = bayesian_model.compute_aleatoric_uncertainty(coords)

Neural Operator Integration
---------------------------

Combine neural operators with meta-learning:

.. code-block:: python

   from ml_research_pipeline.neural_operators import OperatorMetaPINN
   
   # Initialize operator-enhanced model
   operator_model = OperatorMetaPINN(
       pinn_layers=[2, 64, 64, 64, 3],
       operator_type="fourier",  # or "deeponet"
       operator_modes=12
   )
   
   # Training with operator initialization
   for epoch in range(1000):
       tasks = task_generator.generate_task_batch(batch_size=16)
       
       # Operator provides initial parameter estimates
       operator_loss = operator_model.train_operator(tasks)
       
       # Meta-learning refines with physics constraints
       meta_loss = operator_model.meta_update_with_operator(tasks)

Physics Discovery
-----------------

Discover physical relationships automatically:

.. code-block:: python

   from ml_research_pipeline.physics_discovery import IntegratedPhysicsDiscovery
   
   # Initialize physics discovery
   discovery = IntegratedPhysicsDiscovery()
   
   # Analyze flow data for causal relationships
   causal_relationships = discovery.discover_causal_relationships(flow_data)
   
   # Perform symbolic regression
   symbolic_expressions = discovery.discover_symbolic_laws(flow_data)
   
   # Validate discoveries using meta-learning
   validation_scores = discovery.validate_discoveries_with_metalearning(
       discovered_laws=symbolic_expressions,
       meta_model=model
   )

Evaluation and Benchmarking
---------------------------

Comprehensive Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ml_research_pipeline.evaluation import PINNBenchmarkSuite
   
   # Initialize benchmark
   benchmark = PINNBenchmarkSuite()
   
   # Define methods to compare
   methods = {
       "MetaPINN": model,
       "StandardPINN": standard_pinn,
       "TransferLearningPINN": transfer_pinn,
       "FourierNeuralOperator": fno_model,
       "DeepONet": deeponet_model
   }
   
   # Run benchmark
   results = benchmark.run_full_benchmark(
       methods=methods,
       test_tasks=test_tasks,
       metrics=["parameter_accuracy", "adaptation_speed", "physics_consistency"],
       save_dir="results/benchmark"
   )

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ml_research_pipeline.evaluation import StatisticalAnalyzer
   
   # Initialize analyzer
   analyzer = StatisticalAnalyzer()
   
   # Perform method comparison
   comparison_results = analyzer.compare_methods(
       results,
       baseline_method="StandardPINN",
       significance_level=0.05
   )
   
   # Generate statistical report
   report = analyzer.generate_statistical_report(
       comparison_results,
       include_effect_sizes=True,
       include_confidence_intervals=True
   )

Publication Tools
-----------------

Generate publication-ready materials:

.. code-block:: python

   from ml_research_pipeline.papers import PaperPlotGenerator, LaTeXTableGenerator
   
   # Generate plots
   plot_generator = PaperPlotGenerator()
   plot_generator.generate_method_comparison_plot(
       results, save_path="figures/method_comparison.pdf"
   )
   plot_generator.generate_adaptation_curves(
       adaptation_data, save_path="figures/adaptation_curves.pdf"
   )
   
   # Generate LaTeX tables
   table_generator = LaTeXTableGenerator()
   latex_table = table_generator.generate_results_table(
       results, caption="Method comparison results"
   )
   
   # Save table
   with open("tables/results.tex", "w") as f:
       f.write(latex_table)

Best Practices
--------------

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

* Use mixed precision training for memory efficiency
* Implement gradient accumulation for large effective batch sizes
* Cache frequently used computations
* Use distributed training for large-scale experiments

Reproducibility
~~~~~~~~~~~~~~~

* Set random seeds consistently
* Use deterministic algorithms where possible
* Save complete experiment configurations
* Version control datasets and model checkpoints

Debugging
~~~~~~~~~

* Monitor physics residuals during training
* Visualize adaptation trajectories
* Check gradient magnitudes and flows
* Validate against analytical solutions

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Slow Convergence**:
- Reduce adaptation learning rate
- Increase number of adaptation steps
- Check physics loss weighting

**Memory Issues**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

**Numerical Instability**:
- Clip gradients during meta-updates
- Use more stable activation functions
- Reduce learning rates

**Poor Physics Consistency**:
- Increase physics loss weight
- Check boundary condition implementation
- Validate PDE residual computation
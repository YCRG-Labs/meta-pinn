Tutorials
=========

This section provides step-by-step tutorials for using the ML Research Pipeline.

Tutorial 1: Basic Meta-Learning PINN
-------------------------------------

This tutorial demonstrates how to set up and train a basic meta-learning PINN for fluid dynamics tasks.

Setup
~~~~~

First, let's import the necessary modules and set up the environment:

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   
   from ml_research_pipeline.core import MetaPINN, FluidTaskGenerator
   from ml_research_pipeline.config import ExperimentConfig
   from ml_research_pipeline.utils import setup_logging, set_random_seeds
   
   # Setup logging and reproducibility
   setup_logging(level="INFO")
   set_random_seeds(42)
   
   # Check for GPU availability
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

Configuration
~~~~~~~~~~~~~

Create a configuration for the experiment:

.. code-block:: python

   # Load default configuration
   config = ExperimentConfig()
   
   # Customize for this tutorial
   config.model.layers = [2, 64, 64, 64, 3]  # [input, hidden, hidden, hidden, output]
   config.model.activation = "tanh"
   config.training.meta_lr = 0.001
   config.training.adapt_lr = 0.01
   config.training.meta_epochs = 500
   config.training.adaptation_steps = 5
   config.training.batch_size = 8
   
   print("Configuration:")
   print(config)

Task Generation
~~~~~~~~~~~~~~~

Set up the task generator for creating diverse fluid dynamics scenarios:

.. code-block:: python

   # Initialize task generator
   task_generator = FluidTaskGenerator(
       domain_bounds={"x": [0, 1], "y": [0, 1]},
       task_types=["linear_viscosity", "bilinear_viscosity"],
       reynolds_range=[10, 500],
       device=device
   )
   
   # Generate a sample task to visualize
   sample_task = task_generator.generate_single_task(
       task_type="linear_viscosity",
       n_support=50,
       n_query=100
   )
   
   print(f"Sample task config: {sample_task.config}")
   print(f"Support set shape: {sample_task.support_set['coords'].shape}")
   print(f"Query set shape: {sample_task.query_set['coords'].shape}")

Visualize Sample Task
~~~~~~~~~~~~~~~~~~~~~

Let's visualize the generated task:

.. code-block:: python

   def plot_task(task, title="Fluid Task"):
       """Plot the velocity and pressure fields for a task."""
       fig, axes = plt.subplots(1, 3, figsize=(15, 4))
       
       # Extract coordinates and data
       coords = task.support_set['coords'].cpu().numpy()
       data = task.support_set['data'].cpu().numpy()
       
       x, y = coords[:, 0], coords[:, 1]
       u, v, p = data[:, 0], data[:, 1], data[:, 2]
       
       # Plot velocity components and pressure
       scatter_kwargs = {'s': 20, 'alpha': 0.7}
       
       im1 = axes[0].scatter(x, y, c=u, cmap='RdBu_r', **scatter_kwargs)
       axes[0].set_title('u-velocity')
       axes[0].set_xlabel('x')
       axes[0].set_ylabel('y')
       plt.colorbar(im1, ax=axes[0])
       
       im2 = axes[1].scatter(x, y, c=v, cmap='RdBu_r', **scatter_kwargs)
       axes[1].set_title('v-velocity')
       axes[1].set_xlabel('x')
       plt.colorbar(im2, ax=axes[1])
       
       im3 = axes[2].scatter(x, y, c=p, cmap='viridis', **scatter_kwargs)
       axes[2].set_title('Pressure')
       axes[2].set_xlabel('x')
       plt.colorbar(im3, ax=axes[2])
       
       plt.suptitle(title)
       plt.tight_layout()
       plt.show()
   
   # Plot the sample task
   plot_task(sample_task, "Sample Linear Viscosity Task")

Model Initialization
~~~~~~~~~~~~~~~~~~~~

Initialize the meta-learning PINN model:

.. code-block:: python

   # Initialize MetaPINN
   model = MetaPINN(
       layers=config.model.layers,
       activation=config.model.activation,
       meta_lr=config.training.meta_lr,
       adapt_lr=config.training.adapt_lr,
       device=device
   )
   
   print(f"Model architecture:")
   print(model)
   print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

Training Loop
~~~~~~~~~~~~~

Implement the meta-training loop:

.. code-block:: python

   # Training history
   meta_losses = []
   adaptation_accuracies = []
   
   print("Starting meta-training...")
   
   for epoch in range(config.training.meta_epochs):
       # Generate task batch
       task_batch = task_generator.generate_task_batch(
           batch_size=config.training.batch_size,
           n_support=50,
           n_query=100
       )
       
       # Meta-update
       meta_loss = model.meta_update(task_batch)
       meta_losses.append(meta_loss)
       
       # Periodic evaluation
       if epoch % 50 == 0:
           # Generate validation tasks
           val_tasks = task_generator.generate_task_batch(
               batch_size=4,
               n_support=50,
               n_query=100
           )
           
           # Evaluate adaptation performance
           adaptation_accuracy = model.evaluate_adaptation(
               val_tasks,
               adaptation_steps=config.training.adaptation_steps
           )
           adaptation_accuracies.append(adaptation_accuracy)
           
           print(f"Epoch {epoch:3d}: Meta Loss = {meta_loss:.6f}, "
                 f"Adaptation Accuracy = {adaptation_accuracy:.4f}")
   
   print("Meta-training completed!")

Visualize Training Progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot the training progress:

.. code-block:: python

   # Plot training curves
   fig, axes = plt.subplots(1, 2, figsize=(12, 4))
   
   # Meta loss
   axes[0].plot(meta_losses)
   axes[0].set_title('Meta-Learning Loss')
   axes[0].set_xlabel('Epoch')
   axes[0].set_ylabel('Loss')
   axes[0].grid(True)
   
   # Adaptation accuracy
   eval_epochs = np.arange(0, len(meta_losses), 50)[:len(adaptation_accuracies)]
   axes[1].plot(eval_epochs, adaptation_accuracies, 'o-')
   axes[1].set_title('Adaptation Accuracy')
   axes[1].set_xlabel('Epoch')
   axes[1].set_ylabel('Accuracy')
   axes[1].grid(True)
   
   plt.tight_layout()
   plt.show()

Testing Adaptation
~~~~~~~~~~~~~~~~~~

Test the trained model on new tasks:

.. code-block:: python

   # Generate test tasks
   test_tasks = [
       task_generator.generate_single_task("linear_viscosity", n_support=25, n_query=100),
       task_generator.generate_single_task("bilinear_viscosity", n_support=25, n_query=100)
   ]
   
   for i, test_task in enumerate(test_tasks):
       print(f"\nTesting on task {i+1} ({test_task.config['task_type']}):")
       
       # Adapt to the task
       adapted_params = model.adapt_to_task(
           test_task,
           adaptation_steps=10
       )
       
       # Evaluate on query set
       with torch.no_grad():
           predictions = model.forward(test_task.query_coords, adapted_params)
           
           # Compute accuracy metrics
           mse = torch.mean((predictions - test_task.query_data) ** 2)
           physics_residual = model.compute_physics_residual(
               test_task.query_coords, predictions, test_task.config
           )
           
           print(f"  MSE: {mse:.6f}")
           print(f"  Physics Residual: {physics_residual:.6f}")

Visualize Adaptation Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize how well the model adapts to new tasks:

.. code-block:: python

   def plot_adaptation_results(model, task, adaptation_steps=10):
       """Plot adaptation progress for a single task."""
       # Track adaptation progress
       adaptation_losses = []
       
       # Initial prediction (before adaptation)
       with torch.no_grad():
           initial_pred = model.forward(task.query_coords)
           
       # Adapt step by step
       current_params = {name: param.clone() for name, param in model.named_parameters()}
       
       for step in range(adaptation_steps):
           # Adaptation step
           adapted_params = model.adapt_single_step(task, current_params)
           current_params = adapted_params
           
           # Evaluate
           with torch.no_grad():
               pred = model.forward(task.query_coords, adapted_params)
               loss = torch.mean((pred - task.query_data) ** 2)
               adaptation_losses.append(loss.item())
       
       # Final prediction
       final_pred = model.forward(task.query_coords, adapted_params)
       
       # Plot results
       fig, axes = plt.subplots(2, 3, figsize=(15, 8))
       
       # Ground truth
       coords = task.query_coords.cpu().numpy()
       true_data = task.query_data.cpu().numpy()
       initial_data = initial_pred.cpu().numpy()
       final_data = final_pred.cpu().numpy()
       
       x, y = coords[:, 0], coords[:, 1]
       
       # Plot ground truth
       for j, label in enumerate(['u', 'v', 'p']):
           im = axes[0, j].scatter(x, y, c=true_data[:, j], cmap='RdBu_r', s=20)
           axes[0, j].set_title(f'True {label}')
           plt.colorbar(im, ax=axes[0, j])
       
       # Plot final prediction
       for j, label in enumerate(['u', 'v', 'p']):
           im = axes[1, j].scatter(x, y, c=final_data[:, j], cmap='RdBu_r', s=20)
           axes[1, j].set_title(f'Predicted {label}')
           plt.colorbar(im, ax=axes[1, j])
       
       plt.tight_layout()
       plt.show()
       
       # Plot adaptation curve
       plt.figure(figsize=(8, 4))
       plt.plot(adaptation_losses, 'o-')
       plt.title('Adaptation Progress')
       plt.xlabel('Adaptation Step')
       plt.ylabel('MSE Loss')
       plt.grid(True)
       plt.show()
   
   # Visualize adaptation for the first test task
   plot_adaptation_results(model, test_tasks[0])

Save the Model
~~~~~~~~~~~~~~

Save the trained model for future use:

.. code-block:: python

   # Save model checkpoint
   checkpoint = {
       'model_state_dict': model.state_dict(),
       'config': config,
       'meta_losses': meta_losses,
       'adaptation_accuracies': adaptation_accuracies
   }
   
   torch.save(checkpoint, 'tutorial_1_model.pth')
   print("Model saved as 'tutorial_1_model.pth'")

Summary
~~~~~~~

In this tutorial, you learned how to:

1. Set up the ML Research Pipeline environment
2. Configure experiments using the configuration system
3. Generate diverse fluid dynamics tasks
4. Initialize and train a meta-learning PINN
5. Evaluate adaptation performance on new tasks
6. Visualize training progress and results

The trained model can now quickly adapt to new viscosity profiles with just a few gradient steps, demonstrating the power of meta-learning for physics-informed neural networks.

Tutorial 2: Bayesian Uncertainty Quantification
------------------------------------------------

This tutorial shows how to incorporate Bayesian uncertainty quantification into meta-learning PINNs.

Setup and Imports
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy import stats
   
   from ml_research_pipeline.bayesian import BayesianMetaPINN, UncertaintyCalibrator
   from ml_research_pipeline.core import FluidTaskGenerator
   from ml_research_pipeline.utils import setup_logging, set_random_seeds
   
   # Setup
   setup_logging(level="INFO")
   set_random_seeds(42)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Initialize Bayesian Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize Bayesian MetaPINN
   bayesian_model = BayesianMetaPINN(
       layers=[2, 64, 64, 64, 3],
       prior_std=1.0,
       meta_lr=0.001,
       adapt_lr=0.01,
       device=device
   )
   
   print(f"Bayesian model initialized with {sum(p.numel() for p in bayesian_model.parameters())} parameters")

Training with Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Task generator
   task_generator = FluidTaskGenerator(
       domain_bounds={"x": [0, 1], "y": [0, 1]},
       task_types=["linear_viscosity", "exponential_viscosity"],
       device=device
   )
   
   # Training loop with KL divergence
   meta_losses = []
   kl_losses = []
   
   for epoch in range(300):
       task_batch = task_generator.generate_task_batch(batch_size=8)
       
       # Meta-update with KL divergence
       meta_loss, kl_loss = bayesian_model.meta_update_with_kl(task_batch)
       
       meta_losses.append(meta_loss)
       kl_losses.append(kl_loss)
       
       if epoch % 50 == 0:
           print(f"Epoch {epoch}: Meta Loss = {meta_loss:.6f}, KL Loss = {kl_loss:.6f}")

Uncertainty Prediction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate test task
   test_task = task_generator.generate_single_task("linear_viscosity")
   
   # Adapt with uncertainty
   adapted_params = bayesian_model.adapt_to_task(test_task)
   
   # Forward pass with uncertainty quantification
   predictions, uncertainty = bayesian_model.forward_with_uncertainty(
       test_task.query_coords,
       adapted_params,
       n_samples=100
   )
   
   # Decompose uncertainty
   epistemic_unc = bayesian_model.compute_epistemic_uncertainty(
       test_task.query_coords, adapted_params
   )
   aleatoric_unc = bayesian_model.compute_aleatoric_uncertainty(
       test_task.query_coords, adapted_params
   )
   
   print(f"Total uncertainty: {uncertainty.mean():.6f}")
   print(f"Epistemic uncertainty: {epistemic_unc.mean():.6f}")
   print(f"Aleatoric uncertainty: {aleatoric_unc.mean():.6f}")

Uncertainty Calibration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize uncertainty calibrator
   calibrator = UncertaintyCalibrator()
   
   # Generate calibration dataset
   cal_tasks = [task_generator.generate_single_task("linear_viscosity") for _ in range(20)]
   
   # Collect predictions and uncertainties
   all_predictions = []
   all_uncertainties = []
   all_targets = []
   
   for task in cal_tasks:
       adapted_params = bayesian_model.adapt_to_task(task)
       pred, unc = bayesian_model.forward_with_uncertainty(
           task.query_coords, adapted_params
       )
       
       all_predictions.append(pred)
       all_uncertainties.append(unc)
       all_targets.append(task.query_data)
   
   # Calibrate uncertainty
   calibrator.fit(
       torch.cat(all_predictions),
       torch.cat(all_uncertainties),
       torch.cat(all_targets)
   )
   
   # Evaluate calibration
   calibration_error = calibrator.evaluate_calibration()
   print(f"Calibration error: {calibration_error:.6f}")

This tutorial demonstrates the complete workflow for incorporating Bayesian uncertainty quantification into meta-learning PINNs, including training, prediction, and calibration.

Tutorial 3: Neural Operator Integration
----------------------------------------

This tutorial shows how to integrate neural operators (FNO and DeepONet) with meta-learning PINNs for enhanced parameter inference.

Setup and Imports
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   
   from ml_research_pipeline.neural_operators import (
       InverseFourierNeuralOperator, 
       PhysicsInformedDeepONet,
       OperatorMetaPINN
   )
   from ml_research_pipeline.core import FluidTaskGenerator
   from ml_research_pipeline.utils import setup_logging, set_random_seeds
   
   setup_logging(level="INFO")
   set_random_seeds(42)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Initialize Neural Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize Fourier Neural Operator for parameter inference
   fno = InverseFourierNeuralOperator(
       modes=12,
       width=64,
       input_dim=2,  # spatial coordinates
       output_dim=1,  # viscosity field
       device=device
   )
   
   # Initialize Physics-Informed DeepONet
   deeponet = PhysicsInformedDeepONet(
       branch_layers=[100, 128, 128, 128],  # measurement processing
       trunk_layers=[2, 128, 128, 128],     # coordinate encoding
       output_dim=1,  # viscosity at query points
       device=device
   )
   
   print(f"FNO parameters: {sum(p.numel() for p in fno.parameters())}")
   print(f"DeepONet parameters: {sum(p.numel() for p in deeponet.parameters())}")

Operator-Enhanced Meta-Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize operator-enhanced meta-learning model
   operator_meta_pinn = OperatorMetaPINN(
       pinn_layers=[2, 64, 64, 64, 3],
       operator_type="fno",  # or "deeponet"
       operator_config={
           "modes": 12,
           "width": 64
       },
       meta_lr=0.001,
       adapt_lr=0.01,
       device=device
   )
   
   # Task generator with sparse observations
   task_generator = FluidTaskGenerator(
       domain_bounds={"x": [0, 1], "y": [0, 1]},
       task_types=["linear_viscosity", "bilinear_viscosity"],
       sparse_observations=True,  # Enable sparse measurement mode
       n_observations=20,  # Number of sparse measurements
       device=device
   )

Training with Operator Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Training loop with operator-initialized adaptation
   meta_losses = []
   operator_losses = []
   
   for epoch in range(500):
       # Generate task batch with sparse observations
       task_batch = task_generator.generate_task_batch(
           batch_size=8,
           n_support=50,
           n_query=100,
           include_sparse_obs=True
       )
       
       # Joint training of operator and meta-learning
       meta_loss, op_loss = operator_meta_pinn.joint_meta_update(task_batch)
       
       meta_losses.append(meta_loss)
       operator_losses.append(op_loss)
       
       if epoch % 100 == 0:
           print(f"Epoch {epoch}: Meta Loss = {meta_loss:.6f}, "
                 f"Operator Loss = {op_loss:.6f}")

Evaluate Operator-Enhanced Adaptation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate test task with sparse observations
   test_task = task_generator.generate_single_task(
       "exponential_viscosity",
       include_sparse_obs=True
   )
   
   # Standard adaptation (without operator)
   standard_adapted_params = operator_meta_pinn.adapt_to_task(
       test_task,
       use_operator_init=False,
       adaptation_steps=10
   )
   
   # Operator-enhanced adaptation
   operator_adapted_params = operator_meta_pinn.adapt_to_task(
       test_task,
       use_operator_init=True,
       adaptation_steps=5  # Fewer steps needed with operator init
   )
   
   # Compare performance
   with torch.no_grad():
       standard_pred = operator_meta_pinn.forward(
           test_task.query_coords, standard_adapted_params
       )
       operator_pred = operator_meta_pinn.forward(
           test_task.query_coords, operator_adapted_params
       )
       
       standard_mse = torch.mean((standard_pred - test_task.query_data) ** 2)
       operator_mse = torch.mean((operator_pred - test_task.query_data) ** 2)
       
       print(f"Standard adaptation MSE: {standard_mse:.6f}")
       print(f"Operator-enhanced MSE: {operator_mse:.6f}")
       print(f"Improvement: {(standard_mse - operator_mse) / standard_mse * 100:.2f}%")

Tutorial 4: Physics Discovery
------------------------------

This tutorial demonstrates automated physics discovery using causal discovery and symbolic regression to identify novel physical relationships.

Setup and Imports
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   import networkx as nx
   from sympy import symbols, sympify, latex
   
   from ml_research_pipeline.physics_discovery import (
       PhysicsCausalDiscovery,
       NeuralSymbolicRegression,
       IntegratedPhysicsDiscovery
   )
   from ml_research_pipeline.core import FluidTaskGenerator
   from ml_research_pipeline.utils import setup_logging, set_random_seeds
   
   setup_logging(level="INFO")
   set_random_seeds(42)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Generate Physics Discovery Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create task generator with diverse physics scenarios
   task_generator = FluidTaskGenerator(
       domain_bounds={"x": [0, 1], "y": [0, 1]},
       task_types=[
           "linear_viscosity", 
           "bilinear_viscosity", 
           "exponential_viscosity",
           "temperature_dependent"
       ],
       reynolds_range=[10, 1000],
       device=device
   )
   
   # Generate large dataset for physics discovery
   discovery_tasks = []
   for _ in range(100):
       task = task_generator.generate_single_task(
           task_type=np.random.choice([
               "linear_viscosity", 
               "bilinear_viscosity", 
               "exponential_viscosity"
           ]),
           n_support=100,
           n_query=200
       )
       discovery_tasks.append(task)
   
   print(f"Generated {len(discovery_tasks)} tasks for physics discovery")

Causal Discovery
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize causal discovery system
   causal_discovery = PhysicsCausalDiscovery(
       variables=['x', 'y', 'u', 'v', 'p', 'viscosity', 'reynolds'],
       significance_threshold=0.05,
       device=device
   )
   
   # Extract features from all tasks
   all_features = []
   all_targets = []
   
   for task in discovery_tasks:
       # Extract spatial coordinates and flow variables
       coords = task.support_set['coords']
       data = task.support_set['data']
       
       # Compute derived quantities
       x, y = coords[:, 0], coords[:, 1]
       u, v, p = data[:, 0], data[:, 1], data[:, 2]
       
       # Compute gradients and derived physics quantities
       velocity_magnitude = torch.sqrt(u**2 + v**2)
       vorticity = torch.gradient(v, dim=0)[0] - torch.gradient(u, dim=1)[0]
       
       # Create feature matrix
       features = torch.stack([
           x, y, u, v, p, 
           velocity_magnitude, vorticity,
           torch.full_like(x, task.config['reynolds'])
       ], dim=1)
       
       # Target is viscosity
       viscosity = task.compute_viscosity_field(coords)
       
       all_features.append(features)
       all_targets.append(viscosity)
   
   # Combine all data
   combined_features = torch.cat(all_features, dim=0)
   combined_targets = torch.cat(all_targets, dim=0)
   
   # Discover causal relationships
   causal_graph = causal_discovery.discover_relationships(
       combined_features, combined_targets
   )
   
   # Visualize causal graph
   plt.figure(figsize=(10, 8))
   pos = nx.spring_layout(causal_graph)
   nx.draw(causal_graph, pos, with_labels=True, 
           node_color='lightblue', node_size=1500,
           font_size=10, font_weight='bold')
   
   # Add edge labels with causal strengths
   edge_labels = nx.get_edge_attributes(causal_graph, 'strength')
   nx.draw_networkx_edge_labels(causal_graph, pos, edge_labels)
   
   plt.title("Discovered Causal Relationships")
   plt.axis('off')
   plt.show()

Symbolic Regression
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize symbolic regression system
   symbolic_regression = NeuralSymbolicRegression(
       input_variables=['x', 'y', 'u', 'v', 'p', 'reynolds'],
       max_complexity=10,
       population_size=100,
       generations=50,
       device=device
   )
   
   # Discover symbolic expressions for viscosity
   discovered_expressions = symbolic_regression.discover_expressions(
       combined_features[:, :6],  # x, y, u, v, p, reynolds
       combined_targets,
       n_expressions=5
   )
   
   print("Discovered symbolic expressions:")
   for i, (expr, fitness) in enumerate(discovered_expressions):
       print(f"{i+1}. {expr} (fitness: {fitness:.6f})")
       print(f"   LaTeX: ${latex(expr)}$")
       print()

Integrated Physics Discovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize integrated discovery system
   integrated_discovery = IntegratedPhysicsDiscovery(
       causal_discovery=causal_discovery,
       symbolic_regression=symbolic_regression,
       meta_pinn=None,  # Will be set later for validation
       device=device
   )
   
   # Run integrated discovery pipeline
   discovery_results = integrated_discovery.discover_physics_laws(
       discovery_tasks,
       validation_tasks=discovery_tasks[-20:],  # Use last 20 tasks for validation
       max_iterations=10
   )
   
   print("Integrated Physics Discovery Results:")
   print(f"Number of discovered laws: {len(discovery_results['laws'])}")
   print(f"Average validation score: {discovery_results['avg_validation_score']:.4f}")
   
   # Display top discovered laws
   for i, law in enumerate(discovery_results['laws'][:3]):
       print(f"\nLaw {i+1}:")
       print(f"  Expression: {law['expression']}")
       print(f"  Causal strength: {law['causal_strength']:.4f}")
       print(f"  Validation score: {law['validation_score']:.4f}")
       print(f"  Natural language: {law['natural_language']}")

Validate Discovered Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use meta-learning to validate discovered physics laws
   from ml_research_pipeline.core import MetaPINN
   
   # Initialize meta-learning model for validation
   meta_pinn = MetaPINN(
       layers=[2, 64, 64, 64, 3],
       meta_lr=0.001,
       adapt_lr=0.01,
       device=device
   )
   
   # Train on tasks with discovered physics constraints
   validation_tasks = task_generator.generate_task_batch(
       batch_size=20,
       n_support=50,
       n_query=100
   )
   
   # Compare performance with and without discovered physics
   baseline_performance = []
   physics_informed_performance = []
   
   for task in validation_tasks:
       # Baseline adaptation
       baseline_params = meta_pinn.adapt_to_task(task, adaptation_steps=10)
       
       # Physics-informed adaptation with discovered laws
       physics_params = meta_pinn.adapt_to_task_with_discovered_physics(
           task, 
           discovered_laws=discovery_results['laws'][:3],
           adaptation_steps=5
       )
       
       # Evaluate both approaches
       with torch.no_grad():
           baseline_pred = meta_pinn.forward(task.query_coords, baseline_params)
           physics_pred = meta_pinn.forward(task.query_coords, physics_params)
           
           baseline_mse = torch.mean((baseline_pred - task.query_data) ** 2)
           physics_mse = torch.mean((physics_pred - task.query_data) ** 2)
           
           baseline_performance.append(baseline_mse.item())
           physics_informed_performance.append(physics_mse.item())
   
   # Statistical analysis
   from scipy.stats import ttest_rel
   
   t_stat, p_value = ttest_rel(baseline_performance, physics_informed_performance)
   
   print(f"\nValidation Results:")
   print(f"Baseline MSE: {np.mean(baseline_performance):.6f} ± {np.std(baseline_performance):.6f}")
   print(f"Physics-informed MSE: {np.mean(physics_informed_performance):.6f} ± {np.std(physics_informed_performance):.6f}")
   print(f"Improvement: {(np.mean(baseline_performance) - np.mean(physics_informed_performance)) / np.mean(baseline_performance) * 100:.2f}%")
   print(f"Statistical significance: t={t_stat:.4f}, p={p_value:.6f}")

Tutorial 5: Large-Scale Distributed Training
---------------------------------------------

This tutorial covers distributed training for large-scale meta-learning experiments across multiple GPUs and nodes.

Setup for Distributed Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.distributed as dist
   import torch.multiprocessing as mp
   from torch.nn.parallel import DistributedDataParallel as DDP
   import os
   
   from ml_research_pipeline.core import DistributedMetaPINN
   from ml_research_pipeline.utils import setup_distributed, cleanup_distributed
   from ml_research_pipeline.config import ExperimentConfig

Single-Node Multi-GPU Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def train_distributed(rank, world_size, config):
       """Training function for each process."""
       # Setup distributed training
       setup_distributed(rank, world_size)
       
       # Initialize model on specific GPU
       device = torch.device(f"cuda:{rank}")
       model = DistributedMetaPINN(
           config=config,
           device=device,
           rank=rank,
           world_size=world_size
       )
       
       # Wrap model with DDP
       model = DDP(model, device_ids=[rank])
       
       # Initialize task generator (distributed)
       task_generator = FluidTaskGenerator(
           config.data,
           device=device,
           distributed=True,
           rank=rank,
           world_size=world_size
       )
       
       # Training loop
       for epoch in range(config.training.meta_epochs):
           # Generate distributed task batch
           task_batch = task_generator.generate_distributed_batch(
               batch_size=config.training.batch_size // world_size,
               epoch=epoch
           )
           
           # Distributed meta-update
           meta_loss = model.module.distributed_meta_update(task_batch)
           
           if rank == 0 and epoch % 100 == 0:
               print(f"Epoch {epoch}: Meta Loss = {meta_loss:.6f}")
       
       # Cleanup
       cleanup_distributed()
   
   # Launch distributed training
   if __name__ == "__main__":
       config = ExperimentConfig()
       config.training.meta_epochs = 2000
       config.training.batch_size = 64  # Will be split across GPUs
       
       world_size = torch.cuda.device_count()
       mp.spawn(train_distributed, args=(world_size, config), nprocs=world_size)

Multi-Node Training with SLURM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=meta_pinn_training
   #SBATCH --nodes=4
   #SBATCH --ntasks-per-node=4
   #SBATCH --gres=gpu:4
   #SBATCH --time=24:00:00
   
   # Setup environment
   module load python/3.9
   module load cuda/11.8
   
   # Set distributed training environment variables
   export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
   export MASTER_PORT=29500
   export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
   
   # Launch training
   srun python -u distributed_training_script.py \
       --config configs/large_scale_experiment.yaml \
       --nodes $SLURM_NNODES \
       --gpus-per-node $SLURM_NTASKS_PER_NODE

Performance Monitoring and Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ml_research_pipeline.core import CheckpointManager, TrainingMonitor
   
   def train_with_monitoring(rank, world_size, config):
       """Training with comprehensive monitoring and checkpointing."""
       setup_distributed(rank, world_size)
       device = torch.device(f"cuda:{rank}")
       
       # Initialize components
       model = DistributedMetaPINN(config, device, rank, world_size)
       model = DDP(model, device_ids=[rank])
       
       # Initialize monitoring and checkpointing
       if rank == 0:
           checkpoint_manager = CheckpointManager(
               save_dir="checkpoints/large_scale_experiment",
               save_frequency=100,
               max_checkpoints=5
           )
           
           training_monitor = TrainingMonitor(
               log_dir="logs/large_scale_experiment",
               metrics=['meta_loss', 'adaptation_accuracy', 'physics_residual'],
               plot_frequency=50
           )
       
       # Training loop with monitoring
       for epoch in range(config.training.meta_epochs):
           # Generate task batch
           task_batch = task_generator.generate_distributed_batch(
               batch_size=config.training.batch_size // world_size
           )
           
           # Training step
           meta_loss = model.module.distributed_meta_update(task_batch)
           
           if rank == 0:
               # Log metrics
               training_monitor.log_metrics({
                   'meta_loss': meta_loss,
                   'epoch': epoch
               })
               
               # Periodic evaluation
               if epoch % 50 == 0:
                   val_accuracy = evaluate_distributed_model(model, task_generator)
                   training_monitor.log_metrics({
                       'adaptation_accuracy': val_accuracy,
                       'epoch': epoch
                   })
               
               # Save checkpoint
               if epoch % 100 == 0:
                   checkpoint_manager.save_checkpoint({
                       'epoch': epoch,
                       'model_state_dict': model.module.state_dict(),
                       'meta_loss': meta_loss,
                       'config': config
                   })
       
       cleanup_distributed()

This comprehensive tutorial series covers the essential aspects of using the ML Research Pipeline, from basic meta-learning to advanced distributed training and physics discovery.
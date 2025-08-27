Examples
========

This section provides complete examples demonstrating various aspects of the ML Research Pipeline.

Example 1: Complete Meta-Learning Workflow
-------------------------------------------

This example demonstrates a complete meta-learning workflow from task generation to evaluation.

.. literalinclude:: ../examples/comprehensive_report_demo.py
   :language: python
   :caption: Complete Meta-Learning Workflow

Key Features Demonstrated:

* Task generation with multiple viscosity profiles
* Meta-learning PINN training
* Comprehensive evaluation and benchmarking
* Statistical analysis and reporting
* Publication-ready plot generation

Example 2: Bayesian Uncertainty Quantification
-----------------------------------------------

This example shows how to use Bayesian meta-learning PINNs for uncertainty quantification.

.. literalinclude:: ../examples/bayesian_uncertainty_demo.py
   :language: python
   :caption: Bayesian Uncertainty Quantification

Key Features Demonstrated:

* Variational Bayesian neural networks
* Epistemic and aleatoric uncertainty decomposition
* Uncertainty calibration
* Physics-informed uncertainty estimation

Example 3: Neural Operator Integration
---------------------------------------

This example demonstrates integration of neural operators with meta-learning.

.. code-block:: python
   :caption: Neural Operator Integration Example

   import torch
   import numpy as np
   from ml_research_pipeline.neural_operators import OperatorMetaPINN
   from ml_research_pipeline.core import FluidTaskGenerator
   
   # Initialize operator-enhanced meta-learning model
   operator_model = OperatorMetaPINN(
       pinn_layers=[2, 64, 64, 64, 3],
       operator_type="fourier",
       operator_modes=12,
       meta_lr=0.001
   )
   
   # Task generator
   task_generator = FluidTaskGenerator(
       domain_bounds={"x": [0, 1], "y": [0, 1]},
       task_types=["linear_viscosity", "bilinear_viscosity"]
   )
   
   # Training loop with operator initialization
   for epoch in range(500):
       tasks = task_generator.generate_task_batch(batch_size=16)
       
       # Train operator for parameter field prediction
       operator_loss = operator_model.train_operator(tasks)
       
       # Meta-learning with operator initialization
       meta_loss = operator_model.meta_update_with_operator(tasks)
       
       if epoch % 100 == 0:
           print(f"Epoch {epoch}: Operator Loss = {operator_loss:.6f}, "
                 f"Meta Loss = {meta_loss:.6f}")
   
   # Test adaptation with operator initialization
   test_task = task_generator.generate_single_task("exponential_viscosity")
   
   # Operator provides initial parameter estimate
   operator_init = operator_model.operator_predict(test_task.support_set)
   
   # Fast adaptation with operator initialization
   adapted_params = operator_model.adapt_with_operator_init(
       test_task, operator_init, adaptation_steps=3
   )
   
   # Evaluate
   predictions = operator_model.forward(test_task.query_coords, adapted_params)
   accuracy = torch.mean((predictions - test_task.query_data) ** 2)
   print(f"Adaptation accuracy with operator init: {accuracy:.6f}")

Key Features Demonstrated:

* Fourier Neural Operator for parameter field prediction
* Joint training of operators and meta-learning
* Operator-initialized adaptation
* Faster convergence with operator guidance

Example 4: Physics Discovery Pipeline
--------------------------------------

This example shows automated physics discovery using causal discovery and symbolic regression.

.. literalinclude:: ../examples/integrated_physics_discovery_demo.py
   :language: python
   :caption: Physics Discovery Pipeline

Key Features Demonstrated:

* Causal relationship discovery in fluid dynamics
* Symbolic regression for physics law discovery
* Meta-learning validation of discovered physics
* Natural language hypothesis generation

Example 5: Large-Scale Dataset Generation
------------------------------------------

This example demonstrates efficient generation and management of large-scale datasets.

.. literalinclude:: ../examples/large_scale_dataset_demo.py
   :language: python
   :caption: Large-Scale Dataset Generation

Key Features Demonstrated:

* Parallel task generation
* Efficient data storage and caching
* Memory-efficient dataset loading
* Distributed dataset processing

Example 6: Distributed Training
--------------------------------

This example shows how to set up distributed training across multiple GPUs.

.. code-block:: python
   :caption: Distributed Training Example

   import torch
   import torch.distributed as dist
   import torch.multiprocessing as mp
   from ml_research_pipeline.core import DistributedMetaPINN
   from ml_research_pipeline.utils import setup_distributed, cleanup_distributed
   
   def train_distributed(rank, world_size):
       """Distributed training function."""
       # Setup distributed training
       setup_distributed(rank, world_size)
       
       # Initialize distributed model
       model = DistributedMetaPINN(
           layers=[2, 64, 64, 64, 3],
           meta_lr=0.001,
           device=f"cuda:{rank}"
       )
       
       # Task generator (each process generates different tasks)
       task_generator = FluidTaskGenerator(
           domain_bounds={"x": [0, 1], "y": [0, 1]},
           task_types=["linear_viscosity", "bilinear_viscosity"],
           seed=42 + rank  # Different seed per process
       )
       
       # Distributed training loop
       for epoch in range(1000):
           # Generate local task batch
           tasks = task_generator.generate_task_batch(batch_size=8)
           
           # Distributed meta-update with gradient synchronization
           meta_loss = model.distributed_meta_update(tasks)
           
           if rank == 0 and epoch % 100 == 0:
               print(f"Epoch {epoch}: Meta Loss = {meta_loss:.6f}")
       
       # Cleanup
       cleanup_distributed()
   
   def main():
       """Main function for distributed training."""
       world_size = torch.cuda.device_count()
       print(f"Starting distributed training on {world_size} GPUs")
       
       # Spawn processes for distributed training
       mp.spawn(
           train_distributed,
           args=(world_size,),
           nprocs=world_size,
           join=True
       )
   
   if __name__ == "__main__":
       main()

Key Features Demonstrated:

* Multi-GPU distributed training setup
* Gradient synchronization across processes
* Distributed data loading
* Process coordination and cleanup

Example 7: Performance Benchmarking
------------------------------------

This example shows comprehensive performance benchmarking across multiple methods.

.. literalinclude:: ../examples/performance_benchmarking_demo.py
   :language: python
   :caption: Performance Benchmarking

Key Features Demonstrated:

* Multi-method comparison framework
* Statistical significance testing
* Performance profiling and optimization
* Automated benchmark reporting

Example 8: Theoretical Analysis
-------------------------------

This example demonstrates theoretical analysis of sample complexity and convergence rates.

.. literalinclude:: ../examples/theoretical_analysis_demo.py
   :language: python
   :caption: Theoretical Analysis

Key Features Demonstrated:

* Sample complexity analysis
* Convergence rate computation
* Theoretical bound validation
* Mathematical proof generation

Example 9: Publication Generation
----------------------------------

This example shows how to generate publication-ready materials.

.. literalinclude:: ../examples/publication_demo.py
   :language: python
   :caption: Publication Generation

Key Features Demonstrated:

* Publication-quality plot generation
* LaTeX table creation
* Comprehensive report generation
* Statistical result formatting

Example 10: FEniCSx Integration
-------------------------------

This example demonstrates integration with FEniCSx for high-fidelity ground truth generation.

.. literalinclude:: ../examples/fenicsx_integration_demo.py
   :language: python
   :caption: FEniCSx Integration

Key Features Demonstrated:

* FEniCSx solver integration
* Variable viscosity profile handling
* High-fidelity ground truth generation
* Mesh refinement and convergence studies

Running the Examples
--------------------

To run any of these examples:

1. **Install the package**:

   .. code-block:: bash

      pip install -e .

2. **Navigate to the examples directory**:

   .. code-block:: bash

      cd examples/

3. **Run an example**:

   .. code-block:: bash

      python comprehensive_report_demo.py

4. **For distributed examples**:

   .. code-block:: bash

      python -m torch.distributed.launch --nproc_per_node=4 distributed_training_example.py

Expected Outputs
----------------

Each example generates specific outputs:

* **Trained models**: Saved as `.pth` checkpoint files
* **Results**: Numerical results saved as JSON or HDF5 files
* **Plots**: Publication-ready figures in PDF format
* **Tables**: LaTeX tables for inclusion in papers
* **Reports**: Comprehensive analysis reports in PDF format

The examples are designed to be self-contained and can be modified to suit specific research needs. They demonstrate best practices for using the ML Research Pipeline in various scenarios.

Customization
-------------

All examples can be customized by:

* Modifying configuration files in `configs/`
* Changing model architectures and hyperparameters
* Adding new task types or evaluation metrics
* Extending the analysis and visualization components

For more advanced customization, refer to the :doc:`user_guide` and :doc:`api_reference` sections.
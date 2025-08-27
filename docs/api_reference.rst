API Reference
=============

This section provides detailed documentation for all classes and functions in the ML Research Pipeline.

Core Module
-----------

.. automodule:: ml_research_pipeline.core
   :members:
   :undoc-members:
   :show-inheritance:

Meta-Learning
~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.core.MetaPINN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ml_research_pipeline.core.StandardPINN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ml_research_pipeline.core.TransferLearningPINN
   :members:
   :undoc-members:
   :show-inheritance:

Task Generation
~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.core.FluidTaskGenerator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ml_research_pipeline.core.DatasetManager
   :members:
   :undoc-members:
   :show-inheritance:

Physics Solvers
~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.core.FEniCSxSolver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ml_research_pipeline.core.AnalyticalSolutions
   :members:
   :undoc-members:
   :show-inheritance:

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.core.DistributedMetaPINN
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ml_research_pipeline.core.CheckpointManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ml_research_pipeline.core.TrainingMonitor
   :members:
   :undoc-members:
   :show-inheritance:

Bayesian Module
---------------

.. automodule:: ml_research_pipeline.bayesian
   :members:
   :undoc-members:
   :show-inheritance:

Bayesian Meta-Learning
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.bayesian.BayesianMetaPINN
   :members:
   :undoc-members:
   :show-inheritance:

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.bayesian.UncertaintyCalibrator
   :members:
   :undoc-members:
   :show-inheritance:

Neural Operators Module
-----------------------

.. automodule:: ml_research_pipeline.neural_operators
   :members:
   :undoc-members:
   :show-inheritance:

Fourier Neural Operator
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.neural_operators.InverseFourierNeuralOperator
   :members:
   :undoc-members:
   :show-inheritance:

DeepONet
~~~~~~~~

.. autoclass:: ml_research_pipeline.neural_operators.PhysicsInformedDeepONet
   :members:
   :undoc-members:
   :show-inheritance:

Operator Meta-Learning
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.neural_operators.OperatorMetaPINN
   :members:
   :undoc-members:
   :show-inheritance:

Physics Discovery Module
------------------------

.. automodule:: ml_research_pipeline.physics_discovery
   :members:
   :undoc-members:
   :show-inheritance:

Causal Discovery
~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.physics_discovery.PhysicsCausalDiscovery
   :members:
   :undoc-members:
   :show-inheritance:

Symbolic Regression
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.physics_discovery.NeuralSymbolicRegression
   :members:
   :undoc-members:
   :show-inheritance:

Integrated Discovery
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.physics_discovery.IntegratedPhysicsDiscovery
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Module
-----------------

.. automodule:: ml_research_pipeline.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Benchmark Suite
~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.evaluation.PINNBenchmarkSuite
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Metrics
~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.evaluation.EvaluationMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Method Comparison
~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.evaluation.MethodComparison
   :members:
   :undoc-members:
   :show-inheritance:

Performance Analysis
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.evaluation.PerformanceProfiler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ml_research_pipeline.evaluation.PerformanceRegression
   :members:
   :undoc-members:
   :show-inheritance:

Papers Module
-------------

.. automodule:: ml_research_pipeline.papers
   :members:
   :undoc-members:
   :show-inheritance:

Plot Generation
~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.papers.PaperPlotGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Table Generation
~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.papers.LaTeXTableGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Report Generation
~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.papers.ReportGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Module
--------------------

.. automodule:: ml_research_pipeline.config
   :members:
   :undoc-members:
   :show-inheritance:

Base Configuration
~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.config.BaseConfig
   :members:
   :undoc-members:
   :show-inheritance:

Experiment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.config.ExperimentConfig
   :members:
   :undoc-members:
   :show-inheritance:

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.config.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.config.TrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:

Data Configuration
~~~~~~~~~~~~~~~~~~

.. autoclass:: ml_research_pipeline.config.DataConfig
   :members:
   :undoc-members:
   :show-inheritance:

Utilities Module
----------------

.. automodule:: ml_research_pipeline.utils
   :members:
   :undoc-members:
   :show-inheritance:

Logging Utilities
~~~~~~~~~~~~~~~~~

.. autofunction:: ml_research_pipeline.utils.setup_logging

.. autofunction:: ml_research_pipeline.utils.get_logger

Random Utilities
~~~~~~~~~~~~~~~~

.. autofunction:: ml_research_pipeline.utils.set_random_seeds

.. autofunction:: ml_research_pipeline.utils.get_random_state

I/O Utilities
~~~~~~~~~~~~~

.. autofunction:: ml_research_pipeline.utils.save_results

.. autofunction:: ml_research_pipeline.utils.load_results

.. autofunction:: ml_research_pipeline.utils.ensure_dir

Distributed Utilities
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ml_research_pipeline.utils.setup_distributed

.. autofunction:: ml_research_pipeline.utils.cleanup_distributed

.. autofunction:: ml_research_pipeline.utils.get_world_size

.. autofunction:: ml_research_pipeline.utils.get_rank

Theory Module
-------------

.. automodule:: theory
   :members:
   :undoc-members:
   :show-inheritance:

Sample Complexity
~~~~~~~~~~~~~~~~~

.. autoclass:: theory.sample_complexity.SampleComplexityAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Convergence Analysis
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: theory.convergence_analysis.ConvergenceAnalyzer
   :members:
   :undoc-members:
   :show-inheritance:

Mathematical Proofs
~~~~~~~~~~~~~~~~~~~

.. automodule:: theory.proofs.mathematical_proofs
   :members:
   :undoc-members:
   :show-inheritance:
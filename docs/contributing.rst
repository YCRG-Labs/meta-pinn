Contributing
============

We welcome contributions to the ML Research Pipeline! This guide will help you get started with contributing to the project.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/ml-research-pipeline.git
      cd ml-research-pipeline

3. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install in development mode**:

   .. code-block:: bash

      pip install -e ".[dev]"

5. **Install pre-commit hooks**:

   .. code-block:: bash

      pre-commit install

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make your changes** following the coding standards below
3. **Run tests** to ensure everything works:

   .. code-block:: bash

      pytest tests/

4. **Run code quality checks**:

   .. code-block:: bash

      black ml_research_pipeline/
      flake8 ml_research_pipeline/
      mypy ml_research_pipeline/

5. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "Add feature: description of your changes"

6. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/your-feature-name

7. **Create a pull request** on GitHub

Coding Standards
----------------

Code Style
~~~~~~~~~~

We follow PEP 8 with some modifications:

* **Line length**: 88 characters (Black default)
* **Imports**: Use absolute imports, group by standard library, third-party, local
* **Docstrings**: Use Google-style docstrings
* **Type hints**: Required for all public functions and methods

Example:

.. code-block:: python

   from typing import Dict, List, Optional, Tuple
   import torch
   import numpy as np
   
   from ml_research_pipeline.core import BaseModel
   
   
   class ExampleClass(BaseModel):
       """Example class demonstrating coding standards.
       
       This class shows the expected code style and documentation
       format for the ML Research Pipeline.
       
       Args:
           param1: Description of the first parameter.
           param2: Description of the second parameter.
           
       Attributes:
           attribute1: Description of the first attribute.
           attribute2: Description of the second attribute.
       """
       
       def __init__(self, param1: int, param2: Optional[str] = None) -> None:
           """Initialize the example class."""
           super().__init__()
           self.attribute1 = param1
           self.attribute2 = param2 or "default_value"
       
       def example_method(
           self, 
           input_data: torch.Tensor,
           config: Dict[str, float]
       ) -> Tuple[torch.Tensor, Dict[str, float]]:
           """Example method with proper type hints and docstring.
           
           Args:
               input_data: Input tensor with shape (batch_size, features).
               config: Configuration dictionary with hyperparameters.
               
           Returns:
               A tuple containing:
               - output: Processed tensor with same shape as input.
               - metrics: Dictionary of computed metrics.
               
           Raises:
               ValueError: If input_data has wrong shape.
               KeyError: If required config keys are missing.
           """
           if input_data.dim() != 2:
               raise ValueError(f"Expected 2D input, got {input_data.dim()}D")
           
           # Process the data
           output = self._process_data(input_data, config)
           
           # Compute metrics
           metrics = {
               "mean": float(output.mean()),
               "std": float(output.std()),
           }
           
           return output, metrics
       
       def _process_data(
           self, 
           data: torch.Tensor, 
           config: Dict[str, float]
       ) -> torch.Tensor:
           """Private method for data processing."""
           # Implementation details
           return data * config.get("scale_factor", 1.0)

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

All public classes and functions must have comprehensive docstrings:

.. code-block:: python

   def complex_function(
       param1: torch.Tensor,
       param2: Dict[str, Any],
       param3: Optional[List[str]] = None
   ) -> Tuple[torch.Tensor, Dict[str, float]]:
       """One-line summary of the function.
       
       Longer description explaining what the function does,
       its purpose, and any important implementation details.
       
       Args:
           param1: Description of param1, including shape if tensor.
               Expected shape: (batch_size, sequence_length, features).
           param2: Description of param2, including expected keys.
               Required keys: 'learning_rate', 'batch_size'.
               Optional keys: 'momentum', 'weight_decay'.
           param3: Optional parameter description.
               Defaults to None, which means use default behavior.
               
       Returns:
           A tuple containing:
           - result_tensor: Description of the output tensor.
               Shape: (batch_size, output_features).
           - metrics: Dictionary with computed metrics.
               Keys: 'loss', 'accuracy', 'convergence_rate'.
               
       Raises:
           ValueError: If param1 has incompatible shape.
           KeyError: If param2 missing required keys.
           RuntimeError: If computation fails due to numerical issues.
           
       Example:
           >>> import torch
           >>> data = torch.randn(32, 100, 64)
           >>> config = {'learning_rate': 0.001, 'batch_size': 32}
           >>> result, metrics = complex_function(data, config)
           >>> print(f"Result shape: {result.shape}")
           Result shape: torch.Size([32, 10])
           
       Note:
           This function assumes input data is normalized.
           For best performance, use GPU tensors when available.
       """

Testing Standards
-----------------

Test Structure
~~~~~~~~~~~~~~

Tests are organized in the `tests/` directory with the following structure:

.. code-block::

   tests/
   ├── test_core/
   │   ├── test_meta_pinn.py
   │   ├── test_task_generator.py
   │   └── ...
   ├── test_bayesian/
   │   ├── test_bayesian_meta_pinn.py
   │   └── ...
   ├── test_integration/
   │   ├── test_end_to_end.py
   │   └── ...
   └── conftest.py

Test Guidelines
~~~~~~~~~~~~~~~

1. **Test file naming**: `test_<module_name>.py`
2. **Test function naming**: `test_<functionality>_<condition>`
3. **Use fixtures** for common setup
4. **Test edge cases** and error conditions
5. **Include integration tests** for complex workflows

Example test:

.. code-block:: python

   import pytest
   import torch
   import numpy as np
   
   from ml_research_pipeline.core import MetaPINN
   from ml_research_pipeline.core.task_generator import FluidTaskGenerator
   
   
   class TestMetaPINN:
       """Test suite for MetaPINN class."""
       
       @pytest.fixture
       def model(self):
           """Create a test model."""
           return MetaPINN(
               layers=[2, 32, 32, 3],
               meta_lr=0.001,
               adapt_lr=0.01
           )
       
       @pytest.fixture
       def task_generator(self):
           """Create a test task generator."""
           return FluidTaskGenerator(
               domain_bounds={"x": [0, 1], "y": [0, 1]},
               task_types=["linear_viscosity"]
           )
       
       def test_initialization(self, model):
           """Test model initialization."""
           assert isinstance(model, MetaPINN)
           assert len(model.layers) == 4
           assert model.meta_lr == 0.001
           
       def test_forward_pass(self, model):
           """Test forward pass with valid input."""
           x = torch.randn(10, 2)
           output = model.forward(x)
           
           assert output.shape == (10, 3)
           assert not torch.isnan(output).any()
           
       def test_forward_pass_invalid_input(self, model):
           """Test forward pass with invalid input shape."""
           x = torch.randn(10, 3)  # Wrong input dimension
           
           with pytest.raises(ValueError, match="Expected input dimension 2"):
               model.forward(x)
               
       def test_adaptation(self, model, task_generator):
           """Test task adaptation functionality."""
           task = task_generator.generate_single_task(n_support=20, n_query=30)
           
           # Test adaptation
           adapted_params = model.adapt_to_task(task, adaptation_steps=3)
           
           assert isinstance(adapted_params, dict)
           assert len(adapted_params) > 0
           
           # Test that parameters changed
           original_params = dict(model.named_parameters())
           for name, param in adapted_params.items():
               assert not torch.equal(param, original_params[name])
               
       def test_meta_update(self, model, task_generator):
           """Test meta-learning update."""
           tasks = task_generator.generate_task_batch(batch_size=4)
           
           initial_loss = model.meta_update(tasks)
           
           assert isinstance(initial_loss, float)
           assert initial_loss > 0
           
       @pytest.mark.parametrize("batch_size", [1, 4, 8])
       def test_different_batch_sizes(self, model, task_generator, batch_size):
           """Test meta-learning with different batch sizes."""
           tasks = task_generator.generate_task_batch(batch_size=batch_size)
           loss = model.meta_update(tasks)
           
           assert isinstance(loss, float)
           assert loss > 0

Running Tests
~~~~~~~~~~~~~

Run all tests:

.. code-block:: bash

   pytest

Run specific test file:

.. code-block:: bash

   pytest tests/test_core/test_meta_pinn.py

Run with coverage:

.. code-block:: bash

   pytest --cov=ml_research_pipeline --cov-report=html

Run performance tests:

.. code-block:: bash

   pytest tests/test_performance/ -v

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

* **Clear description** of the problem
* **Steps to reproduce** the issue
* **Expected vs actual behavior**
* **Environment details** (Python version, OS, GPU info)
* **Minimal code example** that reproduces the bug

Feature Requests
~~~~~~~~~~~~~~~~

For new features, please provide:

* **Clear description** of the proposed feature
* **Use case** and motivation
* **Proposed API** or interface design
* **Implementation considerations**

Code Contributions
~~~~~~~~~~~~~~~~~~

We welcome contributions in these areas:

* **New algorithms**: Meta-learning variants, physics-informed methods
* **Neural operators**: New operator architectures
* **Evaluation metrics**: Novel evaluation approaches
* **Optimization**: Performance improvements
* **Documentation**: Tutorials, examples, API docs
* **Testing**: Additional test coverage

Documentation Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation improvements are always welcome:

* **API documentation**: Improve docstrings and examples
* **Tutorials**: Step-by-step guides for specific use cases
* **Theory**: Mathematical foundations and derivations
* **Examples**: Complete working examples
* **User guides**: Best practices and troubleshooting

Review Process
--------------

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

1. **Descriptive title** and detailed description
2. **Link to related issues** if applicable
3. **Include tests** for new functionality
4. **Update documentation** as needed
5. **Ensure CI passes** all checks
6. **Request review** from maintainers

Review Criteria
~~~~~~~~~~~~~~~

Pull requests are evaluated on:

* **Code quality**: Follows coding standards
* **Test coverage**: Adequate test coverage for new code
* **Documentation**: Clear documentation and examples
* **Performance**: No significant performance regressions
* **Compatibility**: Maintains backward compatibility
* **Design**: Fits well with existing architecture

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment. Please:

* **Be respectful** and constructive in discussions
* **Focus on the code**, not the person
* **Accept feedback** gracefully
* **Help others** learn and contribute

Communication
~~~~~~~~~~~~~

* **GitHub Issues**: Bug reports and feature requests
* **Pull Requests**: Code contributions and discussions
* **Discussions**: General questions and ideas

Getting Help
~~~~~~~~~~~~

If you need help:

1. **Check the documentation** first
2. **Search existing issues** for similar problems
3. **Create a new issue** with detailed information
4. **Join discussions** for broader questions

Recognition
-----------

Contributors are recognized through:

* **Contributors list** in the repository
* **Changelog entries** for significant contributions
* **Author attribution** in relevant documentation
* **Acknowledgments** in research publications

Thank you for contributing to the ML Research Pipeline! Your contributions help advance the field of physics-informed machine learning.
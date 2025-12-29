Contributing
============

Thank you for your interest in contributing to OpenEvals.

Development Setup
-----------------

Fork and Clone
^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/openevals.git
   cd openevals

Create Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate

Install Development Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Install Pre-commit Hooks
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Code Style
----------

Python
^^^^^^

- Follow PEP 8 style guide
- Use Black for formatting
- Use isort for import sorting
- Maximum line length: 88 characters

.. code-block:: bash

   black .
   isort .

TypeScript (Frontend)
^^^^^^^^^^^^^^^^^^^^^

- Use ESLint and Prettier
- Follow Next.js conventions

.. code-block:: bash

   cd web/frontend
   npm run lint
   npm run format

Testing
-------

Run All Tests
^^^^^^^^^^^^^

.. code-block:: bash

   pytest tests/

Run with Coverage
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pytest --cov=openevals tests/

Test Specific Module
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pytest tests/test_benchmark.py -v

Project Structure
-----------------

.. code-block:: text

   openevals/
   +-- openevals/
   |   +-- core/           # Benchmark orchestration
   |   +-- tasks/          # Benchmark implementations
   |   +-- utils/          # Utilities and metrics
   |   +-- visualization/  # Charts and reporting
   |   +-- scripts/        # CLI entry points
   +-- web/
   |   +-- backend/        # FastAPI server
   |   +-- frontend/       # Next.js app
   +-- docs/               # Documentation
   +-- tests/              # Test suite
   +-- examples/           # Usage examples

Contribution Guidelines
-----------------------

Issues
^^^^^^

- Check existing issues before creating new ones
- Use issue templates when available
- Provide clear reproduction steps for bugs
- Include system information (OS, Python version, GPU)

Pull Requests
^^^^^^^^^^^^^

1. Create a branch from ``main``:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make changes and commit:

   .. code-block:: bash

      git commit -m "Add: description of change"

3. Push and create PR:

   .. code-block:: bash

      git push origin feature/your-feature-name

4. PR requirements:

   - Clear description of changes
   - Tests for new functionality
   - Documentation updates if needed
   - All tests passing

Commit Message Format
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   <type>: <description>

   [optional body]

Types:

- ``Add`` - New feature
- ``Fix`` - Bug fix
- ``Docs`` - Documentation
- ``Refactor`` - Code refactoring
- ``Test`` - Test additions/changes
- ``Chore`` - Build, CI, dependencies

Adding New Benchmarks
---------------------

1. Create task file in ``openevals/tasks/``:

   .. code-block:: python

      from typing import Dict, Any
      from openevals.core.model_loader import ModelWrapper

      class NewTaskBenchmark:
          def __init__(self, config: Dict[str, Any]):
              self.config = config

          def load_data(self):
              pass

          def evaluate(self, model: ModelWrapper) -> Dict[str, Any]:
              return {"overall": {"accuracy": 0.0}}

2. Register in ``openevals/tasks/__init__.py``

3. Add tests in ``tests/test_new_task.py``

4. Document in ``docs/``

Adding New Models
-----------------

1. Add loader in ``openevals/core/model_loader.py``:

   .. code-block:: python

      class NewModelLoader:
          def load_model(self, size: str, variant: str, **kwargs):
              pass

2. Register model type in configuration schema

3. Add tests and documentation

Questions
---------

- Open a GitHub Discussion: https://github.com/heilcheng/openevals/discussions
- Check existing issues and documentation

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.

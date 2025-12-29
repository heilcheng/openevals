Installation
============

This guide covers the installation process for OpenEvals.

Requirements
------------

Hardware
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Model Size
     - Minimum VRAM
     - Recommended
     - With Quantization
   * - 1-3B
     - 4 GB
     - 8 GB
     - 2 GB
   * - 7-9B
     - 8 GB
     - 16 GB
     - 5 GB
   * - 13-14B
     - 16 GB
     - 24 GB
     - 8 GB
   * - 70B+
     - 40 GB
     - 80 GB+
     - 20 GB

Software
^^^^^^^^

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 50 GB+ disk space (models and datasets)

Basic Installation
------------------

Clone the Repository
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/heilcheng/openevals.git
   cd openevals

Create Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -r requirements.txt

Development Installation
------------------------

For contributors who want to modify the codebase:

.. code-block:: bash

   pip install -e .
   pip install pre-commit
   pre-commit install

Optional Dependencies
---------------------

For additional features:

.. code-block:: bash

   # Visualization support
   pip install matplotlib seaborn plotly

   # Statistical analysis
   pip install scipy statsmodels

   # Web platform
   pip install fastapi uvicorn

Configuration
-------------

Authentication
^^^^^^^^^^^^^^

OpenEvals requires a HuggingFace token to download models and datasets.

**Option 1: Environment Variable**

.. code-block:: bash

   export HF_TOKEN=your_huggingface_token

**Option 2: HuggingFace CLI**

.. code-block:: bash

   huggingface-cli login

To get a token, visit https://huggingface.co/settings/tokens.

Docker Installation
-------------------

For containerized deployment:

.. code-block:: bash

   docker build -t openevals .
   docker run --gpus all -it openevals

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**CUDA not available**

Ensure CUDA drivers are installed and ``nvidia-smi`` shows your GPU.

**Out of memory**

Enable quantization in your configuration:

.. code-block:: yaml

   hardware:
     quantization: true

**Model download fails**

Verify your HuggingFace token has the necessary permissions for gated models.

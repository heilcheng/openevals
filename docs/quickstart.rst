Quick Start
===========

This guide will help you run your first benchmark evaluation.

Prerequisites
-------------

- OpenEvals installed (see :doc:`installation`)
- HuggingFace token configured
- GPU with at least 4GB VRAM (or CPU)

Your First Benchmark
--------------------

Download Benchmark Data
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m openevals.scripts.download_data --all

Run a Simple Evaluation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m openevals.scripts.run_benchmark \
     --config configs/benchmark_config.yaml \
     --models gemma-2b \
     --tasks mmlu \
     --visualize

View Results
^^^^^^^^^^^^

Results are saved to ``results/`` with the following structure:

.. code-block:: text

   results/
   +-- 20250128_143022/
       +-- results.yaml       # Raw results
       +-- summary.json       # Aggregated metrics
       +-- visualizations/    # Charts and graphs
       +-- report.md          # Human-readable report

Using the Python API
--------------------

.. code-block:: python

   from openevals.core.benchmark import Benchmark

   # Initialize
   benchmark = Benchmark("config.yaml")
   benchmark.load_models(["gemma-2b"])
   benchmark.load_tasks(["mmlu"])

   # Run and save
   results = benchmark.run_benchmarks()
   benchmark.save_results("results.yaml")

Configuration Example
---------------------

Create a configuration file ``config.yaml``:

.. code-block:: yaml

   models:
     gemma-2b:
       type: gemma
       size: 2b
       variant: it
       quantization: true

   tasks:
     mmlu:
       type: mmlu
       subset: mathematics
       shot_count: 5

   evaluation:
     runs: 1
     batch_size: auto

   output:
     path: ./results
     visualize: true
     export_formats: [json, yaml]

   hardware:
     device: auto
     precision: bfloat16

Using the Web Interface
-----------------------

For a browser-based experience:

.. code-block:: bash

   # Start backend
   cd web/backend && uvicorn app.main:app --port 8000

   # Start frontend (new terminal)
   cd web/frontend && npm install && npm run dev

Access at http://localhost:3000.

Next Steps
----------

- :doc:`data_loading` - Learn about data loading and processing
- :doc:`evaluation_metrics` - Understand evaluation metrics
- :doc:`api/index` - Detailed API reference

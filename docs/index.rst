OpenEvals Documentation
=======================

**OpenEvals** is an open-source evaluation framework for large language models, providing systematic benchmarking across standardized academic tasks.

.. note::

   This project is developed as part of Google Summer of Code 2025 with Google DeepMind.

Overview
--------

OpenEvals provides infrastructure for:

- Evaluating open-weight models on established benchmarks (MMLU, GSM8K, MATH, HumanEval, ARC, TruthfulQA, and more)
- Comparing performance across model families (Gemma, Llama, Mistral, Qwen, DeepSeek, and arbitrary HuggingFace models)
- Measuring computational efficiency metrics (latency, throughput, memory utilization)
- Generating statistical analyses with confidence intervals
- Producing publication-ready visualizations and reports

Supported Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Family
     - Variants
     - Sizes
   * - Gemma
     - Gemma, Gemma 2, Gemma 3
     - 1B - 27B
   * - Llama 3
     - Llama 3, 3.1, 3.2
     - 1B - 405B
   * - Mistral
     - Mistral, Mixtral
     - 7B, 8x7B, 8x22B
   * - Qwen
     - Qwen 2, Qwen 2.5
     - 0.5B - 72B
   * - DeepSeek
     - DeepSeek, DeepSeek-R1
     - 1.5B - 671B
   * - Phi
     - Phi-3
     - Mini, Small, Medium
   * - HuggingFace
     - Any model on Hub
     - Custom

Supported Benchmarks
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Benchmark
     - Category
     - Description
   * - MMLU
     - Knowledge
     - 57 subjects spanning STEM, humanities, social sciences
   * - GSM8K
     - Mathematical
     - Grade school math word problems
   * - MATH
     - Mathematical
     - Competition math (AMC, AIME, Olympiad)
   * - HumanEval
     - Code Generation
     - Python function completion tasks
   * - ARC
     - Reasoning
     - Science questions (Easy and Challenge splits)
   * - TruthfulQA
     - Truthfulness
     - Questions probing common misconceptions
   * - BBH
     - Reasoning
     - BIG-Bench Hard - 23 challenging tasks

Quick Start
-----------

.. code-block:: bash

   # Install
   git clone https://github.com/heilcheng/openevals.git
   cd openevals && pip install -r requirements.txt

   # Set authentication
   export HF_TOKEN=your_token_here

   # Run evaluation
   python -m openevals.scripts.run_benchmark \
     --config configs/benchmark_config.yaml \
     --models llama3-8b \
     --tasks mmlu gsm8k \
     --visualize

Getting Help
------------

- GitHub Issues: https://github.com/heilcheng/openevals/issues
- Documentation: https://heilcheng.github.io/openevals/

Contributing
------------

Contributions are welcome. Please see the :doc:`contributing` guide for details.

Citation
--------

.. code-block:: bibtex

   @software{openevals,
     author = {Hailey Cheng},
     title = {OpenEvals: An Open-Source Evaluation Framework for Large Language Models},
     year = {2025},
     url = {https://github.com/heilcheng/openevals}
   }

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Core Functionality

   data_loading
   evaluation_metrics
   leaderboard

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

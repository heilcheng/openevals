Public Leaderboard
==================

OpenEvals includes a leaderboard system for tracking and comparing model performance.

Overview
--------

The leaderboard provides:

- Ranked model comparisons across benchmarks
- Historical performance tracking
- Public and private leaderboard modes
- Export capabilities for publications

Using the CLI
-------------

View Current Leaderboard
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m openevals.scripts.leaderboard --task mmlu

Submit Results
^^^^^^^^^^^^^^

.. code-block:: bash

   python -m openevals.scripts.leaderboard --submit results.yaml

Filter by Model Family
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m openevals.scripts.leaderboard --family gemma --task gsm8k

Input Formats
-------------

YAML Format
^^^^^^^^^^^

.. code-block:: yaml

   submission:
     model_name: "gemma-2b-it"
     model_family: "gemma"
     model_size: "2b"

   results:
     mmlu:
       overall: 0.65
       mathematics: 0.58
       computer_science: 0.72
     gsm8k:
       overall: 0.45

JSON Format
^^^^^^^^^^^

.. code-block:: json

   {
     "submission": {
       "model_name": "gemma-2b-it",
       "model_family": "gemma",
       "model_size": "2b"
     },
     "results": {
       "mmlu": {"overall": 0.65},
       "gsm8k": {"overall": 0.45}
     }
   }

Customization
-------------

Custom Ranking
^^^^^^^^^^^^^^

Configure ranking criteria:

.. code-block:: python

   from openevals.leaderboard import Leaderboard

   lb = Leaderboard()
   lb.set_ranking_weights({
       "mmlu": 0.3,
       "gsm8k": 0.2,
       "humaneval": 0.3,
       "arc": 0.2
   })

Filtering Options
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Filter by model size
   lb.filter(min_size="7b", max_size="70b")

   # Filter by date
   lb.filter(after="2025-01-01")

   # Filter by benchmark score
   lb.filter(min_mmlu=0.5)

Deployment
----------

Web Interface
^^^^^^^^^^^^^

The web platform includes an interactive leaderboard:

.. code-block:: bash

   cd web/backend && uvicorn app.main:app --port 8000

Access at http://localhost:8000/api/v1/benchmarks/leaderboard.

API Endpoints
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Endpoint
     - Description
   * - ``GET /leaderboard``
     - Retrieve current rankings
   * - ``POST /leaderboard/submit``
     - Submit new results
   * - ``GET /leaderboard/history``
     - View historical rankings
   * - ``GET /leaderboard/export``
     - Export as CSV/JSON

Export Options
--------------

Export for Publications
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m openevals.scripts.leaderboard --export latex --output table.tex

Available formats:

- LaTeX table
- Markdown table
- CSV
- JSON

Example LaTeX Output
^^^^^^^^^^^^^^^^^^^^

.. code-block:: latex

   \begin{table}[h]
   \centering
   \begin{tabular}{lcccc}
   \toprule
   Model & MMLU & GSM8K & HumanEval & ARC \\
   \midrule
   Llama 3 70B & 0.82 & 0.74 & 0.68 & 0.85 \\
   Gemma 27B & 0.75 & 0.68 & 0.62 & 0.78 \\
   \bottomrule
   \end{tabular}
   \caption{Model performance comparison}
   \end{table}

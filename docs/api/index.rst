API Reference
=============

This section provides detailed documentation for all OpenEvals classes and methods.

Core Modules
------------

GemmaBenchmark
^^^^^^^^^^^^^^

Main orchestration class for running benchmarks.

.. code-block:: python

   from openevals.core.benchmark import GemmaBenchmark

   benchmark = GemmaBenchmark("config.yaml")

**Constructor**

.. code-block:: python

   GemmaBenchmark(config_path: str)

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Type
     - Description
   * - config_path
     - str
     - Path to YAML configuration file

**Raises:**

- ``FileNotFoundError`` - Config file does not exist
- ``yaml.YAMLError`` - Invalid YAML syntax

ModelWrapper
^^^^^^^^^^^^

Unified interface for language models.

.. code-block:: python

   from openevals.core.model_loader import ModelWrapper

   wrapper = ModelWrapper("model-name", model, tokenizer)

Main Classes
------------

GemmaBenchmark Methods
^^^^^^^^^^^^^^^^^^^^^^

load_models
"""""""""""

.. code-block:: python

   load_models(model_names: Optional[List[str]] = None) -> None

Load specified models or all models in config.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Type
     - Description
   * - model_names
     - Optional[List[str]]
     - Models to load. If None, loads all.

load_tasks
""""""""""

.. code-block:: python

   load_tasks(task_names: Optional[List[str]] = None) -> None

Load specified tasks or all tasks in config.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Type
     - Description
   * - task_names
     - Optional[List[str]]
     - Tasks to load. If None, loads all.

run_benchmarks
""""""""""""""

.. code-block:: python

   run_benchmarks() -> Dict[str, Dict[str, Any]]

Run all loaded benchmarks for all loaded models.

**Returns:** Nested dictionary with results per model per task.

.. code-block:: python

   {
       "gemma-2b": {
           "mmlu": {
               "overall": {"accuracy": 0.65, "total": 1000},
               "subjects": {"mathematics": {"accuracy": 0.58}}
           }
       }
   }

save_results
""""""""""""

.. code-block:: python

   save_results(output_path: Optional[str] = None) -> str

Save results to disk.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Type
     - Description
   * - output_path
     - Optional[str]
     - Path to save. If None, generates timestamp-based path.

**Returns:** Path where results were saved.

Evaluation Components
---------------------

BenchmarkTask Base Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All tasks implement this interface:

.. code-block:: python

   class BenchmarkTask:
       def __init__(self, config: Dict[str, Any]):
           """Initialize with configuration."""
           pass

       def load_data(self) -> Any:
           """Load benchmark dataset."""
           pass

       def evaluate(self, model: ModelWrapper) -> Dict[str, Any]:
           """Evaluate model on this task."""
           pass

MMLUBenchmark
^^^^^^^^^^^^^

.. code-block:: python

   from openevals.tasks.mmlu import MMLUBenchmark

   config = {"subset": "mathematics", "shot_count": 5}
   benchmark = MMLUBenchmark(config)
   results = benchmark.evaluate(model_wrapper)

**Configuration:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Option
     - Type
     - Description
   * - subset
     - str
     - Subject subset: "all", "mathematics", etc.
   * - shot_count
     - int
     - Few-shot examples (0-10)

**Returns:**

.. code-block:: python

   {
       "overall": {"correct": 650, "total": 1000, "accuracy": 0.65},
       "subjects": {
           "algebra": {"correct": 45, "total": 50, "accuracy": 0.90}
       }
   }

Gsm8kBenchmark
^^^^^^^^^^^^^^

.. code-block:: python

   from openevals.tasks.gsm8k import Gsm8kBenchmark

   config = {"shot_count": 5, "use_chain_of_thought": True}
   benchmark = Gsm8kBenchmark(config)

**Configuration:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Option
     - Type
     - Description
   * - shot_count
     - int
     - Few-shot examples
   * - use_chain_of_thought
     - bool
     - Enable CoT prompting

HumanevalBenchmark
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from openevals.tasks.humaneval import HumanevalBenchmark

   config = {"timeout": 10, "temperature": 0.2}
   benchmark = HumanevalBenchmark(config)

**Configuration:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Option
     - Type
     - Description
   * - timeout
     - int
     - Execution timeout (seconds)
   * - temperature
     - float
     - Sampling temperature
   * - max_new_tokens
     - int
     - Maximum tokens

Utilities
---------

Metrics Functions
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from openevals.utils.metrics import (
       calculate_accuracy,
       calculate_pass_at_k,
       calculate_confidence_interval,
       aggregate_results
   )

calculate_accuracy
""""""""""""""""""

.. code-block:: python

   calculate_accuracy(correct: int, total: int) -> float

calculate_pass_at_k
"""""""""""""""""""

.. code-block:: python

   calculate_pass_at_k(n_samples: int, n_correct: int, k: int) -> float

calculate_confidence_interval
"""""""""""""""""""""""""""""

.. code-block:: python

   calculate_confidence_interval(
       accuracy: float,
       n_samples: int,
       confidence: float = 0.95
   ) -> Tuple[float, float]

Visualization
-------------

ChartGenerator
^^^^^^^^^^^^^^

.. code-block:: python

   from openevals.visualization.charts import ChartGenerator

   generator = ChartGenerator("output/charts")

create_performance_heatmap
""""""""""""""""""""""""""

.. code-block:: python

   create_performance_heatmap(results: Dict) -> str

Generate heatmap of model x task performance.

create_model_comparison_chart
"""""""""""""""""""""""""""""

.. code-block:: python

   create_model_comparison_chart(results: Dict, task_name: str) -> str

Generate bar chart comparing models on a task.

create_efficiency_comparison_chart
""""""""""""""""""""""""""""""""""

.. code-block:: python

   create_efficiency_comparison_chart(results: Dict) -> Dict[str, str]

Generate efficiency charts (latency, memory, throughput).

Exceptions
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Exception
     - Description
   * - ModelLoadingError
     - Model loading failed
   * - TaskInitializationError
     - Task initialization failed
   * - BenchmarkExecutionError
     - Benchmark execution failed
   * - ConfigurationError
     - Invalid configuration

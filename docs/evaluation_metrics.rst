Evaluation Metrics
==================

OpenEvals implements standard evaluation metrics for LLM benchmarking.

Evaluation Framework
--------------------

The evaluation framework consists of:

1. **Task-specific metrics** - Accuracy, Pass@k, exact match
2. **Efficiency metrics** - Latency, throughput, memory
3. **Statistical analysis** - Confidence intervals, significance tests

Standard Metrics
----------------

Accuracy
^^^^^^^^

For multiple choice tasks (MMLU, ARC, TruthfulQA):

.. code-block:: python

   from openevals.utils.metrics import calculate_accuracy

   accuracy = calculate_accuracy(correct=650, total=1000)
   # Returns: 0.65

Exact Match
^^^^^^^^^^^

For generation tasks (GSM8K, MATH):

.. code-block:: python

   from openevals.utils.metrics import exact_match

   score = exact_match(prediction="42", target="42")
   # Returns: 1.0

Pass@k
^^^^^^

For code generation tasks (HumanEval, MBPP):

.. code-block:: python

   from openevals.utils.metrics import calculate_pass_at_k

   pass_at_1 = calculate_pass_at_k(n_samples=10, n_correct=3, k=1)
   pass_at_10 = calculate_pass_at_k(n_samples=10, n_correct=3, k=10)

Safety Metrics
--------------

TruthfulQA Scoring
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from openevals.utils.metrics import truthfulqa_score

   scores = truthfulqa_score(
       predictions,
       mc1_targets,
       mc2_targets
   )
   # Returns: {"mc1": 0.45, "mc2": 0.52}

Custom Scoring Components
-------------------------

Create custom metrics:

.. code-block:: python

   from openevals.utils.metrics import MetricBase

   class CustomMetric(MetricBase):
       def compute(self, predictions, targets):
           # Implement custom scoring logic
           score = self._calculate_score(predictions, targets)
           return {"custom_score": score}

Statistical Analysis
--------------------

Confidence Intervals
^^^^^^^^^^^^^^^^^^^^

Calculate confidence intervals for accuracy:

.. code-block:: python

   from openevals.utils.metrics import calculate_confidence_interval

   ci_lower, ci_upper = calculate_confidence_interval(
       accuracy=0.65,
       n_samples=1000,
       confidence=0.95
   )
   # Returns: (0.62, 0.68)

Significance Testing
^^^^^^^^^^^^^^^^^^^^

Compare model performances:

.. code-block:: python

   from openevals.utils.metrics import paired_bootstrap_test

   p_value = paired_bootstrap_test(
       model_a_scores,
       model_b_scores,
       n_bootstrap=10000
   )

Aggregating Results
-------------------

Aggregate results across multiple runs:

.. code-block:: python

   from openevals.utils.metrics import aggregate_results

   aggregated = aggregate_results(
       results_list,
       method="mean"  # or "median"
   )

Efficiency Metrics
------------------

Latency
^^^^^^^

Time to generate responses:

.. code-block:: python

   from openevals.utils.metrics import measure_latency

   latency = measure_latency(model, prompts, output_length=128)
   # Returns: {"mean": 0.5, "std": 0.1, "p50": 0.45, "p95": 0.7}

Throughput
^^^^^^^^^^

Tokens generated per second:

.. code-block:: python

   from openevals.utils.metrics import measure_throughput

   throughput = measure_throughput(model, prompts)
   # Returns: {"tokens_per_second": 128.5}

Memory Utilization
^^^^^^^^^^^^^^^^^^

GPU memory usage:

.. code-block:: python

   from openevals.utils.metrics import measure_memory

   memory = measure_memory(model)
   # Returns: {"allocated_gb": 4.2, "reserved_gb": 6.0}

Per-Task Metrics
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Task
     - Primary Metric
     - Additional Metrics
   * - MMLU
     - Accuracy
     - Per-subject accuracy
   * - GSM8K
     - Exact Match
     - Chain-of-thought analysis
   * - MATH
     - Accuracy
     - Per-difficulty scores
   * - HumanEval
     - Pass@1, Pass@10
     - Per-problem pass rates
   * - ARC
     - Accuracy
     - Easy/Challenge splits
   * - TruthfulQA
     - MC1, MC2
     - Truthfulness scores

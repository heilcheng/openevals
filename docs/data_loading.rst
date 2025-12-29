Data Loading and Processing
===========================

OpenEvals provides utilities for loading and processing benchmark datasets.

Supported Datasets
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Dataset
     - Source
     - Description
   * - MMLU
     - HuggingFace
     - 57 subjects, multiple choice
   * - GSM8K
     - HuggingFace
     - 8,500 math word problems
   * - MATH
     - HuggingFace
     - Competition mathematics
   * - HumanEval
     - OpenAI
     - 164 Python problems
   * - ARC
     - AI2
     - Science questions
   * - TruthfulQA
     - HuggingFace
     - Truthfulness evaluation

Downloading Data
----------------

Download all datasets:

.. code-block:: bash

   python -m openevals.scripts.download_data --all

Download specific datasets:

.. code-block:: bash

   python -m openevals.scripts.download_data --datasets mmlu gsm8k

Data Validation
---------------

Validate downloaded data:

.. code-block:: python

   from openevals.utils.data_loader import DataValidator

   validator = DataValidator()
   is_valid = validator.validate_dataset("mmlu")
   print(f"MMLU valid: {is_valid}")

Dataset Statistics
------------------

Get statistics for a dataset:

.. code-block:: python

   from openevals.utils.data_loader import get_dataset_stats

   stats = get_dataset_stats("mmlu")
   print(f"Total samples: {stats['total']}")
   print(f"Subjects: {len(stats['subjects'])}")

Custom Data Loading
-------------------

Load custom datasets:

.. code-block:: python

   from openevals.utils.data_loader import CustomDataLoader

   loader = CustomDataLoader()
   dataset = loader.load_from_json("custom_data.json")

Expected format for custom datasets:

.. code-block:: json

   {
     "samples": [
       {
         "question": "What is 2 + 2?",
         "choices": ["3", "4", "5", "6"],
         "answer": "B"
       }
     ]
   }

Data Preprocessing
------------------

Tokenization
^^^^^^^^^^^^

.. code-block:: python

   from openevals.utils.preprocessing import tokenize_dataset

   tokenized = tokenize_dataset(
       dataset,
       tokenizer,
       max_length=2048
   )

Prompt Formatting
^^^^^^^^^^^^^^^^^

Format prompts for evaluation:

.. code-block:: python

   from openevals.utils.preprocessing import format_prompt

   prompt = format_prompt(
       question=sample["question"],
       choices=sample["choices"],
       shot_examples=few_shot_examples
   )

Caching
-------

Downloaded datasets are cached locally:

.. code-block:: text

   ~/.cache/openevals/
   +-- mmlu/
   +-- gsm8k/
   +-- humaneval/

Clear cache:

.. code-block:: bash

   python -m openevals.scripts.clear_cache --datasets mmlu

# API Reference

This document provides detailed API documentation for the Gemma Benchmarking Suite.

## Core Classes

### GemmaBenchmark

The main orchestration class for running benchmarks.

```python
from gemma_benchmark.core.benchmark import GemmaBenchmark

benchmark = GemmaBenchmark(config_path="config.yaml")
```

#### Methods

##### `__init__(config_path: str)`
Initialize the benchmark with a configuration file.

**Parameters:**
- `config_path` (str): Path to YAML configuration file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

##### `load_models(model_names: Optional[List[str]] = None) -> None`
Load specified models or all models in config.

**Parameters:**
- `model_names` (Optional[List[str]]): List of model names to load. If None, loads all models in config.

**Raises:**
- `ImportError`: If model loader module cannot be imported
- `AttributeError`: If model loader class is not found

##### `load_tasks(task_names: Optional[List[str]] = None) -> None`
Load specified tasks or all tasks in config.

**Parameters:**
- `task_names` (Optional[List[str]]): List of task names to load. If None, loads all tasks in config.

**Raises:**
- `ImportError`: If task module cannot be imported
- `AttributeError`: If task class is not found

##### `run_benchmarks() -> Dict[str, Dict[str, Any]]`
Run all loaded benchmarks for all loaded models.

**Returns:**
- `Dict[str, Dict[str, Any]]`: Nested dictionary with results per model per task

**Example:**
```python
results = {
    "gemma-2b": {
        "mmlu": {
            "overall": {"accuracy": 0.65, "total": 1000},
            "subjects": {"mathematics": {"accuracy": 0.58}}
        }
    }
}
```

##### `save_results(output_path: Optional[str] = None) -> str`
Save results to disk.

**Parameters:**
- `output_path` (Optional[str]): Path to save results. If None, generates timestamp-based path.

**Returns:**
- `str`: Path where results were saved

---

### ModelWrapper

Wrapper class for language models providing a unified interface.

```python
from gemma_benchmark.core.model_loader import ModelWrapper

wrapper = ModelWrapper("model-name", model, tokenizer)
```

#### Methods

##### `__init__(model_name: str, model=None, tokenizer=None)`
Initialize the model wrapper.

**Parameters:**
- `model_name` (str): Name of the model
- `model`: The loaded model object
- `tokenizer`: The tokenizer object

##### `generate(prompt: str, max_new_tokens: int = 100, **kwargs) -> str`
Generate text based on a prompt.

**Parameters:**
- `prompt` (str): Input text prompt
- `max_new_tokens` (int): Maximum number of tokens to generate
- `**kwargs`: Additional generation parameters

**Returns:**
- `str`: Generated text

**Raises:**
- `ValueError`: If model or tokenizer is not loaded

---

### Model Loaders

#### GemmaLoader

```python
from gemma_benchmark.core.model_loader import GemmaLoader

loader = GemmaLoader()
model_wrapper = loader.load_model(size="2b", variant="it")
```

##### `load_model(size: str = "2b", variant: str = "it", cache_dir: Optional[str] = None, quantization: bool = True) -> ModelWrapper`

**Parameters:**
- `size` (str): Model size ("2b", "9b", "27b")
- `variant` (str): Model variant ("it" for instruction-tuned)
- `cache_dir` (Optional[str]): Directory to cache model weights
- `quantization` (bool): Whether to use 4-bit quantization

**Returns:**
- `ModelWrapper`: Wrapped model ready for evaluation

**Raises:**
- `ValueError`: If unsupported model size is specified
- `Exception`: If model loading fails

#### MistralLoader

Similar interface to GemmaLoader but for Mistral models.

```python
from gemma_benchmark.core.model_loader import MistralLoader

loader = MistralLoader()
model_wrapper = loader.load_model(size="7b", variant="instruct")
```

---

## Benchmark Tasks

### Base Task Interface

All benchmark tasks implement a common interface:

```python
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
```

### MMLUBenchmark

```python
from gemma_benchmark.tasks.mmlu import MMLUBenchmark

config = {"subset": "mathematics", "shot_count": 5}
benchmark = MMLUBenchmark(config)
results = benchmark.evaluate(model_wrapper)
```

#### Configuration Options

- `subset` (str): Subject subset ("all", "mathematics", "computer_science", etc.)
- `shot_count` (int): Number of few-shot examples (0-10)

#### Return Format

```python
{
    "overall": {
        "correct": 650,
        "total": 1000, 
        "accuracy": 0.65
    },
    "subjects": {
        "algebra": {"correct": 45, "total": 50, "accuracy": 0.90},
        "geometry": {"correct": 30, "total": 50, "accuracy": 0.60}
    }
}
```

### GSM8KBenchmark

```python
from gemma_benchmark.tasks.gsm8k import Gsm8kBenchmark

config = {"shot_count": 5, "use_chain_of_thought": True}
benchmark = Gsm8kBenchmark(config)
results = benchmark.evaluate(model_wrapper, num_samples=100)
```

#### Configuration Options

- `shot_count` (int): Number of few-shot examples
- `use_chain_of_thought` (bool): Enable chain-of-thought prompting

#### Return Format

```python
{
    "overall": {
        "correct": 75,
        "total": 100,
        "accuracy": 0.75
    },
    "config": {
        "shot_count": 5,
        "use_chain_of_thought": True,
        "num_samples": 100
    },
    "failed_examples": [...]  # Sample of failed examples for analysis
}
```

### HumanEvalBenchmark

```python
from gemma_benchmark.tasks.humaneval import HumanevalBenchmark

config = {"timeout": 10, "temperature": 0.2}
benchmark = HumanevalBenchmark(config)
results = benchmark.evaluate(model_wrapper, num_samples=50)
```

#### Configuration Options

- `timeout` (int): Code execution timeout in seconds
- `temperature` (float): Sampling temperature for generation
- `max_new_tokens` (int): Maximum tokens to generate

#### Return Format

```python
{
    "overall": {
        "passed": 20,
        "total": 50,
        "pass_at_1": 0.40
    },
    "config": {
        "temperature": 0.2,
        "max_new_tokens": 256,
        "timeout": 10
    },
    "failed_examples": [...]
}
```

### EfficiencyBenchmark

```python
from gemma_benchmark.tasks.efficiency import EfficiencyBenchmark

config = {
    "sample_prompts": ["Explain AI", "Write code"],
    "output_lengths": [128, 256, 512]
}
benchmark = EfficiencyBenchmark(config)
results = benchmark.evaluate(model_wrapper)
```

#### Configuration Options

- `sample_prompts` (List[str]): List of prompts for testing
- `output_lengths` (List[int]): List of output lengths to test

#### Return Format

```python
{
    "latency": {
        "tokens_128": 1.25,  # seconds
        "tokens_256": 2.50,
        "tokens_512": 5.00
    },
    "memory_usage": {
        "tokens_128": 0.5,   # GB
        "tokens_256": 0.8,
        "tokens_512": 1.2
    },
    "tokens_per_second": {
        "tokens_128": 102.4,
        "tokens_256": 102.4,
        "tokens_512": 102.4
    },
    "system_info": {
        "os": "Linux",
        "cpu_count": 8,
        "memory_total": 32.0,
        "cuda_available": True,
        "gpu_name": ["NVIDIA RTX 4090"]
    }
}
```

---

## Utilities

### Metrics

```python
from gemma_benchmark.utils.metrics import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_pass_at_k,
    calculate_confidence_interval,
    aggregate_results
)
```

#### Functions

##### `calculate_accuracy(correct: int, total: int) -> float`
Calculate simple accuracy metric.

##### `calculate_f1_score(precision: float, recall: float) -> float`
Calculate F1 score from precision and recall.

##### `calculate_pass_at_k(n_samples: int, n_correct: int, k: int) -> float`
Calculate pass@k for code evaluation tasks.

##### `calculate_confidence_interval(accuracy: float, n_samples: int, confidence: float = 0.95) -> Tuple[float, float]`
Calculate confidence interval for accuracy.

##### `aggregate_results(all_run_results: List[Dict[str, Any]]) -> Dict[str, Any]`
Aggregate results from multiple runs with statistics.

### Data Downloading

```python
from gemma_benchmark.utils.data_downloader import (
    download_mmlu_data,
    download_gsm8k_data,
    download_humaneval_data
)
```

#### Functions

##### `download_mmlu_data(target_dir: str = "data/mmlu", force: bool = False)`
Download MMLU dataset using HuggingFace datasets.

##### `download_gsm8k_data(target_dir: str = "data/gsm8k", force: bool = False)`
Download GSM8K dataset.

##### `download_humaneval_data(target_dir: str = "data/humaneval", force: bool = False)`
Download HumanEval dataset.

---

## Visualization

### ChartGenerator

```python
from gemma_benchmark.visualization.charts import ChartGenerator

generator = ChartGenerator("output/charts")
```

#### Methods

##### `create_performance_heatmap(results: Dict[str, Dict[str, Any]]) -> str`
Generate heatmap of model performance across benchmarks.

##### `create_model_comparison_chart(results: Dict[str, Dict[str, Any]], task_name: str) -> str`
Generate bar chart comparing models on a specific task.

##### `create_efficiency_comparison_chart(results: Dict[str, Dict[str, Any]]) -> Dict[str, str]`
Generate efficiency comparison charts (latency, memory, throughput).

##### `create_subject_breakdown_chart(results: Dict[str, Dict[str, Any]], model_name: str, task_name: str) -> str`
Generate chart showing performance breakdown by subject.

---

## Configuration Schema

### Model Configuration

```yaml
models:
  gemma-2b:
    type: gemma                    # Model type
    size: 2b                       # Model size
    variant: it                    # Model variant
    cache_dir: cache/models        # Cache directory (optional)
    quantization: true             # Enable quantization
    device_map: auto               # Device mapping
    max_memory:                    # Max memory per device (optional)
      0: "15GB"
      1: "15GB"
```

### Task Configuration

```yaml
tasks:
  mmlu:
    type: mmlu
    subset: mathematics            # Subject subset
    shot_count: 5                  # Few-shot examples
    
  gsm8k:
    type: gsm8k
    shot_count: 5
    use_chain_of_thought: true
    
  efficiency:
    type: efficiency
    sample_prompts:
      - "Explain quantum computing"
      - "Write Python code"
    output_lengths: [128, 256, 512]
```

### Evaluation Configuration

```yaml
evaluation:
  runs: 3                          # Number of evaluation runs
  batch_size: auto                 # Batch size or "auto"
  statistical_tests: true          # Enable statistical analysis
  confidence_level: 0.95           # Confidence level
```

### Output Configuration

```yaml
output:
  path: results                    # Output directory
  save_predictions: false          # Save individual predictions
  visualize: true                  # Generate visualizations
  export_formats: [json, yaml]     # Export formats
```

### Hardware Configuration

```yaml
hardware:
  device: auto                     # Device selection
  precision: bfloat16              # Floating point precision
  quantization: true               # Enable quantization
  mixed_precision: true            # Mixed precision
  torch_compile: false             # PyTorch compilation
```

---

## Error Handling

### Common Exceptions

#### `ModelLoadingError`
Raised when model loading fails.

#### `TaskInitializationError`
Raised when task initialization fails.

#### `BenchmarkExecutionError`
Raised when benchmark execution fails.

#### `ConfigurationError`
Raised when configuration is invalid.

### Error Recovery

The framework implements graceful error handling:

1. **Model Loading Failures**: Continue with other models
2. **Task Failures**: Mark as failed and continue
3. **Memory Issues**: Automatic fallback to smaller batch sizes
4. **Network Issues**: Retry with exponential backoff

---

## Examples

### Basic Usage

```python
from gemma_benchmark.core.benchmark import GemmaBenchmark

# Initialize and run
benchmark = GemmaBenchmark("config.yaml")
benchmark.load_models(["gemma-2b"])
benchmark.load_tasks(["mmlu"])
results = benchmark.run_benchmarks()

# Save results
benchmark.save_results("results.yaml")
```

### Custom Task Implementation

```python
from gemma_benchmark.core.model_loader import ModelWrapper
from typing import Dict, Any

class CustomBenchmark:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def evaluate(self, model: ModelWrapper) -> Dict[str, Any]:
        # Your evaluation logic here
        return {"overall": {"accuracy": 0.85}}
```

### Model Comparison

```python
# Compare multiple models
benchmark.load_models(["gemma-2b", "gemma-9b", "mistral-7b"])
benchmark.load_tasks(["mmlu", "gsm8k"])
results = benchmark.run_benchmarks()

# Generate comparison charts
from gemma_benchmark.visualization.charts import ChartGenerator
generator = ChartGenerator("charts")
generator.create_performance_heatmap(results)
```
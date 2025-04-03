# Gemma Benchmarking Suite

A comprehensive benchmarking framework for evaluating Gemma models across a variety of tasks and comparing them with other open source models.

## Overview

This benchmarking suite allows researchers and practitioners to:

- Systematically evaluate Gemma models across standard academic benchmarks (MMLU, etc.)
- Compare different Gemma model sizes and variants
- Benchmark Gemma against other open models like Llama 2 and Mistral
- Generate informative visualizations and reports
- Automate the benchmarking process with reproducible scripts

## Features

- **Modular Architecture**: Easily extend with new models, benchmarks, and visualizations
- **Comprehensive Evaluation**: Benchmark models on knowledge, reasoning, coding, and more
- **Efficiency Metrics**: Measure inference latency, memory usage, and tokens per second
- **Visualization Tools**: Generate charts and reports for easy interpretation
- **Statistical Validation**: Conduct multiple runs and compute confidence intervals

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gemma-benchmark.git
cd gemma-benchmark

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run benchmarks using the default configuration:

```bash
python -m gemma_benchmark.scripts.run_benchmark --config configs/benchmark_config.yaml
```

### Advanced Usage

Specify models and tasks to benchmark:

```bash
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b gemma-7b \
  --tasks mmlu efficiency
```

Generate visualizations:

```bash
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --visualize
```

### Configuration

The benchmarking suite is configured using YAML files. Example configuration:

```yaml
# Model configurations
models:
  gemma-2b:
    type: gemma
    size: 2b
    variant: it
  gemma-7b:
    type: gemma
    size: 7b
    variant: it
  mistral-7b:
    type: mistral
    size: 7b
    variant: instruct

# Task configurations
tasks:
  mmlu:
    type: mmlu
    data_path: data/mmlu
    subset: all
    shot_count: 5
  
  efficiency:
    type: efficiency
    sample_prompts:
      - "Explain the theory of relativity"
      - "Write a short story about a robot who discovers emotions"
```

## Extending the Framework

### Adding a New Model

1. Create a model loader in `gemma_benchmark/core/model_loader.py`
2. Implement the `ModelWrapper` interface

### Adding a New Benchmark

1. Create a new task module in `gemma_benchmark/tasks/`
2. Implement the evaluation logic

## License

MIT

## Citations

If you use this benchmarking suite in your research, please cite:

```
@software{gemma_benchmarking_suite,
  author = {Hailey Cheng},
  title = {Gemma Benchmarking Suite},
  year = {2025},
  url = {https://github.com/heilcheng/gemma-benchmark}
}
```
# Gemma Benchmarking Suite

A systematic evaluation framework for large language models, designed to facilitate reproducible benchmarking across standardized academic tasks.

## Overview

This framework provides infrastructure for:

- Evaluating language models on established benchmarks (MMLU, GSM8K, HumanEval, ARC, TruthfulQA)
- Comparing performance across model families (Gemma, Mistral, Llama, and arbitrary HuggingFace models)
- Measuring computational efficiency metrics (latency, throughput, memory utilization)
- Generating statistical analyses with confidence intervals
- Producing publication-ready visualizations and reports

## Requirements

### Hardware

| Model Size | Minimum VRAM | Recommended | With Quantization |
|------------|--------------|-------------|-------------------|
| 2B         | 4 GB         | 8 GB        | 2 GB              |
| 7-9B       | 8 GB         | 16 GB       | 5 GB              |
| 13B        | 16 GB        | 24 GB       | 8 GB              |
| 27B        | 24 GB        | 32 GB+      | 14 GB             |

### Software

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 50 GB+ disk space (models and datasets)

### Model Access

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the license agreement for gated models (e.g., [Gemma](https://huggingface.co/google/gemma-2-2b))
3. Generate an access token with read permissions

## Installation

### Standard Setup

```bash
git clone https://github.com/heilcheng/gemma-benchmark.git
cd gemma-benchmark

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

### Docker

```bash
docker build -t gemma-benchmark .

docker run --gpus all -v $(pwd)/results:/app/results gemma-benchmark \
  python -m gemma_benchmark.scripts.run_benchmark --config /app/configs/benchmark_config.yaml
```

## Quick Start

```bash
# Set HuggingFace authentication
export HF_TOKEN=your_token_here

# Download benchmark datasets
python -m gemma_benchmark.scripts.download_data --all

# Run evaluation
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b \
  --tasks mmlu \
  --visualize
```

## Supported Benchmarks

| Benchmark   | Category         | Description                                    | Metrics                    |
|-------------|------------------|------------------------------------------------|----------------------------|
| MMLU        | Knowledge        | 57 subjects spanning STEM, humanities, social sciences | Per-subject accuracy       |
| MMLU-Pro    | Knowledge        | Enhanced MMLU with more challenging questions  | Accuracy                   |
| GSM8K       | Mathematical     | Grade school math word problems                | Exact match accuracy       |
| MATH        | Mathematical     | Competition math (AMC, AIME, Olympiad)         | Accuracy                   |
| HumanEval   | Code Generation  | Python function completion tasks               | Pass@k                     |
| MBPP        | Code Generation  | Mostly Basic Python Problems                   | Pass@k                     |
| ARC         | Reasoning        | Science questions (Easy and Challenge splits)  | Multiple choice accuracy   |
| HellaSwag   | Reasoning        | Commonsense reasoning about situations         | Accuracy                   |
| WinoGrande  | Reasoning        | Large-scale Winograd Schema Challenge          | Accuracy                   |
| TruthfulQA  | Truthfulness     | Questions probing common misconceptions        | MC1/MC2 accuracy           |
| GPQA        | Expert Knowledge | Graduate-level physics, biology, chemistry     | Accuracy                   |
| IFEval      | Instruction      | Instruction following evaluation               | Strict/Loose accuracy      |
| BBH         | Reasoning        | BIG-Bench Hard - 23 challenging tasks          | Accuracy                   |
| Efficiency  | Performance      | Computational resource utilization             | Tokens/sec, memory, latency|

## Supported Models

| Family       | Variants                                      | Sizes                              |
|--------------|-----------------------------------------------|-----------------------------------|
| Gemma        | Gemma, Gemma 2                               | 2B, 7B, 9B, 27B                   |
| Gemma 3      | Gemma 3                                      | 1B, 4B, 12B, 27B                  |
| Llama 3      | Llama 3, 3.1, 3.2                            | 1B, 3B, 8B, 70B, 405B             |
| Mistral      | Mistral, Mixtral                             | 7B, 8x7B, 8x22B                   |
| Qwen         | Qwen 2, Qwen 2.5                             | 0.5B - 72B                        |
| DeepSeek     | DeepSeek, DeepSeek-R1                        | 1.5B - 671B                       |
| Phi          | Phi-3                                         | Mini, Small, Medium               |
| OLMo         | OLMo                                         | 1B, 7B                            |
| HuggingFace  | Any model on HuggingFace Hub                 | Custom                            |

### Configuration Examples

```yaml
models:
  gemma-2b:
    type: gemma
    size: 2b
    variant: it  # instruction-tuned

  llama3-8b:
    type: llama3
    size: 8b
    variant: instruct

  qwen2.5-7b:
    type: qwen2.5
    size: 7b
    variant: instruct

  deepseek-r1-7b:
    type: deepseek-r1
    size: 7b

  mistral-7b:
    type: mistral
    size: 7b
    variant: instruct

  custom-model:
    type: huggingface
    model_id: "organization/model-name"
```

## Configuration

```yaml
models:
  gemma-2b:
    type: gemma
    size: 2b
    variant: it
    quantization: true
    cache_dir: ./cache/models

tasks:
  mmlu:
    type: mmlu
    subset: all
    shot_count: 5

  gsm8k:
    type: gsm8k
    shot_count: 8

  efficiency:
    type: efficiency
    output_lengths: [128, 256, 512]

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
```

## Usage

### Command Line

```bash
# Single model, single task
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b \
  --tasks mmlu

# Multi-model comparison
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b gemma-9b mistral-7b \
  --tasks mmlu gsm8k humaneval \
  --visualize
```

### Python API

```python
from gemma_benchmark.core.benchmark import GemmaBenchmark

benchmark = GemmaBenchmark("config.yaml")
benchmark.load_models(["gemma-2b"])
benchmark.load_tasks(["mmlu"])

results = benchmark.run_benchmarks()
benchmark.save_results("results.yaml")
```

### Visualization

```python
from gemma_benchmark.visualization.charts import BenchmarkVisualizer

visualizer = BenchmarkVisualizer("./results", style="publication")
visualizer.create_performance_overview(results)
visualizer.create_efficiency_analysis(results)
visualizer.create_statistical_analysis(results, multi_run_data)
```

## Output Structure

```
results/
├── 20250108_143022/
│   ├── results.yaml
│   ├── summary.json
│   ├── visualizations/
│   │   ├── performance_overview.png
│   │   ├── efficiency_analysis.png
│   │   ├── mmlu_comparison.png
│   │   └── statistical_analysis.png
│   └── executive_summary.md
```

## Web Platform

An interactive web interface is available for browser-based evaluation management:

```bash
# Backend (FastAPI)
cd web/backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (Next.js)
cd web/frontend
npm install
npm run dev
```

See `web/README.md` for detailed documentation.

## Authentication

```bash
# Environment variable
export HF_TOKEN=hf_your_token_here

# Interactive setup
python -m gemma_benchmark.auth

# HuggingFace CLI
huggingface-cli login
```

## Performance Optimization

### Memory

```yaml
models:
  large-model:
    quantization: true
    device_map: auto
    max_memory: {0: "15GB", 1: "15GB"}
```

### Speed

```yaml
hardware:
  precision: bfloat16
  torch_compile: true
  mixed_precision: true
```

## Testing

```bash
pytest tests/
pytest tests/test_core.py -v
pytest --cov=gemma_benchmark tests/
```

## Project Structure

```
gemma-benchmark/
├── gemma_benchmark/
│   ├── core/           # Orchestration and model loading
│   ├── tasks/          # Benchmark implementations
│   ├── utils/          # Metrics, validation, data utilities
│   ├── visualization/  # Charts and reporting
│   └── scripts/        # CLI entry points
├── web/                # Web platform (FastAPI + Next.js)
├── configs/            # Configuration examples
├── tests/              # Test suite
└── examples/           # Usage examples
```

## Troubleshooting

| Issue                          | Solution                                          |
|--------------------------------|---------------------------------------------------|
| `CUDA out of memory`           | Enable quantization or reduce batch size          |
| `Repository not found`         | Verify HF token and model access permissions      |
| `No module named 'flash_attn'` | Optional dependency; can be ignored or installed  |
| Authentication errors          | Ensure HF_TOKEN is set with read permissions      |

## Contributing

### Adding Models

1. Implement loader in `gemma_benchmark/core/model_loader.py`
2. Register the model type
3. Add configuration examples

### Adding Benchmarks

1. Implement task in `gemma_benchmark/tasks/`
2. Inherit from `AbstractBenchmark`
3. Register with `BenchmarkFactory`

## Citation

```bibtex
@software{gemma_benchmark,
  author = {Hailey Cheng},
  title = {Gemma Benchmarking Suite: A Systematic Evaluation Framework for Large Language Models},
  year = {2025},
  url = {https://github.com/heilcheng/gemma-benchmark}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Google Research for the Gemma model family
- HuggingFace for model hosting and datasets infrastructure
- Contributors to the benchmark datasets and evaluation methodologies

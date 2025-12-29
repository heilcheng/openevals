# OpenEvals

[![Google Summer of Code](https://img.shields.io/badge/Google%20Summer%20of%20Code-2025-blue?logo=google&logoColor=white)](https://summerofcode.withgoogle.com/)
[![Organization: Google DeepMind](https://img.shields.io/badge/Organization-Google%20DeepMind-4285F4?logo=google&logoColor=white)](https://deepmind.google/)

An open-source evaluation framework for large language models, providing systematic benchmarking across standardized academic tasks.

**[View Full Documentation](https://heilcheng.github.io/openevals/)**

## Overview

OpenEvals provides infrastructure for:

- Evaluating open-weight models on established benchmarks (MMLU, GSM8K, MATH, HumanEval, ARC, TruthfulQA, and more)
- Comparing performance across model families (Gemma, Llama, Mistral, Qwen, DeepSeek, and arbitrary HuggingFace models)
- Measuring computational efficiency metrics (latency, throughput, memory utilization)
- Generating statistical analyses with confidence intervals
- Producing publication-ready visualizations and reports

## Requirements

### Hardware

| Model Size | Minimum VRAM | Recommended | With Quantization |
|------------|--------------|-------------|-------------------|
| 1-3B       | 4 GB         | 8 GB        | 2 GB              |
| 7-9B       | 8 GB         | 16 GB       | 5 GB              |
| 13-14B     | 16 GB        | 24 GB       | 8 GB              |
| 70B+       | 40 GB        | 80 GB+      | 20 GB             |

### Software

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 50 GB+ disk space (models and datasets)

## Installation

```bash
git clone https://github.com/heilcheng/openevals.git
cd openevals

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Supported Models

| Family       | Variants                                      | Sizes                              |
|--------------|-----------------------------------------------|-----------------------------------|
| Gemma        | Gemma, Gemma 2                               | 2B, 7B, 9B, 27B                   |
| Gemma 3      | Gemma 3                                      | 1B, 4B, 12B, 27B                  |
| Llama 3      | Llama 3, 3.1, 3.2                            | 1B - 405B                         |
| Mistral      | Mistral, Mixtral                             | 7B, 8x7B, 8x22B                   |
| Qwen         | Qwen 2, Qwen 2.5                             | 0.5B - 72B                        |
| DeepSeek     | DeepSeek, DeepSeek-R1                        | 1.5B - 671B                       |
| Phi          | Phi-3                                         | Mini, Small, Medium               |
| OLMo         | OLMo                                         | 1B, 7B                            |
| HuggingFace  | Any model on HuggingFace Hub                 | Custom                            |

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

## Quick Start

```bash
# Set HuggingFace authentication
export HF_TOKEN=your_token_here

# Download benchmark datasets
python -m openevals.scripts.download_data --all

# Run evaluation
python -m openevals.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models llama3-8b \
  --tasks mmlu gsm8k \
  --visualize
```

## Configuration

```yaml
models:
  llama3-8b:
    type: llama3
    size: 8b
    variant: instruct
    quantization: true

  qwen2.5-7b:
    type: qwen2.5
    size: 7b
    variant: instruct

tasks:
  mmlu:
    type: mmlu
    subset: all
    shot_count: 5

  gsm8k:
    type: gsm8k
    shot_count: 8

  math:
    type: math
    shot_count: 4

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

## Python API

```python
from openevals.core.benchmark import Benchmark

benchmark = Benchmark("config.yaml")
benchmark.load_models(["llama3-8b", "qwen2.5-7b"])
benchmark.load_tasks(["mmlu", "gsm8k"])

results = benchmark.run_benchmarks()
benchmark.save_results("results.yaml")
```

## Web Platform

An interactive web interface is available for browser-based evaluation.

Features:
- Real-time benchmark progress via WebSocket
- Model configuration management
- Interactive leaderboard and visualizations
- Support for all benchmark tasks

Quick Start:

```bash
# Backend (FastAPI)
cd web/backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# Frontend (Next.js 15)
cd web/frontend
npm install && npm run dev
```

- Web UI: http://localhost:3000
- API Docs: http://localhost:8000/api/docs

See [web/README.md](web/README.md) for detailed API documentation.

## Output Structure

```
results/
+-- 20250128_143022/
    +-- results.yaml
    +-- summary.json
    +-- visualizations/
    |   +-- performance_overview.png
    |   +-- model_comparison.png
    |   +-- task_breakdown.png
    +-- report.md
```

## Testing

```bash
pytest tests/
pytest --cov=openevals tests/
```

## Project Structure

```
openevals/
+-- openevals/
|   +-- core/           # Orchestration and model loading
|   +-- tasks/          # Benchmark implementations
|   +-- utils/          # Metrics, validation, utilities
|   +-- visualization/  # Charts and reporting
|   +-- scripts/        # CLI entry points
+-- web/                # Web platform (FastAPI + Next.js)
+-- configs/            # Configuration examples
+-- tests/              # Test suite
+-- examples/           # Usage examples
```

## Citation

```bibtex
@software{openevals,
  author = {Cheng Hei Lam},
  title = {OpenEvals: An Open-Source Evaluation Framework for Large Language Models},
  year = {2025},
  url = {https://github.com/heilcheng/openevals}
}
```

## Documentation

Full documentation is available at: **https://heilcheng.github.io/openevals/**

- [Getting Started Guide](https://heilcheng.github.io/openevals/installation.html)
- [Quick Start](https://heilcheng.github.io/openevals/quickstart.html)
- [API Reference](https://heilcheng.github.io/openevals/api/index.html)

## Contributing

Contributions are welcome. Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup
- Code style guidelines
- Pull request process
- Issue reporting

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Google Summer of Code 2025 - This project is developed as part of Google Summer of Code with Google DeepMind
- Model developers: Google, Meta, Mistral AI, Alibaba, DeepSeek, Microsoft, Allen AI
- HuggingFace for model hosting and datasets
- Contributors to benchmark datasets and evaluation methodologies

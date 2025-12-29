# OpenEvals

> This project is developed as part of [Google Summer of Code 2025](https://summerofcode.withgoogle.com/), mentored by [Google DeepMind](https://deepmind.google/).

An open-source evaluation framework for large language models, providing systematic benchmarking across standardized academic tasks.

## Overview

OpenEvals provides infrastructure for:

- Evaluating open-weight models on established benchmarks (MMLU, GSM8K, MATH, HumanEval, ARC, TruthfulQA, and more)
- Comparing performance across model families (Gemma, Llama, Mistral, Qwen, DeepSeek, and arbitrary HuggingFace models)
- Measuring computational efficiency metrics (latency, throughput, memory utilization)
- Generating statistical analyses with confidence intervals
- Producing publication-ready visualizations and reports

**[Read the full documentation](https://heilcheng.github.io/openevals/)**

## Requirements

### Software

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 50 GB+ disk space (models and datasets)

### Hardware

*Rule-of-thumb estimates for dense decoder-only models at moderate context length. Long context and batching can dominate VRAM via KV cache.*

| Model Size | Minimum VRAM | Recommended | With Quantization |
|------------|--------------|-------------|-------------------|
| 1-3B       | 4 GB         | 8 GB        | 2 GB              |
| 7-9B       | 8 GB         | 16 GB       | 5 GB              |
| 13-14B     | 16 GB        | 24 GB       | 8 GB              |
| 70B+       | 40 GB        | 80 GB+      | 20 GB             |

## Installation

```bash
git clone https://github.com/heilcheng/openevals.git
cd openevals

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Benchmark Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OpenEvals Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────────┐│
│  │   Input     │    │   Model Under   │    │      Model Response          ││
│  │  Benchmark  │───▶│     Test        │───▶│   (Generated Answer)         ││
│  │   Dataset   │    │  (HuggingFace)  │    │                              ││
│  └─────────────┘    └─────────────────┘    └──────────────┬───────────────┘│
│                                                            │                │
│                                                            ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Evaluation Engine                                    ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │                                                                         ││
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐││
│  │  │ Knowledge      │  │  Code          │  │     Efficiency             │││
│  │  │ Tasks          │  │  Generation    │  │     Metrics                │││
│  │  │                │  │                │  │                            │││
│  │  │ • MMLU         │  │ • HumanEval    │  │ • Latency (ms)             │││
│  │  │ • TruthfulQA   │  │ • MBPP         │  │ • Throughput (tok/s)       │││
│  │  │ • ARC          │  │ • Pass@k       │  │ • Memory (GB)              │││
│  │  │ • GPQA         │  │                │  │ • GPU utilization          │││
│  │  └────────────────┘  └────────────────┘  └────────────────────────────┘││
│  │                                                                         ││
│  │  ┌────────────────┐  ┌────────────────┐                                ││
│  │  │ Mathematical   │  │  Reasoning     │                                ││
│  │  │ Tasks          │  │  Tasks         │                                ││
│  │  │                │  │                │                                ││
│  │  │ • GSM8K        │  │ • HellaSwag    │                                ││
│  │  │ • MATH         │  │ • WinoGrande   │                                ││
│  │  │ • Chain-of-    │  │ • BBH          │                                ││
│  │  │   Thought      │  │ • IFEval       │                                ││
│  │  └────────────────┘  └────────────────┘                                ││
│  │                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                            │                │
│                                                            ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Output: Scores, Reports, Visualizations              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Supported Models

*Model naming convention: Use HuggingFace model IDs or shorthand names as defined in configuration.*

| Family       | Variants                    | Sizes            | Notes |
|--------------|-----------------------------|------------------|-------|
| Gemma        | Gemma, Gemma 2, Gemma 3     | 1B - 27B         | Google |
| Llama 3      | Llama 3, 3.1, 3.2           | 1B - 405B        | Meta |
| Mistral      | Mistral, Mixtral            | 7B, 8x7B, 8x22B  | Mistral AI |
| Qwen         | Qwen 2, Qwen 2.5            | 0.5B - 72B       | Alibaba |
| DeepSeek     | DeepSeek, DeepSeek-R1       | 1.5B - 671B      | DeepSeek |
| Phi          | Phi-3                       | Mini, Small, Med | Microsoft |
| OLMo         | OLMo                        | 1B, 7B           | Allen AI |
| HuggingFace  | Any model on Hub            | Custom           | Any compatible model |

## Supported Benchmarks

| Benchmark   | Category         | Description                                    | Metrics              |
|-------------|------------------|------------------------------------------------|----------------------|
| MMLU        | Knowledge        | 57 subjects spanning STEM, humanities, social sciences | Per-subject accuracy |
| MMLU-Pro    | Knowledge        | Enhanced MMLU with more challenging questions  | Accuracy             |
| GSM8K       | Mathematical     | Grade school math word problems                | Exact match          |
| MATH        | Mathematical     | Competition math (AMC, AIME, Olympiad)         | Accuracy             |
| HumanEval   | Code Generation  | Python function completion tasks               | Pass@k               |
| MBPP        | Code Generation  | Mostly Basic Python Problems                   | Pass@k               |
| ARC         | Reasoning        | Science questions (Easy and Challenge splits)  | Accuracy             |
| HellaSwag   | Reasoning        | Commonsense reasoning about situations         | Accuracy             |
| WinoGrande  | Reasoning        | Large-scale Winograd Schema Challenge          | Accuracy             |
| TruthfulQA  | Truthfulness     | Questions probing common misconceptions        | MC1/MC2 accuracy     |
| GPQA        | Expert Knowledge | Graduate-level physics, biology, chemistry     | Accuracy             |
| IFEval      | Instruction      | Instruction following evaluation               | Strict/Loose acc     |
| BBH         | Reasoning        | BIG-Bench Hard - 23 challenging tasks          | Accuracy             |

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
    use_chain_of_thought: true

evaluation:
  runs: 1
  batch_size: auto
  statistical_tests: true

output:
  path: ./results
  visualize: true
  export_formats: [json, yaml]

hardware:
  device: auto
  precision: bfloat16
  quantization: true
```

## Python API

```python
from openevals.core.benchmark import Benchmark

# Initialize
benchmark = Benchmark("config.yaml")
benchmark.load_models(["llama3-8b", "qwen2.5-7b"])
benchmark.load_tasks(["mmlu", "gsm8k"])

# Run evaluation
results = benchmark.run_benchmarks()
benchmark.save_results("results.yaml")

# Access results
for model_name, model_results in results.items():
    for task_name, task_results in model_results.items():
        accuracy = task_results["overall"]["accuracy"]
        print(f"{model_name} on {task_name}: {accuracy:.4f}")
```

## Web Platform

An interactive web interface is available for browser-based evaluation:

```bash
# Backend
cd web/backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000

# Frontend
cd web/frontend
npm install && npm run dev
```

Access at http://localhost:3000

Features:
- **Dashboard**: Overview of benchmark runs and statistics
- **Benchmarks**: Configure and run evaluations
- **Models**: Manage model configurations
- **Leaderboard**: Compare model performance across tasks
- **Results**: Visualize scores and export reports

## Output Structure

```
results/
├── 20250128_143022/
│   ├── results.yaml
│   ├── summary.json
│   ├── visualizations/
│   │   ├── performance_overview.png
│   │   ├── model_comparison.png
│   │   └── task_breakdown.png
│   └── report.md
```

## Testing

```bash
pytest tests/
pytest --cov=openevals tests/
```

## Project Structure

```
openevals/
├── openevals/                 # Core Python library
│   ├── core/                  # Orchestration and model loading
│   ├── tasks/                 # Benchmark implementations
│   ├── utils/                 # Metrics, validation, utilities
│   ├── visualization/         # Charts and reporting
│   └── scripts/               # CLI entry points
├── web/                       # Web platform (Next.js + FastAPI)
├── configs/                   # Configuration templates
├── data/                      # Benchmark datasets
├── tests/                     # Test suite
├── docs/                      # Sphinx documentation
└── examples/                  # Usage examples
```

## Documentation

Full documentation is available at: https://heilcheng.github.io/openevals/

## Citation

If you use OpenEvals in your research, please cite:

```bibtex
@software{openevals2025,
  author = {Cheng Hei Lam},
  title = {OpenEvals: An Open-Source Evaluation Framework for Large Language Models},
  year = {2025},
  url = {https://github.com/heilcheng/openevals}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

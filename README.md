# Gemma Benchmarking Suite

A production-ready benchmarking framework for evaluating Google's Gemma models and comparing them with other open-source language models on standard academic benchmarks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/google)

## Overview

This benchmarking suite enables researchers and practitioners to:

- **Evaluate Gemma models** across standard academic benchmarks (MMLU, GSM8K, HumanEval)
- **Compare different model sizes** (2B, 9B, 27B) and variants
- **Benchmark against other models** like Llama 2, Mistral, and Claude
- **Generate comprehensive reports** with statistical significance testing
- **Optimize for your hardware** with automatic quantization and memory management
- **Reproduce results** with deterministic evaluation protocols

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/heilcheng/gemma-benchmark.git
cd gemma-benchmark
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Authenticate with HuggingFace (required for Gemma access)
export HF_TOKEN=your_token_here  # Get from https://huggingface.co/settings/tokens
python -c "from gemma_benchmark.auth import setup_huggingface_auth; setup_huggingface_auth()"

# 4. Download datasets and run benchmark
python -m gemma_benchmark.scripts.download_data --mmlu
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b \
  --tasks mmlu \
  --visualize
```

## 📋 Prerequisites

### Hardware Requirements

| Model Size | Min VRAM | Recommended | Memory (Quantized) | Performance |
|------------|----------|-------------|-------------------|-------------|
| **Gemma 2B** | 4GB | 8GB | ~2GB | RTX 3060+ |
| **Gemma 9B** | 8GB | 16GB | ~5GB | RTX 4070+ |
| **Gemma 27B** | 16GB | 32GB+ | ~14GB | A100/H100 |

*Quantized memory usage assumes 4-bit quantization with double quantization*

### Software Requirements

- **Python 3.8+** with pip
- **CUDA 11.8+** (for GPU acceleration)
- **16GB+ RAM** (for larger models)
- **50GB disk space** (for models + datasets)

### Account Setup

1. **HuggingFace Account**: Create at [huggingface.co](https://huggingface.co)
2. **Gemma License**: Accept at [huggingface.co/google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)
3. **Access Token**: Generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## 🛠️ Installation

### Option 1: Standard Installation

```bash
git clone https://github.com/heilcheng/gemma-benchmark.git
cd gemma-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Development Installation

```bash
git clone https://github.com/heilcheng/gemma-benchmark.git
cd gemma-benchmark

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Option 3: Docker Installation

```bash
# Build image
docker build -t gemma-benchmark .

# Run with GPU support
docker run --gpus all -v $(pwd)/results:/app/results gemma-benchmark \
  python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b \
  --tasks mmlu
```

## 🔐 Authentication

### Environment Variable (Recommended)

```bash
export HF_TOKEN=hf_your_token_here
```

### Interactive Setup

```bash
python -c "from gemma_benchmark.auth import setup_huggingface_auth; setup_huggingface_auth()"
```

### Verify Access

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"
```

## 📊 Usage Examples

### Basic Evaluation

```bash
# Evaluate Gemma 2B on MMLU
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b \
  --tasks mmlu
```

### Multi-Model Comparison

```bash
# Compare multiple models
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/benchmark_config.yaml \
  --models gemma-2b gemma-9b mistral-7b \
  --tasks mmlu efficiency \
  --visualize
```

### Subject-Specific Evaluation

```bash
# Evaluate only mathematics subjects
python -m gemma_benchmark.scripts.run_benchmark \
  --config configs/math_config.yaml \
  --models gemma-9b \
  --tasks mmlu
```

### Custom Configuration

```yaml
# configs/custom_config.yaml
models:
  gemma-2b:
    type: gemma
    size: 2b
    variant: it
    quantization: true
    cache_dir: ./cache

tasks:
  mmlu:
    type: mmlu
    subset: mathematics  # or "all"
    shot_count: 5
    
output:
  path: ./results
  visualize: true
```

## 📈 Supported Benchmarks

| Benchmark | Type | Description | Metrics |
|-----------|------|-------------|---------|
| **MMLU** | Knowledge | 57 subjects, 15K questions | Accuracy per subject |
| **GSM8K** | Math Reasoning | Grade school math | Exact match accuracy |
| **HumanEval** | Code Generation | Python programming | Pass@1, Pass@10 |
| **HellaSwag** | Commonsense | Sentence completion | Accuracy |
| **ARC** | Science | AI2 Reasoning Challenge | Accuracy |
| **TruthfulQA** | Truthfulness | Factual accuracy | Truth score |
| **Efficiency** | Performance | Latency, memory, throughput | Tokens/sec, GB usage |

### Adding Custom Benchmarks

```python
# gemma_benchmark/tasks/my_task.py
from ..core.model_loader import ModelWrapper

class MyTaskBenchmark:
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, model: ModelWrapper):
        # Your evaluation logic
        return {"accuracy": 0.85, "details": {...}}
```

## 🎯 Model Support

### Google Gemma Models

```yaml
models:
  gemma-2b:
    type: gemma
    size: 2b      # 2b, 9b, 27b
    variant: it   # it (instruction-tuned)
    
  gemma-9b:
    type: gemma
    size: 9b
    variant: it
```

### Comparison Models

```yaml
models:
  mistral-7b:
    type: mistral
    size: 7b
    variant: instruct
    
  llama2-7b:
    type: llama
    size: 7b
    variant: chat
```

### Custom Models

```yaml
models:
  custom-model:
    type: huggingface
    model_id: "organization/model-name"
    tokenizer_id: "organization/tokenizer-name"  # optional
```

## ⚡ Performance Optimization

### Memory Optimization

```yaml
# Enable 4-bit quantization (reduces memory by ~75%)
models:
  gemma-9b:
    quantization: true
    quantization_config:
      load_in_4bit: true
      bnb_4bit_quant_type: "nf4"
      bnb_4bit_compute_dtype: "bfloat16"
```

### Speed Optimization

```yaml
# Use Flash Attention and optimized settings
hardware:
  precision: bfloat16
  attn_implementation: "flash_attention_2"
  torch_compile: true
```

### Batch Processing

```yaml
# Optimize batch sizes automatically
evaluation:
  batch_size: "auto"  # or specific number
  max_batch_size: 32
  auto_batch_size: true
```

## 📊 Results and Visualization

### Output Structure

```
results/
├── 20250108_143022/
│   ├── results.yaml              # Raw results
│   ├── summary.json              # Aggregated metrics
│   ├── config.yaml               # Configuration used
│   └── visualizations/
│       ├── performance_heatmap.png
│       ├── mmlu_comparison.png
│       └── efficiency_charts.png
```

### Statistical Analysis

Results include:
- **Confidence intervals** (95% by default)
- **Statistical significance** testing
- **Multiple run aggregation** with variance
- **Subject-level breakdowns** for detailed analysis

### Example Results

```yaml
mmlu:
  overall:
    accuracy: 0.647
    confidence_interval: [0.631, 0.663]
    total_questions: 14042
  subjects:
    mathematics:
      accuracy: 0.423
      questions: 1374
    computer_science:
      accuracy: 0.567
      questions: 668
```

## 🔧 Configuration Options

### Complete Configuration Example

```yaml
# Production configuration
models:
  gemma-2b:
    type: gemma
    size: 2b
    variant: it
    cache_dir: ./cache/models
    quantization: true
    device_map: "auto"
    
  gemma-9b:
    type: gemma
    size: 9b
    variant: it
    cache_dir: ./cache/models
    quantization: true
    max_memory: {0: "15GB", 1: "15GB"}

tasks:
  mmlu:
    type: mmlu
    subset: all
    shot_count: 5
    temperature: 0.0
    max_new_tokens: 10
    
  efficiency:
    type: efficiency
    sample_prompts:
      - "Explain quantum computing"
      - "Write a Python function"
    output_lengths: [128, 256, 512]

evaluation:
  runs: 3
  batch_size: 8
  statistical_tests: true
  confidence_level: 0.95

output:
  path: ./results
  save_predictions: true
  visualize: true
  export_formats: ["json", "csv", "yaml"]

hardware:
  device: auto
  precision: bfloat16
  mixed_precision: true
  gradient_checkpointing: true
```

## 🚨 Troubleshooting

### Authentication Issues

```bash
# Check if authenticated
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"

# Re-authenticate
huggingface-cli login

# Check token permissions
python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['auth']['accessToken']['displayName'])"
```

### Memory Issues

```bash
# Enable quantization
export ENABLE_QUANTIZATION=true

# Reduce batch size
export MAX_BATCH_SIZE=4

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Common Error Solutions

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Enable quantization, reduce batch size, or use smaller model |
| `Repository not found` | Check Gemma license acceptance and HF token |
| `Flash attention not available` | Install: `pip install flash-attn` |
| `Dataset download fails` | Check internet connection, try `--force` flag |
| `Model loading timeout` | Increase timeout in config or use cached models |

### Performance Issues

```bash
# Install optimizations
pip install flash-attn
pip install torch-compile

# Check GPU utilization
nvidia-smi

# Profile memory usage
python -m gemma_benchmark.utils.profile_memory --model gemma-2b
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_tasks.py -v
pytest tests/test_integration.py -v

# Run with coverage
pytest --cov=gemma_benchmark tests/

# Test specific model loading
python tests/manual_test_model_loading.py --model gemma-2b
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/heilcheng/gemma-benchmark.git
cd gemma-benchmark

# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run code formatting
black gemma_benchmark/
isort gemma_benchmark/

# Run linting
flake8 gemma_benchmark/
mypy gemma_benchmark/
```

### Adding New Models

1. Create loader in `gemma_benchmark/core/model_loader.py`
2. Add configuration schema
3. Update documentation
4. Add tests

### Adding New Benchmarks

1. Create task in `gemma_benchmark/tasks/`
2. Implement `evaluate()` method
3. Add configuration options
4. Include in test suite

## 📚 Documentation

- **API Reference**: [docs/api.md](docs/api.md)
- **Configuration Guide**: [docs/configuration.md](docs/configuration.md)
- **Benchmark Details**: [docs/benchmarks.md](docs/benchmarks.md)
- **Performance Tuning**: [docs/optimization.md](docs/optimization.md)
- **Examples**: [examples/](examples/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this benchmarking suite in your research, please cite:

```bibtex
@software{gemma_benchmarking_suite,
  author = {Hailey Cheng},
  title = {Gemma Benchmarking Suite: Evaluation Framework},
  year = {2025},
  url = {https://github.com/heilcheng/gemma-benchmark},
  version = {1.0.0}
}
```

## 🙏 Acknowledgments

- **Google Research** for open-sourcing Gemma models
- **HuggingFace** for model hosting and datasets
- **EleutherAI** for evaluation methodology inspiration
- **Community contributors** for feedback and improvements

---

**⭐ Star this repository if it helps your research!**
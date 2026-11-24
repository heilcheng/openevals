# OpenEvals

Benchmarking suite for open-weight language models, written in Rust. It sends benchmark prompts to any OpenAI-compatible server (vLLM, SGLang, llama.cpp, Ollama, TGI, hosted APIs), scores the replies deterministically, and writes resumable, reproducible results.

The suite was originally a Google Summer of Code 2025 project mentored by Google DeepMind. Version 0.2 is a rewrite in Rust with an updated model registry (November 2026).

## Features

- One HTTP backend for every serving stack that exposes `/v1/chat/completions`.
- Asynchronous generation on tokio with a per-model concurrency limit, retries with exponential backoff and `Retry-After` support.
- Append-only JSONL completion cache. Interrupted runs resume where they stopped, and re-scoring a finished run makes no requests.
- Deterministic scoring: fixed few-shot examples, seeded shuffles, no model-in-the-loop grading.
- Benchmarks: MMLU, MMLU-Pro, GPQA, ARC, TruthfulQA (generation and MC1), GSM8K, MATH-500, HumanEval, and an efficiency benchmark (latency, time to first token, tokens per second).
- Wilson 95 percent confidence intervals, repeated runs with mean and standard deviation, per-subject breakdowns.
- Model registry (`models.toml`) with 68 researched open-weight models: Hugging Face repo, parameters, context length, license, reasoning mode and vendor-recommended sampling. See [docs/MODELS.md](docs/MODELS.md) for sources and open questions.
- Thinking-mode control per model (`chat_template_kwargs.enable_thinking`, `reasoning_effort`, arbitrary `extra_body`).

## Install

Requires Rust 1.88 or newer (the minimum supported version, checked in CI). HumanEval additionally needs `python3` on the PATH.

```bash
git clone https://github.com/heilcheng/openevals.git
cd openevals
cargo install --path crates/openevals-cli
```

Or with Docker:

```bash
docker build -t openevals .
docker run --rm openevals models --family qwen
```

## Quickstart

Start a server for a model from the registry, for example with vLLM:

```bash
vllm serve Qwen/Qwen3-8B --port 8000
```

Download the datasets once (they are cached as JSONL and work offline afterwards). Set `HF_TOKEN` for gated datasets such as GPQA.

```bash
openevals download mmlu gsm8k
```

Run the quickstart configuration:

```bash
openevals run -c configs/quickstart.toml
```

Results are written to `results/quickstart/`:

```
results/quickstart/
  results.json        scores, confidence intervals, per-subject accuracy
  leaderboard.md      ranked table
  samples/<model>/    one JSONL file per task with prompts, replies and scores
  cache/<model>.jsonl completion cache used for resuming
```

Re-running the same command continues from the cache. Render a report at any time:

```bash
openevals report results/quickstart --format md   # or csv, json
```

## Commands

| Command | Purpose |
|---|---|
| `openevals run -c CONFIG [--models M..] [--tasks T..] [--limit N] [--dry-run]` | Run benchmarks |
| `openevals models [--family F] [--format table\|json]` | List the model registry |
| `openevals tasks` | List benchmark types |
| `openevals download [TASK..] [--data-dir DIR]` | Cache datasets |
| `openevals report PATH [--format md\|csv\|json]` | Render a leaderboard |
| `openevals validate -c CONFIG` | Check a configuration |

`--registry FILE` replaces the built-in registry with another `models.toml`.

## Configuration

Run configurations are TOML. Unknown keys are rejected.

```toml
[run]
output = "results/demo"
data_dir = "data"
concurrency = 16        # in-flight requests per model
retries = 5
timeout_secs = 600
runs = 1                # repeated runs for mean and standard deviation
seed = 42

[endpoint]
base_url = "http://localhost:8000/v1"
api_key_env = "OPENAI_API_KEY"   # optional

[[models]]
model = "qwen3-8b"               # registry id
thinking = false                 # optional: chat_template_kwargs.enable_thinking
use_recommended_sampling = false # true uses the vendor sampling from the registry

[[models]]
model = "my-finetune"            # not in the registry
served_name = "org/my-finetune"  # required in that case
base_url = "http://other-host:8000/v1"

[[tasks]]
type = "mmlu"
shots = 5
limit = 500
max_tokens = 1024
```

Task keys: `type`, `name`, `subset`, `shots`, `limit` (a seeded random subset, not the first N rows, because some datasets are sorted by subject), `max_tokens`, `temperature`, `data_file`; HumanEval adds `samples`, `k_values`, `timeout`, `python`; TruthfulQA adds `task_type` (`generation` or `mc1`); efficiency adds `sample_prompts`, `output_lengths`, `warmup_runs`, `measurement_runs`. See `configs/` for complete examples, including Ollama.

By default requests use greedy decoding (temperature 0) so scores are reproducible. Reasoning models usually need a larger `max_tokens` so the final answer is not cut off inside the thinking block. Inline `<think>` blocks and separate `reasoning_content` fields are removed before scoring.

## Benchmarks

| Type | Dataset | Scoring | Primary metric |
|---|---|---|---|
| `mmlu` | `cais/mmlu` | Few-shot multiple choice, letter extraction | `accuracy` |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | 10-option multiple choice | `accuracy` |
| `gpqa` | `Idavidrein/gpqa` (gated) | 4-option multiple choice, shuffled | `accuracy` |
| `arc` | `allenai/ai2_arc` | Multiple choice, Easy or Challenge | `accuracy` |
| `truthfulqa` | `truthfulqa/truthful_qa` | `mc1`: shuffled multiple choice. `generation`: substring and negation heuristic | `accuracy` or `truthfulness_rate` |
| `gsm8k` | `openai/gsm8k` | Numeric match after `####` or boxed answer | `accuracy` |
| `math500` | `HuggingFaceH4/MATH-500` | Normalised match of the final `\boxed{}` answer | `accuracy` |
| `humaneval` | `openai/openai_humaneval` | Unit tests in a sandboxed Python process, unbiased pass@k | `pass_at_1` |
| `efficiency` | built-in prompts | Streaming timing | `tokens_per_second` |

MMLU, GSM8K, HumanEval, ARC and TruthfulQA are kept for continuity with earlier results but are saturated for current models. For new comparisons prefer `mmlu_pro`, `gpqa` and `math500`; the benchmark survey in [docs/MODELS.md](docs/MODELS.md) lists the evaluations vendors report in 2026 and which of them are not yet implemented here (LiveCodeBench, AIME 2026, IFBench, SWE-bench Verified, HLE).

## Supported models

The registry lists the latest generation of each family as of 2026-11-24, with one prior-generation entry where it is still widely used. Every repo ID was verified against the Hugging Face Hub API. Use the `id` column in configs. Any other model can be evaluated by giving `served_name`.

| ID | Family | HF repo | Params (active) | Context | License | Reasoning |
|---|---|---|---|---|---|---|
| `muse-spark-open` | Meta Muse | `meta-models/Muse-Spark-Open-816B-A42B` (gated) | 816B (42B) | 1M | Muse-Model-License | always |
| `muse-glimmer-30b` | Meta Muse | `meta-models/Muse-Glimmer-30B` | 29.6B | 128K | Apache-2.0 | always |
| `llama-3.1-8b-instruct` | Meta Llama | `meta-llama/Llama-3.1-8B-Instruct` (gated) | 8.03B | 128K | Llama-3.1-Community | none |
| `qwen3.8-2.4t-a95b` | Qwen | `Qwen/Qwen3.8-2.4T-A95B` | 2446B (95B) | 256K | Qwen3.8-Max-License | always |
| `qwen3.8-flash-next` | Qwen | `Qwen/Qwen3.8-Flash-Next` (gated) | 125B (6B) | 256K | Qwen-Community-1.0 | hybrid |
| `qwen3.8-27b` | Qwen | `Qwen/Qwen3.8-27B` | 27.8B | 256K | Apache-2.0 | hybrid |
| `qwen3.6-35b-a3b` | Qwen | `Qwen/Qwen3.6-35B-A3B` | 35B (3B) | 256K | Apache-2.0 | hybrid |
| `qwen3.5-397b-a17b` | Qwen | `Qwen/Qwen3.5-397B-A17B` | 397B (17B) | 256K | Apache-2.0 | hybrid |
| `qwen3.5-122b-a10b` | Qwen | `Qwen/Qwen3.5-122B-A10B` | 122B (10B) | 256K | Apache-2.0 | hybrid |
| `qwen3.5-9b` | Qwen | `Qwen/Qwen3.5-9B` | 9B | 256K | Apache-2.0 | hybrid |
| `qwen3.5-4b` | Qwen | `Qwen/Qwen3.5-4B` | 4B | 256K | Apache-2.0 | hybrid |
| `qwen3.5-2b` | Qwen | `Qwen/Qwen3.5-2B` | 2B | 256K | Apache-2.0 | hybrid |
| `qwen3-coder-next` | Qwen | `Qwen/Qwen3-Coder-Next` | 80B (3B) | 256K | Apache-2.0 | none |
| `qwen3-8b` | Qwen | `Qwen/Qwen3-8B` | 8.2B | 32K | Apache-2.0 | hybrid |
| `gemma-4-31b-it` | Gemma | `google/gemma-4-31B-it` | 30.7B | 256K | Apache-2.0 | hybrid |
| `gemma-4-26b-a4b-it` | Gemma | `google/gemma-4-26B-A4B-it` | 25.2B (3.8B) | 256K | Apache-2.0 | hybrid |
| `gemma-4-12b-it` | Gemma | `google/gemma-4-12B-it` | 11.95B | 256K | Apache-2.0 | hybrid |
| `gemma-4-e4b-it` | Gemma | `google/gemma-4-E4B-it` | 8B (4.5B) | 128K | Apache-2.0 | hybrid |
| `gemma-4-e2b-it` | Gemma | `google/gemma-4-E2B-it` | 5.1B (2.3B) | 128K | Apache-2.0 | hybrid |
| `gemma-3-27b-it` | Gemma | `google/gemma-3-27b-it` (gated) | 27.4B | 128K | Gemma | none |
| `mistral-large-4` | Mistral | `mistralai/Mistral-Large-4-1T-A52B` | 1000B (52B) | 1M | Mistral-Ecosystem-Open-Weight | always |
| `mistral-medium-3.5` | Mistral | `mistralai/Mistral-Medium-3.5-128B` | 128B | 256K | Modified-MIT | hybrid |
| `mistral-small-4` | Mistral | `mistralai/Mistral-Small-4-119B-2603` | 119B (6.5B) | 256K | Apache-2.0 | hybrid |
| `ministral-3-14b-instruct` | Mistral | `mistralai/Ministral-3-14B-Instruct-2512` | 14B | 256K | Apache-2.0 | none |
| `ministral-3-8b-instruct` | Mistral | `mistralai/Ministral-3-8B-Instruct-2512` | 8.9B | 256K | Apache-2.0 | none |
| `ministral-3-3b-instruct` | Mistral | `mistralai/Ministral-3-3B-Instruct-2512` | 3.85B | 256K | Apache-2.0 | none |
| `devstral-2-123b` | Mistral | `mistralai/Devstral-2-123B-Instruct-2512` | 123B | 256K | Modified-MIT | none |
| `devstral-small-2-24b` | Mistral | `mistralai/Devstral-Small-2-24B-Instruct-2512` | 24B | 256K | Apache-2.0 | none |
| `deepseek-v4-pro-0813` | DeepSeek | `deepseek-ai/DeepSeek-V4-Pro-0813` | 1600B (49B) | 1M | MIT | hybrid |
| `deepseek-v4.1-flash-1017` | DeepSeek | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | 552B (16B) | 1M | MIT | hybrid |
| `deepseek-v3.2` | DeepSeek | `deepseek-ai/DeepSeek-V3.2` | 685B (37B) | 128K | MIT | hybrid |
| `gpt-oss-120b` | gpt-oss | `openai/gpt-oss-120b` | 117B (5.1B) | 128K | Apache-2.0 | always |
| `gpt-oss-20b` | gpt-oss | `openai/gpt-oss-20b` | 21B (3.6B) | 128K | Apache-2.0 | always |
| `phi-4-reasoning-plus` | Phi | `microsoft/Phi-4-reasoning-plus` | 14.7B | 32K | MIT | always |
| `phi-4` | Phi | `microsoft/phi-4` | 14.7B | 16K | MIT | none |
| `phi-4-mini-instruct` | Phi | `microsoft/Phi-4-mini-instruct` | 3.8B | 128K | MIT | none |
| `olmo-3.1-32b-think` | OLMo | `allenai/Olmo-3.1-32B-Think` | 32.2B | 64K | Apache-2.0 | always |
| `olmo-3.1-32b-instruct` | OLMo | `allenai/Olmo-3.1-32B-Instruct` | 32.2B | 64K | Apache-2.0 | none |
| `olmo-3-7b-think` | OLMo | `allenai/Olmo-3-7B-Think` | 7.3B | 64K | Apache-2.0 | always |
| `olmo-3-7b-instruct` | OLMo | `allenai/Olmo-3-7B-Instruct` | 7.3B | 64K | Apache-2.0 | none |
| `kimi-k3.1-pro` | Kimi | `moonshotai/Kimi-K3.1-Pro` | 2800B (104B) | 1M | Kimi-K3.1-Pro-License | always |
| `kimi-k2.6` | Kimi | `moonshotai/Kimi-K2.6` | 1000B (32B) | 256K | Modified-MIT | hybrid |
| `glm-5.3` | GLM | `zai-org/GLM-5.3` | 744B (40B) | 1M | GLM-5.3-License | always |
| `glm-5.3-flash` | GLM | `zai-org/GLM-5.3-Flash` | 320B (18B) | 1M | MIT | always |
| `glm-4.7-flash` | GLM | `zai-org/GLM-4.7-Flash` | 31.2B (3B) | 198K | MIT | hybrid |
| `minimax-m3.1-flash` | MiniMax | `MiniMaxAI/MiniMax-M3.1-Flash` | 428B (23B) | 1M | MiniMax-Community | always |
| `minimax-m2.7` | MiniMax | `MiniMaxAI/MiniMax-M2.7` | 230B (10B) | 200K | Other-MiniMax-M2.7 | always |
| `granite-4.2-30b` | Granite | `ibm-granite/granite-4.2-30b` | 29.3B | 128K | Apache-2.0 | hybrid |
| `granite-4.2-8b` | Granite | `ibm-granite/granite-4.2-8b` | 8.8B | 128K | Apache-2.0 | hybrid |
| `granite-4.2-3b` | Granite | `ibm-granite/granite-4.2-3b` | 3.7B | 128K | Apache-2.0 | hybrid |
| `nemotron-3-ultra-550b-a55b` | Nemotron | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | 550B (55B) | 1M | OpenMDW-1.1 | hybrid |
| `nemotron-3-super-120b-a12b` | Nemotron | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 120B (12B) | 1M | NVIDIA-Nemotron-Open-Model-License | hybrid |
| `nemotron-3.5-lightning-30b-a3b` | Nemotron | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | 30B (3B) | 1M | OpenMDW-1.1 | hybrid |
| `nemotron-3.5-bolt-74b-a9b` | Nemotron | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16` | 74.9B (9B) | 256K | OpenMDW-1.1 | hybrid |
| `nemotron-3-nano-30b-a3b` | Nemotron | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 30B (3.5B) | 256K | NVIDIA-Nemotron-Open-Model-License | hybrid |
| `smollm3-3b` | SmolLM | `HuggingFaceTB/SmolLM3-3B` | 3.08B | 64K | Apache-2.0 | hybrid |
| `falcon-h1r-7b` | Falcon | `tiiuae/Falcon-H1R-7B` | 7.6B | 256K | Falcon-LLM-License | always |
| `hy4` | Hunyuan | `tencent/Hy4` | 770B (49B) | 1M | Tencent-Hy-License | hybrid |
| `hy4-flash` | Hunyuan | `tencent/Hy4-flash` | 220B (12B) | 1M | Tencent-Hy-License | hybrid |
| `hy3` | Hunyuan | `tencent/Hy3` | 295B (21B) | 256K | Apache-2.0 | hybrid |
| `ernie-5.5-flash-preview` | ERNIE | `baidu/ERNIE-5.5-Flash-Preview` | 280B (14B) | 1M | Apache-2.0 | hybrid |
| `lfm2.5-24b-a2b` | LFM | `LiquidAI/LFM2.5-24B-A2B` | 24B (2B) | 128K | LFM-1.0 | always |
| `lfm3-1.7b` | LFM | `LiquidAI/LFM3-1.7B` | 1.73B | 256K | LFM-1.0 | hybrid |
| `seed-oss-36b-instruct` | Seed-OSS | `ByteDance-Seed/Seed-OSS-36B-Instruct` | 36.2B | 512K | Apache-2.0 | always |
| `command-a-plus-05-2026` | Command A | `CohereLabs/command-a-plus-05-2026-bf16` | 218B (25B) | 128K | Apache-2.0 | hybrid |
| `intern-s2-397b` | InternLM | `internlm/Intern-S2-397B` | 397B (17B) | 256K | Apache-2.0 | hybrid |
| `step-5-preview` | Step | `stepfun-ai/Step-5-Preview` | 600B (27B) | 1M | Other-StepFun-Step-5 | always |
| `step-3.7-flash` | Step | `stepfun-ai/Step-3.7-Flash` | 196B (11B) | 256K | Apache-2.0 | always |

Chat template notes, recommended sampling and sources for each family are in [docs/MODELS.md](docs/MODELS.md).

## Security: HumanEval runs model-generated code

The HumanEval benchmark executes code written by the model on your machine. The built-in sandbox is best effort and is **not container-grade isolation**. It runs each program in a separate `python3 -I` process in an empty temporary directory (deleted afterwards), with stdin closed, a wall-clock timeout after which the process is killed, a static check that rejects imports outside an allowlist and calls such as `eval`, `exec`, `open` and `__import__`, and CPU and address-space limits through `ulimit` where the platform honours them (Linux does; macOS ignores the memory limit). The static check is a regex filter and can be bypassed by determined or adversarial code. There is no network, filesystem or syscall isolation.

Run HumanEval for untrusted or unvetted models inside a container or VM, for example with the provided `Dockerfile` (the image runs as an unprivileged user; add `--cap-drop ALL --security-opt no-new-privileges` and a memory limit to `docker run`), or skip the `humaneval` task.

API keys are read only from the environment variables named by `api_key_env` in the config. They are never written to results, caches or logs.

## Caching

Completions are cached per model in `results/cache/<model>.jsonl`, keyed by the model label, the served model name, `thinking`, `reasoning_effort`, `extra_body`, the full prompt and the sampling parameters. Changing any of these re-queries the server. The key does not include `base_url` or the weights behind it, so delete the cache file if you swap a server for a different quantization of the same model name.

## Migrating from the Python version

| Python version | Rust version |
|---|---|
| Loads weights with transformers in-process | Talks to an OpenAI-compatible server; start the model with vLLM, SGLang, llama.cpp or Ollama |
| YAML config with `type`, `size`, `variant` per model | TOML config referencing registry ids or a `served_name` |
| `results.yaml` and matplotlib charts | `results.json`, `leaderboard.md`, CSV export, per-sample JSONL |
| FastAPI and Next.js web platform | Removed. `results.json` is plain JSON for external dashboards |
| GPU memory, CPU and RAM efficiency metrics | Not measured (the model runs in another process); latency, TTFT and throughput are kept |
| TruthfulQA MC2 | Not ported (needs log-probabilities) |

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The integration test starts a mock OpenAI-compatible server and runs the real binary against it with pre-seeded datasets, including retry on HTTP 429, resume from the cache and report generation. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{openevals2025,
  author = {Cheng Hei Lam},
  title = {OpenEvals: An Open-Source Evaluation Framework for Large Language Models},
  year = {2025},
  url = {https://github.com/heilcheng/openevals}
}
```

## License

MIT. See [LICENSE](LICENSE).

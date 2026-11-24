# Contributing to OpenEvals

## Setup

```bash
git clone https://github.com/heilcheng/openevals.git
cd openevals
cargo build
```

## Checks

All three must pass before a pull request is merged:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Layout

| Crate | Purpose |
|---|---|
| `openevals-core` | Types, run configuration, model registry, metrics, cache, results and reports |
| `openevals-backend` | OpenAI-compatible HTTP client with retries and streaming timing |
| `openevals-tasks` | Benchmarks, scorers, dataset access and the runner |
| `openevals-cli` | The `openevals` binary and the mock-server integration test |

## Adding a benchmark

1. Implement the `Task` trait in `crates/openevals-tasks`: `load` builds prompts and references, `score` is a pure function of the reference and the model reply.
2. Register the type in `build_task` and in `TASK_TYPES` (`openevals-core/src/config.rs`), and add its dataset to `dataset_specs`.
3. Add unit tests for the scorer, including malformed model output.

## Adding a model

Edit `models.toml`. Every `hf_repo` must exist on the Hugging Face Hub (`curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/<repo>` returns 200). Record the source in `docs/MODELS.md`. The registry test checks that IDs are unique and fields are sane.

## Commit messages

Use the imperative mood and a short subject line, for example `Add MMLU-Pro scorer`.

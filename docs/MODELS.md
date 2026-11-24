# Open-weight LLM registry (November 2026)

This document describes the models listed in `models.toml`, the registry the openevals benchmarking tool reads. It records what each model is, how it was verified, and the chat template and sampling details that change benchmark results.

## Scope and date

- Snapshot date: 2026-11-24.
- Coverage: general-purpose open-weight text LLMs, including chat models that also accept images. Excluded: base checkpoints (unless no instruct model exists), quantized repackages (FP8, NVFP4, GGUF, MLX), speculative decoding drafters, embedding, reranker, speech, OCR, image and video models, and guard/safety classifiers.
- Selection rule: the newest generation of each family with its main size variants, plus at most one prior-generation entry when it is still widely used as a baseline.
- Result: 68 models across 23 family groups: Meta Muse and Meta Llama, Qwen, Gemma, Mistral, DeepSeek, gpt-oss, Phi, OLMo, Kimi, GLM, MiniMax, Granite, Nemotron, SmolLM, Falcon, Hunyuan (Hy), ERNIE, LFM, Seed-OSS, Command A, InternLM (Intern-S), Step.

## Method

1. Discovery. Each vendor org was listed through the Hugging Face hub API, newest first, for example `https://huggingface.co/api/models?author=Qwen&sort=createdAt&direction=-1&limit=60`. Orgs listed: Qwen, google, meta-llama, meta-models, mistralai, deepseek-ai, openai, microsoft, allenai, moonshotai, zai-org, MiniMaxAI, ibm-granite, nvidia, HuggingFaceTB, tiiuae, tencent, baidu, internlm, LiquidAI, ByteDance-Seed, stepfun-ai, CohereLabs, xai-org. Keyword searches (for example `search=Nemotron-3`, `search=Muse-Spark`) filled gaps. Meta's 2026 open models are published under the `meta-models` org, not `meta-llama`; this was found by tracing `base_model` metadata on derivative repos.
2. Repo verification. 103 candidate repo IDs were checked with `curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/<id>`. All 103 returned 200; none had to be dropped or corrected. The 68 IDs kept in `models.toml` were checked a second time after the file was written (68 of 68 returned 200) and the `gated` field was compared against the API value (no mismatches). Gated repos: `meta-models/Muse-Spark-Open-816B-A42B` (manual), `meta-llama/Llama-3.1-8B-Instruct` (manual), `google/gemma-3-27b-it` (manual), `Qwen/Qwen3.8-Flash-Next` (auto).
3. Metadata. From the API JSON: `license:*` tags and `cardData.license_name`, `safetensors.total`, `gated`, `createdAt`. From `config.json`: `max_position_embeddings` (or the nested `text_config` value for VLMs) and `rope_scaling`. From `generation_config.json`: shipped default temperature, top_p, top_k.
4. Model cards. Each README was read for total and active parameters, stated context length, license name, thinking controls, system role handling and recommended sampling. For gated repos the rendered model page was read instead of the raw README. Chat templates (`chat_template.jinja` or `tokenizer_config.json`) were inspected for the actual toggle names (`enable_thinking`, `reasoning_effort`, `thinking_mode`, `clear_thinking`, `preserve_thinking`).
5. Precedence. Card statements win over `config.json` and `generation_config.json`. When a card is silent, the generation config default is used and the notes say so. Release month is the card's stated date when given, otherwise the repo creation month.

Registry conventions (also in the header of `models.toml`):

- `[model.sampling]` holds the vendor recommendation for the default or non-thinking mode. For `reasoning = "always"` it is the only table.
- `[model.sampling_thinking]` appears only when the vendor gives different numbers for thinking mode.
- Keys the vendor does not state are omitted. Qwen's recommended `presence_penalty` (1.5 in several modes) and Liquid's `repetition_penalty` are recorded in `notes` because the schema has no key for them.


## Model tables and family notes

Params are total / active in billions. Context is the native window in tokens (K = 1024). Reasoning: none, hybrid (toggleable), always. Release is YYYY-MM.

### Meta Muse

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `muse-spark-open` | [meta-models/Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) (gated) | 816 / 42 | 1024K | Muse-Model-License | always | 2026-10 |
| `muse-glimmer-30b` | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 29.6 / 29.6 | 128K | Apache-2.0 | always | 2026-08 |

Meta's 2026 open-weight releases use the Muse name and the `meta-models` org. No new Llama-branded model has appeared on `meta-llama` since Llama 4 (April 2025).

- Muse Spark Open: 816B total, 42B active, multimodal, 1M context, Muse Model License, gated with manual approval. Reasoning effort is set with the `reasoning_effort` request parameter or `chat_template_kwargs`, values low, medium, high, xhigh; the template default is xhigh. Recommended sampling: temperature 1.0, top_p 1.0, top_k disabled.
- Muse Glimmer 30B: dense 29.6B including a perception encoder, distilled from Muse Spark, Apache-2.0, not gated. Reasoning strength goes in the system prompt as `Reasoning strength: <value>` with values low, medium, high, xhigh. Recommended sampling: temperature 1.0, top_p 0.95, top_k 64. Ships with a DFlash drafter repo for speculative decoding.
- Neither card documents a way to turn reasoning off, so both are recorded as `always`.

Sources: https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B, https://huggingface.co/meta-models/Muse-Glimmer-30B

### Meta Llama

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `llama-3.1-8b-instruct` | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (gated) | 8.03 / 8.03 | 128K | Llama-3.1-Community | none | 2024-07 |

Kept as the single prior-generation baseline. Llama 3.1 8B Instruct uses the Llama 3 header-token template (`<|start_header_id|>role<|end_header_id|>`) and supports a system role. The repo is gated, so `generation_config.json` could not be read anonymously and no sampling is recorded. Llama 4 Scout (109B total, 17B active) and Llama 4 Maverick (400B total, 17B active) remain available under the Llama 4 Community License but were left out under the one-prior-entry rule.

Sources: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct

### Qwen

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `qwen3.8-2.4t-a95b` | [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 2446 / 95 | 256K | Qwen3.8-Max-License | always | 2026-08 |
| `qwen3.8-flash-next` | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (gated) | 125 / 6 | 256K | Qwen-Community-1.0 | hybrid | 2026-08 |
| `qwen3.8-27b` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27.8 / 27.8 | 256K | Apache-2.0 | hybrid | 2026-08 |
| `qwen3.6-35b-a3b` | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35 / 3 | 256K | Apache-2.0 | hybrid | 2026-04 |
| `qwen3.5-397b-a17b` | [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | 397 / 17 | 256K | Apache-2.0 | hybrid | 2026-02 |
| `qwen3.5-122b-a10b` | [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 122 / 10 | 256K | Apache-2.0 | hybrid | 2026-02 |
| `qwen3.5-9b` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9 / 9 | 256K | Apache-2.0 | hybrid | 2026-02 |
| `qwen3.5-4b` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 4 / 4 | 256K | Apache-2.0 | hybrid | 2026-02 |
| `qwen3.5-2b` | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2 / 2 | 256K | Apache-2.0 | hybrid | 2026-02 |
| `qwen3-coder-next` | [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 80 / 3 | 256K | Apache-2.0 | none | 2026-01 |
| `qwen3-8b` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.2 / 8.2 | 32K | Apache-2.0 | hybrid | 2025-04 |

Three 2026 generations coexist. Qwen3.8 (August 2026) has only three open checkpoints (27B dense, Flash-Next 125B-A6B, 2.4T-A95B). Qwen3.6 (April 2026) adds the 35B-A3B MoE. Small sizes (9B, 4B, 2B, 0.8B) exist only in Qwen3.5 (February 2026). Qwen3.6-27B is superseded by Qwen3.8-27B and was omitted.

- Thinking is on by default for every size except Qwen3.5-2B and Qwen3.5-0.8B, which default to non-thinking. Toggle with `chat_template_kwargs: {"enable_thinking": false}` on vLLM/SGLang. Output starts with `<think>\n...</think>\n\n`.
- Qwen3.8 adds `reasoning_effort` (xhigh, medium, low; default xhigh) and `preserve_thinking` (keeps reasoning from all past turns; on by default in 3.8, off by default in 3.6).
- Qwen3.8-2.4T-A95B is text only and thinking cannot be disabled. Recommended output budget: 262,144 tokens for reasoning plus 131,072 for the answer.
- Qwen3.8-Flash-Next is gated (auto approval) under the Qwen Community License 1.0; its parameter count is 125B with 6B active plus a 51B n-gram embedding and 4B MTP head (180B in safetensors).
- Recommended sampling, Qwen3.8: thinking temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0; instruct temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5.
- Qwen3.5 and 3.6: thinking general temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5; thinking precise coding temperature 0.6, presence_penalty 0; instruct temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5. The cards advise keeping at least 128K context so thinking is not truncated.
- Qwen3-Coder-Next (80B-A3B) is non-thinking only; sampling (temperature 1.0, top_p 0.95, top_k 40) comes from its generation config.
- Qwen3-8B (prior) supports the `/think` and `/no_think` soft switches; native context 32K (config 40960), 131K with YaRN.

Sources: https://huggingface.co/Qwen/Qwen3.8-27B, https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B, https://huggingface.co/Qwen/Qwen3.8-Flash-Next, https://huggingface.co/Qwen/Qwen3.6-35B-A3B, https://huggingface.co/Qwen/Qwen3.5-397B-A17B, https://huggingface.co/Qwen/Qwen3.5-2B, https://huggingface.co/Qwen/Qwen3-Coder-Next, https://huggingface.co/Qwen/Qwen3-8B

### Gemma

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `gemma-4-31b-it` | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 30.7 / 30.7 | 256K | Apache-2.0 | hybrid | 2026-03 |
| `gemma-4-26b-a4b-it` | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | 25.2 / 3.8 | 256K | Apache-2.0 | hybrid | 2026-03 |
| `gemma-4-12b-it` | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | 11.95 / 11.95 | 256K | Apache-2.0 | hybrid | 2026-05 |
| `gemma-4-e4b-it` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 8 / 4.5 | 128K | Apache-2.0 | hybrid | 2026-03 |
| `gemma-4-e2b-it` | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 5.1 / 2.3 | 128K | Apache-2.0 | hybrid | 2026-03 |
| `gemma-3-27b-it` | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) (gated) | 27.4 / 27.4 | 128K | Gemma | none | 2025-03 |

Gemma 4 (March 2026, 12B added May 2026) moved to Apache-2.0 and is not gated. Sizes: E2B and E4B (effective parameters via per-layer embeddings; 5.1B and 8B stored), 12B unified (text, image, audio, video input), 26B-A4B MoE (25.2B total, 3.8B active), 31B dense. Context: 128K for E2B/E4B, 256K for the rest.

- Gemma 4 adds a native `system` role (Gemma 3 had none and folded system text into the first user turn).
- Thinking is enabled by putting `<|think|>` at the start of the system prompt (`enable_thinking=True` in the HF template). When thinking is off, all sizes except E2B and E4B still emit an empty thought block. Earlier turns' thoughts must be removed from history, except around tool calls.
- Recommended sampling for all sizes and both modes: temperature 1.0, top_p 0.95, top_k 64.
- Gemma 3 27B IT is the prior entry; it is gated (Gemma license acceptance), so its generation config was not read and no sampling is recorded.

Sources: https://huggingface.co/google/gemma-4-26B-A4B-it, https://huggingface.co/google/gemma-4-12B-it, https://huggingface.co/google/gemma-4-31B-it, https://huggingface.co/google/gemma-3-27b-it

### Mistral

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `mistral-large-4` | [mistralai/Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) | 1000 / 52 | 1024K | Mistral-Ecosystem-Open-Weight | always | 2026-09 |
| `mistral-medium-3.5` | [mistralai/Mistral-Medium-3.5-128B](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) | 128 / 128 | 256K | Modified-MIT | hybrid | 2026-03 |
| `mistral-small-4` | [mistralai/Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) | 119 / 6.5 | 256K | Apache-2.0 | hybrid | 2026-03 |
| `ministral-3-14b-instruct` | [mistralai/Ministral-3-14B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) | 14 / 14 | 256K | Apache-2.0 | none | 2025-12 |
| `ministral-3-8b-instruct` | [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) | 8.9 / 8.9 | 256K | Apache-2.0 | none | 2025-12 |
| `ministral-3-3b-instruct` | [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | 3.85 / 3.85 | 256K | Apache-2.0 | none | 2025-12 |
| `devstral-2-123b` | [mistralai/Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) | 123 / 123 | 256K | Modified-MIT | none | 2025-12 |
| `devstral-small-2-24b` | [mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) | 24 / 24 | 256K | Apache-2.0 | none | 2025-12 |

Mistral's 2026 line merges the instruct and reasoning (formerly Magistral) models into single checkpoints with a per-request `reasoning_effort`.

- Mistral Large 4 (September 2026): about 1T total, 49B active (52B with embeddings), 128 routed experts with 4 active plus 1 shared, 1M context, Mistral Ecosystem Open-Weight License (modified MIT). `reasoning_effort` accepts low, medium, high (default high); the template raises an error for any other value, so there is no off switch. Sampling for all efforts: temperature 1.0, top_p 0.95. vLLM needs `--reasoning-parser mistral --tool-call-parser mistral`.
- Mistral Medium 3.5 (dense 128B, 256K, modified MIT) and Mistral Small 4 (119B MoE, 6.5B active, 256K, Apache-2.0): `reasoning_effort` none or high. With high: temperature 0.7 (and top_p 0.95 for Medium 3.5). With none: temperature 0.0 to 0.7 depending on the task, top_p unset. The registry stores 0.7 for both modes and the range in notes. Mistral Small 4's config allows 1M but the card states 256K.
- Ministral 3 (December 2025; 3B, 8B, 14B) instruct checkpoints: temperature below 0.1 for production use. Separate `-Reasoning-2512` checkpoints use temperature 1.0, a provided `SYSTEM_PROMPT.txt` and keep reasoning traces in context.
- Devstral 2 (123B dense, modified MIT) and Devstral Small 2 (24B, Apache-2.0): coding agents, temperature 0.15. Devstral Small 2's config reports 393,216 positions; the card states 256K.
- Mistral templates use `[SYSTEM_PROMPT]`, `[INST]` and `[THINK]` control tokens; `mistral_common` is the reference tokenizer.
- Magistral Small 2509 and Mistral Small 3.2 are superseded by Mistral Small 4 and are not listed.

Sources: https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B, https://huggingface.co/mistralai/Mistral-Medium-3.5-128B, https://huggingface.co/mistralai/Mistral-Small-4-119B-2603, https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512, https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512, https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512

### DeepSeek

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `deepseek-v4-pro-0813` | [deepseek-ai/DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) | 1600 / 49 | 1024K | MIT | hybrid | 2026-08 |
| `deepseek-v4.1-flash-1017` | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 552 / 16 | 1024K | MIT | hybrid | 2026-10 |
| `deepseek-v3.2` | [deepseek-ai/DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | 685 / 37 | 128K | MIT | hybrid | 2025-12 |

DeepSeek V4 (April 2026 preview) introduced 1M context; V4-Pro-0813 is the official Pro release and V4.1-Flash-1017 is the latest Flash update. The original V4-Flash (284B, 13B active) is superseded by V4.1-Flash.

- No Jinja chat template is shipped for V3.2 or V4. Prompts must be built with the repo's `encoding/` Python reference (`encode_messages`) or the `deepseek-recipe` Rust library. This matters for a Rust benchmarking tool: the prompt format has to be ported, not loaded from the tokenizer.
- Modes: `thinking_mode="thinking"` or `"chat"` (chat mode places `</think>` immediately). V4-Pro-0813 effort levels are low, high, max. V4.1-Flash takes a numeric effort 1 to 100 (default 75), rendered as a `Reasoning Effort:` prefix before the system prompt, only in thinking mode.
- V4.1-Flash: 552B backbone with 8B to 16B activated per token (the registry stores 16), multimodal, 45T training tokens. Recommended sampling temperature 1.0, top_p 0.95 or 1.0, max_tokens at least 256K.
- V4-Pro-0813: 1.6T total, 49B active. temperature 1.0 with top_p 0.95 for agentic use and 1.0 otherwise.
- V3.2 (prior): 685B total, 37B active (V3 architecture), DSA sparse attention, 128K. Recommended temperature 1.0, top_p 0.95. The encoder takes `drop_thinking` and `add_default_bos_token`.

Sources: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813, https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro, https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash, https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017, https://huggingface.co/deepseek-ai/DeepSeek-V3.2

### gpt-oss

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `gpt-oss-120b` | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | 117 / 5.1 | 128K | Apache-2.0 | always | 2025-08 |
| `gpt-oss-20b` | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | 21 / 3.6 | 128K | Apache-2.0 | always | 2025-08 |

No newer open-weight OpenAI LLM exists on the hub; `gpt-oss-safeguard-*` are policy classifiers and are excluded.

- Both models must be prompted in the harmony format (system, developer, user, assistant roles; `analysis` and `final` channels). The HF chat template applies it; raw `generate` calls without it degrade badly.
- Reasoning effort is set in the system prompt as `Reasoning: low|medium|high`; there is no off switch.
- Recommended sampling (from the openai/gpt-oss GitHub README): temperature 1.0, top_p 1.0.
- 120b: 117B total, 5.1B active. 20b: 21B total, 3.6B active. MXFP4 MoE weights, 128K context.

Sources: https://huggingface.co/openai/gpt-oss-120b, https://huggingface.co/openai/gpt-oss-20b, https://github.com/openai/gpt-oss

### Phi

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `phi-4-reasoning-plus` | [microsoft/Phi-4-reasoning-plus](https://huggingface.co/microsoft/Phi-4-reasoning-plus) | 14.7 / 14.7 | 32K | MIT | always | 2025-04 |
| `phi-4` | [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) | 14.7 / 14.7 | 16K | MIT | none | 2024-12 |
| `phi-4-mini-instruct` | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8 / 3.8 | 128K | MIT | none | 2025-02 |

No Phi-5 text model was found. The newest Phi release is Phi-4-reasoning-vision-15B (January 2026), a vision-reasoning model with a 16K context that was left out as not a general chat LLM.

- Phi-4-reasoning-plus: ChatML template and a mandatory long system prompt given on the card. Sampling must be temperature 0.8, top_k 50, top_p 0.95, do_sample true; max_new_tokens 32768 for hard problems.
- Phi-4 (14B): 16K context, template `<|im_start|>role<|im_sep|>`. No vendor sampling recommendation.
- Phi-4-mini-instruct (3.8B, 128K): `<|system|>...<|end|>` format, tools placed in the system turn between `<|tool|>` and `<|/tool|>`. Card examples use greedy decoding.

Sources: https://huggingface.co/microsoft/Phi-4-reasoning-plus, https://huggingface.co/microsoft/phi-4, https://huggingface.co/microsoft/Phi-4-mini-instruct, https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B

### OLMo

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `olmo-3.1-32b-think` | [allenai/Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think) | 32.2 / 32.2 | 64K | Apache-2.0 | always | 2025-12 |
| `olmo-3.1-32b-instruct` | [allenai/Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct) | 32.2 / 32.2 | 64K | Apache-2.0 | none | 2025-12 |
| `olmo-3-7b-think` | [allenai/Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) | 7.3 / 7.3 | 64K | Apache-2.0 | always | 2025-11 |
| `olmo-3-7b-instruct` | [allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | 7.3 / 7.3 | 64K | Apache-2.0 | none | 2025-11 |

Olmo 3 (November 2025) and 3.1 (December 2025) are fully open: weights, data, code and intermediate checkpoints. Separate Think and Instruct checkpoints, no toggle. Native context 65,536 (8K base with 8x YaRN in config).

- Recommended sampling for Think and Instruct: temperature 0.6, top_p 0.95; Think uses max_new_tokens 32768.
- The cards define a default system prompt; Think outputs reasoning ending with `</think>` before the answer.
- Olmo-Hybrid 7B (February 2026, hybrid attention research release, 32K context) was found but not listed.

Sources: https://huggingface.co/allenai/Olmo-3.1-32B-Think, https://huggingface.co/allenai/Olmo-3.1-32B-Instruct, https://huggingface.co/allenai/Olmo-3-7B-Think, https://huggingface.co/allenai/Olmo-Hybrid-Instruct-DPO-7B

### Kimi

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `kimi-k3.1-pro` | [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) | 2800 / 104 | 1024K | Kimi-K3.1-Pro-License | always | 2026-11 |
| `kimi-k2.6` | [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) | 1000 / 32 | 256K | Modified-MIT | hybrid | 2026-04 |

- Kimi K3 (June 2026) and K3.1 Pro (November 2026): 2.8T total, 104B active, Kimi Delta Attention, 1M context, MXFP4 weights from QAT, custom licenses (Kimi K3 License, Kimi K3.1 Pro License). Thinking is always on; `reasoning_effort` is a top-level request field with low, high, max (default max). The models were trained with preserved thinking history, so `reasoning_content` from earlier turns should be passed back. The card gives evaluation settings (temperature 1.0, top_p 0.95 for single-step tasks) rather than a general recommendation; the registry uses those. Kimi K3 is superseded by K3.1 Pro and is not listed separately.
- Kimi K2.6 (prior, 1T total, 32B active, 256K, modified MIT): Thinking mode temperature 1.0, Instant mode temperature 0.6, top_p 0.95. Instant mode via `chat_template_kwargs: {"thinking": false}` on vLLM/SGLang.
- Kimi K2.7-Code (June 2026) is a coding variant without Instant mode; not listed.

Sources: https://huggingface.co/moonshotai/Kimi-K3.1-Pro, https://huggingface.co/moonshotai/Kimi-K3, https://huggingface.co/moonshotai/Kimi-K2.6, https://huggingface.co/moonshotai/Kimi-K2.7-Code

### GLM

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `glm-5.3` | [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | 744 / 40 | 1024K | GLM-5.3-License | always | 2026-08 |
| `glm-5.3-flash` | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 320 / 18 | 1024K | MIT | always | 2026-08 |
| `glm-4.7-flash` | [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | 31.2 / 3 | 198K | MIT | hybrid | 2026-01 |

- GLM-5.3 (August 2026) shares the GLM-5.2 base; GLM-5 was 744B total, 40B active. 1M context. The chat template accepts `reasoning_effort` low or high and maps anything else to max, so reasoning cannot be turned off through the template. `clear_thinking` defaults to false (prior reasoning kept); the card asks for `clear_thinking=true` in plain chat. Sampling comes from the generation config (temperature 1.0, top_p 0.95) and matches the card's evaluation settings.
- GLM-5.3-Flash: 320B total, 18B active, natively multimodal, MIT. Same controls as GLM-5.3.
- GLM-4.7-Flash (prior, 30B-A3B, about 200K context, MIT): card default temperature 1.0, top_p 0.95, max new tokens 131072; Preserved Thinking for multi-turn agent tasks.

Sources: https://huggingface.co/zai-org/GLM-5.3, https://huggingface.co/zai-org/GLM-5.3-Flash, https://huggingface.co/zai-org/GLM-5, https://huggingface.co/zai-org/GLM-4.7-Flash

### MiniMax

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `minimax-m3.1-flash` | [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) | 428 / 23 | 1024K | MiniMax-Community | always | 2026-10 |
| `minimax-m2.7` | [MiniMaxAI/MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) | 230 / 10 | 200K | Other-MiniMax-M2.7 | always | 2026-04 |

- MiniMax-M3 (June 2026) and M3.1-Flash (October 2026): about 428B total, 23B activated, multimodal, 1M context, MiniMax Community license. M3 exposes `thinking_mode` enabled, disabled, adaptive in its template (so M3 itself is hybrid). M3.1-Flash instead exposes `reasoning_effort` low, medium, high, xhigh, max rendered as an `<effort>` tag; the card does not describe disabling reasoning, so it is recorded as `always`. Recommended sampling: temperature 1.0, top_p 0.95. M3.1-Flash's safetensors total (220B) is about half the stated parameter count, which suggests packed low-bit weights.
- MiniMax-M2.7 (prior, text only, 200K): an interleaved thinking model; earlier `<think>...</think>` blocks must stay in the history. Recommended temperature 1.0, top_p 0.95, top_k 40, plus a default system prompt from the card. Total and active parameters (230B, 10B) are taken from the MiniMax-M2 card since the M2.7 card does not restate them.
- MiniMax-H3 is an omni-modal video generation system and is out of scope.

Sources: https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash, https://huggingface.co/MiniMaxAI/MiniMax-M3, https://huggingface.co/MiniMaxAI/MiniMax-M2.7, https://huggingface.co/MiniMaxAI/MiniMax-M2

### Granite

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `granite-4.2-30b` | [ibm-granite/granite-4.2-30b](https://huggingface.co/ibm-granite/granite-4.2-30b) | 29.3 / 29.3 | 128K | Apache-2.0 | hybrid | 2026-08 |
| `granite-4.2-8b` | [ibm-granite/granite-4.2-8b](https://huggingface.co/ibm-granite/granite-4.2-8b) | 8.8 / 8.8 | 128K | Apache-2.0 | hybrid | 2026-08 |
| `granite-4.2-3b` | [ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b) | 3.7 / 3.7 | 128K | Apache-2.0 | hybrid | 2026-08 |

Granite 4.2 (August 2026): dense 3B, 8B and 30B reasoning models, Apache-2.0, 128K native context with extension to 512K.

- Three modes: full thinking (default, `enable_thinking=True`), non-thinking (`enable_thinking=False`), low effort (`reasoning_effort="low"`). Earlier turns' reasoning is truncated from history by default.
- The card states temperature 1.0 and top_p 0.95 are required for all modes, including tool calling.
- Granite 4.1 (April 2026) and the Granite Switch 4.1 models are superseded and not listed.

Sources: https://huggingface.co/ibm-granite/granite-4.2-30b, https://huggingface.co/ibm-granite/granite-4.2-8b, https://huggingface.co/ibm-granite/granite-4.2-3b

### Nemotron

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `nemotron-3-ultra-550b-a55b` | [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) | 550 / 55 | 1024K | OpenMDW-1.1 | hybrid | 2026-06 |
| `nemotron-3-super-120b-a12b` | [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 120 / 12 | 1024K | NVIDIA-Nemotron-Open-Model-License | hybrid | 2026-03 |
| `nemotron-3.5-lightning-30b-a3b` | [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | 30 / 3 | 1024K | OpenMDW-1.1 | hybrid | 2026-08 |
| `nemotron-3.5-bolt-74b-a9b` | [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16) | 74.9 / 9 | 256K | OpenMDW-1.1 | hybrid | 2026-10 |
| `nemotron-3-nano-30b-a3b` | [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 30 / 3.5 | 256K | NVIDIA-Nemotron-Open-Model-License | hybrid | 2025-12 |

Nemotron 3 (Nano December 2025, Super March 2026, Ultra June 2026) and 3.5 (Lightning August 2026, Bolt and Super VL October 2026) are hybrid Mamba-Transformer MoE models trained from scratch.

- Thinking toggle: `chat_template_kwargs: {"enable_thinking": true|false}`, default true. Nemotron 3 Ultra additionally needs `force_nonempty_content: true` when calling with tools on the chat completions endpoint.
- Sampling: Super, Ultra, Lightning and Bolt cards recommend temperature 1.0, top_p 0.95 for every task and mode. Nemotron 3 Nano recommends temperature 1.0, top_p 1.0 for reasoning and temperature 0.6, top_p 0.95 for tool calling.
- Licenses: Nemotron 3 Nano and Super use the NVIDIA Nemotron Open Model License; Ultra and the 3.5 models use OpenMDW-1.1.
- Context: cards state up to 1M for Ultra, Super and Lightning even where `config.json` lists 262,144; Bolt 74B-A9B states 256K.
- Bolt 74B-A9B is a vision-language model pruned from Nemotron 3.5 Super VL for local agent use. Bolt E4B is built on the Gemma 4 E4B architecture (64K) and Nemotron 3 Nano 4B also exists; both were left out for size balance.

Sources: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16, https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16, https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16, https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

### SmolLM

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `smollm3-3b` | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3.08 / 3.08 | 64K | Apache-2.0 | hybrid | 2025-07 |

SmolLM3-3B (July 2025) is still the newest SmolLM chat model. Extended thinking is on by default; disable with `/no_think` in the system prompt or `enable_thinking=False`. Recommended temperature 0.6, top_p 0.95. Config is set for 64K; 128K needs a YaRN `rope_scaling` edit.

Sources: https://huggingface.co/HuggingFaceTB/SmolLM3-3B

### Falcon

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `falcon-h1r-7b` | [tiiuae/Falcon-H1R-7B](https://huggingface.co/tiiuae/Falcon-H1R-7B) | 7.6 / 7.6 | 256K | Falcon-LLM-License | always | 2026-01 |

Falcon-H1R-7B is a hybrid Mamba-Transformer reasoning model under the Falcon LLM license. It always emits a `<think>...</think>` block. Recommended temperature 0.6, top_p 0.95, max new tokens up to 65536. The release month (2026-01) is inferred from the citation year; the repo was created in October 2025. Falcon-H1 Instruct (May 2025, 0.5B to 34B) is the older non-reasoning line and is not listed.

Sources: https://huggingface.co/tiiuae/Falcon-H1R-7B, https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct

### Hunyuan

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `hy4` | [tencent/Hy4](https://huggingface.co/tencent/Hy4) | 770 / 49 | 1024K | Tencent-Hy-License | hybrid | 2026-10 |
| `hy4-flash` | [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) | 220 / 12 | 1024K | Tencent-Hy-License | hybrid | 2026-11 |
| `hy3` | [tencent/Hy3](https://huggingface.co/tencent/Hy3) | 295 / 21 | 256K | Apache-2.0 | hybrid | 2026-07 |

Tencent renamed Hunyuan to Hy in 2026.

- Hy4 (October 2026): 770B backbone, 49B active, multimodal, 1M context, built-in DFly2 drafter, Tencent Hy license. `reasoning_effort` accepts only no_think or high (default high); other values raise a template error.
- Hy4 flash (November 2026): 220B total including 64B Engram parameters (155.5B backbone), 12B active, 1M context. Same reasoning switch.
- Hy3 (prior, July 2026): 295B total, 21B active, 256K, Apache-2.0. `reasoning_effort` no_think (default), low, high.
- Recommended sampling for all three: temperature 0.9, top_p 1.0 (generation config also sets top_k -1, meaning disabled).

Sources: https://huggingface.co/tencent/Hy4, https://huggingface.co/tencent/Hy4-flash, https://huggingface.co/tencent/Hy3

### ERNIE

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `ernie-5.5-flash-preview` | [baidu/ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) | 280 / 14 | 1024K | Apache-2.0 | hybrid | 2026-11 |

ERNIE 5.5 Flash Preview (November 2026): 280B total, 14B active MoE, 1M context, Apache-2.0. Thinking is on by default and disabled with `enable_thinking: false`; when on, `reasoning_effort` is low, medium or high (default high). The card does not state sampling; the registry uses the generation config (temperature 1.0, top_p 0.95). ERNIE 4.5 (2025) checkpoints remain available but were not listed.

Sources: https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview, https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking

### LFM

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `lfm2.5-24b-a2b` | [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) | 24 / 2 | 128K | LFM-1.0 | always | 2026-10 |
| `lfm3-1.7b` | [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | 1.73 / 1.73 | 256K | LFM-1.0 | hybrid | 2026-10 |

Liquid AI models use a hybrid convolution and attention architecture and the LFM Open License 1.0.

- LFM2.5-24B-A2B (October 2026): 24B total, 2B active, 128K, always thinks. Recommended temperature 0.1, top_k 50. The card says it is weak on knowledge-heavy tasks without retrieval.
- LFM3-1.7B (October 2026): 256K context, reasoning on by default; `enable_thinking=False` works but the card warns it can be inconsistent. General use: temperature 0.1, top_k 50, repetition_penalty 1.1. Math and STEM: temperature 1.0, top_p 0.95, top_k 20 (stored as `sampling_thinking`).
- LFM2.5-1.2B Instruct and Thinking (January 2026, 32K) and LFM2.5-8B-A1B also exist and were left out.

Sources: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B, https://huggingface.co/LiquidAI/LFM3-1.7B, https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking

### Seed-OSS

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `seed-oss-36b-instruct` | [ByteDance-Seed/Seed-OSS-36B-Instruct](https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct) | 36.2 / 36.2 | 512K | Apache-2.0 | always | 2025-08 |

Seed-OSS-36B-Instruct (August 2025) is still the only open Seed chat LLM. Dense 36B, 512K native context, Apache-2.0. Thinking uses `<seed:think>` tags and a `thinking_budget` template argument; with no budget the model thinks without a limit, and budgets should be multiples of 512. Recommended temperature 1.1, top_p 0.95 (temperature 1, top_p 0.7 for tau-bench).

Sources: https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct

### Command A

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `command-a-plus-05-2026` | [CohereLabs/command-a-plus-05-2026-bf16](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16) | 218 / 25 | 128K | Apache-2.0 | hybrid | 2026-05 |

Command A+ (May 2026): sparse MoE, 218B total, 25B active, 128 experts with 8 active, 128K input and 64K output, Apache-2.0, not gated. Published as `-bf16`, `-fp8` and `-w4a4` repos; there is no repo without a suffix, so the registry points at the BF16 one. Reasoning is on by default and turned off with `reasoning=false` or `reasoning_effort="none"`; thoughts appear between `<|START_THINKING|>` and `<|END_THINKING|>`. vLLM needs `--tool-call-parser cohere_command4 --reasoning-parser cohere_command4`. Sampling (temperature 0.6, top_p 0.95) is taken from the card's example code, not a stated recommendation. The older command-a-reasoning-08-2025 is CC-BY-NC and gated, and is not listed.

Sources: https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16, https://huggingface.co/CohereLabs/command-a-reasoning-08-2025

### InternLM

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `intern-s2-397b` | [internlm/Intern-S2-397B](https://huggingface.co/internlm/Intern-S2-397B) | 397 / 17 | 256K | Apache-2.0 | hybrid | 2026-09 |

Shanghai AI Lab's current general model is Intern-S2-397B (September 2026), a scientific multimodal model on the Qwen3.5-397B-A17B MoE architecture (the `model_type` is `qwen3_5_moe`; the 17B active figure is inferred from that architecture). Thinking is on by default and disabled with `enable_thinking=False`. Recommended temperature 0.8, top_p 0.95, top_k 50, min_p 0. The InternLM-branded chat line has not had a new release since 2025.

Sources: https://huggingface.co/internlm/Intern-S2-397B

### Step

| id | repo | params (total / active) | context | license | reasoning | release |
|---|---|---|---|---|---|---|
| `step-5-preview` | [stepfun-ai/Step-5-Preview](https://huggingface.co/stepfun-ai/Step-5-Preview) | 600 / 27 | 1024K | Other-StepFun-Step-5 | always | 2026-09 |
| `step-3.7-flash` | [stepfun-ai/Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) | 196 / 11 | 256K | Apache-2.0 | always | 2026-05 |

- Step 5 Preview (September 2026): 600B total, 27B active, up to 1M context, custom license (LICENSE.txt). Recommended temperature 1.0, top_p 0.95, top_k 0 (disabled). The template writes `Reasoning: <effort>` into the system turn and supports `preserve_thinking`; the card does not list effort values or an off switch.
- Step 3.7 Flash (May 2026): 196B language backbone plus 1.8B vision encoder, about 11B active, 256K, Apache-2.0. Reasoning levels low, medium, high. No sampling stated and no generation config shipped.

Sources: https://huggingface.co/stepfun-ai/Step-5-Preview, https://huggingface.co/stepfun-ai/Step-3.7-Flash

## Evaluation benchmarks in current use for open models (Nov 2026)

Method: benchmark names were counted across the 66 model cards in the candidate set whose repos were created in 2026, then spot-checked in the result tables of the DeepSeek V4, Qwen3.5/3.8, Gemma 4, Nemotron 3 and GLM-5.3 cards. Dataset IDs were verified with `curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/datasets/<id>`; only IDs that returned 200 are listed. Counts are the number of 2026 cards that mention the benchmark at least once (a loose signal, since some mentions are in comparison tables or training data lists).

### Saturated or legacy (still seen, mostly in base-model tables)

| Benchmark | HF dataset (verified) | Status |
|---|---|---|
| MMLU | `cais/mmlu` | Saturated. DeepSeek V4 base models score 87.8 to 90.1. Reported for base checkpoints, rarely for chat models. |
| MMLU-Redux | `edinburgh-dawg/mmlu-redux-2.0` | Near saturated (about 90 on V4-Pro base); appears in 11 cards. |
| GSM8K | `openai/gsm8k` | Saturated (above 90 for base models). Base-model tables only. |
| HumanEval | `openai/openai_humaneval` | Saturated for chat models; only 4 of the 2026 cards mention it, and mostly for base models. |
| ARC (AI2 Reasoning Challenge) | `allenai/ai2_arc` | Saturated; not reported in 2026 chat model cards. Distinct from ARC-AGI. |
| TruthfulQA | `truthfulqa/truthful_qa` | Legacy; not reported in 2026 chat model cards. |
| HellaSwag | `Rowan/hellaswag` | Saturated; base-model or ablation use only. |
| MATH-500 | `HuggingFaceH4/MATH-500` | Near saturated for reasoning models; only 4 of the 2026 cards. |
| IFEval | `google/IFEval` | Still reported (19 cards) but compressed near the top; IFBench is replacing it. |
| AIME 2025 | `MathArena/aime_2025`, `opencompass/AIME2025` | Near saturated for frontier open models (Nemotron 3 Super reports about 90 without tools); vendors are moving to AIME 2026 and HMMT. |

### Current standards

| Benchmark | What it measures | HF dataset (verified) | 2026 cards mentioning |
|---|---|---|---|
| GPQA Diamond | graduate-level science MCQ | `Idavidrein/gpqa` (gated, auto) | 41 |
| Humanity's Last Exam (HLE), with and without tools | frontier knowledge and reasoning | `cais/hle` (gated, auto) | 37 |
| Terminal-Bench 2.x / 3.0 | agentic terminal tasks | `harborframework/terminal-bench-2.0` (community mirror; card says its leaderboard is unofficial) | 33 |
| LiveCodeBench (v6 and later windows) | contamination-limited competitive coding | `livecodebench/code_generation_lite` | 28 |
| IFBench | out-of-distribution instruction following | `allenai/IFBench_test` | 25 |
| MMLU-Pro | harder 10-option knowledge MCQ | `TIGER-Lab/MMLU-Pro` | 23 |
| MMMU | multimodal reasoning (VLM-capable models) | `MMMU/MMMU` | 21 |
| BFCL (v3/v4) | function calling | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | 18 |
| HMMT (Feb 2025, Feb 2026) | competition math | `MathArena/hmmt_feb_2025`, `MathArena/hmmt_feb_2026` | 16 |
| DeepSWE v1.1 | long-horizon software engineering | none found on HF | 16 |
| SWE-bench Verified | repository bug fixing | `princeton-nlp/SWE-bench_Verified`, `SWE-bench/SWE-bench_Verified` | 15 |
| AA-LCR | long-context reasoning | `ArtificialAnalysis/AA-LCR` | 15 |
| BrowseComp | agentic web browsing | none verified (`openai/BrowseComp` returns 401) | 14 |
| tau2-bench (and tau3) | tool-agent-user conversations | `HuggingFaceH4/tau2-bench-data` (domain data mirror; official harness is on GitHub) | 13 |
| MRCR | multi-round long-context retrieval | `openai/mrcr` | 13 |
| SuperGPQA | broad graduate-level MCQ | `m-a-p/SuperGPQA` | 12 |
| AIME 2026 | competition math, post-cutoff for most models | `MathArena/aime_2026` | 12 |
| GDPval (GDPval-AA) | economically valuable knowledge work | `openai/gdpval` | 12 |
| SWE-bench Pro | harder, less contaminated SWE tasks | `ScaleAI/SWE-bench_Pro` | 9 |
| SWE-bench Multilingual | SWE tasks beyond Python | `SWE-bench/SWE-bench_Multilingual` | not counted |
| MMMLU | multilingual MMLU | `openai/MMMLU` | not counted |
| LongBench v2 | long-context understanding | `zai-org/LongBench-v2` | not counted |
| SimpleQA / SimpleQA Verified | short-form factuality | `basicv8vc/SimpleQA` (community copy), `google/simpleqa-verified` | 6 |
| Arena-Hard (v2) | open-ended chat quality, LLM judge | `lmarena-ai/arena-hard-auto` | 5 |
| SciCode | scientific coding | `SciCode1/SciCode` | 13 |
| Codeforces | competitive programming rating | `open-r1/codeforces` (problem set, not an official rating harness) | not counted |

Other agentic suites that appear in 2026 cards but have no verified HF dataset: OSWorld, NL2Repo, Toolathlon, MCP-Atlas, SWE-Marathon, ProgramBench, CyberGym, Agents' Last Exam. These are harness-based (Docker or VM environments) rather than static datasets, so a Rust runner would need to drive the external harness.

Practical notes for openevals:

- Static, cheap and still discriminative: MMLU-Pro, GPQA Diamond, SuperGPQA, IFBench, AIME 2026, HMMT Feb 2026, LiveCodeBench (recent window), HLE (text subset), MRCR, AA-LCR.
- Reasoning models need large output budgets (vendors evaluate with 64K to 256K max tokens) and vendor sampling; greedy decoding is not the vendor setting for any 2026 reasoning model in this registry.
- Several vendors report results on corrected or "Verified" variants (HLE-Verified, refined SWE-bench Pro), so published numbers are not always comparable with the public dataset.

## Open questions

- Release dates use the card's stated month where given and the repo creation month otherwise. Falcon-H1R-7B (repo October 2025, citation 2026) and the Ministral 3 / Devstral 2 models (repos October to November 2025, version suffix 2512) are the least certain.
- Reasoning mode is recorded as `always` when the card and template show only effort levels without an off switch: Muse Spark Open, Muse Glimmer, Mistral Large 4, GLM-5.3, GLM-5.3-Flash, MiniMax-M3.1-Flash, Kimi K3.1 Pro, Step 5 Preview, Step 3.7 Flash, Seed-OSS-36B. Some of these may accept a "none" effort or a zero thinking budget that is not documented.
- Active parameter counts not stated on the card and taken from related sources: GLM-5.3 (40B, from the GLM-5 card), MiniMax-M2.7 (10B, from the MiniMax-M2 card), Intern-S2-397B (17B, inferred from the Qwen3.5-397B-A17B architecture), DeepSeek V4.1-Flash (stored as the upper bound, 16B, of the stated 8B to 16B range).
- Total parameter counts that disagree with the safetensors total: MiniMax-M3.1-Flash (card 428B, safetensors 220B), DeepSeek V4.1-Flash (552B backbone, safetensors 763B including vision and other weights), Mistral Large 4 (safetensors total not reported by the API).
- Context lengths where card and config disagree: Nemotron 3 Ultra, Super and Lightning (card up to 1M, config 262,144), Mistral Small 4 (config 1M, card 256K), Devstral Small 2 (config 393,216, card 256K), Command A+ (config 200,000, card 128K input). The registry uses the card value except where the card value is an extension.
- Sampling taken from evaluation settings or example code rather than an explicit recommendation: Kimi K3.1 Pro, Command A+. Taken from `generation_config.json`: GLM-5.3, ERNIE 5.5 Flash Preview, Qwen3-Coder-Next, Olmo 3 / 3.1 Instruct.
- Sampling missing: Llama 3.1 8B and Gemma 3 27B (generation config behind the gate), Phi-4, Step 3.7 Flash.
- Licenses recorded as short names; several are custom (Muse Model License, Qwen3.8-Max, Qwen Community 1.0, Kimi K3.1 Pro, GLM-5.3, MiniMax Community, Tencent Hy, OpenMDW-1.1, LFM 1.0, Mistral Ecosystem Open-Weight, StepFun Step 5). Their commercial terms were not reviewed.
- Qwen3.8-Flash-Next reports `model_type` `qwen4_exp`; runtime support outside recent transformers, vLLM and SGLang builds was not checked. The same applies to other new architectures (Kimi K3 KDA, DeepSeek V4.1 CSA2, Hy4, ERNIE 5.5, Step 5, LFM3).
- Families checked with no newer open text LLM found: Llama (no release after Llama 4 on `meta-llama`), SmolLM (SmolLM3), Seed-OSS (36B), gpt-oss, Phi (no Phi-5 text model). xAI's Grok 2 weights (`xai-org/grok-2`, August 2025) exist but were not included because they are a prior-generation release with a custom license and no verified model card metadata.
- Benchmark mention counts are keyword matches over model cards, not a count of reported results.

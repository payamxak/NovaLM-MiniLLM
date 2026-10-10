# NovaLM — End-to-End Project Specification
## A Modern Decoder-Only Mini-LLM Built From Scratch

> This document is the authoritative implementation specification for NovaLM. The goal is not a toy Transformer or a wrapper around Hugging Face; it is a reproducible miniature LLM stack covering data engineering, tokenizer training, model implementation, pretraining, post-training, optimized inference, evaluation, and research ablations.

---

# 0. IMPLEMENTATION GUIDELINES

Implementation should follow these requirements:

1. Read this entire document before implementation.
2. Work milestone-by-milestone; do not skip acceptance tests.
3. Implement core mathematical components manually in PyTorch first.
4. Use optimized third-party kernels only as optional backends after a reference implementation exists.
5. Write unit tests for every mathematical/core component and integration tests for every pipeline stage.
6. Benchmark every optimization against a baseline.
7. Run ablations for architectural choices where practical.
8. Never claim a feature works without a test, benchmark, or reproducible experiment.
9. Keep code modular, typed where practical, documented, and hardware-agnostic.
10. Never hard-code local paths, API keys, or machine-specific assumptions.
11. Keep large datasets/checkpoints out of Git.
12. Record versions, configs, seeds, hardware, and git commit for experiments.
13. If hardware prevents a large experiment, implement/test a smaller reference version and document the limitation.
14. Do not replace the core implementation with `AutoModelForCausalLM`, a pretrained LLM, or a black-box trainer.
15. Correctness comes before optimization.

Preferred stack: Python 3.11/3.12, PyTorch, NumPy, PyYAML, SentencePiece or a custom BPE implementation, safetensors, pytest, ruff, tqdm, optional TensorBoard/W&B, FastAPI/Uvicorn, optional pynvml/psutil, matplotlib. Optional optimized backends: PyTorch SDPA, FlashAttention, torch.compile, optional bitsandbytes for quantization.

---

# 1. PROJECT OBJECTIVE

Build a modern decoder-only causal language model called **NovaLM**.

Release progression:

- `v0.1`: baseline Transformer
- `v0.2`: modern dense architecture
- `v0.3`: optimized training
- `v0.4`: optimized inference
- `v0.5`: Multi-Token Prediction (MTP)
- `v0.6`: Mixture-of-Experts (MoE)
- `v0.7`: Multi-head Latent Attention (MLA)
- `v1.0`: pretrained + SFT + preference-tuned model

The main dense showcase model should target ~30M–100M parameters, depending on available hardware.

The project must demonstrate understanding of Transformer mathematics, tokenizer/data engineering, optimization, memory efficiency, post-training, decoding, evaluation, profiling, reproducibility, and software engineering.

---

# 2. HIGH-LEVEL PIPELINE

```text
Raw Data
  -> cleaning/normalization
  -> quality filtering
  -> language filtering
  -> exact + near deduplication
  -> dataset mixture
  -> tokenizer training
  -> tokenization
  -> sequence packing
  -> sharded dataset
  -> DataLoader
  -> NovaLM
  -> pretraining
  -> evaluation
  -> SFT
  -> DPO (optional GRPO research branch)
  -> chat model
  -> KV-cache inference
  -> sampling
  -> streaming API
  -> benchmark + ablations + report
```

---

# 3. REPOSITORY STRUCTURE

Create:

```text
NovaLM/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── CONTRIBUTING.md
├── CHANGELOG.md
│
├── configs/
│   ├── model/
│   │   ├── novalm_30m.yaml
│   │   ├── novalm_100m.yaml
│   │   ├── novalm_moe.yaml
│   │   └── novalm_mla.yaml
│   ├── data/
│   │   ├── tokenizer.yaml
│   │   ├── pretraining.yaml
│   │   ├── sft.yaml
│   │   └── preference.yaml
│   └── training/
│       ├── default.yaml
│       ├── debug.yaml
│       ├── single_gpu.yaml
│       └── distributed.yaml
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── tokenizer/
│   └── shards/
│
├── novalm/
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py
│   │   └── loader.py
│   ├── tokenizer/
│   │   ├── train.py
│   │   ├── tokenizer.py
│   │   ├── special_tokens.py
│   │   └── chat_template.py
│   ├── data/
│   │   ├── ingest.py
│   │   ├── cleaning.py
│   │   ├── normalization.py
│   │   ├── language_filter.py
│   │   ├── quality.py
│   │   ├── dedup.py
│   │   ├── minhash.py
│   │   ├── mixture.py
│   │   ├── tokenize.py
│   │   ├── packing.py
│   │   ├── shards.py
│   │   ├── dataset.py
│   │   └── collator.py
│   ├── model/
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── lm_head.py
│   │   ├── transformer.py
│   │   └── block.py
│   ├── model/layers/
│   │   ├── rmsnorm.py
│   │   ├── rope.py
│   │   ├── qk_norm.py
│   │   ├── swiglu.py
│   │   └── activations.py
│   ├── model/attention/
│   │   ├── base.py
│   │   ├── causal.py
│   │   ├── mha.py
│   │   ├── gqa.py
│   │   ├── sdpa.py
│   │   ├── flash.py
│   │   ├── mla.py
│   │   └── kv_cache.py
│   ├── model/moe/
│   │   ├── expert.py
│   │   ├── router.py
│   │   ├── topk.py
│   │   ├── moe_layer.py
│   │   └── load_balance.py
│   ├── model/mtp/
│   │   ├── head.py
│   │   └── loss.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── loop.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── amp.py
│   │   ├── checkpoint.py
│   │   ├── distributed.py
│   │   ├── gradient.py
│   │   ├── profiler.py
│   │   ├── metrics.py
│   │   └── seed.py
│   ├── post_training/
│   │   ├── formatting.py
│   │   ├── sft.py
│   │   ├── preference.py
│   │   ├── dpo.py
│   │   └── grpo.py
│   ├── inference/
│   │   ├── engine.py
│   │   ├── generate.py
│   │   ├── sampler.py
│   │   ├── kv_cache.py
│   │   ├── streaming.py
│   │   └── quantization.py
│   ├── evaluation/
│   │   ├── perplexity.py
│   │   ├── generation.py
│   │   ├── benchmark.py
│   │   ├── long_context.py
│   │   ├── multilingual.py
│   │   ├── math.py
│   │   └── code.py
│   └── serving/
│       ├── schemas.py
│       ├── app.py
│       ├── routes.py
│       └── batching.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── tokenize_dataset.py
│   ├── build_shards.py
│   ├── inspect_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── sft.py
│   ├── dpo.py
│   ├── serve.py
│   └── benchmark.py
│
├── tests/
│   ├── unit/
│   │   ├── test_rmsnorm.py
│   │   ├── test_rope.py
│   │   ├── test_attention.py
│   │   ├── test_gqa.py
│   │   ├── test_swiglu.py
│   │   ├── test_moe.py
│   │   ├── test_mla.py
│   │   ├── test_mtp.py
│   │   ├── test_kv_cache.py
│   │   ├── test_tokenizer.py
│   │   └── test_packing.py
│   └── integration/
│       ├── test_forward.py
│       ├── test_training_step.py
│       ├── test_checkpoint_resume.py
│       ├── test_generation.py
│       └── test_api.py
│
├── benchmarks/
│   ├── attention.py
│   ├── kv_cache.py
│   ├── inference.py
│   ├── training.py
│   ├── memory.py
│   ├── moe.py
│   └── quantization.py
│
├── experiments/
│   ├── configs/
│   ├── runs/
│   ├── ablations/
│   └── reports/
│
├── docs/
│   ├── architecture.md
│   ├── data_pipeline.md
│   ├── tokenizer.md
│   ├── training.md
│   ├── inference.md
│   ├── post_training.md
│   ├── evaluation.md
│   ├── ablations.md
│   └── troubleshooting.md
│
├── notebooks/
│   ├── 01_tokenizer_analysis.ipynb
│   ├── 02_training_curves.ipynb
│   ├── 03_attention_benchmark.ipynb
│   ├── 04_ablation_analysis.ipynb
│   └── 05_inference_analysis.ipynb
├── checkpoints/
├── logs/
└── outputs/
```

Large datasets and checkpoints must not be committed.

---

# 4. CONFIGURATION SYSTEM

Implement validated configuration objects for:

- ModelConfig
- AttentionConfig
- MoEConfig
- MTPConfig
- DataConfig
- TokenizerConfig
- TrainingConfig
- OptimizerConfig
- SchedulerConfig
- InferenceConfig
- EvaluationConfig

Example main configuration:

```yaml
model:
  vocab_size: 32000
  hidden_size: 768
  num_layers: 12
  num_attention_heads: 12
  num_kv_heads: 4
  head_dim: 64
  intermediate_size: 2048
  max_seq_length: 2048
  rope_theta: 10000
  norm_eps: 1.0e-6
  tie_embeddings: true
  attention_type: gqa
  qk_norm: true
  mlp_type: swiglu
  moe_enabled: false
  mla_enabled: false
  mtp_enabled: false
```

Parameter count must be computed programmatically and printed at startup.

---

# 5. MODEL SIZE TIERS

## Debug

```yaml
layers: 2
hidden: 128
heads: 4
kv_heads: 2
vocab: 4096
context: 128
```

Purpose: CPU tests and CI.

## NovaLM-30M

Target ~30M parameters. Purpose: first real training on limited hardware.

## NovaLM-100M

Target ~100M parameters. Purpose: main portfolio model.

## Research variants

- NovaLM-100M-MTP
- NovaLM-100M-MoE
- NovaLM-100M-MLA

Do not force every advanced mechanism into one model. Preserve interpretable experimental branches.

---

# 6. TOKENIZER

Train the tokenizer from project data. Do not silently reuse a pretrained LLM tokenizer.

Preferred initial implementation: SentencePiece BPE or an equivalent custom BPE implementation.

Target vocabulary: 32K.

Special tokens:

```text
<|pad|> <|bos|> <|eos|> <|unk|>
<|system|> <|user|> <|assistant|> <|tool|>
<|think|> <|/think|>
```

Required analysis:

- encode/decode round-trip tests
- Persian token fertility
- English token fertility
- mixed Persian-English examples
- code examples
- mathematical notation
- average tokens/word
- unknown-token rate
- common token fragments

---

# 7. DATA ENGINEERING

Pipeline:

```text
ingestion
 -> Unicode normalization
 -> cleaning
 -> quality filtering
 -> language filtering
 -> exact deduplication
 -> near-duplicate detection
 -> dataset mixture
 -> tokenization
 -> sequence packing
 -> sharding
```

Implement configurable quality checks:

- min/max length
- alphabetic ratio
- symbol ratio
- URL density
- repeated-line ratio
- repeated-character ratio
- garbage-character ratio
- empty/invalid documents

Implement exact deduplication with hashes.

Implement optional near-deduplication with MinHash + LSH.

Track licenses and provenance in a dataset manifest.

Initial mixture may be:

```text
English: 40%
Persian: 35%
Code: 15%
Math/science: 10%
```

Treat this only as an initial experiment; measure alternatives.

---

# 8. SEQUENCE PACKING AND SHARDING

Support contexts:

- 512
- 1024
- 2048
- optionally 4096

Avoid unnecessary padding by packing token sequences into fixed-length blocks.

Test:

- input/target shift
- EOS handling
- causal masking
- boundary behavior
- no accidental cross-example leakage where prohibited

Build sharded binary/memory-efficient datasets with manifests containing:

- source
- license
- preprocessing version
- tokenizer version
- token count
- mixture
- checksum
- shard count

---

# 9. CORE MODEL

Implement manually in PyTorch:

- TokenEmbedding
- RMSNorm
- RoPE
- QK-Norm
- causal attention
- GQA
- SwiGLU
- TransformerBlock
- Transformer
- LMHead

Default block:

```text
x
 |
 RMSNorm
 |
 Q/K/V
 |
 QK-Norm
 |
 RoPE
 |
 GQA
 |
 output projection
 |
 + residual
 |
 RMSNorm
 |
 SwiGLU or MoE
 |
 output projection
 |
 + residual
```

Use pre-normalization.

---

# 10. RMSNORM

Implement:

```text
RMS(x) = sqrt(mean(x^2) + eps)
RMSNorm(x) = gamma * x / RMS(x)
```

Requirements:

- numerical stability
- dtype awareness
- gradient correctness
- reference unit test

---

# 11. ROPE

Implement Rotary Position Embeddings manually.

Requirements:

- configurable theta/base
- cached cos/sin
- dynamic extension
- correct broadcasting
- long-context test
- numerical tests

Do not use an opaque external implementation as the reference.

---

# 12. QK-NORM

Normalize Q and K before attention score calculation.

Document the motivation and run an ablation against no QK-Norm.

---

# 13. ATTENTION BACKENDS

Create a common interface and registry:

```text
naive
sdpa
flash
```

Reference: naive causal attention.

Optimized: PyTorch SDPA and optional FlashAttention.

All backends must be numerically compared against the reference within dtype-appropriate tolerance.

Benchmark:

- latency
- memory
- throughput
- sequence length scaling

---

# 14. GROUPED QUERY ATTENTION

Implement GQA manually.

Example:

```text
query heads = 12
KV heads    = 4
```

Test:

- shapes
- causal behavior
- gradients
- cache compatibility
- equivalence to MHA when `num_kv_heads == num_attention_heads`

---

# 15. SWIGLU

Implement:

```text
SwiGLU(x) = W2( SiLU(Wg x) * Wu x )
```

Test shapes, gradients, and finite outputs.

---

# 16. WEIGHT TYING

Support shared input embedding and LM-head weights.

When enabled, prove actual parameter sharing in a test; do not merely copy tensors.

---

# 17. CAUSAL LM OBJECTIVE

For tokens `x0...xT`, train on next-token targets.

Use cross entropy with support for ignored/masked positions.

The implementation must correctly support both pretraining labels and SFT labels.

---

# 18. MULTI-TOKEN PREDICTION (MTP)

Optional advanced objective.

For K future tokens:

```text
h_t -> t+1
h_t -> t+2
...
h_t -> t+K
```

Use:

```text
L = L1 + lambda2*L2 + ... + lambdaK*LK
```

Requirements:

- configurable K
- correct target shifts
- sequence-boundary masking
- per-horizon metrics
- ablation against normal NTP

Do not claim improvement without measurement.

---

# 19. MIXTURE-OF-EXPERTS (RESEARCH BRANCH)

Implement modular MoE:

- Router
- Top-K routing
- Experts
- capacity handling if needed
- load-balancing auxiliary loss
- expert utilization metrics
- optional shared expert

Starting research configuration:

```text
num_experts = 8
top_k = 2
```

Required monitoring:

- expert routing histogram
- expert token counts
- router entropy
- auxiliary loss
- expert collapse detection

Do not implement MoE before the dense model is stable.

---

# 20. MLA (RESEARCH BRANCH)

Implement Multi-head Latent Attention as a separate attention backend.

Requirements:

- explicit equations in docs
- compressed latent KV representation
- inference cache representation
- small reference implementation
- correctness tests
- KV memory benchmark against GQA
- latency benchmark
- quality/perplexity comparison

MLA should initially coexist with, not replace, the stable GQA path.

---

# 21. INITIALIZATION

Define explicit initialization for:

- embeddings
- linear layers
- output projections
- normalization parameters

Check initial activation variance and detect NaNs/Infs.

---

# 22. OPTIMIZER AND SCHEDULER

Use AdamW as the default.

Starting configuration:

```text
learning_rate = 3e-4
betas = (0.9, 0.95)
weight_decay = 0.1
```

Implement parameter groups so normalization and bias parameters can be excluded from weight decay.

Implement gradient clipping.

Scheduler:

```text
linear warmup -> cosine decay
```

Make all values configurable.

---

# 23. MIXED PRECISION, ACCUMULATION, CHECKPOINTING

Support:

- FP32
- FP16
- BF16

Prefer BF16 when supported; use correct scaling for FP16.

Support gradient accumulation and log effective batch size:

```text
micro_batch * accumulation_steps * world_size
```

Support activation/gradient checkpointing as an optional memory optimization and benchmark its memory/throughput tradeoff.

---

# 24. TRAINING ENGINE

The training loop must support:

- forward
- loss
- backward
- accumulation
- clipping
- optimizer step
- scheduler step
- validation
- checkpointing
- resume
- logging
- profiling

Checkpoint state should include:

- model
- optimizer
- scheduler
- step/epoch
- RNG states
- AMP state
- config
- tokenizer version
- git commit if available

Use safetensors for model weights where practical.

Implement and test interrupted/resumed training.

---

# 25. DISTRIBUTED TRAINING

Design cleanly for distributed execution.

Implement single-GPU first, then optional DDP.

Document:

- world size
- rank
- local rank
- distributed sampler
- gradient synchronization

Do not add distributed complexity before the single-GPU implementation is correct.

---

# 26. PROFILING AND METRICS

Track:

- train loss
- validation loss
- perplexity
- learning rate
- gradient norm
- tokens/sec
- samples/sec
- step time
- forward/backward time
- optimizer time
- peak VRAM
- GPU utilization where available
- parameter count
- approximate FLOPs

Optional: PyTorch profiler.

For MoE additionally track expert utilization; for MTP track each horizon loss.

---

# 27. PRETRAINING WORKFLOW

Before real pretraining:

1. Run debug model on CPU.
2. Overfit one batch.
3. Overfit a tiny dataset.
4. Run a short smoke test.
5. Validate checkpoint save/load.
6. Validate resume.
7. Validate generation.
8. Only then launch longer training.

Start context length at 512 and scale to 1024/2048 after correctness is proven.

Maintain a 30M model as the first real target. Scale to 100M only after the complete pipeline works.

---

# 28. INFERENCE ENGINE

Implement generation manually rather than relying only on a high-level `generate()`.

Required decoding:

- greedy
- temperature
- top-k
- top-p/nucleus
- min-p
- repetition penalty

Support:

- max_new_tokens
- EOS
- stop sequences

---

# 29. KV CACHE

Implement per-layer K/V caching.

For GQA, cache KV heads rather than expanded query heads.

Support:

- prefill
- autoregressive decode

Benchmark:

- no cache vs cache
- TTFT (time to first token)
- decode tokens/sec
- cache memory

Cached and uncached generation must agree within numerical tolerance.

---

# 30. STREAMING

Expose generation as a token iterator so the server can stream tokens incrementally.

Do not wait for the full completion before returning output.

---

# 31. CHAT TEMPLATE

Use a documented format such as:

```text
<|system|>
You are NovaLM.
<|user|>
...
<|assistant|>
...
```

Implement it in the tokenizer/chat-template module.

---

# 32. SUPERVISED FINE-TUNING

Pipeline:

```text
instruction dataset
 -> chat formatting
 -> tokenization
 -> assistant-only loss masking
 -> NovaLM Base
 -> SFT
 -> NovaLM-Instruct
```

Do not apply loss to system/user tokens when assistant-only masking is selected.

Evaluate instruction following before and after SFT on a controlled held-out set.

---

# 33. DPO / PREFERENCE OPTIMIZATION

Implement DPO as the primary preference-tuning research branch.

Input format:

```text
prompt
chosen
rejected
```

Implement the objective directly enough that the code clearly exposes:

- policy log-probability
- reference log-probability
- chosen/rejected difference
- beta
- loss

Optional advanced branch: GRPO. Keep it experimental unless a clear reward/evaluation setup exists.

---

# 34. REASONING MODE

Optionally support:

```text
<|think|>
...
<|/think|>
```

Do not claim these tokens create reasoning ability. Any reasoning experiment must document data, objective, evaluation, and limitations.

---

# 35. QUANTIZATION

Provide an optional inference quantization path.

Benchmark at least one low-bit mode and, where hardware/software allows, compare:

```text
FP32
FP16/BF16
INT8
INT4
```

Measure:

- memory
- throughput
- perplexity degradation
- generation quality

---

# 36. FASTAPI SERVING

Expose:

```text
GET  /health
GET  /v1/model/info
POST /v1/completions
POST /v1/chat/completions
```

Support:

- request validation
- generation parameters
- streaming
- safe model selection
- clear errors

Provide a minimal client example.

---

# 37. BATCHING

Implement simple static batching first.

Optional advanced branch: continuous batching.

Benchmark batch sizes where hardware permits.

---

# 38. EVALUATION

At minimum report:

- validation loss
- perplexity
- controlled generation examples
- Persian QA/instruction evaluation
- English evaluation
- code completion if code was included
- math evaluation if a valid benchmark is available
- long-context behavior

For every benchmark record:

- checkpoint
- tokenizer version
- dataset/version
- context length
- decoding settings
- sample count
- seed where applicable

Never fabricate benchmark scores.

---

# 39. ABLATION STUDIES — MANDATORY FOR PORTFOLIO VERSION

Run, where compute permits:

1. LayerNorm vs RMSNorm
2. GELU/ReLU vs SwiGLU
3. absolute positional encoding vs RoPE
4. MHA vs GQA
5. GQA vs MLA
6. NTP vs NTP+MTP
7. Dense vs MoE
8. with/without QK-Norm
9. with/without activation checkpointing
10. tokenizer vocabulary-size comparison if practical
11. data-mixture comparison if practical

For architecture changes report quality, parameter count, and training cost.
For optimizations report throughput, latency, and memory.

---

# 40. BENCHMARK SUITE

Create reproducible scripts for:

## Attention

Compare naive, SDPA, FlashAttention:

- latency
- memory
- throughput
- context scaling

## KV cache

Compare no-cache vs cache.

## Architecture

Compare MHA, GQA, MLA.

## Dense vs MoE

Compare:

- total parameters
- active parameters
- throughput
- memory
- expert utilization
- validation quality

## Quantization

Compare precision modes.

---

# 41. NUMERICAL CORRECTNESS

Every optimized implementation must have a reference comparison.

Examples:

```text
naive attention vs SDPA
naive attention vs FlashAttention
reference GQA vs optimized GQA
uncached generation vs cached generation
save/load output equivalence
```

Use dtype-appropriate tolerances; do not demand exact floating-point equality.

---

# 42. TESTING

Unit tests must cover all mathematical primitives.

Integration tests must cover:

- tokenizer
- dataset
- forward
- backward
- one optimizer step
- checkpoint save/load
- resume
- generation
- KV cache
- API

The debug configuration must make the complete integration suite runnable on CPU.

---

# 43. SANITY CHECKS

Before serious training:

## One-batch overfit

Loss should decrease strongly.

## Tiny-dataset overfit

The model should memorize enough to prove the pipeline works.

## Causal masking

Changing a future token must not change earlier hidden states.

## Shift correctness

Input/target alignment must be verified explicitly.

## Cache equivalence

Cached and uncached generation must agree.

## Save/load

Outputs before/after checkpoint reload must match within tolerance.

---

# 44. FAILURE DETECTION

Detect and fail clearly on:

- NaN loss
- Inf loss
- exploding gradients
- dead/unused experts
- excessive unknown tokens
- empty/corrupt samples
- corrupted shards
- invalid config
- incompatible checkpoint
- context overflow

Never silently continue a broken training run.

---

# 45. REPRODUCIBILITY

Implement a seed utility.

Record:

- Python version
- PyTorch version
- CUDA version
- GPU
- git commit
- config
- seed
- dataset version
- tokenizer version

Each run should contain:

```text
config.yaml
metrics.jsonl
environment.json
command.txt
git_commit.txt
checkpoint metadata
```

---

# 46. EXPERIMENT TRACKING

Use:

```text
experiments/runs/<experiment_name>/
```

Example:

```text
baseline_30m/
gqa_30m/qknorm_30m/mtp_30m/moe_100m/mla_100m
```

Store config, logs, metrics, benchmark results, and generated samples.

---

# 47. DOCUMENTATION

Create:

- `architecture.md`: equations, tensor shapes, data flow
- `data_pipeline.md`: sources, licenses, cleaning, deduplication, packing
- `tokenizer.md`: tokenizer training and analysis
- `training.md`: objective, optimizer, scheduler, precision, checkpointing
- `inference.md`: decoding, KV cache, streaming, quantization
- `post_training.md`: SFT/DPO/GRPO
- `evaluation.md`: methodology and results
- `ablations.md`: hypothesis -> experiment -> result -> interpretation
- `troubleshooting.md`

README must be portfolio-quality with architecture diagrams, features, setup, training, inference, API, evaluation, benchmarks, ablations, limitations, roadmap, reproducibility, and license.

---

# 48. CLI

Provide:

```bash
python scripts/train_tokenizer.py --config configs/data/tokenizer.yaml
python scripts/prepare_data.py --config configs/data/pretraining.yaml
python scripts/build_shards.py --config configs/data/pretraining.yaml
python scripts/train.py --config configs/model/novalm_30m.yaml
python scripts/evaluate.py --checkpoint checkpoints/latest
python scripts/generate.py --checkpoint checkpoints/latest --prompt "..."
python scripts/sft.py --config configs/data/sft.yaml
python scripts/dpo.py --config configs/data/preference.yaml
python scripts/serve.py --checkpoint checkpoints/chat
python scripts/benchmark.py --suite inference
```

Every script must expose `--help`.

---

# 49. CODE QUALITY

Use ruff and pytest. Prefer small modules, explicit interfaces, type hints, meaningful names, no duplicated logic, and no hidden global state.

Avoid giant files, magic numbers, hard-coded paths, silent exception handling, and unnecessary dependencies.

---

# 50. GIT WORKFLOW

Use meaningful incremental commits such as:

```text
feat: add tokenizer
feat: add rmsnorm
feat: add rope
feat: add causal attention
feat: add gqa
feat: add swiglu
feat: add transformer
feat: add training loop
feat: add checkpointing
feat: add kv cache
feat: add mtp
feat: add moe
feat: add mla
feat: add sft
feat: add dpo
feat: add evaluation
perf: optimize attention
perf: optimize dataloader
test: add cache equivalence tests
docs: add architecture guide
```

Do not create one giant commit.

---

# 51. MILESTONES AND ACCEPTANCE CRITERIA

## M0 — Bootstrap

Deliver repository, configs, package structure, testing, linting, README skeleton.

Acceptance: `pytest` passes and `import novalm` works.

## M1 — Tokenizer

Deliver tokenizer training, special tokens, encode/decode, analysis.

Acceptance: round-trip tests pass on Persian/English/code.

## M2 — Data pipeline

Deliver cleaning, filtering, dedup, packing, sharding.

Acceptance: reproducible tiny dataset and valid shards.

## M3 — Reference model

Deliver RMSNorm, RoPE, naive causal attention, SwiGLU, Transformer block, LM head.

Acceptance: forward/backward pass, causal test, one-batch overfit.

## M4 — Modern attention

Deliver GQA, QK-Norm, SDPA, optional FlashAttention.

Acceptance: numerical equivalence tests and benchmark.

## M5 — Training engine

Deliver AdamW, scheduler, AMP, accumulation, checkpoint/resume, logging.

Acceptance: interrupted/resumed training works; tiny loss decreases.

## M6 — Pretraining

Deliver 30M model, validation, metrics, checkpoints, reproducible run.

Acceptance: meaningful validation loss, no NaN/Inf, documented curve.

## M7 — Inference

Deliver generation, sampling, KV cache, streaming.

Acceptance: cache equivalence and measurable speedup.

## M8 — SFT

Deliver chat formatting, assistant-only masking, SFT, evaluation.

Acceptance: controlled instruction-following evaluation improves.

## M9 — DPO

Deliver preference format, DPO implementation, evaluation.

Acceptance: numerically valid loss and preference metrics.

## M10 — MTP

Deliver multi-token heads, loss, evaluation.

Acceptance: correct shifts and ablation.

## M11 — MoE

Deliver router, top-k, experts, balancing, metrics.

Acceptance: expert utilization measured; no silent collapse.

## M12 — MLA

Deliver reference implementation, cache, benchmark.

Acceptance: correctness plus memory/latency comparison with GQA.

## M13 — Quantization

Deliver at least one low-bit inference path and benchmark.

Acceptance: memory and quality comparison.

## M14 — Serving

Deliver FastAPI, streaming, model info, health, validation.

Acceptance: local API works end-to-end.

## M15 — Portfolio release

Deliver complete README, diagrams, benchmark tables, ablations, docs, reproducibility instructions, example outputs, limitations, and release tag.

Acceptance: another engineer can clone and follow the project without private knowledge.

---

# 52. REQUIRED RESEARCH QUESTIONS

Investigate experimentally:

1. Does GQA reduce KV-cache memory without meaningful quality loss at this scale?
2. Does QK-Norm improve training stability?
3. Does SwiGLU improve validation loss at equal parameter budgets?
4. Does MTP improve convergence or representation quality?
5. Does MoE improve quality at similar active compute?
6. How does expert count affect routing stability?
7. Does MLA reduce KV memory relative to GQA at similar quality?
8. What is the memory/throughput tradeoff of activation checkpointing?
9. How much quality is lost under INT8/INT4 inference?
10. How does tokenizer vocabulary size affect Persian tokenization efficiency?
11. How does data mixture affect Persian vs English performance?
12. How does context length affect validation loss and inference cost?

Answers must come from experiments, not assumptions.

---

# 53. DEVELOPMENT ORDER

Do NOT implement every advanced feature at once.

Correct order:

```text
correctness
  -> baseline
  -> measurement
  -> optimization
  -> ablation
```

Recommended implementation order:

1. Repository/bootstrap
2. Config system
3. Tokenizer
4. Data pipeline
5. RMSNorm
6. RoPE
7. naive attention
8. GQA
9. SwiGLU
10. TransformerBlock
11. NovaLM
12. causal loss
13. tests + overfit
14. training engine
15. checkpoint/resume
16. mixed precision
17. SDPA/FlashAttention
18. pretraining
19. generation
20. KV cache
21. streaming
22. SFT
23. DPO
24. MTP
25. MoE
26. MLA
27. quantization
28. FastAPI
29. full benchmarks
30. ablations
31. documentation/release

---

# 54. HARDWARE-AWARE STRATEGY

Maintain a CPU-debug model and a 30M model for constrained GPUs.

For limited VRAM, use:

- shorter context
- smaller microbatch
- gradient accumulation
- activation checkpointing
- BF16/FP16 where supported
- SDPA/FlashAttention
- efficient KV cache

Do not make the project dependent on training a billion-parameter model. Portfolio value comes from implementation quality, experiments, measurements, and reproducibility.

---

# 55. INITIAL TRAINING STRATEGY

Start with:

```text
synthetic data
 -> tiny real corpus
 -> ~1M token scale
 -> ~10M token scale
 -> larger corpus if hardware allows
```

The first objective is correctness, not scale.

The 30M model is the first serious target. The 100M model is the portfolio showcase after the pipeline is stable.

---

# 56. FINAL PORTFOLIO OUTPUT

The repository should communicate, truthfully:

> NovaLM is a modern decoder-only language model implemented from first principles in PyTorch, including a custom data pipeline and tokenizer, RMSNorm, RoPE, QK-Norm, GQA, SwiGLU, optimized attention backends, KV-cache inference, Multi-Token Prediction, experimental MoE/MLA architectures, SFT/DPO post-training, evaluation, profiling, and reproducible benchmarks.

Only advertise features that actually pass their acceptance criteria.

---

# 57. DEFINITION OF DONE

The project is complete only when:

- tokenizer works
- data pipeline works
- exact and optional near dedup work
- packing works
- reference model works
- modern dense model works
- causal masking is tested
- training loop works
- checkpoint resume works
- validation works
- pretraining completes a documented run
- generation works
- KV cache works
- streaming works
- SFT works
- DPO works or is clearly marked experimental
- MTP is implemented and evaluated
- MoE is implemented and evaluated
- MLA is implemented and benchmarked
- quantization is benchmarked
- FastAPI serving works
- unit tests pass
- integration tests pass
- benchmark suite runs
- ablations are documented
- README is complete
- reproducibility instructions are usable

---

# 58. IMPLEMENTATION WORKFLOW

Treat NovaLM as a serious open-source ML engineering and research project.

At the beginning of each milestone:

1. inspect existing code
2. identify what is already implemented
3. implement only the next required layer of functionality
4. add tests
5. run tests
6. run lint/type checks where configured
7. run the relevant smoke test
8. benchmark when the milestone introduces an optimization
9. update documentation
10. record limitations
11. only then proceed

If a critical acceptance criterion fails, stop advancing the architecture and fix the failure first.

The final result must demonstrate that the author understands not only how to use an LLM, but how modern language models are constructed, trained, evaluated, optimized, and served.

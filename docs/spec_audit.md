# Project specification audit

Audit date: 2026-10-07. This audit separates implemented code from completed acceptance evidence. Synthetic checks do not establish language-model quality on a real corpus.

| Milestone | Status | Evidence or remaining work |
|---|---|---|
| M0 Bootstrap | Implemented | All 188 repository-tree entries listed in the spec exist. `python -m pytest -q`, `python -m unittest discover -s tests -v`, package import, and Ruff pass. The workspace has no Git metadata, so commit-based provenance and the prescribed commit workflow are unavailable. |
| M1 Tokenizer | Partial | Locally trained SentencePiece BPE, special tokens, chat formatting, round-trip tests, JSONL/text training API, and analysis CLI exist. A 1,024-token tokenizer was analyzed on four short real English and four short Persian summaries; representative-corpus fertility and unknown-token reports remain absent. |
| M2 Data | Partial | Streaming ingestion, normalization, configurable quality checks, script heuristic/filter, exact and optional MinHash dedup, weighted source mixture, tokenization, packing, collator, binary shards, provenance, and checksums exist. A 12-summary English/Persian corpus was fetched with API-reported licenses/revisions and processed into train/validation shards. Full licensed corpus preparation and mixture studies remain. |
| M3 Reference model | Implemented at debug scale | Manual RMSNorm, RoPE, causal attention, GQA/MHA, SwiGLU, pre-norm decoder, tied embeddings, causal loss, and causal/gradient checks. The recorded one-batch CPU smoke run (seed 7, 30 steps) reduced loss from 3.5377 to 0.1771. LayerNorm/GELU/absolute-position alternatives are configurable and covered by cache-equivalence and parameter-count checks. |
| M4 Modern attention | Partial | Naive, SDPA, and SDPA-dispatched `flash` backends, GQA, QK-Norm, and reference comparisons exist. A separately installed/forced FlashAttention package backend is absent. CUDA smoke benchmark reports are present. |
| M5 Training | Partial | Factored AdamW grouping, warmup/cosine scheduler, AMP helpers, accumulation, clipping, finite-value guards, throughput/validation/gradient-norm metrics, resumable sampler state, activation checkpointing, atomic periodic/final and best-validation checkpoints, RNG/environment state, and DDP-aware trainer/CLI integration are implemented. A bounded profiler mode writes traces. A deterministic CPU CLI test matches continuous and resumed weights exactly. Exact per-rank resume remains limited to single-process runs. |
| M6 Pretraining | Not complete | A 30,027,520-parameter model completed 100 BF16 CUDA optimizer steps on 1,024 real-text tokens and produced a held-out curve. Validation improved early then degraded; final perplexity was 1,104.9 on only 512 targets and was barely below random initialization. This is a pipeline smoke run, not meaningful corpus-scale pretraining. |
| M7 Inference | Partial | Greedy and sampling controls, stop/EOS handling, KV cache, iterator, inference engine, and equal-length static batch generation are implemented. Cached/full output checks, local CUDA cache smoke report, CPU TTFT smoke report, and UTF-8 Windows CLI output check exist. API stream performance and concurrent request batching are unmeasured/unimplemented. |
| M8 SFT | Partial | Chat formatting, assistant-only labels, and SFT CLI pass synthetic smoke checks. No controlled held-out before/after instruction evaluation. |
| M9 DPO | Partial | Explicit policy/reference chosen/rejected objective and synthetic training smoke check exist. GRPO has a clipped policy-loss primitive only; reward design, group sampling, trainer, and evaluation are absent. No real preference evaluation. |
| M10 MTP | Implemented, evaluation incomplete | Horizon heads and shifted losses are integrated into model/trainer; horizon metrics, gradients, and training-step checks pass. No NTP-vs-MTP quality/convergence experiment. |
| M11 MoE | Implemented, evaluation incomplete | Top-k experts, router balancing loss, expert-use/entropy/dead-expert metrics, model/trainer integration, and training-step checks pass. Synthetic forward latency/memory smoke comparison is recorded. No capacity-limited dispatch or quality/throughput study at training scale. |
| M12 MLA | Implemented, benchmark incomplete | Configurable latent attention is integrated, supports cache, trains, and passes cache-equivalence checks. Tiny synthetic GQA/MLA latency/cache-memory comparison is recorded; comparable quality or target-scale study remains. |
| M13 Quantization | Partial | Dynamic INT8 and packed weight-only INT4 paths pass forward checks. A CPU smoke benchmark records state size, latency, synthetic loss, and greedy-token agreement. INT4 dequantizes every forward and is slower in this run. No held-out perplexity/generation-quality comparison. |
| M14 Serving | Partial | FastAPI health/info/completion/chat, parameter validation, stop/sampling controls, SSE streaming, and local TestClient checks exist. Python static batching is implemented, but HTTP request batching, continuous batching, and serving performance measurements remain. |
| M15 Portfolio release | Not complete | README/docs and raw system/real-text smoke reports exist, but there is no useful trained showcase checkpoint, full benchmark suite, controlled corpus-based ablation report, or release tag. Reproducibility is limited by absent Git metadata; raw corpus and checkpoints are intentionally untracked. |

## Architecture ablations

The model exposes and tests LayerNorm/RMSNorm, GELU/SwiGLU, and absolute positions/RoPE switches. MoE/MTP/MLA variants have model configs. Activation checkpointing has a small synthetic memory comparison recorded; repeated target-scale timing remains. Passing implementation tests does not answer the spec's research questions; quality comparisons require matched real data, budgets, and held-out evaluation. QK-Norm, tokenizer-size, and data-mixture comparisons are not yet documented as completed experiments.

## Remaining definition-of-done work

The largest outstanding requirements are corpus-scale licensed training/evaluation data, meaningful pretraining quality, controlled SFT/DPO/MTP/MoE/MLA evaluation, target-scale quantization and architecture benchmarks, DDP trainer integration, serving-side batching, and complete reproducible release metadata. The current Wikipedia summary run proves pipeline operation only and shows overfitting; it does not satisfy those research or quality criteria.

## Additional local verification (2026-10-07)

- `python -m pytest -q`: 81 passed, 26 subtests passed (one upstream Starlette deprecation warning).
- `python -m unittest discover -s tests -v`: 40 tests passed.
- `python -m ruff check novalm scripts tests benchmarks` and `python -m compileall -q novalm scripts benchmarks tests` passed.
- All scripts and benchmark modules returned successfully for `--help`; all five notebook files parsed and their code cells ran.
- A strict load of the existing 30,027,520-parameter smoke checkpoint into the refactored model passed.
- The `inference` benchmark suite dispatches to `benchmarks/inference.py`; a CPU smoke run (prompt 4, decode 4) reported TTFT and post-first-token throughput to a JSON report.
- The packed INT4 CPU smoke report compares FP32, dynamic INT8, and INT4 storage/latency on synthetic tokens; it is not a quality evaluation.
- The deterministic CPU one-batch overfit run is recorded in `experiments/reports/one_batch_overfit_cpu.json` with seed, config, environment, and losses.
- `scripts/inspect_data.py` was added to summarize JSONL samples or checksum-validate shard manifests; both modes passed against bundled Wikipedia smoke artifacts.
- `scripts/train.py` now integrates `torchrun` DDP, rank-sharded deterministic sampling, rank-zero checkpoint/metric writing, and world-size metadata; the world-size-one CPU path remains covered by the full integration suite.
- A two-process `torchrun` smoke was attempted locally but the installed Windows PyTorch 2.6.0 wheel fails during launcher rendezvous because it was built without libuv; this is an environment limitation before NovaLM code starts.
- The profiler smoke wrote a trace, single-process distributed setup returned rank 0/world size 1, and memory/MoE benchmark smoke commands returned valid summaries.

## Additional verification (2026-10-08)

- Generated `docs/novalm_reference.md` from the checked-in source and YAML configurations. It inventories the algorithmic pipeline, every package module, functions, methods, class fields, `self.*` attributes, defaults, and configuration values.
- Added `scripts/generate_reference.py` so the reference can be regenerated after implementation or configuration changes.
- All 16 scripts and 8 benchmark entry points returned successfully for `--help`.
- The CPU one-batch overfit smoke again reduced loss from `3.5377` to `0.1771` and wrote `experiments/reports/one_batch_overfit_cpu_latest.json`.
- `python -m pytest -q`: 83 passed and 26 subtests passed; `python -m unittest discover -s tests -v`: 42 tests passed; Ruff and compileall passed.

# Ablations and architecture measurements

No quality ablation has been run. The available corpus is insufficient for a held-out language-quality comparison, so architectural quality claims would be misleading.

## Architecture smoke benchmark

`benchmarks/architecture.py` compares MHA, GQA, MLA, dense, and MoE reference models at the same small model dimensions. It reports parameter counts, estimated active parameters, forward latency, peak CUDA allocation, and cache bytes. The raw output is in [`architecture_cuda_smoke.json`](../experiments/reports/architecture_cuda_smoke.json).

Run on October 6, 2026 with Python 3.13.15, PyTorch 2.6.0+cu124, NVIDIA GeForce RTX 3060 Laptop GPU, batch 1, sequence 32, two timed iterations after two warmups:

| Variant | Parameters | Estimated active parameters | Forward ms | Peak allocated MiB | Cache KiB |
|---|---:|---:|---:|---:|---:|
| MHA | 434,944 | 434,944 | 2.813 | 10.09 | 64 |
| GQA | 402,176 | 402,176 | 2.852 | 9.92 | 32 |
| MLA | 418,432 | 418,432 | 5.703 | 9.97 | 16 |
| Dense | 402,176 | 402,176 | 2.916 | 9.92 | 32 |
| MoE, 4 experts / top 2 | 1,214,208 | 673,536 | 6.149 | 13.03 | 32 |

This is a smoke measurement at tiny dimensions with two timed iterations; launch overhead and run-to-run noise dominate. It validates that the benchmark runs and records the expected cache-size ordering in this configuration. It does not establish throughput at target scale, quality parity, or a general speed ranking. The reference MoE loops over experts and is intentionally unoptimized.

## Activation-checkpointing smoke benchmark

`benchmarks/training.py` checks baseline and checkpointed training with identical initialization and synthetic token batches. In the saved October 6, 2026 run (Python 3.13.15, PyTorch 2.6.0+cu124, RTX 3060 Laptop GPU, 3.08M parameters, batch 4, context 256, five steps), first and last losses matched exactly between variants. Peak allocated CUDA memory was 169,500,160 bytes (161.65 MiB) for baseline and 98,719,232 bytes (94.15 MiB) with checkpointing, a 41.8% reduction for this setup. The raw result is [`training_checkpoint_cuda_smoke.json`](../experiments/reports/training_checkpoint_cuda_smoke.json). The run is short and synthetic; timing is not interpreted as a speed result, and memory savings will vary with model, batch, and context.

## Dynamic INT8 inference smoke benchmark

`benchmarks/quantization.py` compares FP32 and PyTorch dynamic INT8 on CPU. In the saved October 6, 2026 run (PyTorch 2.6.0+cu124, 402,176 parameters, batch 2, context 32, three timed iterations), serialized state size was 1,617,102 bytes for FP32 and 555,633 bytes for INT8. Greedy outputs agreed on all synthetic prompt positions. The synthetic-token loss delta was -0.000099; it is not a language-quality metric. INT8 was slower in this tiny run (2.270 ms vs 1.557 ms per forward), which is not a general performance result. Raw data is in [`quantization_cpu_smoke.json`](../experiments/reports/quantization_cpu_smoke.json).

`benchmarks/quantization.py` also compares a packed weight-only INT4 reference path. In the October 7, 2026 CPU run (same 402,176-parameter model, batch 1, context 8, one timed iteration), serialized state size was 1,617,102 bytes for FP32, 521,978 bytes for dynamic INT8, and 353,566 bytes for INT4. Forward latency was 1.064 ms, 1.815 ms, and 3.025 ms respectively. INT4 synthetic-token loss delta was +0.0109 and greedy-token agreement was 87.5%. INT4 weights are dequantized on every forward, so this is a storage/correctness reference, not an accelerated kernel. These tiny synthetic measurements do not establish held-out perplexity or generation quality. Raw result: [`quantization_int4_cpu_smoke.json`](../experiments/reports/quantization_int4_cpu_smoke.json).

## Implementation switches

The model exposes LayerNorm/RMSNorm, GELU/SwiGLU, learned absolute positions/RoPE, QK-Norm on/off, MHA/GQA, and MLA. Tests check model forward, parameter counts, and cached/full-prefix equivalence for supported settings. MTP/MoE/MLA have integration tests and runnable config variants. These checks validate implementation behavior, not research outcomes.

## Required controlled studies still outstanding

Use identical data splits, token budgets, seed policy, and recorded training cost for:

- LayerNorm vs RMSNorm; GELU/ReLU vs SwiGLU; absolute positions vs RoPE.
- MHA vs GQA; QK-Norm on/off; GQA vs MLA.
- NTP vs NTP+MTP; dense vs MoE and expert-count variants.
- Vocabulary size and data mixture. Activation checkpointing has a synthetic memory smoke comparison, but still needs repeated target-scale throughput/memory measurement.
- FP32/BF16/INT8/INT4 quality, memory, latency, and throughput.

Do not infer quality improvements from architectural conventions or synthetic smoke tests.

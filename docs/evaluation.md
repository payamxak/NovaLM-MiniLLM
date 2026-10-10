# Evaluation

`scripts/evaluate.py` computes autoregressive next-token cross-entropy and perplexity on packed token shards. Reports identify the checkpoint, model config, shards, and supervised token count. Shard samples advance by one context at a time, so the validation tokens are not repeatedly counted through stride-one overlapping windows.

## Real-text pipeline smoke run

[`wikipedia_30m_smoke.json`](../experiments/reports/wikipedia_30m_smoke.json) records a 100-step, 30,027,520-parameter BF16 run with activation checkpointing on a local RTX 3060 Laptop GPU. Training used eight English/Persian Wikipedia lead summaries totaling 1,024 tokenizer IDs; validation used four separate summaries totaling 640 IDs. The evaluation covered 512 unique next-token targets. Per-page license, revision, and attribution metadata was fetched and saved with the local corpus.

| Model state | Validation loss | Perplexity | Targets |
|---|---:|---:|---:|
| Seed-42 random initialization | 7.0185 | 1,117.1 | 512 |
| Best checkpoint, step 10 | 6.8395 | 934.0 | 512 |
| Final checkpoint, step 100 | 7.0075 | 1,104.9 | 512 |

Train loss decreased from 7.0349 on the first step to 2.6276 on the last, while validation worsened after step 10. This is evidence that the end-to-end tokenizer/data/train/validation/checkpoint path runs on actual English/Persian text, and evidence of overfitting on an extremely small sample. It is not a meaningful language-quality result or the spec's corpus-scale pretraining acceptance. Do not infer model quality from this run.

The 1,024-token tokenizer's small training-sample analysis is in [`wikipedia_tokenizer_analysis.json`](../experiments/reports/wikipedia_tokenizer_analysis.json). It records 0 unknown tokens in four short documents per language; these counts do not represent corpus-level fertility.

## Reproduce evaluation

```bash
python scripts/evaluate.py --checkpoint experiments/runs/wikipedia_30m_smoke_v3/best.pt --config configs/model/novalm_30m_wikipedia_smoke.yaml --shards data/shards/wikipedia_validation --batch-size 1 --output outputs/evaluation.json
```

No third-party benchmark dataset or corpus is bundled. Persian QA/instruction, English benchmarks, code/math tasks, long-context quality, and controlled post-training evaluations still require appropriately licensed held-out datasets, recorded sample counts, prompts/decoding settings, and seeds.

## Systems benchmark

`benchmarks/attention.py` checks naive/SDPA output agreement before recording local latency, token throughput, and peak CUDA allocation. Results are hardware/software-specific and are not general speedup claims. See [ablation measurements](ablations.md) for architecture, activation-checkpointing, and quantization smoke comparisons.

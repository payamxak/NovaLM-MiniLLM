# Troubleshooting

- For a CPU-only smoke check use `configs/model/debug.yaml` and short sequences.
- A token shard must contain at least `context_length + 1` tokens for one next-token example.
- Set model `vocab_size` to the trained tokenizer vocabulary size.
- `fp16` requires CUDA in this trainer; CPU users can use FP32 or BF16 where supported.
- The `flash` config name still goes through PyTorch SDPA, which chooses an available kernel automatically; a forced FlashAttention implementation is not included.
- Do not point preprocessing at unlicensed corpora. The quality filters can reject repeated-line documents, including synthetic corpora with identical lines.

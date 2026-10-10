# Training

NovaLM's single-process trainer uses AdamW with separate decay groups, linear warmup followed by cosine decay, gradient clipping, optional FP16/BF16 autocast, gradient accumulation, finite-value guards, and JSONL metrics. Configure these options in a training YAML and invoke `scripts/train.py` with token-shard directories.

Before longer runs, execute `python scripts/sanity_check.py --output experiments/reports/one_batch_overfit_cpu.json`. It performs a deterministic CPU one-batch overfit check and writes the seed, model config, environment, and initial/final losses.

Training windows contain `context + 1` tokens. The trainer feeds the first `context` tokens to the model and supervises next-token targets from the remaining window. Optional validation shards are evaluated at a configurable optimizer-step interval and again at the final step when it falls between intervals. Validation loss, perplexity, and token count are written beside train loss and tokens per second. You can also request a separate model-only best-validation checkpoint with `--best-checkpoint`.

Activation checkpointing can be enabled with `activation_checkpointing: true` in the training config. It recomputes decoder blocks during backpropagation to reduce saved activations. The reference implementation supports the MoE auxiliary outputs as well. Use `benchmarks/training.py` to measure the memory/time tradeoff on the target hardware.

Training checkpoints contain model, optimizer, scaler, Python/NumPy/PyTorch RNG, training step, and resumable sampler permutation/cursor. The training CLI uses a stateful random sampler and no worker processes so a resumed run can continue at the same next example. The integration suite compares uninterrupted and separate-process resumed runs and checks exact model-weight equality for a deterministic CPU smoke configuration.

Each CLI run records its model and training config, command, Python/PyTorch/CUDA/device details, seed, parameter count, tokenizer hash when present, and a Git commit marker. The current workspace has no Git metadata, so the marker explicitly says the commit is unavailable.

## Distributed execution

`scripts/train.py` detects the standard `torchrun` environment (`WORLD_SIZE`, `RANK`,
and `LOCAL_RANK`), initializes Gloo on CPU or NCCL on CUDA, wraps the model in
`DistributedDataParallel`, and uses a deterministic rank-sharded resumable sampler.
Only rank zero writes checkpoints, metrics, and run metadata. Launch a CPU smoke run
with `torchrun --standalone --nproc-per-node=2 scripts/train.py ... --device cpu`;
CUDA runs use the same command with `--device auto`.

The single-process checkpoint format remains the source of truth. Distributed runs
currently save the rank-zero sampler state, so resume with the same world size and
rank layout; exact per-rank data replay is guaranteed for single-process runs.

## Current limits

Real-corpus pretraining and held-out quality evaluation have not been run. Assistant-only
SFT and DPO components have synthetic checks but no controlled real-data outcomes. See
[the spec audit](spec_audit.md).

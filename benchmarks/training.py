"""Measure activation-checkpointing training time and CUDA memory on synthetic tokens."""
import argparse
import json
import platform
import time
from datetime import UTC, datetime

import torch

from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM
from novalm.training.trainer import Trainer, TrainerConfig


def run_variant(config, training_config, initial_state, batch, device, steps):
    model = NovaLM(config).to(device)
    model.load_state_dict(initial_state)
    trainer = Trainer(model, training_config, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = [trainer.train_step(batch) for _ in range(steps)]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "seconds_per_step": elapsed / steps,
        "tokens_per_second": batch.numel() * steps / elapsed,
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }
    if device.type == "cuda":
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    del trainer, model
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.sequence_length, args.batch_size, args.iterations) < 1:
        parser.error("sequence-length, batch-size, and iterations must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(2026)
    model_config = ModelConfig(vocab_size=512, hidden_size=256, num_layers=4,
                               num_attention_heads=8, num_kv_heads=4,
                               intermediate_size=704,
                               max_seq_length=args.sequence_length,
                               attention_backend="sdpa")
    initial_model = NovaLM(model_config)
    initial_state = {name: value.detach().clone()
                     for name, value in initial_model.state_dict().items()}
    batch = torch.randint(0, model_config.vocab_size,
                          (args.batch_size, args.sequence_length + 1), device=device)
    variants = {}
    for enabled in (False, True):
        train_config = TrainerConfig(
            learning_rate=3e-4, warmup_steps=1, total_steps=args.iterations,
            activation_checkpointing=enabled,
        )
        variants["activation_checkpointing" if enabled else "baseline"] = run_variant(
            model_config, train_config, initial_state, batch, device, args.iterations)
    if not torch.isclose(torch.tensor(variants["baseline"]["first_loss"]),
                         torch.tensor(variants["activation_checkpointing"]["first_loss"]),
                         rtol=1e-5, atol=1e-6):
        raise RuntimeError("checkpointed and baseline training loss differ beyond tolerance")
    print(json.dumps({
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "shape": list(batch.shape),
        "parameter_count": initial_model.parameter_count,
        "iterations": args.iterations,
        "synthetic_data": True,
        "quality_evaluated": False,
        "results": variants,
    }, indent=2))


if __name__ == "__main__":
    main()

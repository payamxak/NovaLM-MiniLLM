"""Synthetic architecture throughput/memory comparison; it does not measure quality."""
import argparse
import json
import platform
import time
from datetime import UTC, datetime

import torch

from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM


def measure(model: NovaLM, ids: torch.Tensor, iterations: int) -> dict:
    model.eval()
    with torch.inference_mode():
        logits = model(ids)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("model produced non-finite benchmark logits")
        for _ in range(2):
            model(ids)
        if ids.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(ids.device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(ids)
        if ids.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        _, cache = model(ids, use_cache=True)
    cache_bytes = sum(item.numel() * item.element_size()
                      for layer_cache in cache
                      for item in (layer_cache if isinstance(layer_cache, tuple)
                                   else (layer_cache,)))
    result = {
        "parameters": model.parameter_count,
        "seconds_per_forward": elapsed / iterations,
        "tokens_per_second": ids.numel() * iterations / elapsed,
        "cache_bytes_for_prompt": cache_bytes,
    }
    if ids.device.type == "cuda":
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(ids.device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.sequence_length, args.batch_size, args.iterations) < 1:
        parser.error("sequence-length, batch-size, and iterations must be positive")

    torch.manual_seed(0)
    common = {"vocab_size": 256, "hidden_size": 128, "num_layers": 2,
              "num_attention_heads": 4, "num_kv_heads": 2, "intermediate_size": 352,
              "max_seq_length": args.sequence_length, "attention_backend": "sdpa"}
    variants = {
        "mha": ModelConfig(**{**common, "num_kv_heads": 4}),
        "gqa": ModelConfig(**common),
        "mla": ModelConfig(**common, attention_type="mla", mla_latent_dim=64,
                            qk_norm=False),
        "dense": ModelConfig(**common),
        "moe": ModelConfig(**common, moe_enabled=True, num_experts=4, moe_top_k=2),
    }
    ids = torch.randint(0, 256, (args.batch_size, args.sequence_length), device=args.device)
    results = {}
    for name, config in variants.items():
        model = NovaLM(config).to(args.device)
        results[name] = measure(model, ids, args.iterations)
        if name == "moe":
            experts = config.num_experts
            expert_params_per_layer = experts * 3 * config.hidden_size * config.intermediate_size
            active_expert_params_per_layer = config.moe_top_k * 3 * config.hidden_size * config.intermediate_size
            results[name]["estimated_active_parameters"] = (
                model.parameter_count - config.num_layers * expert_params_per_layer
                + config.num_layers * active_expert_params_per_layer
            )
        else:
            results[name]["estimated_active_parameters"] = model.parameter_count
        del model
        if ids.device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device) if ids.device.type == "cuda" else None,
        "shape": list(ids.shape),
        "iterations": args.iterations,
        "quality_evaluated": False,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Reproducible local baseline vs SDPA latency check; makes no portable speed claim."""
import argparse
import json
import platform
import time
from datetime import UTC, datetime

import torch

from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    common = {"vocab_size": 256, "hidden_size": 128, "num_layers": 2,
              "num_attention_heads": 4, "num_kv_heads": 2, "intermediate_size": 352,
              "max_seq_length": args.sequence_length}
    ref = NovaLM(ModelConfig(**common, attention_backend="naive")).to(args.device).eval()
    sdpa = NovaLM(ModelConfig(**common, attention_backend="sdpa")).to(args.device).eval()
    sdpa.load_state_dict(ref.state_dict())
    ids = torch.randint(0, 256, (args.batch_size, args.sequence_length), device=args.device)
    with torch.inference_mode():
        torch.testing.assert_close(ref(ids), sdpa(ids), atol=2e-5, rtol=2e-5)
        results = {}
        for name, model in (("naive", ref), ("sdpa", sdpa)):
            for _ in range(2):
                model(ids)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.iterations):
                model(ids)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            results[name] = {"seconds_per_forward": elapsed / args.iterations,
                             "tokens_per_second": args.batch_size * args.sequence_length * args.iterations / elapsed}
            if args.device.startswith("cuda"):
                results[name]["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    print(json.dumps({"timestamp_utc": datetime.now(UTC).isoformat(),
                      "python": platform.python_version(), "torch": torch.__version__,
                      "device": args.device, "shape": [args.batch_size, args.sequence_length],
                      "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()

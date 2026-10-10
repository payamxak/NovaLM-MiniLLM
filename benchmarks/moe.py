"""Measure reference MoE routing use, parameter count, and forward latency."""
import argparse
import json
import time

import torch

from novalm.model.moe.layer import MoELayer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=352)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.tokens < 1 or args.iterations < 1:
        parser.error("tokens and iterations must be positive")
    layer = MoELayer(args.hidden_size, args.intermediate_size,
                     args.experts, args.top_k).to(args.device).eval()
    x = torch.randn(1, args.tokens, args.hidden_size, device=args.device)
    with torch.inference_mode():
        _, _, metrics = layer(x)
        for _ in range(2):
            layer(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.iterations):
            layer(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(json.dumps({"device": args.device, "tokens": args.tokens,
                      "experts": args.experts, "top_k": args.top_k,
                      "parameters": sum(p.numel() for p in layer.parameters()),
                      "expert_token_counts": metrics["expert_token_counts"].tolist(),
                      "router_entropy": float(metrics["router_entropy"]),
                      "dead_experts": int(metrics["dead_experts"]),
                      "milliseconds_per_forward": elapsed * 1000 / args.iterations}, indent=2))


if __name__ == "__main__":
    main()

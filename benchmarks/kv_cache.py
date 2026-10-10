"""Compare full-prefix decode with cached decode and check next-token equivalence."""
import argparse
import json
import platform
import time
from datetime import UTC, datetime

import torch

from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM


def timed_decode(model, prompt, steps, cached):
    generated = prompt
    cache = None
    with torch.inference_mode():
        if cached:
            logits, cache = model(prompt, use_cache=True)
            logits = logits[:, -1]
        started = time.perf_counter()
        for _ in range(steps):
            if not cached:
                logits = model(generated)[:, -1]
            token = logits.argmax(-1, keepdim=True)
            generated = torch.cat((generated, token), dim=1)
            if cached:
                logits, cache = model(token, cache, use_cache=True)
                logits = logits[:, -1]
        if prompt.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    cache_bytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                      for k, v in cache) if cache is not None else 0
    return generated[:, prompt.size(1):], elapsed, cache_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    context = args.prompt_length + args.decode_steps
    cfg = ModelConfig(vocab_size=256, hidden_size=128, num_layers=2,
                      num_attention_heads=4, num_kv_heads=2, intermediate_size=352,
                      max_seq_length=context, attention_backend="sdpa")
    model = NovaLM(cfg).to(args.device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_length), device=args.device)
    plain, plain_time, _ = timed_decode(model, prompt, args.decode_steps, False)
    cached, cached_time, cache_bytes = timed_decode(model, prompt, args.decode_steps, True)
    if not torch.equal(plain, cached):
        raise RuntimeError("cached and uncached greedy output differs")
    print(json.dumps({"timestamp_utc": datetime.now(UTC).isoformat(),
                      "python": platform.python_version(), "torch": torch.__version__,
                      "device": args.device,
                      "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                      "parameter_count": model.parameter_count,
                      "prompt_tokens": args.prompt_length,
                      "decode_tokens": args.decode_steps,
                      "uncached_seconds_per_token": plain_time / args.decode_steps,
                      "cached_seconds_per_token": cached_time / args.decode_steps,
                      "uncached_tokens_per_second": args.decode_steps / plain_time,
                      "cached_tokens_per_second": args.decode_steps / cached_time,
                      "kv_cache_bytes": cache_bytes}, indent=2))


if __name__ == "__main__":
    main()

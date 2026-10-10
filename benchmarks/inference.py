"""Measure cached generation latency, first-token latency, and decode throughput."""
import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from novalm.inference.generate import stream_generate
from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="optional path for the JSON result")
    args = parser.parse_args()
    if args.prompt_length < 1 or args.decode_tokens < 1:
        parser.error("prompt-length and decode-tokens must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    config = ModelConfig(vocab_size=256, hidden_size=128, num_layers=2,
                         num_attention_heads=4, num_kv_heads=2, intermediate_size=352,
                         max_seq_length=args.prompt_length + args.decode_tokens,
                         attention_backend="sdpa")
    model = NovaLM(config).to(args.device).eval()
    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_length), device=args.device)

    first_token_seconds = None
    started = time.perf_counter()
    generated = 0
    with torch.inference_mode():
        for _token in stream_generate(model, prompt, args.decode_tokens, temperature=0):
            generated += 1
            if first_token_seconds is None:
                if prompt.device.type == "cuda":
                    torch.cuda.synchronize()
                first_token_seconds = time.perf_counter() - started
    if prompt.device.type == "cuda":
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started
    decode_seconds = max(total_seconds - (first_token_seconds or total_seconds), 0.0)
    decode_tokens = max(generated - 1, 0)

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device)
        if args.device.startswith("cuda") else None,
        "parameter_count": model.parameter_count,
        "prompt_tokens": args.prompt_length,
        "generated_tokens": generated,
        "time_to_first_token_seconds": first_token_seconds,
        "total_generation_seconds": total_seconds,
        "decode_tokens_per_second": decode_tokens / decode_seconds if decode_seconds else None,
        "post_first_token_seconds_per_token": decode_seconds / decode_tokens
        if decode_tokens else None,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

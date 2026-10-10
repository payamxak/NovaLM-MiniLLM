"""Compare FP32, dynamic INT8, and reference weight-only INT4 on synthetic prompts."""
import argparse
import io
import json
import platform
import time
from datetime import UTC, datetime

import torch
from torch.nn import functional as F

from novalm.inference.quantization import dynamic_int8, quantize_int4
from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM


def evaluate(model, ids, iterations):
    model.eval()
    with torch.inference_mode():
        logits = model(ids).float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1))
        for _ in range(2):
            model(ids)
        started = time.perf_counter()
        for _ in range(iterations):
            model(ids)
        elapsed = time.perf_counter() - started
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        greedy = logits.argmax(dim=-1)
    return {
        "loss_on_synthetic_tokens": float(loss),
        "seconds_per_forward": elapsed / iterations,
        "tokens_per_second": ids.numel() * iterations / elapsed,
        "serialized_state_bytes": buffer.tell(),
        "greedy_tokens": greedy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.sequence_length, args.batch_size, args.iterations) < 1:
        parser.error("sequence-length, batch-size, and iterations must be positive")
    torch.manual_seed(73)
    config = ModelConfig(vocab_size=256, hidden_size=128, num_layers=2,
                         num_attention_heads=4, num_kv_heads=2,
                         intermediate_size=352, max_seq_length=args.sequence_length,
                         attention_backend="sdpa")
    fp32_model = NovaLM(config).eval()
    int8_model = dynamic_int8(fp32_model).eval()
    int4_model = quantize_int4(fp32_model).eval()
    ids = torch.randint(0, config.vocab_size, (args.batch_size, args.sequence_length))
    fp32 = evaluate(fp32_model, ids, args.iterations)
    int8 = evaluate(int8_model, ids, args.iterations)
    int4 = evaluate(int4_model, ids, args.iterations)
    token_agreement = float((fp32["greedy_tokens"] == int8["greedy_tokens"]).float().mean())
    int4_agreement = float((fp32["greedy_tokens"] == int4["greedy_tokens"]).float().mean())
    fp32_loss, int8_loss = fp32.pop("loss_on_synthetic_tokens"), int8.pop("loss_on_synthetic_tokens")
    fp32.pop("greedy_tokens")
    int8.pop("greedy_tokens")
    int4_loss = int4.pop("loss_on_synthetic_tokens")
    int4.pop("greedy_tokens")
    print(json.dumps({
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "shape": list(ids.shape),
        "parameter_count": fp32_model.parameter_count,
        "iterations": args.iterations,
        "synthetic_data": True,
        "quality_evaluated": False,
        "synthetic_token_loss_delta": int8_loss - fp32_loss,
        "greedy_token_agreement": token_agreement,
        "synthetic_int4_loss_delta": int4_loss - fp32_loss,
        "int4_greedy_token_agreement": int4_agreement,
        "results": {"fp32": fp32, "dynamic_int8": int8, "weight_only_int4": int4},
    }, indent=2))


if __name__ == "__main__":
    main()

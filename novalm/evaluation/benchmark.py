"""Basic model-forward latency and token-throughput measurement."""
import time

import torch


@torch.inference_mode()
def benchmark_forward(model, input_ids: torch.Tensor, warmup: int = 2,
                      iterations: int = 10) -> dict[str, float]:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    model.eval()
    for _ in range(warmup):
        model(input_ids)
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    started = time.perf_counter()
    for _ in range(iterations):
        model(input_ids)
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    elapsed = time.perf_counter() - started
    tokens = input_ids.numel() * iterations
    return {"iterations": float(iterations), "elapsed_seconds": elapsed,
            "milliseconds_per_forward": elapsed * 1000 / iterations,
            "tokens_per_second": tokens / elapsed}

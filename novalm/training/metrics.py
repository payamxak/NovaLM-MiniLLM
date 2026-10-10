"""Small deterministic metric helpers used by training and experiment scripts."""
import math


def effective_batch_size(micro_batch: int, accumulation_steps: int,
                         world_size: int = 1) -> int:
    if min(micro_batch, accumulation_steps, world_size) < 1:
        raise ValueError("batch size, accumulation, and world size must be positive")
    return micro_batch * accumulation_steps * world_size


def tokens_per_second(token_count: int, elapsed_seconds: float) -> float:
    if token_count < 0 or elapsed_seconds <= 0:
        raise ValueError("token_count must be nonnegative and elapsed time positive")
    return token_count / elapsed_seconds


def perplexity(loss: float) -> float | None:
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("loss must be finite and nonnegative")
    return math.exp(loss) if loss < 709 else None

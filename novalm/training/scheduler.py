"""Warmup followed by cosine learning-rate decay."""
import math


def warmup_cosine_lr(step: int, learning_rate: float, warmup_steps: int,
                     total_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < 0 or learning_rate <= 0 or warmup_steps < 0 or total_steps < 1:
        raise ValueError("invalid scheduler step or configuration")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if step < warmup_steps:
        scale = (step + 1) / max(warmup_steps, 1)
    else:
        progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        scale = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return learning_rate * scale

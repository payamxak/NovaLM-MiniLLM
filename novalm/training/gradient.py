"""Gradient clipping with explicit non-finite failure handling."""
import torch
from torch import nn


def clip_gradients(model: nn.Module, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    return norm

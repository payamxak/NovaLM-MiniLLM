"""Utilities for the per-layer tuple K/V cache used by NovaLM decode."""
from collections.abc import Sequence

import torch

LayerCache = tuple[torch.Tensor, torch.Tensor]
KVCache = tuple[LayerCache, ...]


def cache_length(cache: Sequence[LayerCache] | None) -> int:
    if not cache:
        return 0
    key = cache[0][0]
    if key.ndim != 4:
        raise ValueError("K/V cache tensors must be [batch, heads, sequence, dim]")
    length = key.size(-2)
    for layer_key, layer_value in cache:
        if layer_key.shape != layer_value.shape or layer_key.size(-2) != length:
            raise ValueError("K/V cache layers have inconsistent shapes or lengths")
    return length


def cache_bytes(cache: Sequence[LayerCache] | None) -> int:
    if not cache:
        return 0
    return sum(key.numel() * key.element_size() + value.numel() * value.element_size()
               for key, value in cache)

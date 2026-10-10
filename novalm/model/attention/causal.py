"""Reference causal mask and mathematically explicit attention implementation."""
import torch


def causal_mask(query_length: int, key_length: int, *, query_offset: int = 0,
                device: torch.device | str | None = None) -> torch.Tensor:
    if query_length < 1 or key_length < 1 or query_offset < 0:
        raise ValueError("attention lengths must be positive and query_offset nonnegative")
    query_positions = query_offset + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    if query_offset + query_length > key_length:
        raise ValueError("key sequence does not cover all query positions")
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def naive_causal_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           allowed: torch.Tensor) -> torch.Tensor:
    """Compute softmax(QKᵀ/sqrt(d))V using the explicit reference path."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be [batch, heads, sequence, dim]")
    if key.shape != value.shape or query.shape[:2] != key.shape[:2]:
        raise ValueError("attention batch/head dimensions and key/value shapes must match")
    if query.size(-1) != key.size(-1) or allowed.shape != (query.size(-2), key.size(-2)):
        raise ValueError("attention dimensions or causal mask shape are incompatible")
    scores = query @ key.transpose(-2, -1) / query.size(-1) ** 0.5
    scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    return scores.softmax(dim=-1) @ value

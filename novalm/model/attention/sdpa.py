"""PyTorch scaled-dot-product attention backend."""
import torch
from torch.nn import functional as F


def sdpa_causal_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          allowed: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, attn_mask=allowed,
                                          is_causal=False)

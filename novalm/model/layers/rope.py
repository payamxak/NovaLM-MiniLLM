import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_length: int = 128, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.head_dim, self.theta = head_dim, theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._cache_length = 0
        self._ensure_cache(max_seq_length, torch.device("cpu"))

    def _ensure_cache(self, length: int, device: torch.device) -> None:
        if length <= self._cache_length and self.cos_cached.device == device:
            return
        # Keep phase construction in fp32 even after the enclosing model is cast.
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device, dtype=torch.float32))
        self.cos_cached = angles.cos()
        self.sin_cached = angles.sin()
        self._cache_length = length

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        # x: [batch, heads, sequence, head_dim]
        length = x.shape[-2]
        end = position_offset + length
        self._ensure_cache(end, x.device)
        cos = self.cos_cached[position_offset:end].to(x.dtype)[None, None, :, :]
        sin = self.sin_cached[position_offset:end].to(x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)

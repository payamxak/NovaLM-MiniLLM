"""Common typing contract for attention backend callables."""
from typing import Protocol

import torch


class AttentionBackend(Protocol):
    def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 allowed: torch.Tensor) -> torch.Tensor: ...

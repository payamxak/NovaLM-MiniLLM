"""Small explicit activation functions used by the feed-forward variants."""
import torch
from torch.nn import functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)

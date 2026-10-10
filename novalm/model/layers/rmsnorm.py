import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0 or eps <= 0:
            raise ValueError("dim and eps must be positive")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate in fp32 for low-precision inputs, then return the input dtype.
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype=x.dtype)

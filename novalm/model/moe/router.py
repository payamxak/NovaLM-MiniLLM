"""Learned softmax router for token-to-expert assignments."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Router(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        if hidden_size <= 0 or num_experts < 2:
            raise ValueError("hidden_size must be positive and num_experts >= 2")
        # Keep the historical `router.weight` state-dict key for checkpoint
        # compatibility with the earlier integrated MoE implementation.
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, tokens):
        return F.linear(tokens, self.weight).softmax(dim=-1)

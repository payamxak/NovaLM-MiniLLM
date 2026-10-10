import torch
from torch import nn

from novalm.model.moe.expert import Expert
from novalm.model.moe.load_balance import load_balance_loss
from novalm.model.moe.router import Router
from novalm.model.moe.topk import topk_routes


class MoELayer(nn.Module):
    """Readable top-k expert reference layer; not capacity limited or optimized."""
    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        if num_experts < 2 or not 1 <= top_k <= num_experts:
            raise ValueError("require num_experts >= 2 and 1 <= top_k <= num_experts")
        self.router = Router(hidden_size, num_experts)
        self.experts = nn.ModuleList(Expert(hidden_size, intermediate_size)
                                     for _ in range(num_experts))
        self.num_experts, self.top_k = num_experts, top_k

    def forward(self, x: torch.Tensor):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probs = self.router(flat)
        weights, indices = topk_routes(probs, self.top_k)
        output = torch.zeros_like(flat)
        aux_loss, counts = load_balance_loss(probs, indices, self.num_experts)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            if rows.numel():
                output.index_add_(0, rows, expert(flat[rows]) * weights[rows, slots, None])
        metrics = {"expert_token_counts": counts.detach(),
                   "router_entropy": (-(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()).detach(),
                   "dead_experts": (counts == 0).sum().detach(),
                   "aux_loss": aux_loss}
        return output.reshape(shape), aux_loss, metrics

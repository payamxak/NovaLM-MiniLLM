"""Top-k expert assignment with renormalized routing weights."""
import torch


def topk_routes(probabilities: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if probabilities.ndim != 2 or not 1 <= top_k <= probabilities.size(-1):
        raise ValueError("probabilities must be [tokens, experts] and top_k in range")
    weights, indices = probabilities.topk(top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny)
    return weights, indices

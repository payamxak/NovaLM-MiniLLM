"""Load-balancing objective and expert-use metrics for top-k routing."""
import torch


def load_balance_loss(probabilities: torch.Tensor, indices: torch.Tensor,
                      num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    if probabilities.ndim != 2 or probabilities.size(-1) != num_experts:
        raise ValueError("probabilities must have shape [tokens, num_experts]")
    counts = torch.bincount(indices.reshape(-1), minlength=num_experts)
    importance = probabilities.mean(dim=0)
    assignments = max(1, indices.numel())
    load = counts.to(probabilities.dtype) / assignments
    return num_experts * torch.sum(importance * load), counts

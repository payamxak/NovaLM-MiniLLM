import torch
from torch.nn import functional as F


def multi_token_loss(logits_by_horizon: list[torch.Tensor], token_ids: torch.Tensor,
                     weights: list[float] | None = None, ignore_index: int = -100):
    """Each auxiliary head at t predicts token t+h, with boundary positions excluded."""
    if weights is None:
        weights = [1.0] * len(logits_by_horizon)
    if len(weights) != len(logits_by_horizon):
        raise ValueError("weights and horizon logits must have equal lengths")
    losses = {}
    for horizon, (logits, weight) in enumerate(zip(logits_by_horizon, weights), start=2):
        valid = token_ids.size(1) - horizon
        if valid <= 0:
            raise ValueError("sequence is too short for requested MTP horizons")
        targets = token_ids[:, horizon:horizon + valid]
        losses[horizon] = F.cross_entropy(
            logits[:, :valid].reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=ignore_index)
    total = sum(weights[i] * losses[h] for i, h in enumerate(losses))
    return total, losses

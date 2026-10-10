"""Experimental group-relative policy objective; reward design is caller-owned."""
import torch


def group_relative_policy_loss(policy_logps: torch.Tensor, old_policy_logps: torch.Tensor,
                               advantages: torch.Tensor,
                               clip_range: float = 0.2) -> torch.Tensor:
    """Clipped token/sample policy objective for externally computed group advantages.

    This is an objective primitive only; it does not define rewards, sampling,
    KL control, or an evaluation benchmark.
    """
    if policy_logps.shape != old_policy_logps.shape or policy_logps.shape != advantages.shape:
        raise ValueError("policy log-probabilities and advantages must have identical shapes")
    if policy_logps.numel() == 0 or not 0 < clip_range < 1:
        raise ValueError("inputs must be nonempty and clip_range must be in (0, 1)")
    ratio = torch.exp(policy_logps - old_policy_logps)
    clipped = ratio.clamp(1 - clip_range, 1 + clip_range)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()

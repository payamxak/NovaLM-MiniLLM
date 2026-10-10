import torch
from torch.nn import functional as F


def completion_ids(tokenizer, prompt: str, completion: str, max_length: int):
    """Encode a preference pair while preserving at least one prompt and target token."""
    if max_length < 2:
        raise ValueError("DPO max_length must leave room for a conditioning token and target")
    prefix = tokenizer.encode(prompt)
    answer = tokenizer.encode(completion, add_eos=True)[:max_length - 1]
    if not prefix:
        prefix = [tokenizer.processor.bos_id()]
    if not answer:
        raise ValueError("preference completion encoded to no target tokens")
    prefix = prefix[-(max_length - len(answer)):]
    ids = prefix + answer
    labels = [-100] * len(prefix) + answer
    return torch.tensor([ids], dtype=torch.long), torch.tensor([labels], dtype=torch.long)


def sequence_log_probability(logits: torch.Tensor, labels: torch.Tensor,
                             ignore_index: int = -100) -> torch.Tensor:
    """Sum token log probabilities for each batch row, excluding masked labels."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have shapes [batch, sequence, vocab] and [batch, sequence]")
    shifted_logits, shifted_labels = logits[:, :-1].float(), labels[:, 1:]
    mask = shifted_labels.ne(ignore_index)
    if not mask.any(dim=1).all():
        raise ValueError("every preference completion must contain a supervised next-token target")
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
             beta: float = 0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct Preference Optimization objective from policy/reference sequence logps."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()
    return loss, {"policy_margin": policy_margin.detach(),
                  "reference_margin": reference_margin.detach(),
                  "accuracy": (logits > 0).float().mean().detach()}

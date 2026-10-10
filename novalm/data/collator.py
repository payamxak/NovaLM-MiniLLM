"""Batch token sequences with explicit next-token inputs, labels, and padding masks."""
import torch


def causal_lm_collate(examples: list[torch.Tensor | list[int]], pad_id: int = 0,
                      ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length examples and mask pad labels; input/target shift is model-side."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    sequences = [torch.as_tensor(example, dtype=torch.long) for example in examples]
    if any(sequence.ndim != 1 or sequence.numel() < 2 for sequence in sequences):
        raise ValueError("every sequence must be one-dimensional with at least two tokens")
    length = max(sequence.numel() for sequence in sequences)
    inputs = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    labels = torch.full((len(sequences), length), ignore_index, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        inputs[index, :sequence.numel()] = sequence
        labels[index, :sequence.numel()] = sequence
    return inputs, labels


def shift_labels(input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                 ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    """Align inputs and targets so logits at t predict labels at t+1."""
    if input_ids.ndim != 2 or input_ids.size(1) < 2:
        raise ValueError("input_ids must have shape [batch, sequence>=2]")
    if labels is None:
        labels = input_ids
    if labels.shape != input_ids.shape:
        raise ValueError("labels and input_ids must have matching shapes")
    inputs = input_ids[:, :-1]
    targets = labels[:, 1:].clone()
    return inputs, targets


__all__ = ["causal_lm_collate", "shift_labels"]

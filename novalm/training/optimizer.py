"""AdamW construction with bias and normalization parameters exempt from decay."""
import torch
from torch import nn


def build_adamw(model: nn.Module, learning_rate: float = 3e-4,
                weight_decay: float = 0.1,
                betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

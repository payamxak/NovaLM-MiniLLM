"""Static same-length prompt batching for decoder-only generation."""
import torch

from novalm.inference.sampler import sample_next_token


@torch.inference_mode()
def batch_generate(model, input_ids: torch.Tensor, max_new_tokens: int,
                   eos_token_id: int | None = None, temperature: float = 1.0,
                   top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
                   repetition_penalty: float = 1.0, pad_token_id: int = 0) -> torch.Tensor:
    """Generate for an equal-length prompt batch, returning prompt plus continuations.

    Variable prompt lengths need an attention padding mask and are rejected because
    the current NovaLM attention API intentionally does not implement one.
    """
    if input_ids.ndim != 2 or input_ids.size(0) < 1 or input_ids.size(1) < 1:
        raise ValueError("input_ids must have shape [batch>=1, sequence>=1]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.size(1) + max_new_tokens > model.config.max_seq_length:
        raise ValueError("prompt plus max_new_tokens exceeds model context length")
    model.eval()
    tokens = input_ids
    logits, cache = model(input_ids, use_cache=True)
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
        next_token = sample_next_token(logits[:, -1], temperature, top_k, top_p, min_p,
                                       tokens, repetition_penalty)
        if eos_token_id is not None:
            pad = torch.full_like(next_token, pad_token_id)
            next_token = torch.where(finished[:, None], pad, next_token)
            finished |= next_token[:, 0].eq(eos_token_id)
        tokens = torch.cat((tokens, next_token), dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break
        logits, cache = model(next_token, past_key_values=cache, use_cache=True)
    return tokens

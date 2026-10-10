from collections.abc import Iterator

import torch

from novalm.inference.sampler import sample_next_token


@torch.inference_mode()
def stream_generate(model, input_ids: torch.Tensor, max_new_tokens: int,
                    eos_token_id: int | None = None, temperature: float = 1.0,
                    top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
                    repetition_penalty: float = 1.0,
                    stop_sequences: list[list[int]] | None = None) -> Iterator[torch.Tensor]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must be nonempty [batch, sequence]")
    if input_ids.size(1) + max_new_tokens > model.config.max_seq_length:
        raise ValueError("prompt plus max_new_tokens exceeds model context length")
    model.eval()
    tokens = input_ids
    logits, cache = model(input_ids, use_cache=True)
    for _ in range(max_new_tokens):
        nxt = sample_next_token(logits[:, -1], temperature, top_k, top_p, min_p,
                                tokens, repetition_penalty)
        tokens = torch.cat((tokens, nxt), dim=1)
        yield nxt
        if eos_token_id is not None and torch.all(nxt == eos_token_id):
            break
        if stop_sequences and tokens.size(0) == 1 and any(
                len(seq) <= tokens.size(1) and tokens[0, -len(seq):].tolist() == seq
            for seq in stop_sequences if seq):
            break
        logits, cache = model(nxt, past_key_values=cache, use_cache=True)


def generate(model, input_ids: torch.Tensor, max_new_tokens: int = 32, **kwargs) -> torch.Tensor:
    tokens = input_ids
    for next_token in stream_generate(model, input_ids, max_new_tokens, **kwargs):
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens

import torch


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                      top_p: float = 1.0, min_p: float = 0.0,
                      repetition_tokens: torch.Tensor | None = None,
                      repetition_penalty: float = 1.0) -> torch.Tensor:
    if temperature < 0 or top_k < 0 or not 0 < top_p <= 1 or not 0 <= min_p <= 1:
        raise ValueError("invalid sampling parameters")
    scores = logits.float().clone()
    if repetition_tokens is not None and repetition_penalty != 1:
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        token_rows = repetition_tokens.unsqueeze(0) if repetition_tokens.ndim == 1 else repetition_tokens
        score_rows = scores.unsqueeze(0) if scores.ndim == 1 else scores
        if token_rows.size(0) != score_rows.size(0):
            raise ValueError("repetition token batch must match logits batch")
        for row in range(score_rows.size(0)):
            for token in token_rows[row].unique():
                value = score_rows[row, token]
                score_rows[row, token] = torch.where(value < 0, value * repetition_penalty,
                                                       value / repetition_penalty)
        if scores.ndim == 1:
            scores = score_rows[0]
    if temperature == 0:
        return scores.argmax(dim=-1, keepdim=True)
    scores /= temperature
    if top_k:
        threshold = scores.topk(min(top_k, scores.size(-1))).values[..., -1, None]
        scores.masked_fill_(scores < threshold, -torch.inf)
    if top_p < 1:
        sorted_scores, sorted_idx = scores.sort(descending=True)
        probs = sorted_scores.softmax(-1)
        remove = probs.cumsum(-1) - probs >= top_p
        scores.scatter_(-1, sorted_idx, sorted_scores.masked_fill(remove, -torch.inf))
    if min_p:
        probs = scores.softmax(-1)
        scores.masked_fill_(probs < probs.max(-1, keepdim=True).values * min_p, -torch.inf)
    return torch.multinomial(scores.softmax(-1), 1)

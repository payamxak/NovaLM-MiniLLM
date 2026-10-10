"""Perplexity measurements on token windows grouped by context length."""
from novalm.evaluation.perplexity import evaluate_loss


def evaluate_context_lengths(model, token_batches: dict[int, list],
                             device="cpu") -> dict[int, dict[str, float]]:
    results = {}
    for context_length, batches in sorted(token_batches.items()):
        if context_length < 2:
            raise ValueError("context lengths must be at least two")
        for batch in batches:
            if batch.size(1) > context_length:
                raise ValueError("batch sequence exceeds its declared context length")
        results[context_length] = evaluate_loss(model, batches, device)
    if not results:
        raise ValueError("at least one context length is required")
    return results

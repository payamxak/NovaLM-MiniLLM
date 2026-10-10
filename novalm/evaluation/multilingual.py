"""Per-language perplexity evaluation from named token-batch groups."""
from novalm.evaluation.perplexity import evaluate_loss


def evaluate_languages(model, language_batches: dict[str, list],
                       device="cpu") -> dict[str, dict[str, float]]:
    if not language_batches:
        raise ValueError("language_batches cannot be empty")
    return {language: evaluate_loss(model, batches, device)
            for language, batches in sorted(language_batches.items())}

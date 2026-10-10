import math

import torch


@torch.inference_mode()
def evaluate_loss(model, batches, device: str | torch.device = "cpu") -> dict[str, float]:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in batches:
        batch = batch.to(device)
        logits = model(batch[:, :-1]).float()
        labels = batch[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
        total_nll += float(nll)
        total_tokens += labels.numel()
    if not math.isfinite(total_nll):
        raise FloatingPointError("non-finite evaluation loss")
    if not total_tokens:
        raise ValueError("evaluation dataset is empty")
    loss = total_nll / total_tokens
    perplexity = math.exp(loss) if loss < 709 else None
    return {"loss": loss, "perplexity": perplexity,
            "tokens": float(total_tokens)}

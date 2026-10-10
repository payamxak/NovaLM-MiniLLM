"""Small high-level inference facade over the hand-written NovaLM decoder."""
from collections.abc import Iterator

import torch

from novalm.inference.generate import generate, stream_generate


class InferenceEngine:
    def __init__(self, model, tokenizer=None) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer

    def stream_ids(self, input_ids: torch.Tensor, **generation_options) -> Iterator[torch.Tensor]:
        return stream_generate(self.model, input_ids, **generation_options)

    def generate_ids(self, input_ids: torch.Tensor, **generation_options) -> torch.Tensor:
        return generate(self.model, input_ids, **generation_options)

    def generate_text(self, prompt: str, **generation_options) -> str:
        if self.tokenizer is None:
            raise RuntimeError("generate_text requires a tokenizer")
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated = self.generate_ids(input_ids, **generation_options)
        return self.tokenizer.decode(generated[0, input_ids.size(1):].tolist())

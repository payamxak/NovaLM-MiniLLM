"""Controlled generation samples with explicit decoding metadata."""
from collections.abc import Iterable

from novalm.inference.engine import InferenceEngine


def generate_examples(engine: InferenceEngine, prompts: Iterable[str], **options) -> list[dict]:
    if engine.tokenizer is None:
        raise RuntimeError("generation evaluation requires a tokenizer")
    return [{"prompt": prompt, "completion": engine.generate_text(prompt, **options)}
            for prompt in prompts]

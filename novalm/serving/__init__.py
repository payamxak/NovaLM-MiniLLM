"""HTTP serving for an explicitly loaded NovaLM model."""
from novalm.serving.app import create_app
from novalm.serving.batching import batch_generate

__all__ = ["batch_generate", "create_app"]

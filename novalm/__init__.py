"""NovaLM: a from-scratch decoder-only language model."""

from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM

__all__ = ["ModelConfig", "NovaLM"]

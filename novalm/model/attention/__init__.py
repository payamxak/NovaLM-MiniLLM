"""Reference and optimized causal attention backends."""
from novalm.model.attention.gqa import GroupedQueryAttention
from novalm.model.attention.mha import MultiHeadAttention

__all__ = ["GroupedQueryAttention", "MultiHeadAttention"]

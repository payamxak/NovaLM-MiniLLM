"""Multi-head attention is the GQA implementation with one KV head per query head."""
from novalm.model.attention.gqa import GroupedQueryAttention


class MultiHeadAttention(GroupedQueryAttention):
    def __init__(self, hidden_size: int, num_heads: int, max_seq_length: int,
                 theta: float = 10000.0, eps: float = 1e-6,
                 backend: str = "naive", qk_norm: bool = True,
                 use_rope: bool = True) -> None:
        super().__init__(hidden_size, num_heads, num_heads, max_seq_length, theta,
                         eps, backend, qk_norm, use_rope)

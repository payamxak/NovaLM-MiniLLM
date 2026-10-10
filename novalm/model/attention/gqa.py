import torch
from torch import nn

from novalm.model.attention.causal import causal_mask, naive_causal_attention
from novalm.model.attention.flash import flash_causal_attention
from novalm.model.attention.sdpa import sdpa_causal_attention
from novalm.model.layers.qk_norm import QKNorm
from novalm.model.layers.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 max_seq_length: int, theta: float, eps: float, backend: str = "naive",
                 qk_norm: bool = True, use_rope: bool = True) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.groups = num_heads // num_kv_heads
        self.backend = backend
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = QKNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.k_norm = QKNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, max_seq_length, theta) if use_rope else None

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, position_offset=past_length)
            k = self.rope(k, position_offset=past_length)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        present = (k, v) if use_cache else None
        key_length = k.size(-2)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        allowed = causal_mask(t, key_length, query_offset=past_length, device=x.device)
        backend = {"naive": naive_causal_attention,
                   "sdpa": sdpa_causal_attention,
                   "flash": flash_causal_attention}[self.backend]
        out = backend(q, k, v, allowed)
        output = self.o_proj(out.transpose(1, 2).contiguous().view(b, t, -1))
        return (output, present) if use_cache else output

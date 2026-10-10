"""Small experimental latent-KV attention; equations are documented in docs/architecture.md."""
import torch
from torch import nn
from torch.nn import functional as F

from novalm.model.layers.rope import RotaryEmbedding


class LatentAttention(nn.Module):
    """Compress each token to a latent vector, then project per-head keys and values."""
    def __init__(self, hidden_size: int, num_heads: int, latent_dim: int,
                 max_seq_length: int = 2048, theta: float = 10000.0,
                 use_rope: bool = True) -> None:
        super().__init__()
        if hidden_size % num_heads or (use_rope and (hidden_size // num_heads) % 2):
            raise ValueError("hidden/head dimensions must divide evenly and RoPE head_dim must be even")
        self.num_heads, self.head_dim = num_heads, hidden_size // num_heads
        self.down = nn.Linear(hidden_size, latent_dim, bias=False)
        self.key_up = nn.Linear(latent_dim, hidden_size, bias=False)
        self.value_up = nn.Linear(latent_dim, hidden_size, bias=False)
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_length, theta) if use_rope else None

    def forward(self, x: torch.Tensor, past_latent: torch.Tensor | None = None,
                use_cache: bool = False):
        b, t, h = x.shape
        q = self.query(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        latent = self.down(x)
        past_len = 0 if past_latent is None else past_latent.size(1)
        if past_latent is not None:
            latent = torch.cat((past_latent, latent), dim=1)
        k = self.key_up(latent).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_up(latent).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, past_len), self.rope(k, 0)
        qpos = past_len + torch.arange(t, device=x.device)
        kpos = torch.arange(latent.size(1), device=x.device)
        allowed = kpos[None, :] <= qpos[:, None]
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        result = self.output(result.transpose(1, 2).reshape(b, t, h))
        return (result, latent) if use_cache else result

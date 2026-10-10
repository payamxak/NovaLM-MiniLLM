from torch import nn

from novalm.model.attention import GroupedQueryAttention
from novalm.model.attention_mla import LatentAttention
from novalm.model.config import ModelConfig
from novalm.model.layers.gelu_mlp import GELUMLP
from novalm.model.layers.rmsnorm import RMSNorm
from novalm.model.layers.swiglu import SwiGLU
from novalm.model.moe.layer import MoELayer


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        norm = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.attn_norm = norm(config.hidden_size, config.norm_eps)
        if config.attention_type == "mla":
            self.attention = LatentAttention(config.hidden_size, config.num_attention_heads,
                                             config.mla_latent_dim, config.max_seq_length,
                                             config.rope_theta, config.position_type == "rope")
        else:
            self.attention = GroupedQueryAttention(
                config.hidden_size, config.num_attention_heads, config.num_kv_heads,
                config.max_seq_length, config.rope_theta, config.norm_eps,
                config.attention_backend, config.qk_norm,
                config.position_type == "rope",
            )
        self.mlp_norm = norm(config.hidden_size, config.norm_eps)
        self.mlp = (MoELayer(config.hidden_size, config.intermediate_size,
                             config.num_experts, config.moe_top_k)
                    if config.moe_enabled else (SwiGLU(config.hidden_size, config.intermediate_size)
                    if config.mlp_type == "swiglu" else GELUMLP(config.hidden_size, config.intermediate_size)))

    def forward(self, x, past_key_value=None, use_cache: bool = False,
                return_aux: bool = False):
        if use_cache and return_aux:
            raise ValueError("cache output and training auxiliary outputs are separate modes")
        attention = self.attention(self.attn_norm(x), past_key_value, use_cache)
        if use_cache:
            attention, present = attention
        x = x + attention
        mlp_result = self.mlp(self.mlp_norm(x))
        aux_loss = x.new_zeros(())
        moe_metrics = None
        if isinstance(self.mlp, MoELayer):
            mlp_result, aux_loss, moe_metrics = mlp_result
        x = x + mlp_result
        if use_cache:
            return x, present
        if return_aux:
            return x, aux_loss, moe_metrics
        return x

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    hidden_size: int = 128
    num_layers: int = 2
    num_attention_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 352
    max_seq_length: int = 128
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    attention_backend: str = "naive"
    qk_norm: bool = True
    attention_type: str = "gqa"
    mla_latent_dim: int = 128
    moe_enabled: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01
    mtp_enabled: bool = False
    mtp_horizons: int = 0
    mtp_loss_weight: float = 0.25
    norm_type: str = "rmsnorm"
    mlp_type: str = "swiglu"
    position_type: str = "rope"

    def __post_init__(self) -> None:
        positive = ("vocab_size", "hidden_size", "num_layers", "num_attention_heads",
                    "num_kv_heads", "intermediate_size", "max_seq_length")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_kv_heads:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be positive")
        if self.position_type == "rope" and self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.attention_backend not in {"naive", "sdpa", "flash"}:
            raise ValueError("attention_backend must be 'naive', 'sdpa' or 'flash'")
        if self.attention_type not in {"gqa", "mla"}:
            raise ValueError("attention_type must be 'gqa' or 'mla'")
        if self.norm_type not in {"rmsnorm", "layernorm"}:
            raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")
        if self.mlp_type not in {"swiglu", "gelu"}:
            raise ValueError("mlp_type must be 'swiglu' or 'gelu'")
        if self.position_type not in {"rope", "absolute"}:
            raise ValueError("position_type must be 'rope' or 'absolute'")
        if self.moe_enabled and self.mlp_type != "swiglu":
            raise ValueError("MoE experts currently use SwiGLU")
        if self.attention_type == "mla" and self.mla_latent_dim <= 0:
            raise ValueError("mla_latent_dim must be positive")
        if self.num_experts < 2 or not 1 <= self.moe_top_k <= self.num_experts:
            raise ValueError("MoE requires num_experts >= 2 and 1 <= moe_top_k <= num_experts")
        if self.mtp_enabled and self.mtp_horizons < 1:
            raise ValueError("mtp_horizons must be >= 1 when MTP is enabled")
        if self.moe_aux_loss_weight < 0 or self.mtp_loss_weight < 0:
            raise ValueError("auxiliary loss weights cannot be negative")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def parameter_count(self) -> int:
        # Exact for this implementation, including optional tied token/output weights.
        h, v, f = self.hidden_size, self.vocab_size, self.intermediate_size
        embedding = v * h
        if self.attention_type == "gqa":
            attention = h * (h + 2 * (self.num_kv_heads * self.head_dim)) + h * h
            qknorm = 2 * self.head_dim if self.qk_norm else 0
        else:
            attention = 2 * h * h + 3 * h * self.mla_latent_dim
            qknorm = 0
        mlp = (self.num_experts * 3 * h * f + h * self.num_experts
               if self.moe_enabled else (3 if self.mlp_type == "swiglu" else 2) * h * f)
        norm_size = h if self.norm_type == "rmsnorm" else 2 * h
        norms = 2 * norm_size
        block = attention + qknorm + mlp + norms
        final_norm = norm_size
        output = 0 if self.tie_embeddings else v * h
        mtp = self.mtp_horizons * h * v if self.mtp_enabled else 0
        positions = self.max_seq_length * h if self.position_type == "absolute" else 0
        return embedding + positions + self.num_layers * block + final_norm + output + mtp


def validate_checkpoint_config(config: ModelConfig, checkpoint: dict) -> None:
    saved = checkpoint.get("model_config")
    if saved is not None and saved != asdict(config):
        differing = [key for key, value in asdict(config).items() if saved.get(key) != value]
        raise ValueError(f"checkpoint model config is incompatible; differing fields: {differing}")

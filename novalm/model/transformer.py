import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from novalm.model.block import TransformerBlock
from novalm.model.config import ModelConfig
from novalm.model.embeddings import TokenEmbedding
from novalm.model.layers.rmsnorm import RMSNorm
from novalm.model.lm_head import LMHead
from novalm.model.mtp.head import MultiTokenPredictionHead


class NovaLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.activation_checkpointing = False
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        self.position_embedding = (nn.Embedding(config.max_seq_length, config.hidden_size)
                                   if config.position_type == "absolute" else None)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        norm = RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.final_norm = norm(config.hidden_size, config.norm_eps)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size)
        self.mtp_head = (MultiTokenPredictionHead(config.hidden_size, config.vocab_size,
                                                 config.mtp_horizons)
                         if config.mtp_enabled else None)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, past_key_values=None,
                use_cache: bool = False, return_aux: bool = False) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if use_cache and return_aux:
            raise ValueError("cache output and training auxiliary outputs are separate modes")
        if past_key_values is None:
            past_length = 0
        elif isinstance(past_key_values[0], torch.Tensor):
            past_length = past_key_values[0].shape[1]
        else:
            past_length = past_key_values[0][0].shape[-2]
        if past_length + input_ids.shape[1] > self.config.max_seq_length:
            raise ValueError("sequence length exceeds model max_seq_length")
        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(past_length, past_length + input_ids.shape[1],
                                     device=input_ids.device)
            x = x + self.position_embedding(positions).unsqueeze(0)
        presents = []
        auxiliary_losses = []
        moe_metrics = []
        for i, block in enumerate(self.blocks):
            cache = None if past_key_values is None else past_key_values[i]
            should_checkpoint = (self.activation_checkpointing and self.training
                                 and not use_cache)
            if should_checkpoint and return_aux:
                def run_block(hidden, current_block=block):
                    hidden, aux_loss, metrics = current_block(hidden, return_aux=True)
                    if metrics is None:
                        empty = hidden.new_empty((0,))
                        scalar = hidden.new_zeros(())
                        return hidden, aux_loss, empty, scalar, scalar
                    return (hidden, aux_loss, metrics["expert_token_counts"],
                            metrics["router_entropy"], metrics["dead_experts"])

                x, aux_loss, expert_counts, router_entropy, dead_experts = checkpoint(
                    run_block, x, use_reentrant=False)
                layer_metrics = None
                if self.config.moe_enabled:
                    layer_metrics = {
                        "expert_token_counts": expert_counts,
                        "router_entropy": router_entropy,
                        "dead_experts": dead_experts,
                    }
                result = (x, aux_loss, layer_metrics)
            elif should_checkpoint:
                result = checkpoint(block, x, use_reentrant=False)
            else:
                result = block(x, cache, use_cache, return_aux)
            if use_cache:
                x, present = result
                presents.append(present)
            elif return_aux:
                x, aux_loss, layer_metrics = result
                auxiliary_losses.append(aux_loss)
                if layer_metrics is not None:
                    moe_metrics.append(layer_metrics)
            else:
                x = result
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        if return_aux:
            aux = {"moe_loss": torch.stack(auxiliary_losses).mean(),
                   "moe_metrics": moe_metrics,
                   "mtp_logits": self.mtp_head(hidden) if self.mtp_head is not None else None}
            return logits, aux
        return (logits, tuple(presents)) if use_cache else logits

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor,
                   ignore_index: int = -100) -> torch.Tensor:
    """Next-token cross entropy: logits at t predict input_ids at t+1."""
    return nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1), ignore_index=ignore_index,
    )

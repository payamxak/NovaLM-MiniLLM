"""Validated configuration dataclasses shared by the NovaLM command-line tools."""
from dataclasses import dataclass

from novalm.model.config import ModelConfig


@dataclass(frozen=True)
class AttentionConfig:
    backend: str = "naive"
    qk_norm: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"naive", "sdpa", "flash"}:
            raise ValueError("backend must be naive, sdpa, or flash")


@dataclass(frozen=True)
class MoEConfig:
    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2
    aux_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.num_experts < 2 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("MoE requires num_experts >= 2 and 1 <= top_k <= num_experts")
        if self.aux_loss_weight < 0:
            raise ValueError("aux_loss_weight cannot be negative")


@dataclass(frozen=True)
class MTPConfig:
    enabled: bool = False
    horizons: int = 0
    loss_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.horizons < 0 or (self.enabled and self.horizons < 1):
            raise ValueError("enabled MTP requires at least one horizon")
        if self.loss_weight < 0:
            raise ValueError("loss_weight cannot be negative")


@dataclass(frozen=True)
class DataConfig:
    block_size: int = 512
    shard_tokens: int = 1_000_000
    seed: int = 0
    near_dedup: bool = False
    near_dedup_threshold: float = 0.85

    def __post_init__(self) -> None:
        if self.block_size < 2 or self.shard_tokens < 1:
            raise ValueError("block_size must be >= 2 and shard_tokens must be positive")
        if not 0 <= self.near_dedup_threshold <= 1:
            raise ValueError("near_dedup_threshold must be in [0, 1]")


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 32_000
    character_coverage: float = 0.9995

    def __post_init__(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size must be at least 8")
        if not 0 < self.character_coverage <= 1:
            raise ValueError("character_coverage must be in (0, 1]")


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate/grad_clip must be positive and weight_decay nonnegative")
        if len(self.betas) != 2 or any(not 0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")


@dataclass(frozen=True)
class SchedulerConfig:
    warmup_steps: int = 10
    total_steps: int = 100
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.total_steps < 1:
            raise ValueError("warmup_steps must be nonnegative and total_steps positive")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")


@dataclass(frozen=True)
class TrainingConfig:
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    activation_checkpointing: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")


@dataclass(frozen=True)
class InferenceConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0 or self.top_k < 0:
            raise ValueError("max_new_tokens and top_k cannot be negative")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("invalid sampling temperature/top_p/min_p")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    max_batches: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_batches is not None and self.max_batches < 1:
            raise ValueError("max_batches must be positive when set")


__all__ = [
    "AttentionConfig",
    "DataConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "MTPConfig",
    "MoEConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TokenizerConfig",
    "TrainingConfig",
]

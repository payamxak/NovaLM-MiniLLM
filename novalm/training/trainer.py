import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from novalm.evaluation.perplexity import evaluate_loss
from novalm.model.mtp.loss import multi_token_loss
from novalm.training.amp import autocast_context, make_grad_scaler
from novalm.training.checkpoint import atomic_torch_save, load_torch_checkpoint
from novalm.training.gradient import clip_gradients
from novalm.training.metrics import tokens_per_second
from novalm.training.optimizer import build_adamw
from novalm.training.scheduler import warmup_cosine_lr


@dataclass
class TrainerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 10
    total_steps: int = 100
    min_lr_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    precision: str = "fp32"
    activation_checkpointing: bool = False


class Trainer:
    def __init__(self, model: nn.Module, config: TrainerConfig | None = None,
                 device: str | torch.device = "cpu", *, rank: int = 0,
                 world_size: int = 1) -> None:
        if config is None:
            config = TrainerConfig()
        if config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if config.learning_rate <= 0 or config.weight_decay < 0 or config.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive; weight_decay cannot be negative")
        if config.warmup_steps < 0 or config.total_steps < 1:
            raise ValueError("warmup_steps must be >= 0 and total_steps must be positive")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.model, self.config = model.to(device), config
        self.rank, self.world_size = rank, world_size
        self.is_main_process = rank == 0
        self._base_model.activation_checkpointing = config.activation_checkpointing
        self.device, self.step = torch.device(device), 0
        self.tokenizer_version: str | None = None
        try:
            self.git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                             text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            self.git_commit = None
        if config.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        self.optimizer = build_adamw(self.model, config.learning_rate, config.weight_decay, config.betas)
        self.scaler = make_grad_scaler(self.device, config.precision)
        self._micro_step = 0
        self.last_metrics: dict[str, float | list[int]] = {}
        self._pending_data_state: dict = {}

    @property
    def _base_model(self) -> nn.Module:
        """Return the underlying model when training through DistributedDataParallel."""
        return self.model.module if hasattr(self.model, "module") else self.model

    @property
    def model_config(self):
        return getattr(self._base_model, "config", None)

    def _lr(self) -> float:
        c, step = self.config, self.step
        return warmup_cosine_lr(step, c.learning_rate, c.warmup_steps,
                                c.total_steps, c.min_lr_ratio)

    def train_step(self, batch: torch.Tensor, labels: torch.Tensor | None = None) -> float:
        self.model.train()
        batch = batch.to(self.device)
        if batch.ndim != 2 or batch.size(1) < 2:
            raise ValueError("batch must have shape [batch, sequence>=2]")
        if labels is not None and labels.shape != batch.shape:
            raise ValueError("labels must have the same shape as input tokens")
        if self._micro_step % self.config.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)
        with autocast_context(self.device, self.config.precision):
            logits, auxiliary = self.model(batch[:, :-1], return_aux=True)
            targets = batch[:, 1:] if labels is None else labels[:, 1:].to(self.device)
            if not targets.ne(-100).any():
                raise ValueError("batch contains no supervised next-token targets")
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                      targets.reshape(-1), ignore_index=-100)
            if auxiliary["moe_loss"] is not None:
                loss = loss + self.model_config.moe_aux_loss_weight * auxiliary["moe_loss"]
            metric_values: dict[str, float | list[int]] = {}
            if auxiliary["moe_metrics"]:
                metric_values["expert_token_counts"] = [
                    int(v) for v in torch.stack(
                        [entry["expert_token_counts"] for entry in auxiliary["moe_metrics"]]
                    ).sum(0).cpu().tolist()
                ]
                metric_values["router_entropy"] = float(torch.stack(
                    [entry["router_entropy"] for entry in auxiliary["moe_metrics"]]).mean().detach())
                metric_values["dead_experts"] = int(sum(
                    int(entry["dead_experts"]) for entry in auxiliary["moe_metrics"]))
            if labels is None and auxiliary["mtp_logits"] is not None:
                mtp_loss, horizon_losses = multi_token_loss(
                    auxiliary["mtp_logits"], batch,
                    [self.model_config.mtp_loss_weight] * len(auxiliary["mtp_logits"]),
                )
                loss = loss + mtp_loss
                for horizon, horizon_loss in horizon_losses.items():
                    metric_values[f"mtp_loss_h{horizon}"] = float(horizon_loss.detach())
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        self.last_metrics = metric_values
        scaled_loss = loss / self.config.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        self._micro_step += 1
        if self._micro_step % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_gradients(self.model, self.config.grad_clip)
            self.last_metrics["grad_norm"] = float(grad_norm.detach())
            lr = self._lr()
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
        return float(loss.detach())

    def save_checkpoint(self, path: str | Path, model_config: dict | None = None,
                        data_state: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if model_config is None and self.model_config is not None:
            model_config = asdict(self.model_config)
        state = {"model": self._base_model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(), "step": self.step,
                    "micro_step": self._micro_step,
                    "data_state": data_state or {},
                    "config": asdict(self.config), "model_config": model_config,
                    "tokenizer_version": self.tokenizer_version,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "environment": {"timestamp_utc": datetime.now(UTC).isoformat(),
                                    "python": platform.python_version(), "torch": torch.__version__,
                                    "cuda": torch.version.cuda, "git_commit": self.git_commit,
                                    "device": str(self.device),
                                    "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None}}
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        atomic_torch_save(state, path)

    def load_checkpoint(self, path: str | Path) -> None:
        # Keep CPU RNG state on CPU; load_state_dict moves model/optimizer tensors
        # to their parameter devices as appropriate.
        state = load_torch_checkpoint(path, map_location="cpu")
        saved_tokenizer = state.get("tokenizer_version")
        if self.tokenizer_version and saved_tokenizer and self.tokenizer_version != saved_tokenizer:
            raise ValueError("checkpoint tokenizer version does not match current training shards")
        self.tokenizer_version = saved_tokenizer or self.tokenizer_version
        if self.model_config is not None:
            from novalm.model.config import validate_checkpoint_config
            validate_checkpoint_config(self.model_config, state)
        self._base_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.step = state["step"]
        self._micro_step = state.get("micro_step", self.step)
        self._pending_data_state = state.get("data_state", {})
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def fit(self, batches, max_steps: int, checkpoint_path: str | Path | None = None,
            log_path: str | Path | None = None, validation_batches=None,
            eval_every_steps: int = 0, max_eval_batches: int | None = None,
            data_sampler=None, checkpoint_every_steps: int = 1,
            best_checkpoint_path: str | Path | None = None,
            profiler=None) -> list[float]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if checkpoint_every_steps < 1:
            raise ValueError("checkpoint_every_steps must be positive")
        if eval_every_steps < 0:
            raise ValueError("eval_every_steps cannot be negative")
        if max_eval_batches is not None and max_eval_batches < 1:
            raise ValueError("max_eval_batches must be positive")
        if validation_batches is None and eval_every_steps:
            raise ValueError("validation batches are required when eval_every_steps is set")
        if best_checkpoint_path is not None and validation_batches is None:
            raise ValueError("validation batches are required for best-checkpoint selection")
        if best_checkpoint_path is not None and eval_every_steps == 0:
            raise ValueError("eval_every_steps must be positive for best-checkpoint selection")
        if data_sampler is not None and "sampler" in self._pending_data_state:
            data_sampler.load_state_dict(self._pending_data_state["sampler"])
            self._pending_data_state = {}
        history = []
        iterator = iter(batches)
        target_step = self.step + max_steps
        last_eval_step = -1
        last_saved_step = self.step
        best_validation_loss = float("inf")
        if best_checkpoint_path is not None and Path(best_checkpoint_path).is_file():
            saved_best = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
            saved_config = saved_best.get("model_config")
            current_config = asdict(self.model_config) if self.model_config is not None else None
            if saved_config != current_config:
                raise ValueError("best checkpoint model config does not match current model")
            best_validation_loss = float(saved_best["metrics"]["validation_loss"])
        while self.step < target_step:
            step_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                loss = self.train_step(batch[0], batch[1])
            else:
                loss = self.train_step(batch)
            history.append(loss)
            validation_metrics = {}
            if (validation_batches is not None and eval_every_steps
                    and self.step > 0
                    and (self.step % eval_every_steps == 0 or self.step == target_step)
                    and self.step != last_eval_step):
                def validation_iterator():
                    for index, val_batch in enumerate(validation_batches):
                        if max_eval_batches is not None and index >= max_eval_batches:
                            break
                        yield val_batch

                evaluated = evaluate_loss(self.model, validation_iterator(), self.device)
                validation_metrics = {
                    "validation_loss": evaluated["loss"],
                    "validation_perplexity": evaluated["perplexity"],
                    "validation_tokens": evaluated["tokens"],
                }
                if (best_checkpoint_path is not None
                        and evaluated["loss"] < best_validation_loss):
                    best_path = Path(best_checkpoint_path)
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    if self.is_main_process:
                        atomic_torch_save({
                            "model": self._base_model.state_dict(),
                            "model_config": asdict(self.model_config)
                            if self.model_config is not None else None,
                            "tokenizer_version": self.tokenizer_version,
                            "best_validation_step": self.step,
                            "metrics": validation_metrics,
                            "environment": {
                                "python": platform.python_version(),
                                "torch": torch.__version__,
                                "cuda": torch.version.cuda,
                                "device": str(self.device),
                                "git_commit": self.git_commit,
                            },
                        }, best_path)
                    best_validation_loss = evaluated["loss"]
                last_eval_step = self.step
            if log_path and self.is_main_process:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                elapsed = time.perf_counter() - step_started
                batch_size, sequence_length = batch[0].shape if isinstance(batch, (tuple, list)) else batch.shape
                tokens = batch_size * max(0, sequence_length - 1)
                with Path(log_path).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": self.step, "train_loss": loss,
                                             "step_seconds": elapsed,
                                             "tokens_per_second": tokens_per_second(tokens, max(elapsed, 1e-9)),
                                             "learning_rate": self.optimizer.param_groups[0]["lr"],
                                             **self.last_metrics,
                                             **validation_metrics}) + "\n")
            if profiler is not None:
                profiler.step()
            if (checkpoint_path and self.is_main_process and self.step
                    and self.step % checkpoint_every_steps == 0
                    and self.step != last_saved_step
                    and self._micro_step % self.config.gradient_accumulation_steps == 0):
                data_state = ({"sampler": data_sampler.state_dict()}
                              if data_sampler is not None else None)
                self.save_checkpoint(checkpoint_path, data_state=data_state)
                last_saved_step = self.step
        if checkpoint_path and self.is_main_process and self.step != last_saved_step:
            data_state = ({"sampler": data_sampler.state_dict()}
                          if data_sampler is not None else None)
            self.save_checkpoint(checkpoint_path, data_state=data_state)
        return history

"""Autocast and gradient-scaler helpers for supported training precisions."""
import torch


def autocast_context(device: torch.device, precision: str):
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 autocast requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device.type, dtype=dtype, enabled=precision != "fp32")


def make_grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler:
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 gradient scaling requires CUDA")
    return torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

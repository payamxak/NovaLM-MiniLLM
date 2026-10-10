"""Optional torch.distributed setup helpers; single-process runs need no setup."""
import os
from datetime import timedelta

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def initialize_distributed(backend: str | None = None) -> tuple[int, int, int]:
    """Initialize from torchrun environment and return (rank, local_rank, world_size)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("invalid distributed rank environment")
    if world_size > 1 and not torch.distributed.is_initialized():
        selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        if selected_backend == "nccl":
            torch.cuda.set_device(local_rank)
        # Explicitly disable libuv when constructing the store.  Some Windows
        # wheels are built without libuv even though TCPStore supports its
        # regular socket backend; env:// would otherwise fail before DDP starts.
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        try:
            store = torch.distributed.TCPStore(
                master_addr, master_port, world_size, rank == 0,
                timeout=timedelta(minutes=5), use_libuv=False,
            )
            torch.distributed.init_process_group(
                backend=selected_backend, store=store, rank=rank, world_size=world_size,
            )
        except (RuntimeError, TypeError) as exc:
            if "libuv" not in str(exc).lower() and "use_libuv" not in str(exc):
                raise
            torch.distributed.init_process_group(backend=selected_backend, init_method="env://")
    return rank, local_rank, world_size


def wrap_ddp(model: nn.Module, device: torch.device, local_rank: int,
             world_size: int) -> nn.Module:
    if world_size <= 1:
        return model
    if not torch.distributed.is_initialized():
        raise RuntimeError("initialize_distributed must run before wrapping DDP")
    device_ids = [local_rank] if device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids)

"""Opt-in bounded PyTorch profiler context manager."""
from contextlib import nullcontext
from pathlib import Path

import torch


def profiler_context(enabled: bool, output_dir: str | Path = "outputs/profile",
                     wait: int = 1, warmup: int = 1, active: int = 2):
    if not enabled:
        return nullcontext(None)
    if min(wait, warmup, active) < 0 or active == 0:
        raise ValueError("profiler schedule must be nonnegative with active > 0")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(path)),
        record_shapes=True, profile_memory=True,
    )

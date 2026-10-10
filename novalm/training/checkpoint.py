"""Atomic checkpoint persistence helpers."""
import os
import tempfile
from pathlib import Path

import torch


def atomic_torch_save(state: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
        torch.save(state, temp_name)
        os.replace(temp_name, destination)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError("checkpoint must be a training state containing model weights")
    return state

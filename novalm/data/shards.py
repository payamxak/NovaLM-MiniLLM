import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def write_token_shards(blocks: Iterable[list[int]], output_dir: str | Path,
                       shard_tokens: int = 1_000_000, metadata: dict | None = None) -> dict:
    """Write little-endian uint32 token shards plus a checksum manifest."""
    out = Path(output_dir)
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be positive")
    out.mkdir(parents=True, exist_ok=True)
    shard: list[int] = []
    entries = []

    def flush(index: int, values: list[int]) -> None:
        if not values:
            return
        path = out / f"tokens-{index:05d}.bin"
        arr = np.asarray(values, dtype="<u4")
        arr.tofile(path)
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        entries.append({"path": path.name, "tokens": len(values), "sha256": checksum.hexdigest()})

    for block in blocks:
        shard.extend(block)
        while len(shard) >= shard_tokens:
            flush(len(entries), shard[:shard_tokens])
            shard = shard[shard_tokens:]
    flush(len(entries), shard)
    manifest = {"dtype": "uint32-le", "total_tokens": sum(e["tokens"] for e in entries),
                "shard_count": len(entries), "shards": entries, **(metadata or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_shard_manifest(path: str | Path) -> None:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counted = 0
    for entry in manifest.get("shards", []):
        shard = manifest_path.parent / entry["path"]
        if not shard.is_file():
            raise ValueError(f"missing token shard: {shard}")
        digest = hashlib.sha256()
        with shard.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {shard}")
        size = shard.stat().st_size
        if size % 4 or size // 4 != entry["tokens"]:
            raise ValueError(f"token count/byte size mismatch: {shard}")
        counted += entry["tokens"]
    if counted != manifest.get("total_tokens"):
        raise ValueError("manifest total_tokens does not match shard entries")
    if manifest.get("shard_count", len(manifest.get("shards", []))) != len(manifest.get("shards", [])):
        raise ValueError("manifest shard_count does not match shard entries")


class TokenShardDataset:
    def __init__(self, path: str | Path, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.path, self.block_size = Path(path), block_size
        self.tokens = np.memmap(self.path, mode="r", dtype="<u4")

    def __len__(self) -> int:
        # Each sample consumes one context plus its next-token target. Advance by
        # one full context so target tokens are not redundantly counted in every window.
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index: int):
        import torch
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.block_size
        return torch.from_numpy(np.array(
            self.tokens[start:start + self.block_size + 1], dtype=np.int64))

    def close(self) -> None:
        mmap = getattr(self.tokens, "_mmap", None)
        if mmap is not None:
            mmap.close()

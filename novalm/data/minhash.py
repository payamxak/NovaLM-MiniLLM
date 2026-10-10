import hashlib
import re
from collections.abc import Iterable, Iterator


def shingles(text: str, width: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.casefold())
    if not words:
        return set()
    if len(words) <= width:
        return {" ".join(words)}
    return {" ".join(words[i:i + width]) for i in range(len(words) - width + 1)}


def _signature(items: set[str], num_perm: int) -> tuple[int, ...]:
    if not items:
        return tuple([2**64 - 1] * num_perm)
    return tuple(min(int.from_bytes(hashlib.blake2b(
        seed.to_bytes(4, "little") + item.encode(), digest_size=8).digest(), "little")
                      for item in items) for seed in range(num_perm))


def near_deduplicate(records: Iterable[dict], threshold: float = 0.85,
                     num_perm: int = 64, bands: int = 8) -> Iterator[dict]:
    """Deterministic MinHash/LSH candidate filter followed by exact Jaccard verification."""
    if not 0 < threshold <= 1 or num_perm < 1 or bands < 1 or num_perm % bands:
        raise ValueError("threshold must be in (0,1] and num_perm divisible by bands")
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    kept_sets: list[set[str]] = []
    rows_per_band = num_perm // bands
    for record in records:
        text = str(record.get("text", ""))
        current = shingles(text)
        if not current:
            continue
        sig = _signature(current, num_perm)
        candidates: set[int] = set()
        for band in range(bands):
            key = (band, sig[band * rows_per_band:(band + 1) * rows_per_band])
            candidates.update(buckets.get(key, []))
        duplicate = any(len(current & kept_sets[i]) / len(current | kept_sets[i]) >= threshold
                        for i in candidates)
        if duplicate:
            continue
        index = len(kept_sets)
        kept_sets.append(current)
        for band in range(bands):
            key = (band, sig[band * rows_per_band:(band + 1) * rows_per_band])
            buckets.setdefault(key, []).append(index)
        yield dict(record)

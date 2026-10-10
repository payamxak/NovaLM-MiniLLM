"""Streaming local text ingestion with explicit source and license provenance."""
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from novalm.data.cleaning import clean_document


def iter_text_documents(paths: Iterable[str | Path], *, source_name: str | None = None,
                        license_name: str = "unknown", language: str | None = None,
                        attribution_manifest: str | None = None,
                        recursive: bool = True) -> Iterator[dict[str, str]]:
    """Yield normalized UTF-8 text files with provenance; directories are sorted."""
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if path.is_file() else sorted(
            path.rglob("*.txt") if recursive else path.glob("*.txt"))
        for candidate in candidates:
            if candidate.suffix.casefold() != ".txt":
                continue
            text = clean_document(candidate.read_text(encoding="utf-8", errors="replace"))
            if not text:
                continue
            yield {
                "text": text,
                "source": source_name or str(candidate),
                "license": license_name,
                "language": language or "unknown",
                "attribution_manifest": attribution_manifest or "",
            }


def read_attribution_manifest(path: str | Path) -> dict:
    """Read a JSON provenance manifest and fail on malformed/non-object manifests."""
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read attribution manifest {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("attribution manifest must contain a JSON object")
    return value

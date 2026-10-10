"""Validated streaming reader for prompt/chosen/rejected preference records."""
import json
from collections.abc import Iterator
from pathlib import Path


def iter_preference_records(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid preference JSON at line {line_number}: {exc}") from exc
            if not isinstance(record, dict) or any(
                    not isinstance(record.get(key), str) or not record[key]
                    for key in ("prompt", "chosen", "rejected")):
                raise ValueError(
                    f"preference record at line {line_number} needs nonempty prompt/chosen/rejected strings")
            yield {key: record[key] for key in ("prompt", "chosen", "rejected")}

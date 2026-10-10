import hashlib
from collections.abc import Iterable, Iterator

from novalm.data.cleaning import normalize_text


def exact_deduplicate(records: Iterable[dict]) -> Iterator[dict]:
    seen: set[str] = set()
    for record in records:
        text = normalize_text(str(record.get("text", "")))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text and digest not in seen:
            seen.add(digest)
            result = dict(record)
            result["text"] = text
            result["content_hash"] = digest
            yield result

"""Tokenization helpers that preserve record provenance and tokenizer identity."""
import hashlib
from pathlib import Path

from novalm.tokenizer.tokenizer import NovaTokenizer


def tokenize_record(record: dict, tokenizer: NovaTokenizer, tokenizer_sha256: str | None = None) -> dict:
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        raise ValueError("record must contain nonempty text")
    result = dict(record)
    result["token_ids"] = tokenizer.encode(record["text"], add_eos=True)
    result["tokenizer_vocab_size"] = tokenizer.vocab_size
    result["tokenizer_sha256"] = tokenizer_sha256 or tokenizer_digest(tokenizer.model_path)
    if not result["token_ids"]:
        raise ValueError("tokenizer produced an empty token sequence")
    return result


def tokenizer_digest(model_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(model_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

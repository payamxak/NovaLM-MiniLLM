"""Tokenizer training API for UTF-8 text files or prepared JSONL text records."""
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

from novalm.config.schema import TokenizerConfig
from novalm.tokenizer.tokenizer import NovaTokenizer


def _jsonl_texts(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as source:
        for source_line in source:
            row = json.loads(source_line)
            yield str(row.get("text", ""))


def _text_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8", errors="replace") as source:
        for source_line in source:
            yield source_line.rstrip("\r\n")


def train_tokenizer(input_files: list[str | Path], model_prefix: str | Path,
                    config: TokenizerConfig | None = None) -> NovaTokenizer:
    if config is None:
        config = TokenizerConfig()
    if not input_files:
        raise ValueError("at least one tokenizer training file is required")
    paths = [Path(path) for path in input_files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"tokenizer input files not found: {missing}")
    if not any(path.suffix.casefold() == ".jsonl" for path in paths):
        return NovaTokenizer.train(paths, model_prefix, config.vocab_size,
                                   config.character_coverage)

    with tempfile.TemporaryDirectory(prefix="novalm-tokenizer-") as temp_dir:
        corpus = Path(temp_dir) / "corpus.txt"
        with corpus.open("w", encoding="utf-8") as output:
            for path in paths:
                if path.suffix.casefold() == ".jsonl":
                    texts = _jsonl_texts(path)
                else:
                    texts = _text_lines(path)
                for text in texts:
                    for line in str(text).splitlines():
                        for offset in range(0, len(line), 4000):
                            piece = line[offset:offset + 4000]
                            if piece.strip():
                                output.write(piece + "\n")
        if not corpus.stat().st_size:
            raise ValueError("tokenizer corpus contains no nonempty text")
        return NovaTokenizer.train([corpus], model_prefix, config.vocab_size,
                                   config.character_coverage)

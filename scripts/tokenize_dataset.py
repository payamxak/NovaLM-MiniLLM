import argparse
import hashlib
import json
from pathlib import Path

from novalm.data.tokenize import tokenize_record
from novalm.tokenizer.tokenizer import NovaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize prepared JSONL text records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tokenizer = NovaTokenizer(args.tokenizer)
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            row = tokenize_record(row, tokenizer, tokenizer_hash)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

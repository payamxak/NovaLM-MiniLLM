import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from novalm.tokenizer.tokenizer import NovaTokenizer


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Measure tokenizer fertility and common fragments")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", help="Optional JSONL with text and optional language fields")
    parser.add_argument("--output", default="outputs/tokenizer_analysis.json")
    args = parser.parse_args()
    tokenizer = NovaTokenizer(args.tokenizer)
    if args.samples:
        samples = [json.loads(line) for line in Path(args.samples).read_text(encoding="utf-8").splitlines()]
    else:
        samples = [
            {"language": "en", "text": "The quick brown fox jumps over a language model."},
            {"language": "fa", "text": "مدل زبانی می‌تواند متن فارسی را پردازش کند."},
            {"language": "mixed", "text": "This is متن فارسی mixed with English."},
            {"language": "code", "text": "def add(a: int, b: int) -> int:\n    return a + b"},
            {"language": "math", "text": "E = mc^2; integral_0^1 x^2 dx = 1/3"},
        ]
    report, fragments = {}, Counter()
    for row in samples:
        text, group = row["text"], row.get("language", "unknown")
        ids = tokenizer.encode(text)
        entry = report.setdefault(group, {"documents": 0, "words": 0,
                                          "tokens": 0, "unknown_tokens": 0})
        entry["documents"] += 1
        entry["words"] += len(text.split())
        entry["tokens"] += len(ids)
        entry["unknown_tokens"] += sum(token == tokenizer.processor.unk_id() for token in ids)
        fragments.update(tokenizer.processor.id_to_piece(token) for token in ids)
    for entry in report.values():
        entry["tokens_per_word"] = entry["tokens"] / max(entry["words"], 1)
        entry["unknown_rate"] = entry["unknown_tokens"] / max(entry["tokens"], 1)
    result = {"vocab_size": tokenizer.vocab_size, "groups": report,
              "common_fragments": fragments.most_common(30)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

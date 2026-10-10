"""Inspect prepared JSONL documents or validate and summarize token shards."""
import argparse
import json
from collections import Counter
from pathlib import Path

from novalm.data.shards import verify_shard_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="JSONL file or shard manifest.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="maximum JSONL records to inspect; 0 reads all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit cannot be negative")
    path = Path(args.path)
    if not path.is_file():
        parser.error(f"file does not exist: {path}")

    if path.name == "manifest.json":
        verify_shard_manifest(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({
            "kind": "token_shards",
            "manifest": str(path),
            "total_tokens": manifest["total_tokens"],
            "shard_count": manifest["shard_count"],
            "source": manifest.get("source"),
            "licenses": manifest.get("licenses", []),
            "languages": manifest.get("languages", []),
            "checksum_validation": "passed",
        }, indent=2, ensure_ascii=False))
        return

    count = 0
    languages: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    token_counts: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if args.limit and count >= args.limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                parser.error(f"invalid JSON at {path}:{line_number}: {exc}")
            count += 1
            languages[str(row.get("language", "unknown"))] += 1
            sources[str(row.get("source", row.get("mixture_source", "unknown")))] += 1
            content = row.get("token_ids", row.get("text", ""))
            if isinstance(content, list):
                token_counts.append(len(content))
            elif isinstance(content, str):
                token_counts.append(len(content.split()))

    print(json.dumps({
        "kind": "jsonl",
        "path": str(path),
        "records_inspected": count,
        "languages": dict(languages),
        "sources": dict(sources),
        "mean_tokens_or_words": sum(token_counts) / len(token_counts) if token_counts else None,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from novalm.data.packing import pack_sequences
from novalm.data.shards import write_token_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack tokenized JSONL records into uint32 shards")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--eos-id", type=int, default=2)
    parser.add_argument("--shard-tokens", type=int, default=1_000_000)
    args = parser.parse_args()
    metadata = {"source": str(args.input), "block_size": args.block_size,
                "preprocessing_version": "0.1.0",
                "licenses": [], "languages": [], "document_count": 0,
                "tokenizer_vocab_sizes": [], "tokenizer_sha256": [], "mixture_counts": {}}
    metadata["attribution_manifests"] = []

    def sequences():
        with Path(args.input).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                metadata["document_count"] += 1
                for key in ("license", "language"):
                    if row.get(key) and row[key] not in metadata[key + "s"]:
                        metadata[key + "s"].append(row[key])
                if row.get("tokenizer_vocab_size") is not None and row["tokenizer_vocab_size"] not in metadata["tokenizer_vocab_sizes"]:
                    metadata["tokenizer_vocab_sizes"].append(row["tokenizer_vocab_size"])
                if row.get("tokenizer_sha256") and row["tokenizer_sha256"] not in metadata["tokenizer_sha256"]:
                    metadata["tokenizer_sha256"].append(row["tokenizer_sha256"])
                mixture = row.get("mixture_source")
                if mixture:
                    metadata["mixture_counts"][mixture] = metadata["mixture_counts"].get(mixture, 0) + 1
                attribution = row.get("attribution_manifest")
                if attribution and attribution not in metadata["attribution_manifests"]:
                    metadata["attribution_manifests"].append(attribution)
                yield row["token_ids"]

    blocks = pack_sequences(sequences(), args.block_size, args.eos_id)
    manifest = write_token_shards(blocks, args.output_dir, args.shard_tokens, metadata)
    if len(manifest["tokenizer_vocab_sizes"]) > 1 or len(manifest["tokenizer_sha256"]) > 1:
        raise ValueError("input records mix tokenizer versions; rebuild shards from a single tokenizer")
    print(f"wrote {manifest['total_tokens']} tokens in {len(manifest['shards'])} shards")


if __name__ == "__main__":
    main()

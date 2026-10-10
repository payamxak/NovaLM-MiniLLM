import argparse
import itertools
import json
from collections import Counter
from dataclasses import fields
from pathlib import Path

import yaml

from novalm.data.cleaning import clean_document
from novalm.data.dedup import exact_deduplicate
from novalm.data.language_filter import detect_language
from novalm.data.minhash import near_deduplicate
from novalm.data.mixture import weighted_stream
from novalm.data.quality import QualityConfig, quality_rejection


def _paths(input_files) -> list[Path]:
    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    found = []
    for source in input_files:
        path = Path(source)
        found.extend([path] if path.is_file() else path.rglob("*.txt"))
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, filter, deduplicate, and optionally mix local UTF-8 text sources")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    quality_values = dict(cfg.get("quality", {}))
    for field in fields(QualityConfig):
        if field.name in cfg and field.name not in quality_values:
            quality_values[field.name] = cfg[field.name]
    quality = QualityConfig(**quality_values)
    rejected = Counter()
    source_configs = cfg.get("sources")
    if source_configs is None:
        source_configs = {cfg.get("source_name", "local_corpus"): cfg}
    if not isinstance(source_configs, dict) or not source_configs:
        parser.error("sources must be a nonempty mapping of source names to settings")

    streams = {}
    file_count = 0
    for source_name, source_config in source_configs.items():
        if not isinstance(source_config, dict):
            parser.error(f"source {source_name!r} settings must be a mapping")
        source_paths = _paths(source_config.get("input_files", []))
        file_count += len(source_paths)
        allowed_languages = source_config.get("languages", cfg.get("languages"))
        if isinstance(allowed_languages, str):
            allowed_languages = [allowed_languages]

        def records(paths=source_paths, name=source_name, settings=source_config,
                    languages=allowed_languages):
            for path in paths:
                with path.open(encoding="utf-8", errors="replace") as stream:
                    text = clean_document(stream.read())
                reason = quality_rejection(text, quality)
                language = detect_language(text)
                if reason is None and languages and language not in languages:
                    reason = "language"
                if reason:
                    rejected[reason] += 1
                    continue
                yield {
                    "text": text,
                    "source": settings.get("source_name", str(path)),
                    "license": settings.get("license", cfg.get("license", "unknown")),
                    "language": language,
                    "mixture_source": name,
                    "attribution_manifest": settings.get(
                        "attribution_manifest", cfg.get("attribution_manifest")),
                }

        streams[source_name] = records()

    if not file_count:
        parser.error("no input .txt files found in configured sources")

    weights = cfg.get("mixture_weights")
    if weights is not None:
        unknown = set(weights) - set(streams)
        if unknown:
            parser.error(f"mixture_weights names unknown sources: {sorted(unknown)}")
        ordered_records = weighted_stream(streams, weights, seed=cfg.get("mixture_seed", 0))
    else:
        ordered_records = itertools.chain.from_iterable(streams.values())
    records = exact_deduplicate(ordered_records)
    if cfg.get("near_dedup", False):
        records = near_deduplicate(records,
                                   threshold=cfg.get("near_dedup_threshold", 0.85),
                                   num_perm=cfg.get("near_dedup_num_perm", 64),
                                   bands=cfg.get("near_dedup_bands", 8))
    output = Path(cfg.get("output_file", "data/processed/train.jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    mixture_counts = Counter()
    with output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if row.get("mixture_source"):
                mixture_counts[row["mixture_source"]] += 1
    if count == 0:
        parser.error("no documents survived configured cleaning and quality filters")
    print(f"accepted {count} documents from {file_count} files "
          f"({dict(rejected)} rejected); mixture={dict(mixture_counts)}; wrote {output}")


if __name__ == "__main__":
    main()

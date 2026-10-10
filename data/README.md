# Data pipeline

Place only appropriately licensed UTF-8 `.txt` files under `raw/`, grouped into source directories if you want a weighted mixture. Record the actual source and license in the config; the default `unknown` license value is provenance, not permission to use a dataset.

`scripts/prepare_data.py` normalizes and cleans documents, applies configurable length/alphabetic/symbol/URL/repetition/garbage checks, detects English/Persian/mixed script, optionally filters languages, computes exact content hashes, optionally applies MinHash/LSH near deduplication, and records source/license/language provenance. It accepts the legacy `input_files` config or multiple named `sources` with `mixture_weights` and a deterministic `mixture_seed`. Weighted source records are interleaved without replacement; the requested ratios are sampling weights, so finite or small sources can exhaust before the target proportions are reached. The final counts are printed and can be checked in output records.

Then train a project-specific SentencePiece model, tokenize the prepared JSONL, and pack it into uint32 shards. `scripts/build_shards.py` writes a manifest with source, license, preprocessing version, tokenizer hash/vocabulary, mixture counts, attribution manifest paths, shard counts, token counts, and checksums. Keep large datasets and derived shards out of version control. No corpus is bundled with this repository, and no corpus-scale pretraining run is claimed.

For a small, reproducible API smoke sample, `scripts/fetch_wikipedia_sample.py` fetches page summaries and records the API-reported page license, revision, title, and source URL in `attribution.json`. Review those per-page records and terms before reuse or redistribution. The 12-summary development sample used in the smoke report is not committed; its run metadata is in `experiments/reports/wikipedia_30m_smoke.json`.
# Multilingual pretraining data is now installed at the paths consumed by `scripts/train.py`:
#   data/shards/train
#   data/shards/validation
#   data/tokenizer/novalm.model
#   data/tokenizer/novalm.vocab
#
# The source corpus and resumable preparation scripts remain under ../datas.

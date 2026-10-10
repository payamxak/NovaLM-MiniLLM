# Data pipeline

The starter pipeline accepts local UTF-8 `.txt` files. `prepare_data.py` normalizes Unicode and whitespace, applies configurable length/alphabetic/URL/repetition checks, performs exact SHA-256 deduplication, and emits JSONL records with source and license fields. Train a project-owned SentencePiece BPE tokenizer, tokenize the JSONL records, and pack token IDs with EOS separators into uint32 memory-mappable shard files. Shard manifests record token count and SHA-256 checksums.

The optional MinHash/LSH deduplicator and weighted source sampler are wired into the preparation CLI when `mixture_weights` is configured. The current Persian/English detector is a Unicode-script heuristic, not a reliable language identification model. The shard manifest records license/language/source metadata and verifies file checksums before training/evaluation. Corpus-level quality/fertility analysis is still needed. Only data for which you have usage rights should be placed in `data/raw`.

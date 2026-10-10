# Tokenizer

`scripts/train_tokenizer.py` trains project-specific SentencePiece BPE with the configured vocabulary size and NovaLM special tokens. `scripts/analyze_tokenizer.py` reports token fertility and unknown-token rates by supplied category, plus frequent pieces. Without `--samples`, the analysis uses illustrative English, Persian, mixed-language, code, and math snippets; treat those results as a smoke check, not a corpus-level tokenizer evaluation.

Sample JSONL format for analysis: `{"language":"fa","text":"..."}`. The tokenizer must be trained on data you are permitted to use. Special token IDs are PAD=0, BOS=1, EOS=2, UNK=3, followed by user-defined conversation and reasoning tokens.

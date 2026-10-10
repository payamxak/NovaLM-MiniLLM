# Run after collection to make train/validation files and tokenizer text.
python datas/split_corpus.py
python scripts/train_tokenizer.py --config datas/tokenizer.yaml
python scripts/tokenize_dataset.py --input datas/processed/train.jsonl --tokenizer datas/tokenizer/novalm.model --output datas/processed/train_tokens.jsonl
python scripts/tokenize_dataset.py --input datas/processed/validation.jsonl --tokenizer datas/tokenizer/novalm.model --output datas/processed/validation_tokens.jsonl
python scripts/build_shards.py --input datas/processed/train_tokens.jsonl --output-dir datas/shards/train --block-size 512 --shard-tokens 1000000
python scripts/build_shards.py --input datas/processed/validation_tokens.jsonl --output-dir datas/shards/validation --block-size 512 --shard-tokens 1000000

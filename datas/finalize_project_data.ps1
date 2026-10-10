# Complete the multilingual corpus and copy it to the directories consumed by scripts/train.py.
python datas/split_multilingual.py
python scripts/train_tokenizer.py --config datas/tokenizer.yaml
python scripts/tokenize_dataset.py --input datas/processed/train.jsonl --tokenizer datas/tokenizer/novalm.model --output datas/processed/train_tokens.jsonl
python scripts/tokenize_dataset.py --input datas/processed/validation.jsonl --tokenizer datas/tokenizer/novalm.model --output datas/processed/validation_tokens.jsonl
if (Test-Path data/shards/train) { Remove-Item data/shards/train -Recurse -Force }
if (Test-Path data/shards/validation) { Remove-Item data/shards/validation -Recurse -Force }
python scripts/build_shards.py --input datas/processed/train_tokens.jsonl --output-dir data/shards/train --block-size 512 --shard-tokens 1000000
python scripts/build_shards.py --input datas/processed/validation_tokens.jsonl --output-dir data/shards/validation --block-size 512 --shard-tokens 1000000

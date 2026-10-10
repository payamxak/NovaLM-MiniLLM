# English pretraining corpus

For `configs/model/novalm_30m.yaml` (8,192 vocabulary, 512-token context, about 30M parameters), the target is **600 million tokenizer tokens** (about 20 tokens per parameter). This folder contains a resumable English Wikipedia collector and the preparation configuration. It does not train the model.

The source is `wikimedia/wikipedia`, config `20231101.en`, accessed through the Hugging Face Dataset Server. Wikipedia text is CC BY-SA 4.0; retain `raw/wiki_en/attribution.json` when redistributing.

```powershell
python datas/collect_wikipedia.py --target-tokens 600000000
python scripts/prepare_data.py --config datas/preprocess.yaml
Get-Content datas/processed/train.jsonl | ForEach-Object { $_ } | Out-File datas/processed/all_text.txt -Encoding utf8
python scripts/train_tokenizer.py --config datas/tokenizer.yaml
python scripts/tokenize_dataset.py --input datas/processed/train.jsonl --tokenizer datas/tokenizer/novalm.model --output datas/processed/train_tokens.jsonl
python scripts/tokenize_dataset.py --input datas/processed/validation.jsonl --tokenizer datas/tokenizer/novalm.model --output datas/processed/validation_tokens.jsonl
python scripts/build_shards.py --input datas/processed/train_tokens.jsonl --output-dir datas/shards/train --block-size 512 --shard-tokens 1000000
python scripts/build_shards.py --input datas/processed/validation_tokens.jsonl --output-dir datas/shards/validation --block-size 512 --shard-tokens 1000000
```

The collector saves progress and resumes after interruption. The final manifest from `build_shards.py` records the exact token count.

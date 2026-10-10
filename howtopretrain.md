# How to pretrain NovaLM

This guide describes how to prepare data and run causal language-model pretraining for this repository's NovaLM. The training objective is to predict the next token in ordinary text. Pretraining does not by itself teach the model to follow chat instructions; supervised fine-tuning (SFT) is a later step.

## What the data should look like

Start with a large collection of text documents you have the right to use for model training. Good sources can include encyclopedias, books, educational material, public reports, technical writing, and appropriately licensed code. Use material that resembles the language and tasks you want the model to handle. Record the actual source, license, language, and attribution details for every source. A web page being publicly readable does not automatically grant permission to use it for training.

For this repository's preparation script, raw input is UTF-8 `.txt` files. Each file is treated as one document; put files in source-specific folders if combining datasets. For example:

```text
data/raw/
  english_wikipedia/
    article_001.txt
    article_002.txt
  persian_books/
    book_001.txt
  code/
    example_001.txt
```

Documents should contain the actual readable content, not HTML, navigation menus, repeated boilerplate, or metadata pasted into the text. Keep source metadata in your preparation configuration and attribution manifests. Avoid private, sensitive, low-quality, spam, or duplicated material. Hold out a portion of documents for validation before training so nearly identical text cannot appear in both splits.

The pipeline turns these files into JSONL records with fields like:

```json
{"text":"A document's cleaned text goes here.","source":"Example source","license":"CC BY-SA 4.0","language":"en","mixture_source":"english_wikipedia"}
```

Then each record receives `token_ids` from the project tokenizer. The packer concatenates token sequences with an EOS separator and writes fixed-size blocks to binary shards. The trainer reads those shards as `uint32` token IDs; it predicts each next token using windows of `context + 1` tokens. You do not need to make question/answer pairs for pretraining: those are for later SFT or preference training.

## How much data to aim for

The main starter target in this repository is `configs/model/novalm_30m.yaml`: an approximately 30-million-parameter model with an 8,192-token vocabulary and 512-token context. The data notes set a goal of about **600 million training tokens** (roughly 20 tokens per parameter). This is a useful project target, not a guarantee that the model will reach a particular quality. More diverse, clean, well-matched data is generally more useful than simply collecting more low-quality text.

The checked-in `datas/manifest.json` reports about **31.2 million training tokens** and 0.32 million validation tokens at its preparation date, October 8, 2026. That is enough to exercise the pipeline and run experiments, but is far below the 600-million-token goal. The manifest describes English and Persian Wikipedia data; it does not indicate that a full-scale pretraining run has been completed.

## Prepare and tokenize the corpus

1. Put licensed `.txt` documents under `data/raw/` (or another path you configure). Create a train/validation split by document before processing. Keep the validation documents separate and do not use them to fit the tokenizer.
2. Edit `configs/data/pretraining_mixture.yaml` for multiple sources, or `configs/data/pretraining.yaml` for one source. Set real paths, source names, languages, and license identifiers. The example mixture config has placeholder licenses that must be replaced. Tune quality filters to your corpus; the built-in language detector is a script heuristic, not a robust language-identification model.
3. Prepare clean, deduplicated text records. The preparation tool accepts `.txt` files under the configured source paths:

   ```powershell
   python scripts/prepare_data.py --config configs/data/pretraining_mixture.yaml
   ```

   For separate train and validation sets, use separate configs and output files, each pointing only at its own split. Confirm the prepared record counts and metadata before continuing. Near-deduplication is optional and can use significant time and memory.
4. Train a tokenizer on the **training split only**. Set `input_files` in `configs/data/tokenizer.yaml` to the prepared training text. SentencePiece BPE is used, and its vocabulary size must match the model config. The 30M config uses 8,192 tokens, so use that size rather than the generic 32,000-token default:

   ```powershell
   python scripts/train_tokenizer.py --config configs/data/tokenizer.yaml
   ```

   The configured `model_prefix` determines where `novalm.model` and `novalm.vocab` are written. Keep the model file; the same tokenizer must be used for train and validation data and at inference time.
5. Tokenize each prepared split with that tokenizer. For example, if your prepared files are `data/processed/train.jsonl` and `data/processed/validation.jsonl`:

   ```powershell
   python scripts/tokenize_dataset.py --input data/processed/train.jsonl --tokenizer data/tokenizer/novalm.model --output data/processed/train_tokens.jsonl
   python scripts/tokenize_dataset.py --input data/processed/validation.jsonl --tokenizer data/tokenizer/novalm.model --output data/processed/validation_tokens.jsonl
   ```

   Ensure the tokenizer path matches `model_prefix` in your config.
6. Pack the tokenized records into shards. The block size should agree with the model context (512 for `novalm_30m.yaml`):

   ```powershell
   python scripts/build_shards.py --input data/processed/train_tokens.jsonl --output-dir data/shards/train --block-size 512 --shard-tokens 1000000
   python scripts/build_shards.py --input data/processed/validation_tokens.jsonl --output-dir data/shards/validation --block-size 512 --shard-tokens 1000000
   ```

   Each shard directory gets a manifest with token counts, tokenizer metadata, provenance, and file checksums. Use `python scripts/inspect_data.py data/shards/train/manifest.json` to inspect it. Keep raw corpora, processed corpora, and large shards out of Git.

### Ready-made Wikipedia data in this repository

The `datas/` directory contains resumable English and Persian Wikipedia collection and finalization scripts. The prepared shard paths under `data/shards/` are directly usable by the trainer. To continue collection and rebuild the combined project data, see the resume commands in `datas/manifest.json` and run the documented `datas/finalize_project_data.ps1` finalization step after collection. Preserve the attribution files in `datas/raw/` and review the terms and attribution requirements for each source before redistributing data or model artifacts. The current corpus is a useful starting point, not a substitute for a larger and more varied target corpus.

## Run pretraining

Start with a small correctness run, then increase the run length after confirming the data, tokenizer, and validation loss behave as expected. For a quick pipeline check, use `configs/model/debug.yaml` and `configs/training/debug.yaml`. For the 30M model on a compatible GPU, use the single-GPU training config:

```powershell
python scripts/train.py `
  --config configs/model/novalm_30m.yaml `
  --training-config configs/training/single_gpu.yaml `
  --shards data/shards/train `
  --validation-shards data/shards/validation `
  --steps 10000 `
  --batch-size 1 `
  --eval-every 250 `
  --eval-batches 50 `
  --save-every 500 `
  --checkpoint checkpoints/novalm_30m_pretrain.pt `
  --best-checkpoint checkpoints/novalm_30m_best.pt `
  --device auto
```

The step count is an example; choose it based on the amount of data and hardware, and make `total_steps` in the training YAML agree with the planned run because it controls the learning-rate schedule. Effective batch size is microbatch size multiplied by `gradient_accumulation_steps` (and by world size when using distributed training). Reduce batch size or context if memory is limited; use accumulation and activation checkpointing where appropriate. `bf16` requires compatible hardware; use `fp32` for CPU debugging. For distributed training, launch the same script with `torchrun` as described in `docs/training.md`.

To continue an interrupted run with its saved optimizer and RNG state, use the same model/training configuration and data shards and add `--resume`:

```powershell
python scripts/train.py --config configs/model/novalm_30m.yaml --training-config configs/training/single_gpu.yaml --shards data/shards/train --validation-shards data/shards/validation --steps 1000 --batch-size 1 --checkpoint checkpoints/novalm_30m_pretrain.pt --resume --device auto
```

When resuming, `--steps` is the number of additional optimizer steps. Keep the world size and rank layout unchanged for distributed resume.

## Check that training is useful

- Review the shard manifests and check that training and validation use the same tokenizer vocabulary and have valid checksums.
- Monitor training and validation loss, validation perplexity, tokens per second, and generated samples. Training loss falling while validation loss rises can indicate overfitting or a data mismatch.
- Keep validation documents out of the training shards. Evaluate on more than one subject or language if the model is intended to cover multiple areas.
- Save run configs, logs, checkpoints, dataset provenance, tokenizer hash, and a corpus snapshot/version so the result can be reproduced.
- After pretraining, evaluate the base model before SFT. Use the SFT pipeline separately if you want a chat or instruction-following model.

See [the data pipeline guide](docs/data_pipeline.md), [training guide](docs/training.md), and [project README](README.md) for implementation details and current limitations.

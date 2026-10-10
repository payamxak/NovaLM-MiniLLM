import argparse
from pathlib import Path

import yaml

from novalm.config.schema import TokenizerConfig
from novalm.tokenizer.train import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the project SentencePiece BPE tokenizer")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "input_files" not in cfg or "model_prefix" not in cfg:
        parser.error("config must define input_files and model_prefix")
    tokenizer = train_tokenizer(
        [Path(value) for value in cfg["input_files"]], cfg["model_prefix"],
        TokenizerConfig(vocab_size=cfg.get("vocab_size", 32000),
                        character_coverage=cfg.get("character_coverage", 0.9995)),
    )
    print(f"trained vocabulary with {tokenizer.vocab_size} tokens at {tokenizer.model_path}")


if __name__ == "__main__":
    main()

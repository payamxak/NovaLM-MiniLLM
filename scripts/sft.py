import argparse
import hashlib
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from novalm.config import load_model_config
from novalm.model.config import validate_checkpoint_config
from novalm.model.transformer import NovaLM
from novalm.post_training.formatting import SFTDataset, sft_collate
from novalm.tokenizer.tokenizer import NovaTokenizer
from novalm.training.trainer import Trainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant-only supervised fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-config", default="configs/training/debug.yaml")
    parser.add_argument("--data", required=True, help="JSONL rows containing messages")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint", default="checkpoints/sft.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps and batch size must be positive")
    config = load_model_config(args.config)
    tokenizer = NovaTokenizer(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        parser.error("tokenizer vocabulary size does not match model config")
    dataset = SFTDataset(args.data, tokenizer, config.max_seq_length)
    if not len(dataset):
        parser.error("SFT dataset is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda rows: sft_collate(rows, tokenizer.processor.pad_id()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NovaLM(config)
    base_state = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    validate_checkpoint_config(config, base_state)
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if base_state.get("tokenizer_version") and base_state["tokenizer_version"] != tokenizer_hash:
        parser.error("tokenizer file does not match base checkpoint tokenizer version")
    model.load_state_dict(base_state["model"])
    with Path(args.training_config).open(encoding="utf-8") as stream:
        trainer_config = TrainerConfig(**yaml.safe_load(stream))
    trainer = Trainer(model, trainer_config, device)
    trainer.tokenizer_version = base_state.get("tokenizer_version")
    trainer.fit(loader, args.steps, args.checkpoint)
    print(f"saved SFT checkpoint at optimizer step {trainer.step}: {args.checkpoint}")


if __name__ == "__main__":
    main()

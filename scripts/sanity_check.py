"""Run deterministic CPU checks before attempting a longer training job."""
import argparse
import json
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from novalm.model.config import ModelConfig
from novalm.model.transformer import NovaLM
from novalm.training.seed import set_seed
from novalm.training.trainer import Trainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", help="optional path for a JSON run report")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    set_seed(7)
    cfg = ModelConfig(vocab_size=32, hidden_size=32, num_layers=1,
                      num_attention_heads=4, num_kv_heads=2,
                      intermediate_size=64, max_seq_length=16)
    model = NovaLM(cfg)
    trainer = Trainer(model, TrainerConfig(learning_rate=0.01, weight_decay=0,
                                          warmup_steps=2, total_steps=args.steps), "cpu")
    batch = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * 2)
    losses = [trainer.train_step(batch) for _ in range(args.steps)]
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "seed": 7,
        "steps": args.steps,
        "model_config": asdict(cfg),
        "parameter_count": model.parameter_count,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "passed": losses[-1] < losses[0] * 0.5,
    }
    print(f"one-batch overfit: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not losses[-1] < losses[0] * 0.5:
        raise SystemExit("one-batch overfit check failed to reduce loss by 50%")


if __name__ == "__main__":
    main()

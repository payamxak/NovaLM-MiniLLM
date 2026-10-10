"""Small, explicit DPO fine-tuning loop for JSONL prompt/chosen/rejected records."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from novalm.config import load_model_config
from novalm.model.config import validate_checkpoint_config
from novalm.model.transformer import NovaLM
from novalm.post_training.dpo import completion_ids, dpo_loss, sequence_log_probability
from novalm.tokenizer.tokenizer import NovaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct Preference Optimization fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-config", default="configs/training/debug.yaml")
    parser.add_argument("--data", required=True, help="JSONL rows with prompt/chosen/rejected")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--checkpoint", default="checkpoints/dpo.pt")
    args = parser.parse_args()
    if args.steps <= 0 or args.beta <= 0:
        parser.error("steps and beta must be positive")
    cfg, tokenizer = load_model_config(args.config), NovaTokenizer(args.tokenizer)
    if tokenizer.vocab_size != cfg.vocab_size:
        parser.error("tokenizer vocabulary size does not match model config")
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    if not rows:
        parser.error("preference dataset is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = NovaLM(cfg).to(device)
    base = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    validate_checkpoint_config(cfg, base)
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if base.get("tokenizer_version") and base["tokenizer_version"] != tokenizer_hash:
        parser.error("tokenizer file does not match base checkpoint tokenizer version")
    policy.load_state_dict(base["model"])
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    with Path(args.training_config).open(encoding="utf-8") as stream:
        training = yaml.safe_load(stream)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=training.get("learning_rate", 3e-4),
                                  weight_decay=training.get("weight_decay", 0.1),
                                  betas=tuple(training.get("betas", (0.9, 0.95))))
    metrics_path = Path(args.checkpoint).parent / "dpo_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps):
        row = rows[step % len(rows)]
        cids, clabels = completion_ids(tokenizer, row["prompt"], row["chosen"], cfg.max_seq_length)
        rids, rlabels = completion_ids(tokenizer, row["prompt"], row["rejected"], cfg.max_seq_length)
        cids, clabels, rids, rlabels = (v.to(device) for v in (cids, clabels, rids, rlabels))
        if not clabels[:, 1:].ne(-100).any() or not rlabels[:, 1:].ne(-100).any():
            raise ValueError("chosen/rejected completion must contain at least one next-token target")
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        pc = sequence_log_probability(policy(cids), clabels)
        pr = sequence_log_probability(policy(rids), rlabels)
        with torch.no_grad():
            rc = sequence_log_probability(reference(cids), clabels)
            rr = sequence_log_probability(reference(rids), rlabels)
        loss, metrics = dpo_loss(pc, pr, rc, rr, args.beta)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite DPO loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), training.get("grad_clip", 1.0))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite DPO gradient norm")
        optimizer.step()
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": step + 1, "loss": float(loss),
                                     "preference_accuracy": float(metrics["accuracy"])}) + "\n")
    output = Path(args.checkpoint)
    torch.save({"model": policy.state_dict(), "model_config": cfg.__dict__,
                "tokenizer_version": base.get("tokenizer_version"),
                "training_method": "dpo", "steps": args.steps, "beta": args.beta}, output)
    print(f"saved DPO policy checkpoint: {output}")


if __name__ == "__main__":
    main()

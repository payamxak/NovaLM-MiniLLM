import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from novalm.config import load_model_config
from novalm.data.shards import TokenShardDataset, verify_shard_manifest
from novalm.evaluation.perplexity import evaluate_loss
from novalm.model.config import validate_checkpoint_config
from novalm.model.transformer import NovaLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate next-token loss/perplexity on token shards")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()
    config = load_model_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NovaLM(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint_config(config, checkpoint)
    model.load_state_dict(checkpoint["model"])
    paths = sorted(Path(args.shards).glob("tokens-*.bin"))
    if not paths:
        parser.error("no token shards found")
    verify_shard_manifest(Path(args.shards) / "manifest.json")
    dataset = ConcatDataset([TokenShardDataset(path, config.max_seq_length) for path in paths])
    if not len(dataset):
        parser.error("evaluation shards contain too few tokens for one context/target window")
    metrics = evaluate_loss(model, DataLoader(dataset, batch_size=args.batch_size), device)
    metrics.update({"checkpoint": str(args.checkpoint), "config": str(args.config),
                    "shards": [p.name for p in paths]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for shard in dataset.datasets:
        shard.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

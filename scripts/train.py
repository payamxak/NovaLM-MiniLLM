import argparse
import json
import platform
from pathlib import Path

import torch
import yaml
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader

from novalm.config import load_model_config
from novalm.data.shards import TokenShardDataset, verify_shard_manifest
from novalm.model.transformer import NovaLM
from novalm.training.distributed import initialize_distributed, wrap_ddp
from novalm.training.profiler import profiler_context
from novalm.training.sampler import DistributedResumableSampler, ResumableRandomSampler
from novalm.training.seed import set_seed
from novalm.training.trainer import Trainer, TrainerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain NovaLM on token shards")
    parser.add_argument("--config", required=True, help="Model YAML")
    parser.add_argument("--training-config", default="configs/training/debug.yaml")
    parser.add_argument("--shards", required=True, help="Directory of tokens-*.bin shards")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100,
                        help="Write an optimizer checkpoint every N steps and at run completion")
    parser.add_argument("--best-checkpoint", help="Optional model checkpoint path for best validation loss")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--resume", action="store_true", help="Resume model, optimizer, scheduler step and RNG state")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-shards", help="Optional directory of validation token shards")
    parser.add_argument("--eval-every", type=int, default=100, help="Validate every N optimizer steps")
    parser.add_argument("--eval-batches", type=int, default=10, help="Maximum validation batches per evaluation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--profile", action="store_true", help="write a bounded torch.profiler trace")
    parser.add_argument("--profile-dir", default="outputs/profile")
    args = parser.parse_args()
    if (args.steps <= 0 or args.batch_size <= 0 or args.eval_every < 0
            or args.eval_batches <= 0 or args.save_every <= 0):
        parser.error("steps, batch size, eval-batches, and save-every must be positive; eval-every cannot be negative")
    if args.best_checkpoint and not args.validation_shards:
        parser.error("--best-checkpoint requires --validation-shards")
    rank, local_rank, world_size = initialize_distributed()
    set_seed(args.seed)
    model_config = load_model_config(args.config)
    model = NovaLM(model_config)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if world_size > 1 and args.device == "cpu" and torch.distributed.is_initialized():
        # Explicit CPU runs remain valid under torchrun and use the gloo backend.
        device = "cpu"
    else:
        device = ((f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
                  if args.device == "auto" else args.device)
    model = model.to(device)
    if world_size > 1:
        model = wrap_ddp(model, torch.device(device), local_rank, world_size)
    if rank == 0:
        print(f"parameters: {model.module.parameter_count if world_size > 1 else model.parameter_count:,}")
    data_cfg = yaml.safe_load(Path(args.training_config).read_text(encoding="utf-8"))
    training_config = TrainerConfig(**data_cfg)
    files = sorted(Path(args.shards).glob("tokens-*.bin"))
    if not files:
        parser.error("no token shards found")
    verify_shard_manifest(Path(args.shards) / "manifest.json")
    manifest = yaml.safe_load((Path(args.shards) / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_vocab_sizes = manifest.get("tokenizer_vocab_sizes", [])
    if tokenizer_vocab_sizes and tokenizer_vocab_sizes != [model_config.vocab_size]:
        parser.error(f"model vocab_size={model_config.vocab_size} does not match shard tokenizer vocab {tokenizer_vocab_sizes}")
    train_shards = [TokenShardDataset(path, model_config.max_seq_length) for path in files]
    dataset = ConcatDataset(train_shards)
    if not len(dataset):
        parser.error("training shards contain too few tokens for one context/target window")
    sampler = (DistributedResumableSampler(dataset, world_size, rank, seed=args.seed)
               if world_size > 1 else ResumableRandomSampler(dataset, seed=args.seed))
    loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler, args.batch_size,
                                                            drop_last=False))
    validation_loader = None
    if args.validation_shards:
        validation_dir = Path(args.validation_shards)
        validation_files = sorted(validation_dir.glob("tokens-*.bin"))
        if not validation_files:
            parser.error("no validation token shards found")
        verify_shard_manifest(validation_dir / "manifest.json")
        validation_manifest = yaml.safe_load((validation_dir / "manifest.json").read_text(encoding="utf-8"))
        validation_vocab_sizes = validation_manifest.get("tokenizer_vocab_sizes", [])
        if validation_vocab_sizes and validation_vocab_sizes != [model_config.vocab_size]:
            parser.error("model vocab_size does not match validation shard tokenizer vocab")
        validation_dataset = ConcatDataset([
            TokenShardDataset(path, model_config.max_seq_length) for path in validation_files
        ])
        if not len(validation_dataset):
            parser.error("validation shards contain too few tokens for one context/target window")
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    trainer = Trainer(model, training_config, device, rank=rank, world_size=world_size)
    trainer.tokenizer_version = manifest.get("tokenizer_sha256", [None])[0]
    if args.resume:
        trainer.load_checkpoint(args.checkpoint)
    if args.profile and args.steps < 4:
        parser.error("--profile requires at least 4 steps for the default profiler schedule")
    with profiler_context(args.profile and rank == 0, args.profile_dir) as profiler:
        losses = trainer.fit(loader, args.steps, args.checkpoint,
                             Path(args.checkpoint).parent / "metrics.jsonl",
                             validation_batches=validation_loader,
                             eval_every_steps=args.eval_every if validation_loader is not None else 0,
                             max_eval_batches=args.eval_batches, data_sampler=sampler,
                             checkpoint_every_steps=args.save_every,
                             best_checkpoint_path=args.best_checkpoint,
                             profiler=profiler)
    if rank == 0:
        run_dir = Path(args.checkpoint).parent
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "training_config.yaml").write_text(Path(args.training_config).read_text(encoding="utf-8"), encoding="utf-8")
        environment = {"python": platform.python_version(), "torch": torch.__version__,
                       "cuda": torch.version.cuda,
                       "gpu": (torch.cuda.get_device_name(device)
                               if torch.cuda.is_available() and str(device).startswith("cuda") else None),
                       "seed": args.seed, "parameter_count": (model.module.parameter_count
                       if world_size > 1 else model.parameter_count),
                       "effective_batch_size": args.batch_size * training_config.gradient_accumulation_steps * world_size,
                       "world_size": world_size, "rank": rank, "device": device}
        (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        (run_dir / "command.txt").write_text("python " + " ".join(__import__("sys").argv) + "\n", encoding="utf-8")
        commit = trainer.git_commit or "unavailable (workspace is not a git repository)"
        (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        print(f"finished {trainer.step} optimizer steps; last loss {losses[-1]:.4f}")
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()

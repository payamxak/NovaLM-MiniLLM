import argparse
import hashlib
import sys
from pathlib import Path

import torch

from novalm.config import load_model_config
from novalm.inference.generate import generate
from novalm.model.config import validate_checkpoint_config
from novalm.model.transformer import NovaLM
from novalm.tokenizer.tokenizer import NovaTokenizer


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate from a NovaLM checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, tokenizer = load_model_config(args.config), NovaTokenizer(args.tokenizer)
    model = NovaLM(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint_config(config, state)
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if state.get("tokenizer_version") and state["tokenizer_version"] != tokenizer_hash:
        parser.error("tokenizer file does not match checkpoint tokenizer version")
    model.load_state_dict(state["model"])
    ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    result = generate(model, ids, args.max_new_tokens, eos_token_id=2, temperature=0.8, top_p=0.95)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()

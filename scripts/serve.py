import argparse
import hashlib
from pathlib import Path

import torch
import uvicorn

from novalm.config import load_model_config
from novalm.model.config import validate_checkpoint_config
from novalm.model.transformer import NovaLM
from novalm.serving.app import create_app
from novalm.tokenizer.tokenizer import NovaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a NovaLM checkpoint over HTTP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_model_config(args.config)
    model = NovaLM(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint_config(config, checkpoint)
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if checkpoint.get("tokenizer_version") and checkpoint["tokenizer_version"] != tokenizer_hash:
        parser.error("tokenizer file does not match checkpoint tokenizer version")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    app = create_app(model, NovaTokenizer(args.tokenizer))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

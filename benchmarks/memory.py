"""Report parameter and KV-cache memory estimates for configured architectures."""
import argparse
import json

import torch

from novalm.config import load_model_config
from novalm.model.transformer import NovaLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model/debug.yaml")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or args.context_length < 1:
        parser.error("batch-size and context-length must be positive")
    config = load_model_config(args.config)
    if args.context_length > config.max_seq_length:
        parser.error("context length exceeds configured model maximum")
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]
    model = NovaLM(config).to(dtype=dtype)
    parameters = model.parameter_count
    element_size = torch.tensor([], dtype=dtype).element_size()
    is_gqa = config.attention_type == "gqa"
    cache_heads = config.num_kv_heads if is_gqa else config.mla_latent_dim
    cache_width = config.head_dim if is_gqa else 1
    cache_tensors = 2 if is_gqa else 1
    cache_bytes = (cache_tensors * config.num_layers * args.batch_size * args.context_length
                   * cache_heads * cache_width * element_size)
    print(json.dumps({"parameters": parameters,
                      "parameter_bytes_estimate": parameters * element_size,
                      "kv_cache_bytes_estimate": cache_bytes,
                      "dtype": args.dtype,
                      "context_length": args.context_length,
                      "batch_size": args.batch_size,
                      "attention_type": config.attention_type}, indent=2))


if __name__ == "__main__":
    main()

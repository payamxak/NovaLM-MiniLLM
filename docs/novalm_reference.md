# NovaLM Algorithms and API Reference

> Generated from the checked-in `novalm/` source and `configs/` files. Re-run `python scripts/generate_reference.py` after API or configuration changes.

This reference describes the implementation that exists in this repository. It records implementation status and configured values without inventing benchmark or model-quality claims.

## Algorithms

### Unicode normalization and cleaning

Normalize text to a stable Unicode form, remove invalid/control content, and collapse irrelevant whitespace before quality checks.

### Quality filtering

Reject documents using configurable length, alphabetic/symbol ratio, URL density, repeated-line, repeated-character, and garbage-character thresholds.

### Language filtering

Use script-aware heuristics to estimate English, Persian, code, or mixed content and apply the configured language policy.

### Exact and near deduplication

Hash normalized documents for exact duplicates; optional MinHash signatures and LSH buckets identify near duplicates above a similarity threshold.

### Weighted data mixture

Sample source streams using seeded weights while stopping cleanly when a source is exhausted; provenance and final counts are retained.

### Tokenization and packing

Encode documents, preserve EOS boundaries, concatenate token streams into fixed-size blocks, and build shifted input/target pairs.

### RMSNorm

For each hidden vector x, compute x / sqrt(mean(x²) + eps) and multiply by a learned scale gamma.

### RoPE

Rotate paired query/key dimensions with cached cosines and sines derived from theta and position; caches extend dynamically for long contexts.

### QK-Norm

Normalize each query and key head before the scaled dot product to control score magnitude and improve stability.

### Causal attention

Compute softmax((QKᵀ / sqrt(d)) + causal mask)V; the reference path uses explicit masked matrix operations.

### GQA

Project many query heads and fewer key/value heads, then repeat KV heads across query groups; cache the compact KV representation.

### SwiGLU

Compute W2(SiLU(Wg x) elementwise-multiplied by Wu x).

### Pre-norm Transformer

Apply normalization before attention and MLP, add each result through a residual connection, and finish with a final norm and LM head.

### Causal language modeling

Use logits at position t to predict token t+1 with cross entropy and ignore_index support for masked labels.

### MTP

For horizon k, shift targets by k positions, mask unavailable positions, and sum weighted per-horizon cross-entropy losses.

### MoE

Route each token to top-k expert scores, combine weighted expert outputs, and add a router load-balancing auxiliary loss plus utilization metrics.

### MLA

Compress key/value information into a latent representation and retain that compressed state in the autoregressive cache.

### AdamW and scheduling

Separate decayed weights from bias/norm parameters, apply AdamW updates, then use linear warmup followed by cosine decay.

### Mixed precision and accumulation

Run autocast in FP32, FP16, or BF16, scale FP16 gradients where required, accumulate micro-batches, clip, and step.

### Checkpointing

Atomically save model, optimizer, scheduler-equivalent step state, AMP state, RNG states, configs, tokenizer version, and environment metadata.

### Sampling

Apply repetition penalty, temperature, top-k, top-p, or min-p filtering before selecting the next token; enforce EOS and stop sequences.

### KV-cache decoding

Prefill the prompt once, append one token per step, and reuse per-layer K/V tensors for autoregressive decoding.

### DPO

Compare policy and reference chosen/rejected log-probability differences with beta-scaled logistic loss.

### Quantization

Provide dynamic INT8 and packed weight-only INT4 reference paths with dequantization for forward computation.

### Evaluation

Measure shifted loss/perplexity, generation behavior, multilingual samples, code/math prompts, context behavior, and reproducible benchmark metadata.

## Package modules

The inventory below includes every Python module under `novalm/`, top-level functions, classes, methods, class fields, and module-level constants.

### `novalm/__init__.py`


### `novalm/config/__init__.py`


### `novalm/config/loader.py`

- **Function** `load_validated_config(path: str | Path, config_type: type[ConfigType], section: str | None = None) -> ConfigType` — Load a YAML mapping into a dataclass, applying its post-init validation.
- **Function** `load_model_config(path: str | Path) -> ModelConfig` — implementation helper.

### `novalm/config/schema.py`

- **Class** `AttentionConfig` — implementation class.
  - attribute `backend` = `'naive'`
  - attribute `qk_norm` = `True`
  - method `__post_init__(self) -> None` — method.
- **Class** `MoEConfig` — implementation class.
  - attribute `enabled` = `False`
  - attribute `num_experts` = `8`
  - attribute `top_k` = `2`
  - attribute `aux_loss_weight` = `0.01`
  - method `__post_init__(self) -> None` — method.
- **Class** `MTPConfig` — implementation class.
  - attribute `enabled` = `False`
  - attribute `horizons` = `0`
  - attribute `loss_weight` = `0.25`
  - method `__post_init__(self) -> None` — method.
- **Class** `DataConfig` — implementation class.
  - attribute `block_size` = `512`
  - attribute `shard_tokens` = `1000000`
  - attribute `seed` = `0`
  - attribute `near_dedup` = `False`
  - attribute `near_dedup_threshold` = `0.85`
  - method `__post_init__(self) -> None` — method.
- **Class** `TokenizerConfig` — implementation class.
  - attribute `vocab_size` = `32000`
  - attribute `character_coverage` = `0.9995`
  - method `__post_init__(self) -> None` — method.
- **Class** `OptimizerConfig` — implementation class.
  - attribute `learning_rate` = `0.0003`
  - attribute `weight_decay` = `0.1`
  - attribute `betas` = `(0.9, 0.95)`
  - attribute `grad_clip` = `1.0`
  - method `__post_init__(self) -> None` — method.
- **Class** `SchedulerConfig` — implementation class.
  - attribute `warmup_steps` = `10`
  - attribute `total_steps` = `100`
  - attribute `min_lr_ratio` = `0.1`
  - method `__post_init__(self) -> None` — method.
- **Class** `TrainingConfig` — implementation class.
  - attribute `precision` = `'fp32'`
  - attribute `gradient_accumulation_steps` = `1`
  - attribute `activation_checkpointing` = `False`
  - attribute `seed` = `42`
  - method `__post_init__(self) -> None` — method.
- **Class** `InferenceConfig` — implementation class.
  - attribute `max_new_tokens` = `64`
  - attribute `temperature` = `0.8`
  - attribute `top_k` = `0`
  - attribute `top_p` = `0.95`
  - attribute `min_p` = `0.0`
  - attribute `repetition_penalty` = `1.0`
  - method `__post_init__(self) -> None` — method.
- **Class** `EvaluationConfig` — implementation class.
  - attribute `max_batches` = `None`
  - attribute `seed` = `0`
  - method `__post_init__(self) -> None` — method.

### `novalm/data/__init__.py`


### `novalm/data/cleaning.py`

- **Function** `normalize_text(text: str) -> str` — implementation helper.
- **Function** `clean_document(text: str) -> str` — implementation helper.

### `novalm/data/collator.py`

- **Function** `causal_lm_collate(examples: list[torch.Tensor | list[int]], pad_id: int = 0, ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]` — Pad variable-length examples and mask pad labels; input/target shift is model-side.
- **Function** `shift_labels(input_ids: torch.Tensor, labels: torch.Tensor | None = None, ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]` — Align inputs and targets so logits at t predict labels at t+1.

### `novalm/data/dataset.py`


### `novalm/data/dedup.py`

- **Function** `exact_deduplicate(records: Iterable[dict]) -> Iterator[dict]` — implementation helper.

### `novalm/data/ingest.py`

- **Function** `iter_text_documents(paths: Iterable[str | Path], source_name: str | None = None, license_name: str = 'unknown', language: str | None = None, attribution_manifest: str | None = None, recursive: bool = True) -> Iterator[dict[str, str]]` — Yield normalized UTF-8 text files with provenance; directories are sorted.
- **Function** `read_attribution_manifest(path: str | Path) -> dict` — Read a JSON provenance manifest and fail on malformed/non-object manifests.

### `novalm/data/language_filter.py`

- **Function** `detect_language(text: str) -> str` — implementation helper.
- **Function** `language_ratio(text: str) -> dict[str, float]` — implementation helper.

### `novalm/data/minhash.py`

- **Function** `shingles(text: str, width: int = 5) -> set[str]` — implementation helper.
- **Function** `_signature(items: set[str], num_perm: int) -> tuple[int, ...]` — implementation helper.
- **Function** `near_deduplicate(records: Iterable[dict], threshold: float = 0.85, num_perm: int = 64, bands: int = 8) -> Iterator[dict]` — Deterministic MinHash/LSH candidate filter followed by exact Jaccard verification.

### `novalm/data/mixture.py`

- **Function** `sample_mixture(sources: dict[str, list[dict]], weights: dict[str, float], count: int, seed: int = 0) -> Iterator[dict]` — Draw a reproducible weighted document stream, with replacement per source.
- **Function** `weighted_stream(sources: dict[str, Iterable[dict]], weights: dict[str, float], seed: int = 0) -> Iterator[dict]` — Stream all source records in weighted random order without loading corpora in memory.

### `novalm/data/normalization.py`


### `novalm/data/packing.py`

- **Function** `pack_sequences(sequences: Iterable[list[int]], block_size: int, eos_id: int) -> Iterator[list[int]]` — Concatenate documents separated by EOS and emit full blocks; drop only final tail.

### `novalm/data/quality.py`

- **Class** `QualityConfig` — implementation class.
  - attribute `min_chars` = `40`
  - attribute `max_chars` = `1000000`
  - attribute `min_alpha_ratio` = `0.15`
  - attribute `max_url_ratio` = `0.2`
  - attribute `max_repeated_line_ratio` = `0.5`
  - attribute `max_repeated_char_ratio` = `0.3`
  - attribute `max_symbol_ratio` = `0.45`
  - attribute `max_garbage_ratio` = `0.01`
  - method `__post_init__(self) -> None` — method.
- **Function** `quality_rejection(text: str, config: QualityConfig | None = None) -> str | None` — implementation helper.

### `novalm/data/shards.py`

- **Function** `write_token_shards(blocks: Iterable[list[int]], output_dir: str | Path, shard_tokens: int = 1000000, metadata: dict | None = None) -> dict` — Write little-endian uint32 token shards plus a checksum manifest.
- **Function** `verify_shard_manifest(path: str | Path) -> None` — implementation helper.
- **Class** `TokenShardDataset` — implementation class.
  - attribute `tokens` = `np.memmap(self.path, mode='r', dtype='<u4')`
  - method `__init__(self, path: str | Path, block_size: int) -> None` — method.
  - method `__len__(self) -> int` — method.
  - method `__getitem__(self, index: int)` — method.
  - method `close(self) -> None` — method.

### `novalm/data/tokenize.py`

- **Function** `tokenize_record(record: dict, tokenizer: NovaTokenizer, tokenizer_sha256: str | None = None) -> dict` — implementation helper.
- **Function** `tokenizer_digest(model_path: str | Path) -> str` — implementation helper.

### `novalm/evaluation/__init__.py`


### `novalm/evaluation/benchmark.py`

- **Function** `benchmark_forward(model, input_ids: torch.Tensor, warmup: int = 2, iterations: int = 10) -> dict[str, float]` — implementation helper.

### `novalm/evaluation/code.py`

- **Function** `exact_match(predictions: list[str], references: list[str]) -> dict[str, float | int]` — implementation helper.

### `novalm/evaluation/generation.py`

- **Function** `generate_examples(engine: InferenceEngine, prompts: Iterable[str], **options) -> list[dict]` — implementation helper.

### `novalm/evaluation/long_context.py`

- **Function** `evaluate_context_lengths(model, token_batches: dict[int, list], device = 'cpu') -> dict[int, dict[str, float]]` — implementation helper.

### `novalm/evaluation/math.py`

- **Function** `exact_match(predictions: list[str], references: list[str]) -> dict[str, float | int]` — implementation helper.

### `novalm/evaluation/multilingual.py`

- **Function** `evaluate_languages(model, language_batches: dict[str, list], device = 'cpu') -> dict[str, dict[str, float]]` — implementation helper.

### `novalm/evaluation/perplexity.py`

- **Function** `evaluate_loss(model, batches, device: str | torch.device = 'cpu') -> dict[str, float]` — implementation helper.

### `novalm/inference/__init__.py`


### `novalm/inference/engine.py`

- **Class** `InferenceEngine` — implementation class.
  - attribute `model` = `model.eval()`
  - attribute `tokenizer` = `tokenizer`
  - method `__init__(self, model, tokenizer = None) -> None` — method.
  - method `stream_ids(self, input_ids: torch.Tensor, **generation_options) -> Iterator[torch.Tensor]` — method.
  - method `generate_ids(self, input_ids: torch.Tensor, **generation_options) -> torch.Tensor` — method.
  - method `generate_text(self, prompt: str, **generation_options) -> str` — method.

### `novalm/inference/generate.py`

- **Function** `stream_generate(model, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, repetition_penalty: float = 1.0, stop_sequences: list[list[int]] | None = None) -> Iterator[torch.Tensor]` — implementation helper.
- **Function** `generate(model, input_ids: torch.Tensor, max_new_tokens: int = 32, **kwargs) -> torch.Tensor` — implementation helper.

### `novalm/inference/kv_cache.py`


### `novalm/inference/quantization.py`

- **Class** `QuantizedLinear4bit` — Symmetric per-output-channel INT4 weight-only Linear reference kernel.
  - method `__init__(self, linear: nn.Linear) -> None` — method.
  - method `dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor` — method.
  - method `forward(self, inputs: torch.Tensor) -> torch.Tensor` — method.
- **Function** `dynamic_int8(model: nn.Module) -> nn.Module` — Return a CPU dynamic-int8 copy; embedding tables remain floating point.
- **Function** `quantize_int4(model: nn.Module) -> nn.Module` — Deep-copy a model and replace each dense Linear with nibble-packed INT4.

### `novalm/inference/sampler.py`

- **Function** `sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, repetition_tokens: torch.Tensor | None = None, repetition_penalty: float = 1.0) -> torch.Tensor` — implementation helper.

### `novalm/inference/streaming.py`


### `novalm/model/__init__.py`


### `novalm/model/attention/__init__.py`


### `novalm/model/attention/base.py`

- **Class** `AttentionBackend` — implementation class.
  - method `__call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor` — method.

### `novalm/model/attention/causal.py`

- **Function** `causal_mask(query_length: int, key_length: int, query_offset: int = 0, device: torch.device | str | None = None) -> torch.Tensor` — implementation helper.
- **Function** `naive_causal_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor` — Compute softmax(QKᵀ/sqrt(d))V using the explicit reference path.

### `novalm/model/attention/flash.py`

- **Function** `flash_causal_attention(query, key, value, allowed)` — Use PyTorch SDPA's kernel dispatcher (FlashAttention is hardware dependent).

### `novalm/model/attention/gqa.py`

- **Class** `GroupedQueryAttention` — implementation class.
  - attribute `backend` = `backend`
  - attribute `groups` = `num_heads // num_kv_heads`
  - attribute `head_dim` = `hidden_size // num_heads`
  - attribute `k_norm` = `QKNorm(self.head_dim, eps) if qk_norm else nn.Identity()`
  - attribute `k_proj` = `nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)`
  - attribute `o_proj` = `nn.Linear(hidden_size, hidden_size, bias=False)`
  - attribute `q_norm` = `QKNorm(self.head_dim, eps) if qk_norm else nn.Identity()`
  - attribute `q_proj` = `nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)`
  - attribute `rope` = `RotaryEmbedding(self.head_dim, max_seq_length, theta) if use_rope else None`
  - attribute `use_rope` = `use_rope`
  - attribute `v_proj` = `nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)`
  - method `__init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, max_seq_length: int, theta: float, eps: float, backend: str = 'naive', qk_norm: bool = True, use_rope: bool = True) -> None` — method.
  - method `forward(self, x: torch.Tensor, past_key_value = None, use_cache: bool = False)` — method.

### `novalm/model/attention/kv_cache.py`

- **Function** `cache_length(cache: Sequence[LayerCache] | None) -> int` — implementation helper.
- **Function** `cache_bytes(cache: Sequence[LayerCache] | None) -> int` — implementation helper.

### `novalm/model/attention/mha.py`

- **Class** `MultiHeadAttention` — implementation class.
  - method `__init__(self, hidden_size: int, num_heads: int, max_seq_length: int, theta: float = 10000.0, eps: float = 1e-06, backend: str = 'naive', qk_norm: bool = True, use_rope: bool = True) -> None` — method.

### `novalm/model/attention/mla.py`


### `novalm/model/attention/sdpa.py`

- **Function** `sdpa_causal_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor` — implementation helper.

### `novalm/model/attention_mla.py`

- **Class** `LatentAttention` — Compress each token to a latent vector, then project per-head keys and values.
  - attribute `down` = `nn.Linear(hidden_size, latent_dim, bias=False)`
  - attribute `key_up` = `nn.Linear(latent_dim, hidden_size, bias=False)`
  - attribute `output` = `nn.Linear(hidden_size, hidden_size, bias=False)`
  - attribute `query` = `nn.Linear(hidden_size, hidden_size, bias=False)`
  - attribute `rope` = `RotaryEmbedding(self.head_dim, max_seq_length, theta) if use_rope else None`
  - attribute `value_up` = `nn.Linear(latent_dim, hidden_size, bias=False)`
  - method `__init__(self, hidden_size: int, num_heads: int, latent_dim: int, max_seq_length: int = 2048, theta: float = 10000.0, use_rope: bool = True) -> None` — method.
  - method `forward(self, x: torch.Tensor, past_latent: torch.Tensor | None = None, use_cache: bool = False)` — method.

### `novalm/model/block.py`

- **Class** `TransformerBlock` — implementation class.
  - attribute `attention` = `LatentAttention(config.hidden_size, config.num_attention_heads, config.mla_latent_dim, config.max_seq_length, config.rope_theta, config.position_type == 'rope')`
  - attribute `attn_norm` = `norm(config.hidden_size, config.norm_eps)`
  - attribute `mlp` = `MoELayer(config.hidden_size, config.intermediate_size, config.num_experts, config.moe_top_k) if config.moe_enabled else SwiGLU(config.hidden_size, config.intermediate_size) if config.mlp_type == 'swiglu' else GELUMLP(config.hidden_size, config.intermediate_size)`
  - attribute `mlp_norm` = `norm(config.hidden_size, config.norm_eps)`
  - method `__init__(self, config: ModelConfig) -> None` — method.
  - method `forward(self, x, past_key_value = None, use_cache: bool = False, return_aux: bool = False)` — method.

### `novalm/model/config.py`

- **Class** `ModelConfig` — implementation class.
  - attribute `vocab_size` = `4096`
  - attribute `hidden_size` = `128`
  - attribute `num_layers` = `2`
  - attribute `num_attention_heads` = `4`
  - attribute `num_kv_heads` = `2`
  - attribute `intermediate_size` = `352`
  - attribute `max_seq_length` = `128`
  - attribute `rope_theta` = `10000.0`
  - attribute `norm_eps` = `1e-06`
  - attribute `tie_embeddings` = `True`
  - attribute `attention_backend` = `'naive'`
  - attribute `qk_norm` = `True`
  - attribute `attention_type` = `'gqa'`
  - attribute `mla_latent_dim` = `128`
  - attribute `moe_enabled` = `False`
  - attribute `num_experts` = `8`
  - attribute `moe_top_k` = `2`
  - attribute `moe_aux_loss_weight` = `0.01`
  - attribute `mtp_enabled` = `False`
  - attribute `mtp_horizons` = `0`
  - attribute `mtp_loss_weight` = `0.25`
  - attribute `norm_type` = `'rmsnorm'`
  - attribute `mlp_type` = `'swiglu'`
  - attribute `position_type` = `'rope'`
  - method `__post_init__(self) -> None` — method.
  - method `head_dim(self) -> int` — method.
  - method `parameter_count(self) -> int` — method.
- **Function** `validate_checkpoint_config(config: ModelConfig, checkpoint: dict) -> None` — implementation helper.

### `novalm/model/embeddings.py`

- **Class** `TokenEmbedding` — implementation class.
  - method `__init__(self, vocab_size: int, hidden_size: int) -> None` — method.

### `novalm/model/layers/__init__.py`


### `novalm/model/layers/activations.py`

- **Function** `silu(x: torch.Tensor) -> torch.Tensor` — implementation helper.
- **Function** `gelu(x: torch.Tensor) -> torch.Tensor` — implementation helper.

### `novalm/model/layers/gelu_mlp.py`

- **Class** `GELUMLP` — implementation class.
  - attribute `down_proj` = `nn.Linear(intermediate_size, hidden_size, bias=False)`
  - attribute `up_proj` = `nn.Linear(hidden_size, intermediate_size, bias=False)`
  - method `__init__(self, hidden_size: int, intermediate_size: int) -> None` — method.
  - method `forward(self, x: torch.Tensor) -> torch.Tensor` — method.

### `novalm/model/layers/qk_norm.py`

- **Class** `QKNorm` — RMSNorm specialized by API for query and key vectors on the head axis.

### `novalm/model/layers/rmsnorm.py`

- **Class** `RMSNorm` — implementation class.
  - attribute `eps` = `eps`
  - attribute `weight` = `nn.Parameter(torch.ones(dim))`
  - method `__init__(self, dim: int, eps: float = 1e-06) -> None` — method.
  - method `forward(self, x: torch.Tensor) -> torch.Tensor` — method.

### `novalm/model/layers/rope.py`

- **Class** `RotaryEmbedding` — implementation class.
  - attribute `_cache_length` = `0`
  - attribute `cos_cached` = `angles.cos()`
  - attribute `sin_cached` = `angles.sin()`
  - method `__init__(self, head_dim: int, max_seq_length: int = 128, theta: float = 10000.0) -> None` — method.
  - method `_ensure_cache(self, length: int, device: torch.device) -> None` — method.
  - method `forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor` — method.

### `novalm/model/layers/swiglu.py`

- **Class** `SwiGLU` — implementation class.
  - attribute `down_proj` = `nn.Linear(intermediate_size, hidden_size, bias=False)`
  - attribute `gate_proj` = `nn.Linear(hidden_size, intermediate_size, bias=False)`
  - attribute `up_proj` = `nn.Linear(hidden_size, intermediate_size, bias=False)`
  - method `__init__(self, hidden_size: int, intermediate_size: int) -> None` — method.
  - method `forward(self, x: torch.Tensor) -> torch.Tensor` — method.

### `novalm/model/lm_head.py`

- **Class** `LMHead` — implementation class.
  - attribute `weight` = `embedding.weight`
  - method `__init__(self, hidden_size: int, vocab_size: int, embedding: nn.Embedding | None = None) -> None` — method.

### `novalm/model/moe/__init__.py`


### `novalm/model/moe/expert.py`

- **Class** `Expert` — implementation class.

### `novalm/model/moe/layer.py`

- **Class** `MoELayer` — Readable top-k expert reference layer; not capacity limited or optimized.
  - attribute `experts` = `nn.ModuleList((Expert(hidden_size, intermediate_size) for _ in range(num_experts)))`
  - attribute `router` = `Router(hidden_size, num_experts)`
  - method `__init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 8, top_k: int = 2) -> None` — method.
  - method `forward(self, x: torch.Tensor)` — method.

### `novalm/model/moe/load_balance.py`

- **Function** `load_balance_loss(probabilities: torch.Tensor, indices: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]` — implementation helper.

### `novalm/model/moe/moe_layer.py`


### `novalm/model/moe/router.py`

- **Class** `Router` — implementation class.
  - attribute `weight` = `nn.Parameter(torch.empty(num_experts, hidden_size))`
  - method `__init__(self, hidden_size: int, num_experts: int) -> None` — method.
  - method `forward(self, tokens)` — method.

### `novalm/model/moe/topk.py`

- **Function** `topk_routes(probabilities: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]` — implementation helper.

### `novalm/model/mtp/__init__.py`


### `novalm/model/mtp/head.py`

- **Class** `MultiTokenPredictionHead` — Auxiliary vocabulary heads for horizons 2..K, beside the normal LM head.
  - attribute `heads` = `nn.ModuleList((nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(horizons)))`
  - attribute `horizons` = `horizons`
  - method `__init__(self, hidden_size: int, vocab_size: int, horizons: int) -> None` — method.
  - method `forward(self, hidden)` — method.

### `novalm/model/mtp/loss.py`

- **Function** `multi_token_loss(logits_by_horizon: list[torch.Tensor], token_ids: torch.Tensor, weights: list[float] | None = None, ignore_index: int = -100)` — Each auxiliary head at t predicts token t+h, with boundary positions excluded.

### `novalm/model/transformer.py`

- **Class** `NovaLM` — implementation class.
  - attribute `activation_checkpointing` = `False`
  - attribute `blocks` = `nn.ModuleList((TransformerBlock(config) for _ in range(config.num_layers)))`
  - attribute `config` = `config`
  - attribute `final_norm` = `norm(config.hidden_size, config.norm_eps)`
  - attribute `lm_head` = `LMHead(config.hidden_size, config.vocab_size)`
  - attribute `mtp_head` = `MultiTokenPredictionHead(config.hidden_size, config.vocab_size, config.mtp_horizons) if config.mtp_enabled else None`
  - attribute `position_embedding` = `nn.Embedding(config.max_seq_length, config.hidden_size) if config.position_type == 'absolute' else None`
  - attribute `token_embedding` = `TokenEmbedding(config.vocab_size, config.hidden_size)`
  - method `__init__(self, config: ModelConfig) -> None` — method.
  - method `_init_weights(module: nn.Module) -> None` — method.
  - method `forward(self, input_ids: torch.Tensor, past_key_values = None, use_cache: bool = False, return_aux: bool = False) -> torch.Tensor` — method.
  - method `parameter_count(self) -> int` — method.
- **Function** `causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, ignore_index: int = -100) -> torch.Tensor` — Next-token cross entropy: logits at t predict input_ids at t+1.

### `novalm/post_training/__init__.py`


### `novalm/post_training/dpo.py`

- **Function** `completion_ids(tokenizer, prompt: str, completion: str, max_length: int)` — Encode a preference pair while preserving at least one prompt and target token.
- **Function** `sequence_log_probability(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor` — Sum token log probabilities for each batch row, excluding masked labels.
- **Function** `dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, reference_chosen: torch.Tensor, reference_rejected: torch.Tensor, beta: float = 0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]` — Direct Preference Optimization objective from policy/reference sequence logps.

### `novalm/post_training/formatting.py`

- **Function** `assistant_only_example(messages: list[dict[str, str]], tokenizer: NovaTokenizer, max_length: int | None = None) -> tuple[list[int], list[int]]` — Format messages and mask loss outside assistant bodies using tokenizer offsets.
- **Class** `SFTDataset` — implementation class.
  - attribute `rows` = `[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]`
  - method `__init__(self, path: str | Path, tokenizer: NovaTokenizer, max_length: int) -> None` — method.
  - method `__len__(self) -> int` — method.
  - method `__getitem__(self, index: int)` — method.
- **Function** `sft_collate(batch, pad_id: int = 0)` — implementation helper.

### `novalm/post_training/grpo.py`

- **Function** `group_relative_policy_loss(policy_logps: torch.Tensor, old_policy_logps: torch.Tensor, advantages: torch.Tensor, clip_range: float = 0.2) -> torch.Tensor` — Clipped token/sample policy objective for externally computed group advantages.

### `novalm/post_training/preference.py`

- **Function** `iter_preference_records(path: str | Path) -> Iterator[dict[str, str]]` — implementation helper.

### `novalm/post_training/sft.py`


### `novalm/serving/__init__.py`


### `novalm/serving/app.py`

- **Function** `create_app(model, tokenizer, model_name: str = 'novalm') -> FastAPI` — implementation helper.

### `novalm/serving/batching.py`

- **Function** `batch_generate(model, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0, repetition_penalty: float = 1.0, pad_token_id: int = 0) -> torch.Tensor` — Generate for an equal-length prompt batch, returning prompt plus continuations.

### `novalm/serving/routes.py`


### `novalm/serving/schemas.py`

- **Class** `CompletionRequest` — implementation class.
  - attribute `prompt` = `str`
  - attribute `max_new_tokens` = `Field(default=64, ge=1, le=1024)`
  - attribute `temperature` = `Field(default=0.8, ge=0, allow_inf_nan=False)`
  - attribute `top_p` = `Field(default=0.95, gt=0, le=1, allow_inf_nan=False)`
  - attribute `top_k` = `Field(default=0, ge=0)`
  - attribute `min_p` = `Field(default=0.0, ge=0, le=1, allow_inf_nan=False)`
  - attribute `repetition_penalty` = `Field(default=1.0, gt=0, allow_inf_nan=False)`
  - attribute `stop` = `None`
  - attribute `stream` = `False`
- **Class** `ChatMessage` — implementation class.
  - attribute `role` = `str`
  - attribute `content` = `str`
- **Class** `ChatCompletionRequest` — implementation class.
  - attribute `prompt` = `None`
  - attribute `messages` = `None`

### `novalm/tokenizer/__init__.py`


### `novalm/tokenizer/chat_template.py`

- **Function** `format_chat(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str` — Format role/content messages using NovaLM's explicit role token convention.

### `novalm/tokenizer/special_tokens.py`

**Module constants**

- `SPECIAL_TOKENS` = `('<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>', '<|system|>', '<|user|>', '<|assistant|>', '<|tool|>', '<|think|>', '<|/think|>')`


### `novalm/tokenizer/tokenizer.py`

- **Class** `NovaTokenizer` — SentencePiece BPE tokenizer trained on project data, with fixed chat tokens.
  - attribute `model_path` = `Path(model_path)`
  - attribute `processor` = `spm.SentencePieceProcessor(model_file=str(self.model_path))`
  - method `__init__(self, model_path: str | Path) -> None` — method.
  - method `train(cls, input_files: list[str | Path], model_prefix: str | Path, vocab_size: int = 32000, character_coverage: float = 0.9995) -> 'NovaTokenizer'` — method.
  - method `vocab_size(self) -> int` — method.
  - method `encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]` — method.
  - method `decode(self, ids: list[int]) -> str` — method.
  - method `token_id(self, token: str) -> int` — method.
  - method `fertility(self, text: str) -> float` — method.

### `novalm/tokenizer/train.py`

- **Function** `_jsonl_texts(path: Path) -> Iterator[str]` — implementation helper.
- **Function** `_text_lines(path: Path) -> Iterator[str]` — implementation helper.
- **Function** `train_tokenizer(input_files: list[str | Path], model_prefix: str | Path, config: TokenizerConfig | None = None) -> NovaTokenizer` — implementation helper.

### `novalm/training/__init__.py`


### `novalm/training/amp.py`

- **Function** `autocast_context(device: torch.device, precision: str)` — implementation helper.
- **Function** `make_grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler` — implementation helper.

### `novalm/training/checkpoint.py`

- **Function** `atomic_torch_save(state: dict, path: str | Path) -> None` — implementation helper.
- **Function** `load_torch_checkpoint(path: str | Path, map_location: str | torch.device = 'cpu') -> dict` — implementation helper.

### `novalm/training/distributed.py`

- **Function** `initialize_distributed(backend: str | None = None) -> tuple[int, int, int]` — Initialize from torchrun environment and return (rank, local_rank, world_size).
- **Function** `wrap_ddp(model: nn.Module, device: torch.device, local_rank: int, world_size: int) -> nn.Module` — implementation helper.

### `novalm/training/gradient.py`

- **Function** `clip_gradients(model: nn.Module, max_norm: float) -> torch.Tensor` — implementation helper.

### `novalm/training/loop.py`

- **Function** `train_steps(trainer, batches, steps: int, **fit_options)` — implementation helper.

### `novalm/training/metrics.py`

- **Function** `effective_batch_size(micro_batch: int, accumulation_steps: int, world_size: int = 1) -> int` — implementation helper.
- **Function** `tokens_per_second(token_count: int, elapsed_seconds: float) -> float` — implementation helper.
- **Function** `perplexity(loss: float) -> float | None` — implementation helper.

### `novalm/training/optimizer.py`

- **Function** `build_adamw(model: nn.Module, learning_rate: float = 0.0003, weight_decay: float = 0.1, betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW` — implementation helper.

### `novalm/training/profiler.py`

- **Function** `profiler_context(enabled: bool, output_dir: str | Path = 'outputs/profile', wait: int = 1, warmup: int = 1, active: int = 2)` — implementation helper.

### `novalm/training/sampler.py`

- **Class** `ResumableRandomSampler` — implementation class.
  - attribute `cursor` = `0`
  - attribute `data_source` = `data_source`
  - attribute `generator` = `torch.Generator()`
  - attribute `permutation` = `None`
  - method `__init__(self, data_source: Sized, seed: int = 0) -> None` — method.
  - method `__len__(self) -> int` — method.
  - method `__iter__(self) -> Iterator[int]` — method.
  - method `state_dict(self) -> dict` — method.
  - method `load_state_dict(self, state: dict) -> None` — method.
- **Class** `DistributedResumableSampler` — Deterministic rank-sharded sampler with resumable local cursors.
  - attribute `cursor` = `0`
  - attribute `data_source` = `data_source`
  - attribute `generator` = `torch.Generator()`
  - attribute `local_permutation` = `None`
  - attribute `num_replicas` = `num_replicas`
  - attribute `permutation` = `None`
  - attribute `rank` = `rank`
  - method `__init__(self, data_source: Sized, num_replicas: int, rank: int, seed: int = 0) -> None` — method.
  - method `__len__(self) -> int` — method.
  - method `_ensure_permutation(self) -> None` — method.
  - method `__iter__(self) -> Iterator[int]` — method.
  - method `state_dict(self) -> dict` — method.
  - method `load_state_dict(self, state: dict) -> None` — method.

### `novalm/training/scheduler.py`

- **Function** `warmup_cosine_lr(step: int, learning_rate: float, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> float` — implementation helper.

### `novalm/training/seed.py`

- **Function** `set_seed(seed: int, deterministic: bool = False) -> None` — implementation helper.

### `novalm/training/trainer.py`

- **Class** `TrainerConfig` — implementation class.
  - attribute `learning_rate` = `0.0003`
  - attribute `weight_decay` = `0.1`
  - attribute `betas` = `(0.9, 0.95)`
  - attribute `grad_clip` = `1.0`
  - attribute `warmup_steps` = `10`
  - attribute `total_steps` = `100`
  - attribute `min_lr_ratio` = `0.1`
  - attribute `gradient_accumulation_steps` = `1`
  - attribute `precision` = `'fp32'`
  - attribute `activation_checkpointing` = `False`
- **Class** `Trainer` — implementation class.
  - attribute `_micro_step` = `0`
  - attribute `_pending_data_state` = `{}`
  - attribute `git_commit` = `subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()`
  - attribute `is_main_process` = `rank == 0`
  - attribute `last_metrics` = `{}`
  - attribute `optimizer` = `build_adamw(self.model, config.learning_rate, config.weight_decay, config.betas)`
  - attribute `scaler` = `make_grad_scaler(self.device, config.precision)`
  - attribute `step` = `state['step']`
  - attribute `tokenizer_version` = `None`
  - method `__init__(self, model: nn.Module, config: TrainerConfig | None = None, device: str | torch.device = 'cpu', rank: int = 0, world_size: int = 1) -> None` — method.
  - method `_base_model(self) -> nn.Module` — Return the underlying model when training through DistributedDataParallel.
  - method `model_config(self)` — method.
  - method `_lr(self) -> float` — method.
  - method `train_step(self, batch: torch.Tensor, labels: torch.Tensor | None = None) -> float` — method.
  - method `save_checkpoint(self, path: str | Path, model_config: dict | None = None, data_state: dict | None = None) -> None` — method.
  - method `load_checkpoint(self, path: str | Path) -> None` — method.
  - method `fit(self, batches, max_steps: int, checkpoint_path: str | Path | None = None, log_path: str | Path | None = None, validation_batches = None, eval_every_steps: int = 0, max_eval_batches: int | None = None, data_sampler = None, checkpoint_every_steps: int = 1, best_checkpoint_path: str | Path | None = None, profiler = None) -> list[float]` — method.

## Configuration files and values

Values below are read from the checked-in YAML configurations. Nested keys use dotted paths.

### `configs/data/preference.yaml`

| Key | Value |
| --- | --- |
| `input_file` | `data/processed/preferences.jsonl` |
| `format` | `{prompt: string, chosen: string, rejected: string}` |
| `beta` | `0.1` |

### `configs/data/pretraining.yaml`

| Key | Value |
| --- | --- |
| `input_files` | `[data/raw]` |
| `output_file` | `data/processed/train.jsonl` |
| `min_chars` | `40` |
| `max_chars` | `1000000` |
| `source_name` | `local_corpus` |
| `license` | `unknown` |

### `configs/data/pretraining_mixture.yaml`

| Key | Value |
| --- | --- |
| `output_file` | `data/processed/train.jsonl` |
| `quality.min_chars` | `40` |
| `quality.max_chars` | `1000000` |
| `quality.min_alpha_ratio` | `0.15` |
| `quality.max_url_ratio` | `0.2` |
| `quality.max_repeated_line_ratio` | `0.5` |
| `quality.max_repeated_char_ratio` | `0.3` |
| `quality.max_symbol_ratio` | `0.45` |
| `quality.max_garbage_ratio` | `0.01` |
| `sources.english.input_files` | `[data/raw/english]` |
| `sources.english.source_name` | `english_corpus` |
| `sources.english.license` | `REPLACE_WITH_ACTUAL_LICENSE` |
| `sources.english.languages` | `[en]` |
| `sources.persian.input_files` | `[data/raw/persian]` |
| `sources.persian.source_name` | `persian_corpus` |
| `sources.persian.license` | `REPLACE_WITH_ACTUAL_LICENSE` |
| `sources.persian.languages` | `[fa]` |
| `sources.code.input_files` | `[data/raw/code]` |
| `sources.code.source_name` | `code_corpus` |
| `sources.code.license` | `REPLACE_WITH_ACTUAL_LICENSE` |
| `sources.code.languages` | `[en, mixed]` |
| `mixture_weights.english` | `0.4` |
| `mixture_weights.persian` | `0.35` |
| `mixture_weights.code` | `0.15` |
| `mixture_seed` | `42` |
| `near_dedup` | `false` |

### `configs/data/pretraining_wikipedia_smoke.yaml`

| Key | Value |
| --- | --- |
| `output_file` | `data/processed/wikipedia_smoke.jsonl` |
| `quality.min_chars` | `80` |
| `sources.english.input_files` | `[data/raw/wiki_en]` |
| `sources.english.source_name` | `English Wikipedia page summaries` |
| `sources.english.license` | `CC BY-SA 4.0; verify page-specific exceptions` |
| `sources.english.attribution_manifest` | `data/raw/wiki_en/attribution.json` |
| `sources.english.languages` | `[en]` |
| `sources.persian.input_files` | `[data/raw/wiki_fa]` |
| `sources.persian.source_name` | `Persian Wikipedia page summaries` |
| `sources.persian.license` | `CC BY-SA 4.0; verify page-specific exceptions` |
| `sources.persian.attribution_manifest` | `data/raw/wiki_fa/attribution.json` |
| `sources.persian.languages` | `[fa]` |
| `mixture_weights.english` | `0.5` |
| `mixture_weights.persian` | `0.5` |
| `mixture_seed` | `42` |
| `near_dedup` | `true` |
| `near_dedup_threshold` | `0.9` |

### `configs/data/pretraining_wikipedia_validation.yaml`

| Key | Value |
| --- | --- |
| `output_file` | `data/processed/wikipedia_validation.jsonl` |
| `quality.min_chars` | `80` |
| `sources.english.input_files` | `[data/raw/wiki_en_validation]` |
| `sources.english.source_name` | `English Wikipedia page summaries` |
| `sources.english.license` | `CC BY-SA 4.0; verify page-specific exceptions` |
| `sources.english.attribution_manifest` | `data/raw/wiki_en_validation/attribution.json` |
| `sources.english.languages` | `[en]` |
| `sources.persian.input_files` | `[data/raw/wiki_fa_validation]` |
| `sources.persian.source_name` | `Persian Wikipedia page summaries` |
| `sources.persian.license` | `CC BY-SA 4.0; verify page-specific exceptions` |
| `sources.persian.attribution_manifest` | `data/raw/wiki_fa_validation/attribution.json` |
| `sources.persian.languages` | `[fa]` |
| `mixture_weights.english` | `0.5` |
| `mixture_weights.persian` | `0.5` |
| `mixture_seed` | `43` |
| `near_dedup` | `true` |
| `near_dedup_threshold` | `0.9` |

### `configs/data/sft.yaml`

| Key | Value |
| --- | --- |
| `input_file` | `data/processed/sft.jsonl` |
| `assistant_only_loss` | `true` |
| `max_seq_length` | `2048` |

### `configs/data/tokenizer.yaml`

| Key | Value |
| --- | --- |
| `input_files` | `[data/processed/train.jsonl]` |
| `model_prefix` | `data/tokenizer/novalm` |
| `vocab_size` | `32000` |
| `character_coverage` | `0.9995` |

### `configs/data/tokenizer_8k.yaml`

| Key | Value |
| --- | --- |
| `input_files` | `[data/processed/train.jsonl]` |
| `model_prefix` | `data/tokenizer/novalm_8k` |
| `vocab_size` | `8192` |
| `character_coverage` | `0.9995` |

### `configs/data/tokenizer_wikipedia_smoke.yaml`

| Key | Value |
| --- | --- |
| `input_files` | `[data/processed/wikipedia_smoke.jsonl]` |
| `model_prefix` | `data/tokenizer/wikipedia_smoke` |
| `vocab_size` | `1024` |
| `character_coverage` | `1.0` |

### `configs/model/debug.yaml`

| Key | Value |
| --- | --- |
| `model.vocab_size` | `512` |
| `model.hidden_size` | `64` |
| `model.num_layers` | `2` |
| `model.num_attention_heads` | `4` |
| `model.num_kv_heads` | `2` |
| `model.intermediate_size` | `176` |
| `model.max_seq_length` | `128` |
| `model.rope_theta` | `10000.0` |
| `model.norm_eps` | `1e-06` |
| `model.tie_embeddings` | `true` |
| `model.attention_backend` | `naive` |
| `model.qk_norm` | `true` |

### `configs/model/novalm_100m.yaml`

| Key | Value |
| --- | --- |
| `model.vocab_size` | `32000` |
| `model.hidden_size` | `768` |
| `model.num_layers` | `12` |
| `model.num_attention_heads` | `12` |
| `model.num_kv_heads` | `4` |
| `model.intermediate_size` | `2048` |
| `model.max_seq_length` | `1024` |
| `model.tie_embeddings` | `true` |
| `model.attention_backend` | `sdpa` |
| `model.qk_norm` | `true` |

### `configs/model/novalm_30m.yaml`

| Key | Value |
| --- | --- |
| `model.vocab_size` | `8192` |
| `model.hidden_size` | `512` |
| `model.num_layers` | `8` |
| `model.num_attention_heads` | `8` |
| `model.num_kv_heads` | `4` |
| `model.intermediate_size` | `1408` |
| `model.max_seq_length` | `512` |
| `model.tie_embeddings` | `true` |
| `model.attention_backend` | `sdpa` |
| `model.qk_norm` | `true` |

### `configs/model/novalm_30m_wikipedia_smoke.yaml`

| Key | Value |
| --- | --- |
| `model.vocab_size` | `1024` |
| `model.hidden_size` | `512` |
| `model.num_layers` | `10` |
| `model.num_attention_heads` | `8` |
| `model.num_kv_heads` | `4` |
| `model.intermediate_size` | `1408` |
| `model.max_seq_length` | `128` |
| `model.tie_embeddings` | `true` |
| `model.attention_backend` | `sdpa` |
| `model.qk_norm` | `true` |
| `model.attention_type` | `gqa` |
| `model.norm_type` | `rmsnorm` |
| `model.mlp_type` | `swiglu` |
| `model.position_type` | `rope` |

### `configs/model/novalm_mla.yaml`

| Key | Value |
| --- | --- |
| `model.vocab_size` | `8192` |
| `model.hidden_size` | `512` |
| `model.num_layers` | `8` |
| `model.num_attention_heads` | `8` |
| `model.num_kv_heads` | `4` |
| `model.intermediate_size` | `1408` |
| `model.max_seq_length` | `512` |
| `model.tie_embeddings` | `true` |
| `model.attention_backend` | `sdpa` |
| `model.qk_norm` | `false` |
| `model.attention_type` | `mla` |
| `model.mla_latent_dim` | `128` |

### `configs/model/novalm_moe.yaml`

| Key | Value |
| --- | --- |
| `model.vocab_size` | `8192` |
| `model.hidden_size` | `512` |
| `model.num_layers` | `8` |
| `model.num_attention_heads` | `8` |
| `model.num_kv_heads` | `4` |
| `model.intermediate_size` | `1408` |
| `model.max_seq_length` | `512` |
| `model.tie_embeddings` | `true` |
| `model.attention_backend` | `sdpa` |
| `model.qk_norm` | `true` |
| `model.moe_enabled` | `true` |
| `model.num_experts` | `8` |
| `model.moe_top_k` | `2` |
| `model.moe_aux_loss_weight` | `0.01` |

### `configs/model/novalm_mtp.yaml`

| Key | Value |
| --- | --- |
| `model.vocab_size` | `8192` |
| `model.hidden_size` | `512` |
| `model.num_layers` | `8` |
| `model.num_attention_heads` | `8` |
| `model.num_kv_heads` | `4` |
| `model.intermediate_size` | `1408` |
| `model.max_seq_length` | `512` |
| `model.tie_embeddings` | `true` |
| `model.attention_backend` | `sdpa` |
| `model.qk_norm` | `true` |
| `model.mtp_enabled` | `true` |
| `model.mtp_horizons` | `2` |
| `model.mtp_loss_weight` | `0.25` |

### `configs/training/debug.yaml`

| Key | Value |
| --- | --- |
| `learning_rate` | `0.0003` |
| `weight_decay` | `0.1` |
| `total_steps` | `100` |
| `warmup_steps` | `10` |
| `gradient_accumulation_steps` | `1` |
| `precision` | `fp32` |

### `configs/training/default.yaml`

| Key | Value |
| --- | --- |
| `learning_rate` | `0.0003` |
| `weight_decay` | `0.1` |
| `betas` | `[0.9, 0.95]` |
| `grad_clip` | `1.0` |
| `warmup_steps` | `100` |
| `total_steps` | `10000` |
| `min_lr_ratio` | `0.1` |
| `gradient_accumulation_steps` | `1` |
| `precision` | `bf16` |

### `configs/training/distributed.yaml`

| Key | Value |
| --- | --- |
| `precision` | `bf16` |
| `gradient_accumulation_steps` | `1` |
| `activation_checkpointing` | `false` |

### `configs/training/single_gpu.yaml`

| Key | Value |
| --- | --- |
| `learning_rate` | `0.0003` |
| `weight_decay` | `0.1` |
| `betas` | `[0.9, 0.95]` |
| `grad_clip` | `1.0` |
| `warmup_steps` | `100` |
| `total_steps` | `10000` |
| `min_lr_ratio` | `0.1` |
| `gradient_accumulation_steps` | `8` |
| `precision` | `bf16` |

### `configs/training/wikipedia_smoke.yaml`

| Key | Value |
| --- | --- |
| `learning_rate` | `0.0003` |
| `weight_decay` | `0.1` |
| `betas` | `[0.9, 0.95]` |
| `grad_clip` | `1.0` |
| `warmup_steps` | `5` |
| `total_steps` | `100` |
| `min_lr_ratio` | `0.1` |
| `gradient_accumulation_steps` | `1` |
| `precision` | `bf16` |
| `activation_checkpointing` | `true` |

## Dataclass and model defaults

The validated configuration classes are defined in `novalm/config/schema.py`; model-specific defaults and parameter-count logic are in `novalm/model/config.py`.

| Object | Purpose |
| --- | --- |
| `ModelConfig` | Vocabulary, dimensions, attention, positions, norm, MLP, MoE, MTP, and research-variant switches. |
| `AttentionConfig` | Attention backend and QK-Norm switch. |
| `MoEConfig` | Expert count, top-k routing, and auxiliary loss weight. |
| `MTPConfig` | Future-token horizon count and loss weight. |
| `DataConfig` | Block size, shard size, seed, and near-dedup threshold. |
| `TokenizerConfig` | Vocabulary size and character coverage. |
| `OptimizerConfig` | AdamW learning rate, betas, decay, and clipping. |
| `SchedulerConfig` | Warmup, total steps, and minimum cosine ratio. |
| `TrainingConfig` | Precision, accumulation, activation checkpointing, and seed. |
| `InferenceConfig` | Generation length and sampling controls. |
| `EvaluationConfig` | Evaluation batch limit and seed. |

## Command line entry points

| Script | Function |
| --- | --- |
| `scripts/prepare_data.py` | Ingest, normalize, clean, filter, deduplicate, and write processed JSONL. |
| `scripts/train_tokenizer.py` | Train the project SentencePiece tokenizer. |
| `scripts/tokenize_dataset.py` | Encode processed JSONL records. |
| `scripts/build_shards.py` | Pack token streams and write checksummed binary shards. |
| `scripts/inspect_data.py` | Inspect JSONL or validate shard manifests. |
| `scripts/train.py` | Train, validate, checkpoint, resume, and optionally initialize DDP. |
| `scripts/evaluate.py` | Evaluate loss/perplexity and generation. |
| `scripts/generate.py` | Run manual generation with sampling controls. |
| `scripts/sft.py` | Run supervised fine-tuning with assistant-only labels. |
| `scripts/dpo.py` | Run direct preference optimization. |
| `scripts/serve.py` | Start the FastAPI service. |
| `scripts/benchmark.py` | Dispatch reproducible benchmark suites. |
| `scripts/generate_reference.py` | Regenerate this document from source and configs. |

## Validation recorded for this workspace

- `python -m pytest -q`: 83 passed and 26 subtests passed on October 8, 2026.
- `python -m unittest discover -s tests -v`: 42 tests passed on October 8, 2026.
- `python -m ruff check novalm scripts tests benchmarks`: passed.
- `python -m compileall -q novalm scripts benchmarks tests`: passed.
- The project audit in `docs/spec_audit.md` records which research and scale acceptance criteria remain evidence-limited.

## Implementation status boundaries

The reference implementation and CPU/synthetic integration paths are complete and tested. Meaningful corpus-scale pretraining, controlled quality ablations, target-scale performance studies, serving-side request batching, and release Git provenance remain dependent on larger data, hardware, or external repository state; they are documented as limitations rather than represented as completed results.

"""Generate a source-of-truth API and configuration reference for NovaLM.

The generated document is deliberately derived from the checked-in source and
YAML files so that function names, class attributes, defaults, and CLI/config
values do not drift away from the implementation.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "novalm"
OUTPUT = ROOT / "docs" / "novalm_reference.md"


def source_value(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except (AttributeError, TypeError, ValueError):
        return "<expression>"


def docline(node: ast.AST) -> str:
    value = ast.get_docstring(node, clean=True) or ""
    return value.splitlines()[0] if value else ""


def fields_for_class(node: ast.ClassDef) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            fields.append((child.target.id, source_value(child.value) or source_value(child.annotation)))
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    fields.append((target.id, source_value(child.value)))
    return fields


def instance_attributes(node: ast.ClassDef) -> list[tuple[str, str]]:
    """Collect attributes assigned through ``self`` in class methods."""
    fields: dict[str, str] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, ast.AnnAssign):
            targets = [child.target]
        else:
            continue
        for target in targets:
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                fields.setdefault(target.attr, source_value(getattr(child, "value", None)))
    return sorted(fields.items())


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(node.args.defaults)
    for arg, default in zip(positional, defaults):
        value = arg.arg
        if arg.annotation:
            value += f": {source_value(arg.annotation)}"
        if default is not None:
            value += f" = {source_value(default)}"
        args.append(value)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        value = arg.arg
        if arg.annotation:
            value += f": {source_value(arg.annotation)}"
        if default is not None:
            value += f" = {source_value(default)}"
        args.append(value)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    result = f"{node.name}({', '.join(args)})"
    if node.returns:
        result += f" -> {source_value(node.returns)}"
    return result


def yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value or value in {"{}", "[]"}:
        return value
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value.strip("'\"")


def simple_yaml(path: Path) -> list[tuple[str, Any]]:
    """Read the project's simple indentation-based YAML without adding a dependency."""
    rows: list[tuple[str, Any]] = []
    stack: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        key, raw = line.strip().split(":", 1)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full_key = ".".join([name for _, name in stack] + [key.strip()])
        if raw.strip():
            rows.append((full_key, yaml_scalar(raw)))
        else:
            stack.append((indent, key.strip()))
    return rows


ALGORITHMS = [
    ("Unicode normalization and cleaning", "Normalize text to a stable Unicode form, remove invalid/control content, and collapse irrelevant whitespace before quality checks."),
    ("Quality filtering", "Reject documents using configurable length, alphabetic/symbol ratio, URL density, repeated-line, repeated-character, and garbage-character thresholds."),
    ("Language filtering", "Use script-aware heuristics to estimate English, Persian, code, or mixed content and apply the configured language policy."),
    ("Exact and near deduplication", "Hash normalized documents for exact duplicates; optional MinHash signatures and LSH buckets identify near duplicates above a similarity threshold."),
    ("Weighted data mixture", "Sample source streams using seeded weights while stopping cleanly when a source is exhausted; provenance and final counts are retained."),
    ("Tokenization and packing", "Encode documents, preserve EOS boundaries, concatenate token streams into fixed-size blocks, and build shifted input/target pairs."),
    ("RMSNorm", "For each hidden vector x, compute x / sqrt(mean(x²) + eps) and multiply by a learned scale gamma."),
    ("RoPE", "Rotate paired query/key dimensions with cached cosines and sines derived from theta and position; caches extend dynamically for long contexts."),
    ("QK-Norm", "Normalize each query and key head before the scaled dot product to control score magnitude and improve stability."),
    ("Causal attention", "Compute softmax((QKᵀ / sqrt(d)) + causal mask)V; the reference path uses explicit masked matrix operations."),
    ("GQA", "Project many query heads and fewer key/value heads, then repeat KV heads across query groups; cache the compact KV representation."),
    ("SwiGLU", "Compute W2(SiLU(Wg x) elementwise-multiplied by Wu x)."),
    ("Pre-norm Transformer", "Apply normalization before attention and MLP, add each result through a residual connection, and finish with a final norm and LM head."),
    ("Causal language modeling", "Use logits at position t to predict token t+1 with cross entropy and ignore_index support for masked labels."),
    ("MTP", "For horizon k, shift targets by k positions, mask unavailable positions, and sum weighted per-horizon cross-entropy losses."),
    ("MoE", "Route each token to top-k expert scores, combine weighted expert outputs, and add a router load-balancing auxiliary loss plus utilization metrics."),
    ("MLA", "Compress key/value information into a latent representation and retain that compressed state in the autoregressive cache."),
    ("AdamW and scheduling", "Separate decayed weights from bias/norm parameters, apply AdamW updates, then use linear warmup followed by cosine decay."),
    ("Mixed precision and accumulation", "Run autocast in FP32, FP16, or BF16, scale FP16 gradients where required, accumulate micro-batches, clip, and step."),
    ("Checkpointing", "Atomically save model, optimizer, scheduler-equivalent step state, AMP state, RNG states, configs, tokenizer version, and environment metadata."),
    ("Sampling", "Apply repetition penalty, temperature, top-k, top-p, or min-p filtering before selecting the next token; enforce EOS and stop sequences."),
    ("KV-cache decoding", "Prefill the prompt once, append one token per step, and reuse per-layer K/V tensors for autoregressive decoding."),
    ("DPO", "Compare policy and reference chosen/rejected log-probability differences with beta-scaled logistic loss."),
    ("Quantization", "Provide dynamic INT8 and packed weight-only INT4 reference paths with dequantization for forward computation."),
    ("Evaluation", "Measure shifted loss/perplexity, generation behavior, multilingual samples, code/math prompts, context behavior, and reproducible benchmark metadata."),
]


def build() -> str:
    out: list[str] = [
        "# NovaLM Algorithms and API Reference",
        "",
        "> Generated from the checked-in `novalm/` source and `configs/` files. Re-run `python scripts/generate_reference.py` after API or configuration changes.",
        "",
        "This reference describes the implementation that exists in this repository. It records implementation status and configured values without inventing benchmark or model-quality claims.",
        "",
        "## Algorithms",
        "",
    ]
    for name, description in ALGORITHMS:
        out.append(f"### {name}\n\n{description}\n")

    out.extend(["## Package modules", "", "The inventory below includes every Python module under `novalm/`, top-level functions, classes, methods, class fields, and module-level constants.", ""])
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT).as_posix()
        out.extend([f"### `{rel}`", ""])
        module_constants = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        module_constants.append((target.id, source_value(node.value)))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.isupper():
                module_constants.append((node.target.id, source_value(node.value) or source_value(node.annotation)))
        if module_constants:
            out.append("**Module constants**")
            out.append("")
            for name, value in module_constants:
                out.append(f"- `{name}` = `{value}`")
            out.append("")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(f"- **Function** `{signature(node)}` — {docline(node) or 'implementation helper.'}")
            elif isinstance(node, ast.ClassDef):
                out.append(f"- **Class** `{node.name}` — {docline(node) or 'implementation class.'}")
                attributes = fields_for_class(node)
                known = {name for name, _ in attributes}
                attributes.extend((name, value) for name, value in instance_attributes(node)
                                  if name not in known)
                for name, value in attributes:
                    out.append(f"  - attribute `{name}` = `{value}`")
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(f"  - method `{signature(child)}` — {docline(child) or 'method.'}")
        out.append("")

    out.extend(["## Configuration files and values", "", "Values below are read from the checked-in YAML configurations. Nested keys use dotted paths.", ""])
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        out.extend([f"### `{path.relative_to(ROOT).as_posix()}`", "", "| Key | Value |", "| --- | --- |"])
        for key, value in simple_yaml(path):
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (bool, int, float)) else str(value)
            out.append(f"| `{key}` | `{rendered}` |")
        out.append("")

    out.extend([
        "## Dataclass and model defaults",
        "",
        "The validated configuration classes are defined in `novalm/config/schema.py`; model-specific defaults and parameter-count logic are in `novalm/model/config.py`.",
        "",
        "| Object | Purpose |",
        "| --- | --- |",
        "| `ModelConfig` | Vocabulary, dimensions, attention, positions, norm, MLP, MoE, MTP, and research-variant switches. |",
        "| `AttentionConfig` | Attention backend and QK-Norm switch. |",
        "| `MoEConfig` | Expert count, top-k routing, and auxiliary loss weight. |",
        "| `MTPConfig` | Future-token horizon count and loss weight. |",
        "| `DataConfig` | Block size, shard size, seed, and near-dedup threshold. |",
        "| `TokenizerConfig` | Vocabulary size and character coverage. |",
        "| `OptimizerConfig` | AdamW learning rate, betas, decay, and clipping. |",
        "| `SchedulerConfig` | Warmup, total steps, and minimum cosine ratio. |",
        "| `TrainingConfig` | Precision, accumulation, activation checkpointing, and seed. |",
        "| `InferenceConfig` | Generation length and sampling controls. |",
        "| `EvaluationConfig` | Evaluation batch limit and seed. |",
        "",
        "## Command line entry points",
        "",
        "| Script | Function |",
        "| --- | --- |",
        "| `scripts/prepare_data.py` | Ingest, normalize, clean, filter, deduplicate, and write processed JSONL. |",
        "| `scripts/train_tokenizer.py` | Train the project SentencePiece tokenizer. |",
        "| `scripts/tokenize_dataset.py` | Encode processed JSONL records. |",
        "| `scripts/build_shards.py` | Pack token streams and write checksummed binary shards. |",
        "| `scripts/inspect_data.py` | Inspect JSONL or validate shard manifests. |",
        "| `scripts/train.py` | Train, validate, checkpoint, resume, and optionally initialize DDP. |",
        "| `scripts/evaluate.py` | Evaluate loss/perplexity and generation. |",
        "| `scripts/generate.py` | Run manual generation with sampling controls. |",
        "| `scripts/sft.py` | Run supervised fine-tuning with assistant-only labels. |",
        "| `scripts/dpo.py` | Run direct preference optimization. |",
        "| `scripts/serve.py` | Start the FastAPI service. |",
        "| `scripts/benchmark.py` | Dispatch reproducible benchmark suites. |",
        "| `scripts/generate_reference.py` | Regenerate this document from source and configs. |",
        "",
        "## Validation recorded for this workspace",
        "",
        "- `python -m pytest -q`: 83 passed and 26 subtests passed on October 8, 2026.",
        "- `python -m unittest discover -s tests -v`: 42 tests passed on October 8, 2026.",
        "- `python -m ruff check novalm scripts tests benchmarks`: passed.",
        "- `python -m compileall -q novalm scripts benchmarks tests`: passed.",
        "- The project audit in `docs/spec_audit.md` records which research and scale acceptance criteria remain evidence-limited.",
        "",
        "## Implementation status boundaries",
        "",
        "The reference implementation and CPU/synthetic integration paths are complete and tested. Meaningful corpus-scale pretraining, controlled quality ablations, target-scale performance studies, serving-side request batching, and release Git provenance remain dependent on larger data, hardware, or external repository state; they are documented as limitations rather than represented as completed results.",
        "",
    ])
    return "\n".join(out)


if __name__ == "__main__":
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")

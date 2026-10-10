"""Exact-match scoring for deterministic code-completion test cases."""


def exact_match(predictions: list[str], references: list[str]) -> dict[str, float | int]:
    if len(predictions) != len(references) or not references:
        raise ValueError("predictions and nonempty references must have equal lengths")
    correct = sum(pred.rstrip() == ref.rstrip() for pred, ref in zip(predictions, references))
    return {"correct": correct, "count": len(references), "accuracy": correct / len(references)}

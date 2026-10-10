"""Exact-match scoring for answer-keyed math evaluation examples."""


def exact_match(predictions: list[str], references: list[str]) -> dict[str, float | int]:
    if len(predictions) != len(references) or not references:
        raise ValueError("predictions and nonempty references must have equal lengths")
    correct = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
    return {"correct": correct, "count": len(references), "accuracy": correct / len(references)}

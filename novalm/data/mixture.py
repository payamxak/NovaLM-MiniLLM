import random
from collections.abc import Iterable, Iterator


def sample_mixture(sources: dict[str, list[dict]], weights: dict[str, float],
                   count: int, seed: int = 0) -> Iterator[dict]:
    """Draw a reproducible weighted document stream, with replacement per source."""
    if count < 0 or not weights or any(value < 0 for value in weights.values()):
        raise ValueError("count and source weights must be nonnegative")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one mixture weight must be positive")
    names = [name for name, weight in weights.items() if weight > 0]
    if any(not sources.get(name) for name in names):
        raise ValueError("every positively weighted source needs documents")
    rng = random.Random(seed)
    values = [weights[name] for name in names]
    for _ in range(count):
        name = rng.choices(names, weights=values, k=1)[0]
        row = dict(rng.choice(sources[name]))
        row["mixture_source"] = name
        yield row


def weighted_stream(sources: dict[str, Iterable[dict]], weights: dict[str, float],
                    seed: int = 0) -> Iterator[dict]:
    """Stream all source records in weighted random order without loading corpora in memory."""
    if not weights or any(weight < 0 for weight in weights.values()):
        raise ValueError("source weights must be nonnegative and nonempty")
    active = {name: iter(sources[name]) for name, weight in weights.items()
              if weight > 0 and name in sources}
    if not active or any(name not in sources for name, weight in weights.items() if weight > 0):
        raise ValueError("every positively weighted source needs an iterable")
    rng = random.Random(seed)
    while active:
        names = list(active)
        values = [weights[name] for name in names]
        name = rng.choices(names, weights=values, k=1)[0]
        try:
            record = next(active[name])
        except StopIteration:
            del active[name]
            continue
        row = dict(record)
        row["mixture_source"] = name
        yield row

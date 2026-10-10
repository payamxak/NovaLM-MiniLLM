"""Training-loop entry point for library callers."""


def train_steps(trainer, batches, steps: int, **fit_options):
    if steps < 1:
        raise ValueError("steps must be positive")
    return trainer.fit(batches, max_steps=steps, **fit_options)

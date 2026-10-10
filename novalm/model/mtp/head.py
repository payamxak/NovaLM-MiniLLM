from torch import nn


class MultiTokenPredictionHead(nn.Module):
    """Auxiliary vocabulary heads for horizons 2..K, beside the normal LM head."""
    def __init__(self, hidden_size: int, vocab_size: int, horizons: int) -> None:
        super().__init__()
        if horizons < 1:
            raise ValueError("horizons must be >= 1")
        self.horizons = horizons
        self.heads = nn.ModuleList(nn.Linear(hidden_size, vocab_size, bias=False)
                                   for _ in range(horizons))

    def forward(self, hidden):
        return [head(hidden) for head in self.heads]

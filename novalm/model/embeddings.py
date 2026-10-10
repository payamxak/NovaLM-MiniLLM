"""Token embedding layer kept separate from the decoder stack."""
from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        if vocab_size <= 0 or hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        super().__init__(vocab_size, hidden_size)

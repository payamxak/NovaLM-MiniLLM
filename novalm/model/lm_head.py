"""Language-model output projection with optional true parameter tying."""
from torch import nn


class LMHead(nn.Linear):
    def __init__(self, hidden_size: int, vocab_size: int,
                 embedding: nn.Embedding | None = None) -> None:
        super().__init__(hidden_size, vocab_size, bias=False)
        if embedding is not None:
            if embedding.embedding_dim != hidden_size or embedding.num_embeddings != vocab_size:
                raise ValueError("embedding shape must match LM head dimensions")
            self.weight = embedding.weight

"""Random sampler whose permutation and cursor can be checkpointed exactly."""
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized, seed: int = 0) -> None:
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.permutation: torch.Tensor | None = None
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        if self.permutation is None or self.cursor >= len(self.permutation):
            self.permutation = torch.randperm(len(self), generator=self.generator)
            self.cursor = 0
        while self.cursor < len(self.permutation):
            index = int(self.permutation[self.cursor])
            self.cursor += 1
            yield index

    def state_dict(self) -> dict:
        return {
            "size": len(self),
            "permutation": None if self.permutation is None else self.permutation.clone(),
            "cursor": self.cursor,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("size") != len(self):
            raise ValueError("sampler checkpoint dataset size does not match current dataset")
        permutation = state.get("permutation")
        if permutation is not None:
            permutation = torch.as_tensor(permutation, dtype=torch.int64, device="cpu")
            if (permutation.numel() != len(self)
                    or not torch.equal(torch.sort(permutation).values,
                                       torch.arange(len(self), dtype=torch.int64))):
                raise ValueError("sampler checkpoint permutation is invalid")
        cursor = int(state.get("cursor", 0))
        if cursor < 0 or (permutation is not None and cursor > len(permutation)):
            raise ValueError("sampler checkpoint cursor is invalid")
        self.permutation = permutation
        self.cursor = cursor
        self.generator.set_state(state["generator_state"].cpu())


class DistributedResumableSampler(Sampler[int]):
    """Deterministic rank-sharded sampler with resumable local cursors.

    A single global permutation is generated from ``seed`` and every rank takes
    its strided slice.  This keeps the ordering reproducible across processes
    while ensuring that each example is consumed by exactly one rank per epoch.
    """

    def __init__(self, data_source: Sized, num_replicas: int, rank: int,
                 seed: int = 0) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("num_replicas must be positive and rank in range")
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.permutation: torch.Tensor | None = None
        self.local_permutation: torch.Tensor | None = None
        self.cursor = 0

    def __len__(self) -> int:
        size = len(self.data_source)
        return (size + self.num_replicas - 1 - self.rank) // self.num_replicas

    def _ensure_permutation(self) -> None:
        if self.permutation is None or self.local_permutation is None or self.cursor >= len(self.local_permutation):
            self.permutation = torch.randperm(len(self.data_source), generator=self.generator)
            self.local_permutation = self.permutation[self.rank::self.num_replicas].clone()
            self.cursor = 0

    def __iter__(self) -> Iterator[int]:
        self._ensure_permutation()
        assert self.local_permutation is not None
        while self.cursor < len(self.local_permutation):
            index = int(self.local_permutation[self.cursor])
            self.cursor += 1
            yield index

    def state_dict(self) -> dict:
        return {
            "size": len(self.data_source),
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "permutation": None if self.permutation is None else self.permutation.clone(),
            "local_permutation": (None if self.local_permutation is None
                                   else self.local_permutation.clone()),
            "cursor": self.cursor,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("size") != len(self.data_source):
            raise ValueError("sampler checkpoint dataset size does not match current dataset")
        if state.get("num_replicas", self.num_replicas) != self.num_replicas:
            raise ValueError("sampler checkpoint world size does not match current run")
        if state.get("rank", self.rank) != self.rank:
            raise ValueError("sampler checkpoint rank does not match current run")
        permutation = state.get("permutation")
        local = state.get("local_permutation")
        if permutation is not None:
            permutation = torch.as_tensor(permutation, dtype=torch.int64, device="cpu")
            expected = torch.arange(len(self.data_source), dtype=torch.int64)
            if permutation.numel() != len(expected) or not torch.equal(torch.sort(permutation).values, expected):
                raise ValueError("sampler checkpoint permutation is invalid")
        if local is not None:
            local = torch.as_tensor(local, dtype=torch.int64, device="cpu")
            if permutation is None or not torch.equal(local, permutation[self.rank::self.num_replicas]):
                raise ValueError("sampler checkpoint local permutation is invalid")
        cursor = int(state.get("cursor", 0))
        if cursor < 0 or (local is not None and cursor > len(local)):
            raise ValueError("sampler checkpoint cursor is invalid")
        self.permutation = permutation
        self.local_permutation = local
        self.cursor = cursor
        self.generator.set_state(torch.as_tensor(state["generator_state"], dtype=torch.uint8).cpu())

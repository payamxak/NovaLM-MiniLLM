import json
from pathlib import Path

from torch.utils.data import Dataset

from novalm.tokenizer.tokenizer import NovaTokenizer


def assistant_only_example(messages: list[dict[str, str]], tokenizer: NovaTokenizer,
                           max_length: int | None = None) -> tuple[list[int], list[int]]:
    """Format messages and mask loss outside assistant bodies using tokenizer offsets."""
    ids: list[int] = []
    labels: list[int] = []
    for msg in messages:
        role_id = tokenizer.token_id(f"<|{msg['role']}|>")
        prefix = tokenizer.encode("\n")
        body = tokenizer.encode(msg["content"] + "\n")
        segment = [role_id, *prefix, *body]
        ids.extend(segment)
        labels.extend(([-100] * (1 + len(prefix))) +
                      (body if msg["role"] == "assistant" else [-100] * len(body)))
    if max_length is not None:
        # Keep the latest conversation turn and its assistant targets when context is long.
        ids, labels = ids[-max_length:], labels[-max_length:]
    return ids, labels


class SFTDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: NovaTokenizer,
                 max_length: int) -> None:
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        messages = self.rows[index]["messages"]
        ids, labels = assistant_only_example(messages, self.tokenizer, self.max_length)
        import torch
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def sft_collate(batch, pad_id: int = 0):
    import torch
    max_length = max(ids.numel() for ids, _ in batch)
    inputs, labels = [], []
    for ids, target in batch:
        pad = max_length - ids.numel()
        inputs.append(torch.nn.functional.pad(ids, (0, pad), value=pad_id))
        labels.append(torch.nn.functional.pad(target, (0, pad), value=-100))
    return torch.stack(inputs), torch.stack(labels)

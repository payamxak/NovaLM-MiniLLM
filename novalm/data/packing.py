from collections.abc import Iterable, Iterator


def pack_sequences(sequences: Iterable[list[int]], block_size: int, eos_id: int) -> Iterator[list[int]]:
    """Concatenate documents separated by EOS and emit full blocks; drop only final tail."""
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    buffer: list[int] = []
    for seq in sequences:
        if not seq:
            continue
        buffer.extend(seq)
        if seq[-1] != eos_id:
            buffer.append(eos_id)
        while len(buffer) >= block_size:
            yield buffer[:block_size]
            buffer = buffer[block_size:]

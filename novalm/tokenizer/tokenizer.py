from pathlib import Path

import sentencepiece as spm

from novalm.tokenizer.special_tokens import SPECIAL_TOKENS


class NovaTokenizer:
    """SentencePiece BPE tokenizer trained on project data, with fixed chat tokens."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @classmethod
    def train(cls, input_files: list[str | Path], model_prefix: str | Path,
              vocab_size: int = 32000, character_coverage: float = 0.9995) -> "NovaTokenizer":
        prefix = Path(model_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=[str(p) for p in input_files], model_prefix=str(prefix),
            model_type="bpe", vocab_size=vocab_size, character_coverage=character_coverage,
            pad_id=0, bos_id=1, eos_id=2, unk_id=3,
            pad_piece=SPECIAL_TOKENS[0], bos_piece=SPECIAL_TOKENS[1],
            eos_piece=SPECIAL_TOKENS[2], unk_piece=SPECIAL_TOKENS[3],
            user_defined_symbols=list(SPECIAL_TOKENS[4:]), hard_vocab_limit=False,
            minloglevel=2,
        )
        return cls(prefix.with_suffix(".model"))

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.processor.encode(text, out_type=int)
        return ([self.processor.bos_id()] if add_bos else []) + ids + (
            [self.processor.eos_id()] if add_eos else [])

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)

    def token_id(self, token: str) -> int:
        return self.processor.piece_to_id(token)

    def fertility(self, text: str) -> float:
        words = text.split()
        return len(self.encode(text)) / max(1, len(words))

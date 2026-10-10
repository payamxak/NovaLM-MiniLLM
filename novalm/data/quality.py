import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityConfig:
    min_chars: int = 40
    max_chars: int = 1_000_000
    min_alpha_ratio: float = 0.15
    max_url_ratio: float = 0.2
    max_repeated_line_ratio: float = 0.5
    max_repeated_char_ratio: float = 0.3
    max_symbol_ratio: float = 0.45
    max_garbage_ratio: float = 0.01

    def __post_init__(self) -> None:
        if self.min_chars < 0 or self.max_chars < self.min_chars:
            raise ValueError("length bounds must satisfy 0 <= min_chars <= max_chars")
        ratios = (self.min_alpha_ratio, self.max_url_ratio, self.max_repeated_line_ratio,
                  self.max_repeated_char_ratio, self.max_symbol_ratio, self.max_garbage_ratio)
        if any(not 0 <= value <= 1 for value in ratios):
            raise ValueError("quality ratios must be in [0, 1]")


def quality_rejection(text: str, config: QualityConfig | None = None) -> str | None:
    if config is None:
        config = QualityConfig()
    if not text.strip():
        return "empty"
    if not config.min_chars <= len(text) <= config.max_chars:
        return "length"
    if sum(ch.isalpha() for ch in text) / len(text) < config.min_alpha_ratio:
        return "low_alphabetic_ratio"
    if sum(not (ch.isalpha() or ch.isdigit() or ch.isspace()) for ch in text) / len(text) > config.max_symbol_ratio:
        return "symbol_density"
    if sum(ord(ch) < 32 and ch not in "\n\t" or ord(ch) == 0xFFFD for ch in text) / len(text) > config.max_garbage_ratio:
        return "garbage_characters"
    urls = len(re.findall(r"https?://|www\.", text, flags=re.IGNORECASE))
    if urls / max(1, len(text.split())) > config.max_url_ratio:
        return "url_density"
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and 1 - len(set(lines)) / len(lines) > config.max_repeated_line_ratio:
        return "repeated_lines"
    if len(text) and max((text.count(c) for c in set(text)), default=0) / len(text) > config.max_repeated_char_ratio:
        return "repeated_character"
    return None

from collections import Counter


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    persian = sum("\u0600" <= c <= "\u06ff" for c in letters)
    ratio = persian / len(letters)
    if ratio >= 0.7:
        return "fa"
    if ratio <= 0.1:
        return "en"
    return "mixed"


def language_ratio(text: str) -> dict[str, float]:
    counts = Counter("fa" if "\u0600" <= c <= "\u06ff" else "other"
                     for c in text if c.isalpha())
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}

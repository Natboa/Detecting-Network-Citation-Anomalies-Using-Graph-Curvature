"""Rebuild abstract text from OpenAlex abstract_inverted_index (task S1.1)."""


def invert_abstract(inv_idx: dict[str, list[int]] | None) -> str | None:
    if not inv_idx:
        return None
    words_at = {}
    for word, positions in inv_idx.items():
        for i in positions:
            words_at[i] = word
    if not words_at:
        return None
    return " ".join(words_at[i] for i in sorted(words_at))

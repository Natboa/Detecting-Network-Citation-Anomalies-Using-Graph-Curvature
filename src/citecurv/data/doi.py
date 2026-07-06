"""DOI normalization (task S1.1).

Rules: lowercase; strip resolver prefixes and the doi: scheme; strip surrounding
whitespace and trailing punctuation; anything not starting with "10." is not a
DOI and returns None. Version suffixes (".v2") are kept — loose version-blind
matching is a labels-side join decision (task S1.2), not a normalization rule.
"""

_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def normalize_doi(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    for prefix in _PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.strip().rstrip(".,;")
    if not s.startswith("10."):
        return None
    return s

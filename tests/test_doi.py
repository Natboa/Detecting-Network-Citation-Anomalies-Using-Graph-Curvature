"""S1.1: DOI normalization rules (IMPLEMENTATION_PLAN.md §5, task S1.1)."""

from citecurv.data.doi import normalize_doi

CASES = [
    ("HTTPS://DOI.ORG/10.1234/AbC.5", "10.1234/abc.5"),   # prefix + case
    ("doi:10.1/x", "10.1/x"),                             # doi: prefix
    ("https://dx.doi.org/10.99/Y", "10.99/y"),            # dx variant
    ("  10.5555/abc; ", "10.5555/abc"),                   # whitespace + trailing punctuation
    ("10.1/x.v2", "10.1/x.v2"),                           # version suffix survives (loose join is labels-side)
    ("", None),
    (None, None),
    ("not-a-doi", None),                                  # no 10. prefix
]


def test_normalize_doi_cases():
    for raw, expected in CASES:
        assert normalize_doi(raw) == expected, f"input: {raw!r}"

"""Retraction Watch snapshot: loading, reason parsing, family counts (task S0.6).

Snapshot source is the Crossref GitLab distribution (the old Crossref Labs
endpoint is dead). The CSV refreshes every working day ~23:00 UTC; the project
freezes one dated copy and records its SHA256 in rwdb_counts.json. License
wording rule: the data is "publicly available via Crossref" — never CC0.

Reason values are semicolon-joined, each prefixed with '+'. The primary
positive-label family (decision L12) is any reason containing
"Referencing/Attributions"; the broadened sensitivity vocabulary is recorded
by summarize() so the exact broadened list is chosen from real literals.
"""

from pathlib import Path

DEFAULT_URL = (
    "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv"
)
REQUIRED_COLUMNS = {
    "RetractionDOI", "OriginalPaperDOI", "Reason", "RetractionDate", "RetractionNature",
}
REF_ATTR_SUBSTRING = "referencing/attributions"
# Candidate vocabulary for the broadened sensitivity family (final pick at S4.6/G3).
# "peer review" deliberately wide: the live vocabulary has no "Fake Peer Review"
# literal — the real ones are Concerns/Issues about / Compromised / Taken via.
BROADENED_SUBSTRINGS = (
    "paper mill", "peer review", "reference", "attribution", "citation",
)


def parse_reasons(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and raw != raw):  # None or NaN
        return []
    parts = (p.strip().lstrip("+").strip() for p in str(raw).split(";"))
    return [p for p in parts if p]


def _read_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8", encoding_errors="replace")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"unexpected RWDB schema, missing columns: {sorted(missing)}")
    return df


def load_labels(csv_path, families=None):
    """One row per (paper, reason): doi_norm, reason, retraction_date, retraction_nature.

    families: optional iterable of lowercase substrings; keeps rows whose
    reason contains any of them.
    """
    from citecurv.data.doi import normalize_doi

    df = _read_csv(csv_path)
    out = (
        df.assign(
            doi_norm=df["OriginalPaperDOI"].map(normalize_doi),
            reason=df["Reason"].map(parse_reasons),
        )[["doi_norm", "reason", "RetractionDate", "RetractionNature"]]
        .rename(columns={"RetractionDate": "retraction_date",
                         "RetractionNature": "retraction_nature"})
        .explode("reason")
        .dropna(subset=["reason"])
        .reset_index(drop=True)
    )
    if families:
        fams = tuple(f.lower() for f in families)
        keep = out["reason"].str.lower().map(lambda r: any(f in r for f in fams))
        out = out[keep].reset_index(drop=True)
    return out


def summarize(csv_path) -> dict:
    """Counts for rwdb_counts.json: totals, Ref/Attr family, broadened vocabulary."""
    df = _read_csv(csv_path)
    labels = load_labels(csv_path)
    lower = labels["reason"].str.lower()

    ref_attr = labels[lower.str.contains(REF_ATTR_SUBSTRING, regex=False)]
    raw_reason = df["Reason"].fillna("").str.lower()
    broad = labels[lower.map(lambda r: any(s in r for s in BROADENED_SUBSTRINGS))]

    return {
        "total_records": int(len(df)),
        "original_doi_nonnull_rate": round(float(df["OriginalPaperDOI"].notna().mean()), 4),
        "ref_attr_family": {
            "records": int(raw_reason.str.contains(REF_ATTR_SUBSTRING, regex=False).sum()),
            "unique_papers_with_doi": int(ref_attr["doi_norm"].dropna().nunique()),
            "matching_reason_literals": {
                k: int(v) for k, v in ref_attr["reason"].value_counts().items()
            },
        },
        "broadened_vocabulary": {
            k: int(v) for k, v in broad["reason"].value_counts().items()
        },
    }


def download_snapshot(dest: str | Path, url: str = DEFAULT_URL) -> Path:
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest

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


# ---------- S1.2: works <-> labels join ----------

_VERSION_SUFFIX = r"\.v\d+$"
JOIN_MODES = ("strict", "loose")


def strip_version(doi_norm: str | None) -> str | None:
    """Drop one trailing version suffix (".v2", ".v13") from a normalized DOI."""
    import re

    if doi_norm is None or (isinstance(doi_norm, float) and doi_norm != doi_norm):
        return None
    return re.sub(_VERSION_SUFFIX, "", doi_norm)


def join_labels(works_df, labels_df, mode: str = "strict"):
    """Mark each work that matches a labeled paper; report strict and loose counts.

    works_df: needs `doi_norm` (already normalized by doi.normalize_doi); every
    other column is carried through unchanged, row order and length preserved.
    labels_df: needs `doi_norm` (e.g. load_labels output after the family
    filter; several rows per paper are fine). Filter labels by family and by
    RetractionNature BEFORE calling: this function does not decide what a
    positive is.
    mode: "strict" = normalized-DOI equality; "loose" = strict, plus equality
    after strip_version() on both sides (a strict match always wins).

    Returns (joined, stats). joined = works_df + `label` (int 0/1), `label_doi`
    (the labels-side doi_norm that matched, else None), `match` ("strict",
    "loose" or ""). stats has both match counts whatever the mode, so the
    strict-vs-loose delta can be recorded (task S1.2):
    mode, works, works_with_doi, label_papers, strict_matches, loose_matches,
    loose_only, labeled_works.

    Leak argument: labels are outcome data joined after scoring (S4.1); the
    join reads no graph state, so it cannot leak future events into features.
    """
    if mode not in JOIN_MODES:
        raise ValueError(f"join mode must be one of {JOIN_MODES}, got {mode!r}")

    label_dois = sorted(set(labels_df["doi_norm"].dropna()))
    strict_set = set(label_dois)
    loose_map = {}
    for d in label_dois:  # sorted, so ties resolve deterministically
        loose_map.setdefault(strip_version(d), d)

    works_doi = works_df["doi_norm"]
    strict_hit = works_doi.map(lambda d: d if d in strict_set else None)
    loose_hit = works_doi.map(lambda d: loose_map.get(strip_version(d)) if d else None)
    loose_hit = strict_hit.where(strict_hit.notna(), loose_hit)

    chosen = strict_hit if mode == "strict" else loose_hit
    joined = works_df.copy()
    joined["label"] = chosen.notna().astype(int)
    joined["label_doi"] = chosen.astype(object).where(chosen.notna(), None)
    joined["match"] = [
        "strict" if s is not None and s == s else ("loose" if c is not None and c == c else "")
        for s, c in zip(strict_hit, chosen)
    ]

    n_strict = int(strict_hit.notna().sum())
    n_loose = int(loose_hit.notna().sum())
    stats = {
        "mode": mode,
        "works": int(len(works_df)),
        "works_with_doi": int(works_doi.notna().sum()),
        "label_papers": len(label_dois),
        "strict_matches": n_strict,
        "loose_matches": n_loose,
        "loose_only": n_loose - n_strict,
        "labeled_works": int(joined["label"].sum()),
    }
    return joined, stats


# ---------- snapshot refresh ----------

def snapshot_delta(old_csv, new_csv, family: str = REF_ATTR_SUBSTRING) -> dict:
    """Record-level comparison of two RWDB snapshots (for a dated refresh note).

    Rows with no Record ID are blank lines in the CSV, counted as blank_rows_*
    and never as records (summarize() counts them in total_records). Reason
    changes compare parsed reason sets, so a pure formatting change ('+' prefix,
    trailing ';') is not a change. Family DOI counts cover every
    RetractionNature, like summarize(); added family DOIs are broken down by the
    original publication year recorded in the new snapshot.
    """
    from citecurv.data.doi import normalize_doi
    from citecurv.data.slice_query import parse_rwdb_date

    def prep(path):
        df = _read_csv(path)
        blank = df["Record ID"].isna()
        df = df[~blank].copy()
        df.index = df["Record ID"].astype(str).str.strip()
        if df.index.duplicated().any():
            raise ValueError(f"duplicate Record ID values in RWDB snapshot {path}")
        df["rset"] = df["Reason"].map(lambda r: frozenset(parse_reasons(r)))
        df["fam"] = df["rset"].map(lambda rs: any(family in r.lower() for r in rs))
        df["doi"] = df["OriginalPaperDOI"].map(normalize_doi)
        return df, int(blank.sum())

    old, blank_old = prep(old_csv)
    new, blank_new = prep(new_csv)
    common = old.index.intersection(new.index)
    o, n = old.loc[common], new.loc[common]
    fam_old = set(old.loc[old["fam"], "doi"].dropna())
    fam_new = set(new.loc[new["fam"], "doi"].dropna())
    added = fam_new - fam_old

    first_year = (
        new[new["doi"].isin(added)].drop_duplicates("doi")
        .set_index("doi")["OriginalPaperDate"].map(parse_rwdb_date)
    )
    by_year = {}
    for ts in first_year:
        key = int(ts.year) if ts == ts and ts is not None else None
        by_year[key] = by_year.get(key, 0) + 1

    return {
        "records_old": int(len(old)),
        "records_new": int(len(new)),
        "blank_rows_old": blank_old,
        "blank_rows_new": blank_new,
        "records_added": int(len(new.index.difference(old.index))),
        "records_removed": int(len(old.index.difference(new.index))),
        "records_reason_changed": int((o["rset"] != n["rset"]).sum()),
        "family_gained": int((~o["fam"] & n["fam"]).sum()),
        "family_lost": int((o["fam"] & ~n["fam"]).sum()),
        "family_records_old": int(old["fam"].sum()),
        "family_records_new": int(new["fam"].sum()),
        "family_dois_old": len(fam_old),
        "family_dois_new": len(fam_new),
        "family_dois_added": len(added),
        "family_dois_removed": len(fam_old - fam_new),
        "family_dois_added_by_orig_year": dict(sorted(
            by_year.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)
        )),
    }

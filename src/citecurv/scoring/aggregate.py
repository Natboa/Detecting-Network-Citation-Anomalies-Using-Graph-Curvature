"""Scores file, per-paper / per-author rollups (FR8) and filtered export (FR9) - task S3.3.

scores.parquet (CONTRACTS.md, NFR10): one row per scored event with its four
named components, the combined total, the cold-start flag, and the run stamp
(model_variant, seed, config_hash) so any flag traces back to its inputs.
Extra breakdown columns (z_<component>, n_components_used, ...) are kept after
the contract columns.

Rollups (fixed here, plan S3.3): a paper's score is the mean of its top-n edge
totals (n = 5 by default; fewer edges -> mean of all of them), reported with
the number of its edge totals strictly above a threshold and its edge count.
role picks which edges belong to a paper: "cited" (the paper is dst; the
default, matching the cartel/citation-receiving view), "citing" (src), or
"both" (each event counts once for each distinct endpoint). An author's score
is the mean of the top-n paper scores of the author's papers.

Leak argument: rollups and exports only read finished scores and metadata for
reporting. They are never fed back as features, so they may aggregate over any
time range without affecting what an event's score could see.
"""

import datetime as _dt
import hashlib
import json
from pathlib import Path

import numpy as np

COMPONENTS = ["s_embed", "s_geom", "s_temp", "s_curvdyn"]
STAMP_COLUMNS = ["model_variant", "seed", "config_hash"]
INPUT_COLUMNS = ["event_id", "src_id", "dst_id", "t", *COMPONENTS, "total", "cold_start"]
SCORES_COLUMNS = INPUT_COLUMNS + STAMP_COLUMNS
ROLES = ("cited", "citing", "both")
ENDPOINTS = ("either", "src", "dst")
EPOCH = _dt.date(1970, 1, 1)


def _canonical(obj):
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_canonical(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    raise TypeError(f"config_hash cannot canonicalise {type(obj).__name__}")


def config_hash(cfg: dict) -> str:
    """First 12 hex chars of sha256 over canonical JSON (sorted keys, no spaces).

    Tuples hash like lists and numpy scalars like their Python values, so the
    hash depends on the configuration's content, not on how it was loaded.
    """
    text = json.dumps(_canonical(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _check_columns(df, required, what):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing contract columns: {missing}")


def write_scores(df, path, model_variant, seed, config_hash):
    """Validate, stamp and write scores.parquet; returns the path.

    df must hold event_id, src_id, dst_id, t, s_embed, s_geom, s_temp,
    s_curvdyn, total, cold_start with unique event_id. The stamp columns are
    added; if df already has one with a different value, ValueError.
    """
    _check_columns(df, INPUT_COLUMNS, "scores frame")
    if df["event_id"].duplicated().any():
        raise ValueError("scores frame has duplicate event_id values")
    if df["cold_start"].isna().any():
        raise ValueError("cold_start must be a bool for every row")
    stamps = {"model_variant": str(model_variant), "seed": int(seed), "config_hash": str(config_hash)}
    for col, value in stamps.items():
        if col in df.columns and (df[col] != value).any():
            raise ValueError(f"existing {col} column differs from the stamp {value!r}")

    out = df.reset_index(drop=True).copy()
    for col, value in stamps.items():
        out[col] = value
    out["event_id"] = out["event_id"].astype(np.int64)
    out["t"] = out["t"].astype(np.int64)
    out["seed"] = out["seed"].astype(np.int64)
    out["cold_start"] = out["cold_start"].astype(bool)
    out[["src_id", "dst_id"]] = out[["src_id", "dst_id"]].astype(str)
    for c in [*COMPONENTS, "total"]:
        out[c] = out[c].astype(np.float64)
    extras = [c for c in out.columns if c not in SCORES_COLUMNS]
    out = out[SCORES_COLUMNS + extras]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def read_scores(path):
    """Read a scores.parquet file and check its contract columns."""
    import pandas as pd

    df = pd.read_parquet(path)
    _check_columns(df, SCORES_COLUMNS, f"scores file {path}")
    return df


def paper_rollup(
    scores,
    role="cited",
    top_n=5,
    threshold=None,
    exclude_cold_start=True,
    score_col="total",
):
    """Per-paper score: work_id, mean_top_n, n_above, n_events.

    mean_top_n: mean of the paper's top_n edge scores. n_above: edges with
    score > threshold (nullable Int64, all NA when threshold is None).
    n_events: edges counted for the paper. Rows with a NaN score are dropped;
    cold-start rows too unless exclude_cold_start=False. Sorted by mean_top_n
    descending, then work_id ascending.
    """
    import pandas as pd

    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    df = scores
    if exclude_cold_start and "cold_start" in df.columns:
        df = df[~df["cold_start"].astype(bool)]
    df = df[df[score_col].notna()]

    def edges(id_col):
        return pd.DataFrame({
            "event_id": df["event_id"].to_numpy(),
            "work_id": df[id_col].to_numpy(),
            "score": df[score_col].to_numpy(dtype=float),
        })

    if role == "cited":
        long = edges("dst_id")
    elif role == "citing":
        long = edges("src_id")
    else:
        long = pd.concat([edges("src_id"), edges("dst_id")], ignore_index=True)
        long = long.drop_duplicates(["event_id", "work_id"])

    long = long.sort_values(["work_id", "score"], ascending=[True, False], kind="mergesort")
    grouped = long.groupby("work_id", sort=True)["score"]
    out = pd.DataFrame({
        "mean_top_n": long.groupby("work_id", sort=True).head(top_n).groupby("work_id")["score"].mean(),
        "n_events": grouped.size().astype(np.int64),
    })
    if threshold is None:
        out["n_above"] = pd.array([pd.NA] * len(out), dtype="Int64")
    else:
        out["n_above"] = (long["score"] > threshold).groupby(long["work_id"]).sum().astype("Int64")
    out = out.reset_index()[["work_id", "mean_top_n", "n_above", "n_events"]]
    return out.sort_values(
        ["mean_top_n", "work_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def author_rollup(paper_scores, authorships, top_n=5, score_col="mean_top_n"):
    """Per-author score (FR8): author_id, mean_top_n, n_papers.

    paper_scores: output of paper_rollup (work_id + score_col).
    authorships: work_id, author_id (one row per pair; duplicates ignored).
    mean_top_n: mean of the author's top_n paper scores; n_papers: the
    author's papers that have a score. Authors with no scored paper are left
    out. Sorted by mean_top_n descending, then author_id ascending.
    """
    import pandas as pd

    pairs = authorships[["work_id", "author_id"]].drop_duplicates()
    m = pairs.merge(paper_scores[["work_id", score_col]], on="work_id", how="inner")
    m = m[m[score_col].notna()].rename(columns={score_col: "score"})
    m = m.sort_values(["author_id", "score"], ascending=[True, False], kind="mergesort")
    out = pd.DataFrame({
        "mean_top_n": m.groupby("author_id").head(top_n).groupby("author_id")["score"].mean(),
        "n_papers": m.groupby("author_id").size().astype(np.int64),
    }).reset_index()
    return out.sort_values(
        ["mean_top_n", "author_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)[["author_id", "mean_top_n", "n_papers"]]


def _to_days(value) -> int:
    """Days since 1970-01-01 from an int, an ISO date string, or a date/datetime."""
    if isinstance(value, str):
        value = _dt.date.fromisoformat(value)
    if isinstance(value, _dt.datetime):
        value = value.date()
    if isinstance(value, _dt.date):
        return (value - EPOCH).days
    return int(value)


def _concept_items(value):
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(";") if p.strip()]
    return [str(v) for v in value]


def _works_matching_fields(works, fields):
    wanted = set(fields)
    hit = np.zeros(len(works), dtype=bool)
    if "field" not in works.columns and "concepts" not in works.columns:
        raise ValueError("works needs a 'field' or 'concepts' column to filter by field")
    if "field" in works.columns:
        hit |= works["field"].isin(wanted).to_numpy()
    if "concepts" in works.columns:
        hit |= np.array([bool(wanted.intersection(_concept_items(v))) for v in works["concepts"]])
    return set(works["work_id"][hit])


def _endpoint_mask(df, work_ids, endpoint):
    src = df["src_id"].isin(work_ids).to_numpy()
    dst = df["dst_id"].isin(work_ids).to_numpy()
    return {"either": src | dst, "src": src, "dst": dst}[endpoint]


def export_csv(
    scores,
    path,
    t_from=None,
    t_to=None,
    works=None,
    fields=None,
    venues=None,
    min_total=None,
    endpoint="either",
) -> int:
    """Filter scores to a CSV (FR9); returns the number of rows written.

    scores: a scores DataFrame or a scores.parquet path.
    t_from / t_to: inclusive bounds on t, as days since epoch or ISO dates.
    fields: keep events whose endpoint paper has works.field in fields or a
    concept in fields; venues: endpoint paper has works.venue_id in venues.
    endpoint: which paper must match, "either" (default), "src" or "dst".
    min_total: keep total >= min_total. Filters combine with AND.
    Rows are written sorted by total descending, then event_id; a `date`
    column (ISO) is inserted after t for spreadsheet readers. Cold-start rows
    are kept and marked by their cold_start column.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {ENDPOINTS}, got {endpoint!r}")
    if isinstance(scores, (str, Path)):
        scores = read_scores(scores)
    df = scores
    mask = np.ones(len(df), dtype=bool)
    if t_from is not None:
        mask &= (df["t"] >= _to_days(t_from)).to_numpy()
    if t_to is not None:
        mask &= (df["t"] <= _to_days(t_to)).to_numpy()
    if min_total is not None:
        mask &= (df["total"] >= min_total).to_numpy()
    if fields is not None or venues is not None:
        if works is None:
            raise ValueError("field/venue filters need the works table (works=...)")
        if fields is not None:
            mask &= _endpoint_mask(df, _works_matching_fields(works, fields), endpoint)
        if venues is not None:
            hit = works["venue_id"].isin(set(venues)).to_numpy()
            mask &= _endpoint_mask(df, set(works["work_id"][hit]), endpoint)

    out = df[mask].sort_values(["total", "event_id"], ascending=[False, True], kind="mergesort")
    out = out.drop(columns=["date"], errors="ignore")
    dates = [(EPOCH + _dt.timedelta(days=int(t))).isoformat() for t in out["t"]]
    out.insert(out.columns.get_loc("t") + 1, "date", dates)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")
    return int(len(out))

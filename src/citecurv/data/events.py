"""Edges -> the contract events frame, and per-event labels from Retraction Watch (CONTRACTS.md).

This module is the one place where `event_id` is defined for real data. Every later artifact
(curvature cache, TGN scores, components, scores, event labels) is joined on it, so it must not
depend on anything incidental to the pull.

`build_events(edges_df)` takes the S1.1 `edges.parquet` frame (`src_id, dst_id, t, doi_pair_key`;
src cites dst, t = the citing work's publication date in days since 1970-01-01) and returns the
`events` frame `event_id, src_id, dst_id, t`:

- event_id: rows are sorted by (t, src_id, dst_id) and numbered 0..n-1 (int64). The key uses
  contract columns only, so the ids do not depend on the order in which OpenAlex returned pages,
  on how year buckets were concatenated, or on a resumed pull. The ids are stable for one fixed
  edges table; adding or removing works shifts the ids of later rows, which is why S4.0 freezes
  the pull with a manifest. The resulting row order is the canonical order (t, event_id).
- Duplicate (src_id, dst_id) pairs. OpenAlex `referenced_works` lists each referenced work once,
  so a pair can repeat only when the same work is read twice (for example a record updated
  between two sessions of a resumed pull, possibly with another publication date). Policy
  (dedupe=True, the default): keep the first row of the pair in (t, src_id, dst_id) order, that
  is the earliest date. `doi_pair_key` is `src|dst`, so it adds nothing to the key and is dropped.
- Self-citation loops (src_id == dst_id: a work listing itself among its references, a data error
  in the source record) are kept, because removing rows is a data decision that belongs in the
  dataset manifest, not in an id builder. Note for consumers: a loop has an undefined
  Ollivier-Ricci curvature (30_curvature counts them) and counts once in both degrees of the same
  node. `attrs["n_self_loops"]` reports how many there are.

Leak note: this module computes no feature. Ordering ties at the same t are broken by the ids of
the two works, which carry no information about the future of the graph; the same-timestamp
policy of CONTRACTS.md (an event may use ts <= t except its own row) is what downstream modules
rely on, not the tie order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["event_id", "src_id", "dst_id", "t"]
LABEL_COLUMNS = ["event_id", "label"]
_EDGE_KEY = ["t", "src_id", "dst_id"]


def _require(df, cols, what):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing columns {missing}")


def _as_int64(s: pd.Series, name: str) -> np.ndarray:
    if s.isna().any():
        raise ValueError(f"column {name!r} has null values")
    v = pd.to_numeric(s, errors="raise").to_numpy()
    if not np.issubdtype(v.dtype, np.integer):
        if not np.all(np.isfinite(v)) or not np.all(v == np.round(v)):
            raise ValueError(f"column {name!r} must hold integer values")
    return v.astype(np.int64)


def _as_str(s: pd.Series, name: str) -> pd.Series:
    if s.isna().any():
        raise ValueError(f"column {name!r} has null values")
    return s.astype(str)


def build_events(edges_df: pd.DataFrame, *, dedupe: bool = True) -> pd.DataFrame:
    """Contract events frame from an S1.1 edges frame (see the module docstring for the rules).

    edges_df: needs `src_id`, `dst_id`, `t` (integer days); other columns (`doi_pair_key`) are
    ignored. Any row order. Null ids or times raise ValueError.
    dedupe: drop repeated (src_id, dst_id) pairs, keeping the earliest row (default True).
    Returns a new DataFrame with columns event_id (int64, 0..n-1), src_id (str), dst_id (str),
    t (int64), sorted by (t, event_id), with a RangeIndex. attrs: n_edges_in,
    n_duplicates_dropped, n_self_loops (counts for the dataset manifest; not written to parquet
    by write_events).
    """
    _require(edges_df, ["src_id", "dst_id", "t"], "edges frame")
    ev = pd.DataFrame({
        "src_id": _as_str(edges_df["src_id"], "src_id").to_numpy(),
        "dst_id": _as_str(edges_df["dst_id"], "dst_id").to_numpy(),
        "t": _as_int64(edges_df["t"], "t"),
    })
    n_in = len(ev)
    ev = ev.sort_values(_EDGE_KEY, kind="mergesort").reset_index(drop=True)
    if dedupe:
        ev = ev.drop_duplicates(["src_id", "dst_id"], keep="first").reset_index(drop=True)
    ev.insert(0, "event_id", np.arange(len(ev), dtype=np.int64))
    ev["t"] = ev["t"].astype(np.int64)
    ev.attrs = {
        "n_edges_in": int(n_in),
        "n_duplicates_dropped": int(n_in - len(ev)),
        "n_self_loops": int((ev["src_id"] == ev["dst_id"]).sum()),
    }
    return ev


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Check the events contract; returns the four columns typed (int64 ids and times, str ids).

    Raises ValueError for a missing column, a null value, a non-integer t or event_id, or a
    duplicate event_id. Row order is kept.
    """
    _require(events, EVENT_COLUMNS, "events frame")
    out = pd.DataFrame({
        "event_id": _as_int64(events["event_id"], "event_id"),
        "src_id": _as_str(events["src_id"], "src_id").to_numpy(),
        "dst_id": _as_str(events["dst_id"], "dst_id").to_numpy(),
        "t": _as_int64(events["t"], "t"),
    })
    if out["event_id"].duplicated().any():
        raise ValueError("events frame has duplicate event_id values")
    return out


def write_events(events: pd.DataFrame, path) -> Path:
    """Validate and write `events.parquet` in canonical (t, event_id) order; returns the path."""
    ev = validate_events(events).sort_values(["t", "event_id"], kind="mergesort")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ev.reset_index(drop=True).to_parquet(path, index=False)
    return path


def read_events(path) -> pd.DataFrame:
    """Read and validate `events.parquet` (raises ValueError when the file breaks the contract)."""
    return validate_events(pd.read_parquet(path))


def event_labels_from_rwdb(events: pd.DataFrame, works: pd.DataFrame, labels_df: pd.DataFrame,
                           mode: str = "strict", *, families=None, natures=None) -> pd.DataFrame:
    """Weak event labels under rule L12: positive when the citing or the cited paper is labelled.

    events: contract events frame. works: `work_id`, `doi_norm` for every work whose DOI is known
    (works.parquet; papers that are only referenced, never pulled, have no DOI there and so can
    never be positive; 10_data adds the work ids of labelled DOIs from the S1.3 lookup to close
    that gap). labels_df: Retraction Watch labels with `doi_norm` (rwdb.load_labels output, or the
    labels.parquet contract). families: optional lowercase substrings matched against a `reason`
    column (e.g. ["referencing/attributions"]); natures: optional RetractionNature values matched
    against a `retraction_nature` column. A filter whose column is absent raises ValueError rather
    than being skipped. Without filters labels_df is taken as already filtered.
    mode: "strict" or "loose" DOI matching, as in rwdb.join_labels.

    Returns DataFrame event_id (int64), label (int64 0/1) in the row order of `events`.
    attrs["stats"]: the join_labels stats plus labeled_works_in_events, positive_events,
    positive_via_src, positive_via_dst.

    PROVISIONAL(G3): which families and natures count is a pending decision (configs labels.natures).
    Leak note: labels are outcome data joined after scoring (S4.1); never use them as features.
    """
    from citecurv.data.rwdb import JOIN_MODES, join_labels

    if mode not in JOIN_MODES:
        raise ValueError(f"join mode must be one of {JOIN_MODES}, got {mode!r}")
    _require(events, ["event_id", "src_id", "dst_id"], "events frame")
    _require(works, ["work_id", "doi_norm"], "works frame")
    _require(labels_df, ["doi_norm"], "labels frame")

    lab = labels_df
    if families:
        if "reason" not in lab.columns:
            raise ValueError("families filter needs a 'reason' column (rwdb.load_labels output)")
        fams = tuple(str(f).lower() for f in families)
        lab = lab[lab["reason"].fillna("").str.lower().map(lambda r: any(f in r for f in fams))]
    if natures:
        if "retraction_nature" not in lab.columns:
            raise ValueError("natures filter needs a 'retraction_nature' column (rwdb.load_labels output)")
        lab = lab[lab["retraction_nature"].isin(list(natures))]

    id_doi = works[["work_id", "doi_norm"]].drop_duplicates("work_id")
    joined, stats = join_labels(id_doi, lab, mode=mode)
    positive = set(joined.loc[joined["label"] == 1, "work_id"].astype(str))

    via_src = events["src_id"].astype(str).isin(positive).to_numpy()
    via_dst = events["dst_id"].astype(str).isin(positive).to_numpy()
    out = pd.DataFrame({
        "event_id": _as_int64(events["event_id"], "event_id"),
        "label": (via_src | via_dst).astype(np.int64),
    })
    in_events = positive & (set(events["src_id"].astype(str)) | set(events["dst_id"].astype(str)))
    out.attrs["stats"] = {
        **stats,
        "labeled_works_in_events": len(in_events),
        "positive_events": int(out["label"].sum()),
        "positive_via_src": int(via_src.sum()),
        "positive_via_dst": int(via_dst.sum()),
    }
    return out

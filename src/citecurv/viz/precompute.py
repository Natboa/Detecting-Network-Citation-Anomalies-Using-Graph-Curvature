"""Numbers the explorer page shows, computed in Python so the page only renders (no statistics in JavaScript).

Everything here is a pure function of the pipeline outputs (CONTRACTS.md). Conventions:
- Higher score = more anomalous. Display order = total descending, ties by the canonical order (t, event_id).
- Cold-start events (and rows without a finite total) are excluded from scoring (CONTRACTS.md), so they get no rank
  and enter no reference distribution, metric, heatmap, or rollup.
- The "flag" threshold is the total of the last event inside the top `flag_share` (default 1 %) of ranked events,
  the population of the D4 fairness check (IMPLEMENTATION_PLAN.md S3.4a).

Leak note: these are evaluation-time summaries of finished scores; no score is recomputed here, so nothing can leak
into a score. Reference histograms use all scored events in the file (stated on the page), not only training rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

COMPONENTS = ("s_embed", "s_geom", "s_temp", "s_curvdyn")
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 fallback
_MEAN_KEYS = ("mean", "mu", "means", "loc", "center")
_STD_KEYS = ("std", "sd", "sigma", "stds", "scale")


def num(x, sig: int = 6):
    """JSON-safe rounded float (None for NaN/inf/None), deterministic text form."""
    if x is None or x is pd.NA:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return float(f"{f:.{sig}g}")


# ---------------------------------------------------------------- ranking


def rank_events(scores: pd.DataFrame) -> pd.DataFrame:
    """All rows in display order with `rank` (nullable Int64, 1-based) and `scorable` columns added."""
    df = scores.copy()
    if "cold_start" not in df.columns:
        df["cold_start"] = False
    df["cold_start"] = df["cold_start"].fillna(False).astype(bool)
    df["total"] = pd.to_numeric(df["total"], errors="coerce").astype(float)
    key = (-df["total"]).fillna(np.inf)
    df = df.assign(_k=key).sort_values(["_k", "t", "event_id"], kind="mergesort").drop(columns="_k")
    scorable = (~df["cold_start"]) & np.isfinite(df["total"])
    df["scorable"] = scorable.to_numpy()
    rank = pd.array(np.cumsum(scorable.to_numpy()), dtype="Int64")
    rank[~scorable.to_numpy()] = pd.NA
    df["rank"] = rank
    return df.reset_index(drop=True)


def flag_count(n_scored: int, flag_share: float) -> int:
    return 0 if n_scored == 0 else max(1, math.ceil(round(flag_share * n_scored, 9)))


# ---------------------------------------------------------------- z statistics and weights


def _component_dict(x, names=None) -> dict:
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, Mapping):
        return dict(x)
    if isinstance(x, (Sequence, np.ndarray)) and not isinstance(x, str):
        names = list(names) if names is not None else list(COMPONENTS)
        if len(x) != len(names):
            raise TypeError(f"expected {len(names)} values aligned with {names}, got {len(x)}")
        return dict(zip(names, x))
    raise TypeError(f"cannot read per-component values from {type(x).__name__}")


def _pick(obj, keys):
    for k in keys:
        if isinstance(obj, Mapping):
            if k in obj:
                return obj[k]
        elif hasattr(obj, k) and not callable(getattr(obj, k)):
            return getattr(obj, k)
    return None


def coerce_zstats(z) -> dict | None:
    """Frozen z statistics as {component: (mean, std)}.

    Accepts {c: (m, s)}, {c: {"mean": m, "std": s}}, {"mean": {c: m}, "std": {c: s}}, a DataFrame with components on
    the index (or columns) and mean/std on the other axis, or an object (e.g. a ZStats dataclass) with mean/std
    attributes holding dicts, Series, or arrays aligned with `components`/COMPONENTS.
    """
    if z is None:
        return None
    if isinstance(z, pd.DataFrame):
        if set(COMPONENTS) & set(map(str, z.index)):
            z = {str(c): z.loc[c].to_dict() for c in z.index}
        elif set(COMPONENTS) & set(map(str, z.columns)):
            z = {str(c): z[c].to_dict() for c in z.columns}
        else:
            raise TypeError("zstats DataFrame must have the component names on its index or columns")
    if isinstance(z, Mapping) and set(z) & set(COMPONENTS):
        out = {}
        for c in COMPONENTS:
            if c not in z:
                continue
            v = z[c]
            if isinstance(v, pd.Series):
                v = v.to_dict()
            if isinstance(v, Mapping):
                m, s = _pick(v, _MEAN_KEYS), _pick(v, _STD_KEYS)
            else:
                m, s = v
            out[c] = (float(m), float(s))
        return out
    m, s = _pick(z, _MEAN_KEYS), _pick(z, _STD_KEYS)
    if m is not None and s is not None:
        names = _pick(z, ("components", "names", "columns"))
        m, s = _component_dict(m, names), _component_dict(s, names)
        return {c: (float(m[c]), float(s[c])) for c in COMPONENTS if c in m and c in s}
    raise TypeError(f"cannot read z statistics from {type(z).__name__}")


def coerce_weights(w) -> dict | None:
    """Combiner weights as {component: float}; dict, Series, 4-sequence, or an object with a `weights` attribute."""
    if w is None:
        return None
    if not isinstance(w, (Mapping, pd.Series, Sequence, np.ndarray)) and hasattr(w, "weights"):
        return coerce_weights(w.weights)
    d = _component_dict(w)
    return {c: float(d[c]) for c in COMPONENTS if c in d}


def fallback_zstats(ranked: pd.DataFrame) -> dict:
    """Mean/std (ddof 0) over scored rows of the scores file: used only when frozen statistics are not given."""
    d = ranked[ranked["scorable"]]
    out = {}
    for c in COMPONENTS:
        if c in d.columns:
            v = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            out[c] = (float(v.mean()), float(v.std())) if len(v) else (0.0, 1.0)
    return out


def zscores(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in COMPONENTS:
        if c not in df.columns or c not in stats:
            out[c] = np.nan
            continue
        m, s = stats[c]
        raw = pd.to_numeric(df[c], errors="coerce").astype(float)
        out[c] = (raw - m) / s if s and math.isfinite(s) else raw * 0.0
    return out


# ---------------------------------------------------------------- distributions and curves


def histogram(values, bins: int = 40, clip=(0.005, 0.995)) -> dict | None:
    """Counts over the central quantile range; values outside are counted as under/over (shown on the page)."""
    v = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    lo, hi = (float(np.quantile(v, clip[0])), float(np.quantile(v, clip[1]))) if len(v) > 50 else (v.min(), v.max())
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 0.5)
        lo, hi = lo - pad, hi + pad
    inside = v[(v >= lo) & (v <= hi)]
    counts, _ = np.histogram(inside, bins=bins, range=(lo, hi))
    return {"lo": num(lo), "hi": num(hi), "counts": [int(c) for c in counts], "under": int((v < lo).sum()),
            "over": int((v > hi).sum()), "n": int(len(v)), "mean": num(v.mean()), "median": num(np.median(v))}


def percentile_of(sorted_ref: np.ndarray, x) -> float | None:
    if x is None or not np.isfinite(x) or len(sorted_ref) == 0:
        return None
    return num(np.searchsorted(sorted_ref, x, side="right") / len(sorted_ref), 4)


def _downsample(m: int, max_points: int) -> np.ndarray:
    if m <= max_points:
        return np.arange(m)
    half = max_points // 2
    lin = np.linspace(0, m - 1, half).round()
    geo = np.geomspace(1, m, half).round() - 1
    return np.unique(np.concatenate([lin, geo, [0, m - 1]]).astype(int))


def ranking_curves(scores, labels, max_points: int = 300) -> dict:
    """ROC and precision-recall over distinct score thresholds (ties grouped, scikit-learn convention).

    AUROC = trapezoid area under ROC; AP = sum over thresholds of (R_i - R_{i-1}) * P_i. NaN scores rank last.
    Curves are thinned to about `max_points` points for drawing; AUROC/AP use every threshold.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    s = np.where(np.isfinite(s), s, -np.inf)
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    thr = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tps = np.cumsum(yy)[thr].astype(float)
    fps = (thr + 1) - tps
    n_pos, n_neg = tps[-1], fps[-1]
    out = {"n": int(len(s)), "n_pos": int(n_pos), "auroc": None, "ap": None, "roc": [], "pr": []}
    if n_pos == 0 or n_neg == 0:
        return out
    tpr, fpr = tps / n_pos, fps / n_neg
    prec, rec = tps / (tps + fps), tps / n_pos
    out["ap"] = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    out["auroc"] = float(_trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    keep = _downsample(len(thr), max_points)
    out["roc"] = [[0.0, 0.0]] + [[round(float(fpr[i]), 5), round(float(tpr[i]), 5)] for i in keep]
    out["pr"] = [[round(float(rec[i]), 5), round(float(prec[i]), 5)] for i in keep]
    return out


def precision_at(labels_in_rank_order, ks) -> dict:
    y = np.asarray(labels_in_rank_order, dtype=int)
    out = {}
    for k in ks:
        if k <= len(y):
            out[str(k)] = {"p": num(y[:k].mean(), 4), "found": int(y[:k].sum())}
    return out


# ---------------------------------------------------------------- D4 fairness heatmap


def flag_heatmap(df: pd.DataFrame, flag_share: float = 0.01, limit: float = 0.5) -> dict:
    """Share of the top-`flag_share` flags held by each (field, year) cell; FAIL when a cell holds > `limit`.

    `df`: scored rows only, columns event_id, t, total, field (the citing paper's field: t is its publication date).
    """
    d = df.assign(_k=-df["total"].astype(float)).sort_values(["_k", "t", "event_id"], kind="mergesort")
    n = len(d)
    n_flags = flag_count(n, flag_share)
    d = d.assign(flag=np.arange(n) < n_flags, year=pd.to_datetime(d["t"], unit="D").dt.year,
                 field=d["field"].fillna("unknown").astype(str))
    sizes = d.groupby("field").size()
    fields = sorted(sizes.index, key=lambda f: (-int(sizes[f]), f))
    years = list(range(int(d.year.min()), int(d.year.max()) + 1))
    ev = d.groupby(["field", "year"]).size()
    fl = d[d.flag].groupby(["field", "year"]).size()
    events = [[int(ev.get((f, y), 0)) for y in years] for f in fields]
    flags = [[int(fl.get((f, y), 0)) for y in years] for f in fields]
    share = [[num(c / n_flags, 4) for c in row] for row in flags]
    rate = [[num(c / e, 4) if e else None for c, e in zip(rf, re)] for rf, re in zip(flags, events)]
    best = max(((share[i][j] or 0.0, i, j) for i in range(len(fields)) for j in range(len(years))), default=(0, 0, 0))
    return {
        "fields": fields, "years": years, "events": events, "flags": flags, "share": share, "rate": rate,
        "n": int(n), "n_flags": int(n_flags), "flag_share": flag_share, "limit": limit,
        "threshold": num(d["total"].iloc[n_flags - 1]) if n_flags else None,
        "max_share": best[0], "max_cell": [fields[best[1]], years[best[2]]] if fields else None,
        "passed": bool(best[0] <= limit),
    }


# ---------------------------------------------------------------- rollups (FR8, S3.3 definition)


def incident_long(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (paper, scored event it takes part in), with its role ('src' cites, 'dst' is cited)."""
    d = scores
    if "scorable" in d.columns:
        d = d[d["scorable"]]
    else:
        cold = d["cold_start"].fillna(False).astype(bool) if "cold_start" in d.columns else False
        d = d[~cold & np.isfinite(d["total"].astype(float))]
    cols = ["event_id", "t", "total"]
    long = pd.concat([d[cols].assign(paper=d["src_id"].astype(str), role="src"),
                      d[cols].assign(paper=d["dst_id"].astype(str), role="dst")], ignore_index=True)
    long = long.drop_duplicates(["paper", "event_id"])
    long = long.assign(_k=-long["total"]).sort_values(["paper", "_k", "t", "event_id"], kind="mergesort")
    long["k"] = long.groupby("paper").cumcount()
    return long.drop(columns="_k")


def paper_rollup(scores: pd.DataFrame, threshold: float, top: int = 5, long: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-paper score = mean of its top-`top` incident edge totals (fewer if it has fewer edges), plus the count of
    incident edges with total >= threshold. Sorted best first (score, count, id)."""
    long = incident_long(scores) if long is None else long
    g = long.groupby("paper")
    out = pd.DataFrame({
        "score": long[long.k < top].groupby("paper")["total"].mean(),
        "n_above": long[long.total >= threshold].groupby("paper").size(),
        "n_events": g.size(),
        "n_as_src": long[long.role == "src"].groupby("paper").size(),
        "n_as_dst": long[long.role == "dst"].groupby("paper").size(),
    }).fillna({"n_above": 0, "n_as_src": 0, "n_as_dst": 0})
    for c in ("n_above", "n_events", "n_as_src", "n_as_dst"):
        out[c] = out[c].astype(int)
    out = out.rename_axis("paper_id").reset_index()
    return out.sort_values(["score", "n_above", "paper_id"], ascending=[False, False, True], kind="mergesort",
                           ignore_index=True)


def author_rollup(long: pd.DataFrame, authorships: pd.DataFrame, threshold: float, top: int = 5) -> pd.DataFrame:
    """Per-author score = mean of the top-`top` totals over the distinct events incident to any of the author's
    papers, plus the count of such events with total >= threshold. Only each paper's top-`top` events and its
    events above threshold can matter, so the join is pre-filtered to those rows."""
    small = long[(long.k < top) | (long.total >= threshold)]
    a = authorships[["work_id", "author_id"]].astype(str).drop_duplicates()
    j = small.merge(a, left_on="paper", right_on="work_id").drop_duplicates(["author_id", "event_id"])
    j = j.assign(_k=-j["total"]).sort_values(["author_id", "_k", "t", "event_id"], kind="mergesort")
    j["ka"] = j.groupby("author_id").cumcount()
    out = pd.DataFrame({
        "score": j[j.ka < top].groupby("author_id")["total"].mean(),
        "n_above": j[j.total >= threshold].groupby("author_id").size(),
        "n_papers": a.groupby("author_id").size(),
    })
    out = out[out["score"].notna()].fillna({"n_above": 0})
    out["n_above"] = out["n_above"].astype(int)
    out["n_papers"] = out["n_papers"].astype(int)
    out = out.rename_axis("author_id").reset_index()
    return out.sort_values(["score", "n_above", "author_id"], ascending=[False, False, True], kind="mergesort",
                           ignore_index=True)


# ---------------------------------------------------------------- cartel injection track


def coerce_cartels(cartels):
    """(members DataFrame[cartel_id, work_id, role, level], injected event ids or None).

    Accepts a members DataFrame (member column work_id / member_id / node_id / paper_id; role and level optional,
    level may also be called obviousness) or a dict {"members": DataFrame, "event_ids": iterable}.
    """
    event_ids = None
    if isinstance(cartels, Mapping):
        event_ids = cartels.get("event_ids", cartels.get("truth_ids"))
        cartels = cartels["members"]
    m = cartels.copy()
    for alt in ("member_id", "node_id", "paper_id"):
        if "work_id" not in m.columns and alt in m.columns:
            m = m.rename(columns={alt: "work_id"})
    if "level" not in m.columns and "obviousness" in m.columns:
        m = m.rename(columns={"obviousness": "level"})
    if "cartel_id" not in m.columns or "work_id" not in m.columns:
        raise ValueError("cartels need columns cartel_id and work_id (or member_id/node_id/paper_id)")
    if "role" not in m.columns:
        m["role"] = "member"
    if "level" not in m.columns:
        m["level"] = "all"
    m["work_id"] = m["work_id"].astype(str)
    m["role"] = m["role"].fillna("member").astype(str)
    m["level"] = m["level"].fillna("all").astype(str)
    if event_ids is not None:
        event_ids = {int(e) for e in event_ids}
    return m, event_ids


def injected_events(members: pd.DataFrame, event_ids, stream: pd.DataFrame, labels: dict | None) -> pd.DataFrame:
    """Injected events with their cartel: given ids, else events inside one cartel's member set (label 1 if known)."""
    s = stream[["event_id", "src_id", "dst_id", "t"]].astype({"src_id": str, "dst_id": str})
    if event_ids is not None:
        s = s[s.event_id.isin(event_ids)]
    mem = members[["cartel_id", "work_id"]]
    j = s.merge(mem.rename(columns={"work_id": "src_id"}), on="src_id").merge(
        mem.rename(columns={"work_id": "dst_id", "cartel_id": "cartel_dst"}), on="dst_id")
    j = j[j.cartel_id == j.cartel_dst].drop(columns="cartel_dst")
    if event_ids is None and labels:
        j = j[j.event_id.map(lambda e: labels.get(int(e)) == 1)]
    return j.drop_duplicates("event_id").sort_values(["cartel_id", "t", "event_id"], kind="mergesort")


def recall_by_level(levels, ranks, ks) -> dict:
    """{level: {str(k): share of that level's injected events with rank <= k}} (unranked = not caught)."""
    df = pd.DataFrame({"level": list(levels), "rank": list(ranks)})
    out = {}
    for lvl, g in df.groupby("level", sort=True):
        r = pd.to_numeric(g["rank"], errors="coerce")
        out[str(lvl)] = {str(k): num((r <= k).sum() / len(g), 6) for k in ks}
    return out

"""Forman screen at event times, screen thresholds, distribution report (task S2.2).

For every citation event e = (src -> dst, t) the screen value is the unweighted
directed Forman-Ricci curvature (Saucan et al., 2019) on the graph as it stood
at the event's time:

    F(e) = 2 - in_deg_t(src) - out_deg_t(dst)

where the degrees are taken on the graph formed by every OTHER event row with
ts <= t (CONTRACTS.md same-timestamp policy: rows of other events at the same t
are included, the scored row itself is not).

Edge cases, handled explicitly:

- The scored row src -> dst is an out-edge of src and an in-edge of dst, so it
  never enters in(src) or out(dst) unless it is a self-loop (src == dst). For a
  self-loop row, the row itself is removed from both counts; other self-loop
  rows of the same node with ts <= t (earlier, or same-t duplicates) count as
  one in-edge and one out-edge of that node, as in a NetworkX DiGraph.
- Duplicate (x, y) rows: by default a pair counts once from its first
  occurrence on (distinct-neighbour degree of a simple DiGraph). This matches
  forman_directed() on the time-t snapshot DiGraph and the DiGraph that
  EventStore.subgraph_at() hands to the ORC stage. count_duplicates=True
  counts rows instead (the multigraph degree that EventStore.degrees_at()
  reports).

Unknown values (unknown_without_refs=True; ledger G13, 2026-11-18). The event
table cannot tell a cited work that has no references from one whose reference
list is not known: OpenAlex serves no record for some cited ids, returns an
empty list for others, and both give out_deg_t(dst) = 0 and F = 2 - in_deg_t(src),
the maximum of the screen. On the pilot 40,173 of the 45,227 events at F >= 2
cited such a work. With the option on, F is NaN whenever out_deg_t(dst) = 0 at
the event's time: no evidence instead of a value (docs/score_components.md D10).
The rule reads the same counts as F, so it uses nothing after t(e); a cited
work whose first known reference is dated after t(e) is unknown at t(e).
NaN rows enter no threshold, are never escalated (so they get no ORC value)
and are left out of the distribution report. The default is off: the function
then returns the plain Forman value for every row.

Leak argument (NFR3): F(e) reads only rows with ts <= t(e) and never the
scored row itself; nothing after t(e) enters. Same-t rows of other events are
included by the project-wide policy; a module needing ts < t must filter itself.

Thresholds are fitted on reference (training) rows only and then frozen and
applied to every row, as for the combiner's ZStats: validation/test values never
move a cut-off.

Cost: one factorisation, one sort of composite (node, time) keys and two
binary searches per row, O(E log E) overall, vectorised in NumPy.
"""

import numpy as np
import pandas as pd

CURVATURE_COLUMNS = [
    "event_id", "src_id", "dst_id", "t", "forman", "orc", "escalated", "radius",
    "computed_at_stage", "ref_known",
]
# extra columns of the curvature stage's closed mode (pipeline.curvature_stage, option A of 2026-11-26): the reference
# overlap counts, the closed-form directed ORC and the flag for rows where its premise (uncited citing work) fails
ORC_DIR_COLUMNS = ["n_refs_dst", "n_shared", "orc_dir", "citing_cited_at_t"]
EVENT_COLUMNS = ["event_id", "src_id", "dst_id", "t"]
DAYS_EPOCH = np.datetime64("1970-01-01", "D")


def _canonical(events):
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events is missing column(s) {missing}; need {EVENT_COLUMNS}")
    ev = events[EVENT_COLUMNS].astype({"event_id": "int64", "t": "int64"})
    if ev["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    return ev.sort_values(["t", "event_id"], kind="mergesort").reset_index(drop=True)


def _count_upto(sorted_keys, node, tt, span):
    """Number of keys k with node*span <= k <= node*span + tt (per query row)."""
    base = node * span
    return np.searchsorted(sorted_keys, base + tt, "right") - np.searchsorted(sorted_keys, base, "left")


def _with_context(ev, context):
    """(all rows in canonical order, boolean mask of the scored rows). `ev` is canonical; context rows
    are validated like events and must not reuse an event id."""
    if context is None or len(context) == 0:
        if context is not None:
            _canonical(context)  # still reject a frame without the event columns
        return ev, np.ones(len(ev), dtype=bool)
    ctx = _canonical(context)
    if ctx["event_id"].isin(ev["event_id"]).any():
        raise ValueError("context rows must not reuse an event_id of events")
    rows = pd.concat([ev.assign(_scored=True), ctx.assign(_scored=False)], ignore_index=True)
    rows = rows.sort_values(["t", "event_id"], kind="mergesort").reset_index(drop=True)
    return rows[EVENT_COLUMNS], rows["_scored"].to_numpy(dtype=bool)


def forman_screen(events, *, count_duplicates=False, unknown_without_refs=False, context=None, return_degrees=False) -> pd.DataFrame:
    """Unweighted directed Forman curvature of every event at its own time.

    events: DataFrame with event_id (unique), src_id, dst_id, t (int days).
    count_duplicates: False (default) = distinct-neighbour degrees; True = row
    counts. See the module docstring for the self-loop / duplicate rules and
    the leak argument.
    unknown_without_refs: True writes NaN where out_deg_t(dst) is 0 (the module
    docstring, "Unknown values"); False (default) keeps F = 2 - in_deg_t(src).
    context: optional frame with the same four columns and event ids of its own
    (e.g. citecurv.data.cited.reference_events: the reference lists of the cited
    works). Its rows take part in the degrees exactly as event rows do (ts <= t,
    duplicate and self-loop rules) and are not scored: the result has the rows
    of `events` only and equals the screen of the two frames together,
    restricted to them.

    Returns the curvature-cache frame (CURVATURE_COLUMNS) in canonical order
    (t, event_id): forman filled, orc NaN, escalated False, radius NaN,
    computed_at_stage "screen", ref_known = the cited work has an out-edge at t
    (out_deg_t(dst) > 0 after the self-loop rule), whatever unknown_without_refs
    says; with it on, ref_known == ~isnan(forman).
    """
    ev = _canonical(events)
    n = len(ev)
    forman = np.empty(n, dtype=np.float64)
    ref_known = np.zeros(n, dtype=bool)
    in_deg = np.zeros(n, dtype=np.float64)
    out_deg = np.zeros(n, dtype=np.float64)
    if n:
        rows, scored = _with_context(ev, context)
        m = len(rows)
        codes, uniques = pd.factorize(pd.concat([rows["src_id"], rows["dst_id"]], ignore_index=True))
        s_all = codes[:m].astype(np.int64)
        d_all = codes[m:].astype(np.int64)
        tt_all = rows["t"].to_numpy(np.int64)
        tt_all = tt_all - tt_all.min()
        span = int(tt_all.max()) + 1
        if len(uniques) * span >= 2**62:
            raise ValueError("node count x time span too large for int64 composite keys")

        if count_duplicates:
            ps, pdst, pt = s_all, d_all, tt_all
        else:
            # rows are time-sorted, so keep="first" keeps each pair's first occurrence
            first = ~pd.DataFrame({"s": s_all, "d": d_all}).duplicated(keep="first").to_numpy()
            ps, pdst, pt = s_all[first], d_all[first], tt_all[first]
        in_keys = np.sort(pdst * span + pt)   # in-edges keyed by their head
        out_keys = np.sort(ps * span + pt)    # out-edges keyed by their tail
        # degrees are read for the scored rows only; every row (context included) is in the keys
        s, d, tt = s_all[scored], d_all[scored], tt_all[scored]
        in_src = _count_upto(in_keys, s, tt, span)
        out_dst = _count_upto(out_keys, d, tt, span)

        loop = s == d
        if loop.any():
            if count_duplicates:
                own = loop  # the row itself was counted once in each degree
            else:
                loop_all = s_all == d_all  # self-loop rows of the node, context rows included
                loop_keys = np.sort(s_all[loop_all] * span + tt_all[loop_all])
                n_loops_upto = _count_upto(loop_keys, s[loop], tt[loop], span)
                own = np.zeros(n, dtype=bool)
                own[np.flatnonzero(loop)[n_loops_upto == 1]] = True  # no other self-loop row
            in_src = in_src - own
            out_dst = out_dst - own
        forman[:] = 2.0 - in_src - out_dst
        ref_known[:] = out_dst > 0  # the cited work has a known reference at t (G13), after the self-loop rule
        in_deg[:] = in_src
        out_deg[:] = out_dst
        if unknown_without_refs:
            forman[out_dst == 0] = np.nan

    out = ev.copy()
    out["forman"] = forman
    out["orc"] = np.nan
    out["escalated"] = np.zeros(n, dtype=bool)
    out["radius"] = np.nan
    out["computed_at_stage"] = "screen"
    out["ref_known"] = ref_known
    if return_degrees:  # the two counts behind F, for the G7 counting controls (not cache columns)
        out["in_deg_src"] = in_deg
        out["out_deg_dst"] = out_deg
        return out[CURVATURE_COLUMNS + ["in_deg_src", "out_deg_dst"]]
    return out[CURVATURE_COLUMNS]


def spot_check(events, curv, n=20, seed=0, *, unknown_without_refs=False, context=None) -> pd.DataFrame:
    """Recompute F for n random events with direct boolean filtering (an
    independent code path, O(E) per event) and compare with curv["forman"].
    Uses the default distinct-neighbour semantics. unknown_without_refs: the
    option curv was computed with; when True the recomputed value is NaN where
    the cited work has no reference at t, and NaN matches NaN. context: the
    context rows curv was computed with (they are read, not sampled). S2.2 verify step."""
    ev = _canonical(events)
    if context is not None and len(context):
        ev = pd.concat([ev, _canonical(context)], ignore_index=True)
    rng = np.random.default_rng(seed)
    ids = rng.choice(curv["event_id"].to_numpy(), size=min(n, len(curv)), replace=False)
    cached = curv.set_index("event_id")["forman"]
    by_id = ev.set_index("event_id")
    rows = []
    for eid in ids:
        r = by_id.loc[eid]
        ctx = ev[(ev["t"] <= r["t"]) & (ev["event_id"] != eid)]
        in_s = ctx.loc[ctx["dst_id"] == r["src_id"], "src_id"].nunique()
        out_d = ctx.loc[ctx["src_id"] == r["dst_id"], "dst_id"].nunique()
        val = np.nan if unknown_without_refs and out_d == 0 else 2.0 - in_s - out_d
        got = float(cached.loc[eid])
        rows.append({"event_id": int(eid), "forman": got, "recomputed": val,
                     "match": bool(got == val or (np.isnan(got) and np.isnan(val)))})
    return pd.DataFrame(rows)


def _cut(values, pct_low, pct_high):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    return {"lo": float(np.quantile(v, pct_low)), "hi": float(np.quantile(v, pct_high)),
            "n_ref": int(v.size)}


def screen_thresholds(curv, ref_mask, pct_low, pct_high, group_col=None, *, inclusive=False) -> dict:
    """Two-sided percentile cut-offs on `forman`, fitted on reference rows only.

    pct_low / pct_high are quantile levels in [0, 1] (e.g. 0.01 / 0.99 for the
    extreme 1 % in each tail); numpy linear interpolation. ref_mask is a boolean
    array/Series aligned with curv marking the reference (training) rows.
    group_col: optional column (e.g. year or field) for per-group cut-offs;
    groups without reference rows fall back to the global cut-offs. Group keys
    are stored as str so the result round-trips through JSON.
    inclusive: False flags F < lo or F > hi; True flags F <= lo or F >= hi.
    Forman values are integers with heavy ties (F = 2 is the maximum), so the
    realised escalation rate can differ from the nominal tail mass in either
    mode; escalation_budget() reports the realised rate.

    Leak argument: only reference rows enter the quantiles; the dict is frozen
    and applied unchanged to validation/test rows.
    """
    if not 0.0 <= pct_low < pct_high <= 1.0:
        raise ValueError("need 0 <= pct_low < pct_high <= 1")
    ref = np.asarray(ref_mask, dtype=bool)
    if ref.shape != (len(curv),):
        raise ValueError("ref_mask must align with curv rows")
    glob = _cut(curv["forman"].to_numpy()[ref], pct_low, pct_high)
    if glob is None:
        raise ValueError("no reference rows with a forman value")
    th = {"pct_low": float(pct_low), "pct_high": float(pct_high), "inclusive": bool(inclusive),
          "group_col": group_col, "global": glob, "groups": {}}
    if group_col is not None:
        ref_rows = curv.loc[ref, [group_col, "forman"]]
        for g, sub in ref_rows.groupby(group_col, sort=True):
            cut = _cut(sub["forman"], pct_low, pct_high)
            if cut is not None:
                th["groups"][str(g)] = cut
    return th


def apply_screen(curv, thresholds) -> pd.DataFrame:
    """Copy of curv with `escalated` set from frozen thresholds (screen_thresholds).
    Rows with NaN forman are never escalated. Other columns are unchanged."""
    out = curv.copy()
    f = out["forman"].to_numpy(dtype=float)
    lo = np.full(len(out), thresholds["global"]["lo"])
    hi = np.full(len(out), thresholds["global"]["hi"])
    gcol = thresholds.get("group_col")
    if gcol is not None and thresholds["groups"]:
        keys = out[gcol].astype(str)
        groups = thresholds["groups"]
        lo = np.array(keys.map({g: c["lo"] for g, c in groups.items()}), dtype=float)
        hi = np.array(keys.map({g: c["hi"] for g, c in groups.items()}), dtype=float)
        missing = np.isnan(lo)  # group without reference rows -> global cut-offs
        lo[missing] = thresholds["global"]["lo"]
        hi[missing] = thresholds["global"]["hi"]
    with np.errstate(invalid="ignore"):
        if thresholds.get("inclusive"):
            flag = (f <= lo) | (f >= hi)
        else:
            flag = (f < lo) | (f > hi)
    out["escalated"] = flag & ~np.isnan(f)
    return out


def _pct_label(q):
    return f"p{100 * q:g}"


REPORT_PERCENTILES = (0.0, 0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.95, 0.98, 0.99, 1.0)


def screen_report(curv, by=None, percentiles=REPORT_PERCENTILES) -> pd.DataFrame:
    """Distribution of `forman`: n, mean, std and the requested percentiles.

    by: None (one row "all"), a column name of curv, or "year" (derived from t,
    days since 1970-01-01, when curv has no "year" column). Raw material for
    the S2.3 threshold choice.
    """
    df = curv
    if by is None:
        key = pd.Series("all", index=curv.index)
    elif by in curv.columns:
        key = curv[by]
    elif by == "year":
        key = pd.Series((DAYS_EPOCH + curv["t"].to_numpy(np.int64)).astype("datetime64[Y]").astype(int) + 1970,
                        index=curv.index)
    else:
        raise ValueError(f"unknown grouping {by!r}")
    rows = {}
    for g, sub in df["forman"].groupby(key, sort=True, dropna=False):  # NaN group kept
        v = sub.dropna().to_numpy(float)
        row = {"n": int(v.size), "mean": float(v.mean()) if v.size else np.nan,
               "std": float(v.std(ddof=1)) if v.size > 1 else np.nan}
        for q in percentiles:
            row[_pct_label(q)] = float(np.quantile(v, q)) if v.size else np.nan
        rows[g] = row
    rep = pd.DataFrame.from_dict(rows, orient="index")
    rep.index.name = by or "group"
    return rep

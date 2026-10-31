"""Statistical baseline tier, CATCHSYNC-inspired (task S4.4, decision L17).

Four per-event features, each usable as a scoring rule on its own, plus a rank-average combination. CATCHSYNC
(Jiang et al., 2014) flags nodes whose neighbours behave in a synchronised and atypical way; the features below
carry that idea to single citation events with plain counting statistics.

==============  =======================================================================================
feature         definition for event e = (src cites dst at t); higher = more anomalous
==============  =======================================================================================
self_citation   1.0 if src and dst share at least one author, 0.0 if not, NaN if either paper has no
                author rows or no ``authorships`` table is given.
venue_gini      Gini coefficient of the citing-venue distribution of dst over the whole venue universe
                (V = distinct non-null ``works.venue_id``), using the citations of dst with ts <= t except
                e itself. Citations from papers without a venue are skipped. NaN below ``min_citers``.
burst_z         Poisson z-score of dst's citation count in the trailing window (t - W, t] (e excluded)
                against dst's own rate before the window. NaN when dst has no citation at ts <= t - W.
ref_overlap     max Jaccard between src's reference list and the reference lists of up to
                ``max_candidates`` most recent earlier citers of dst, with dst removed from both lists.
                NaN when dst has no other citer at ts <= t.
==============  =======================================================================================

Timestamp policy (CONTRACTS.md): every feature of e uses graph state with ts <= t and never e's own row. Other
events with the same t (other papers published the same day, or src's other references) are visible.

Streaming, not static: unlike the PyGOD and CurvGAD tiers, which see only the cutoff snapshot, these features are
computed over the full event stream (train, val and test together) exactly as the TGN sees earlier test events
when it scores later ones. Compute them once over all events, then select the rows to score.

Scale: all four features are O(E log E) plus O(E * max_candidates) set intersections for ref_overlap. Nothing is
quadratic in E; the per-dst loops touch each event a bounded number of times.

Typical use::

    feats = compute_features(events, works, authorships)
    test = feats[feats.event_id.isin(test_ids)]
    ref = feats[feats.event_id.isin(train_ids)]
    combined = rank_average(test, reference=ref)          # DataFrame(event_id, score)
    one_rule = rule_scores(test, "burst_z", reference=ref)  # DataFrame(event_id, score)
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ("self_citation", "venue_gini", "burst_z", "ref_overlap")
EVENT_COLUMNS = ("event_id", "src_id", "dst_id", "t")


@dataclass
class _Prepared:
    event_id: np.ndarray  # int64, input row order
    src: np.ndarray  # int64 node codes
    dst: np.ndarray
    t: np.ndarray  # int64 days
    src_ids: np.ndarray  # original ids (object)
    dst_ids: np.ndarray
    n_nodes: int


def _prepare(events: pd.DataFrame) -> _Prepared:
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events is missing columns {missing}; expected {list(EVENT_COLUMNS)}")
    if events["event_id"].duplicated().any():
        raise ValueError("events.event_id must be unique")
    if events["src_id"].isna().any() or events["dst_id"].isna().any():
        raise ValueError("events.src_id / dst_id must not be null")
    n = len(events)
    codes, uniques = pd.factorize(pd.concat([events["src_id"], events["dst_id"]], ignore_index=True))
    codes = codes.astype(np.int64)
    return _Prepared(
        event_id=events["event_id"].to_numpy(np.int64),
        src=codes[:n],
        dst=codes[n:],
        t=events["t"].to_numpy(np.int64),
        src_ids=events["src_id"].to_numpy(object),
        dst_ids=events["dst_id"].to_numpy(object),
        n_nodes=len(uniques),
    )


def count_le(codes: np.ndarray, times: np.ndarray, q_codes: np.ndarray, q_times: np.ndarray) -> np.ndarray:
    """For each query i: number of events j with codes[j] == q_codes[i] and times[j] <= q_times[i].

    One sort plus two binary searches per query (keys = code * span + time), so O((E + Q) log E).
    """
    codes = np.asarray(codes, np.int64)
    times = np.asarray(times, np.int64)
    q_codes = np.asarray(q_codes, np.int64)
    q_times = np.asarray(q_times, np.int64)
    if len(codes) == 0:
        return np.zeros(len(q_codes), np.int64)
    tmin, tmax = int(times.min()), int(times.max())
    span = tmax - tmin + 2
    keys = np.sort(codes * span + (times - tmin))
    start = np.searchsorted(keys, q_codes * span - 1, side="right")
    rel = np.clip(q_times - tmin, -1, span - 2)
    stop = np.searchsorted(keys, q_codes * span + rel, side="right")
    return (stop - start).astype(np.int64)


# ---------------------------------------------------------------- (a) self-citation


def _self_citation(p: _Prepared, authorships: pd.DataFrame | None) -> np.ndarray:
    out = np.full(len(p.event_id), np.nan)
    if authorships is None:
        return out
    auth = authorships.dropna(subset=["work_id", "author_id"])
    sets = auth.groupby("work_id")["author_id"].agg(frozenset).to_dict()
    for i, (s, d) in enumerate(zip(p.src_ids, p.dst_ids)):
        a = sets.get(s)
        b = sets.get(d)
        if a is not None and b is not None:
            out[i] = 0.0 if a.isdisjoint(b) else 1.0
    return out


def self_citation(events: pd.DataFrame, authorships: pd.DataFrame | None) -> np.ndarray:
    """1.0 if src and dst share an author, 0.0 if not, NaN if unknown (array in input row order).

    authorships: DataFrame (work_id, author_id), one row per (work, author); None -> all NaN.

    Leak argument: the author lists of src and dst are attributes of the two endpoint papers, fixed when each
    paper is published (dst at or before t, src at t). No other event, label or later state is read, so the
    value at t is the value any observer at t could compute.
    """
    return _self_citation(_prepare(events), authorships)


# ---------------------------------------------------------------- (b) citing-venue concentration


def gini_from_counts(counts, n_venues: int) -> float:
    """Gini coefficient over ``n_venues`` categories: the positive ``counts`` plus (n_venues - k) zeros.

    Uses sum_ij |x_i - x_j| = 2 sum_i (2i - k - 1) x_(i) over the sorted observed counts, and the zero padding
    adds 2 (V - k) n. Equivalent to 1 - k (1 - G_obs) / V where G_obs is the Gini over observed venues only.
    """
    s = np.sort(np.asarray(counts, dtype=float))
    k = len(s)
    n = float(s.sum())
    if n <= 0 or n_venues <= 0:
        return float("nan")
    if k > n_venues:
        raise ValueError(f"{k} observed venues exceed the universe size {n_venues}")
    i = np.arange(1, k + 1)
    d_v = 2.0 * float(np.sum((2 * i - k - 1) * s)) + 2.0 * (n_venues - k) * n
    return d_v / (2.0 * n_venues * n)


def _venue_codes(p: _Prepared, works: pd.DataFrame) -> tuple[np.ndarray, int]:
    w = works.dropna(subset=["venue_id"]).drop_duplicates("work_id")
    vcode, vuniq = pd.factorize(w["venue_id"])
    lookup = dict(zip(w["work_id"], vcode))
    src_v = np.fromiter((lookup.get(s, -1) for s in p.src_ids), dtype=np.int64, count=len(p.src_ids))
    return src_v, len(vuniq)


def _venue_gini(p: _Prepared, works: pd.DataFrame | None, min_citers: int) -> np.ndarray:
    n = len(p.event_id)
    out = np.full(n, np.nan)
    if works is None or n == 0:
        return out
    src_v, V = _venue_codes(p, works)
    if V == 0:
        return out
    min_citers = max(int(min_citers), 1)
    order = np.lexsort((p.t, p.dst))
    d_s, t_s, v_s = p.dst[order], p.t[order], src_v[order]
    new_block = np.r_[True, (d_s[1:] != d_s[:-1]) | (t_s[1:] != t_s[:-1])]
    new_group = np.r_[True, d_s[1:] != d_s[:-1]]
    starts = np.flatnonzero(new_block)
    ends = np.r_[starts[1:], n]
    v_list = v_s.tolist()
    # Per dst: counts per venue, the sorted list of positive counts, n, and D = sum_ij |x_i - x_j| over the
    # zero-padded universe. Raising venue a from c to c+1 changes D by 2 (V - 1 - 2 #{x_j > c}); the sorted list
    # stays sorted by replacing the last c with c+1 (or prepending a 1 for a new venue). O(log k) per event.
    counts: dict[int, int] = {}
    s: list[int] = []
    n_known = 0
    d_v = 0
    for lo, hi in zip(starts.tolist(), ends.tolist()):
        if new_group[lo]:
            counts, s, n_known, d_v = {}, [], 0, 0
        for j in range(lo, hi):  # the whole same-t block enters the state first
            v = v_list[j]
            if v < 0:
                continue
            c = counts.get(v, 0)
            pos = bisect_right(s, c)
            d_v += 2 * (V - 1 - 2 * (len(s) - pos))
            if c == 0:
                s.insert(0, 1)
            else:
                s[pos - 1] = c + 1
            counts[v] = c + 1
            n_known += 1
        if n_known == 0:
            continue
        k = len(s)
        for j in range(lo, hi):  # then each event is scored with its own row left out
            v = v_list[j]
            if v >= 0:
                g = k - bisect_left(s, counts[v])  # venues with count >= c, v included
                d_loo, n_loo = d_v - 2 * (V - 2 * g + 1), n_known - 1
            else:
                d_loo, n_loo = d_v, n_known
            if n_loo >= min_citers:
                out[order[j]] = d_loo / (2.0 * V * n_loo)
    return out


def venue_gini(events: pd.DataFrame, works: pd.DataFrame | None, min_citers: int = 3) -> np.ndarray:
    """Gini coefficient of the venues of papers that cited dst at ts <= t, e excluded (array, input order).

    The Gini is taken over the full venue universe (V = distinct non-null works.venue_id), with zero counts for
    venues that never cited dst. A Gini over the observed venues only would be 0 when every citer comes from
    one venue, which is the most concentrated case; the zero-padded version gives 1 - 1/V there and decreases as
    citations spread over more venues. V is a constant of the dataset, so it shifts values but never changes
    the ranking between events. Same-t blocks are handled by adding the whole block, then removing each event's
    own contribution in closed form (O(log k) per event). NaN when fewer than ``min_citers`` known-venue citers
    remain, or when ``works`` is None.

    Leak argument: the state for e is the multiset of venues of papers citing dst with ts <= t, minus e's own
    row; a citing paper's venue is fixed at its publication (<= t). V is computed from static metadata and does
    not affect ranks. Events after t never enter the counts (the per-dst sweep runs in time order and stops the
    state at the current timestamp block).
    """
    return _venue_gini(_prepare(events), works, min_citers)


# ---------------------------------------------------------------- (c) citation burst


def _burst_z(p: _Prepared, window_days: int) -> np.ndarray:
    W = int(window_days)
    if W <= 0:
        raise ValueError("window_days must be positive")
    n = len(p.event_id)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    le_t = count_le(p.dst, p.t, p.dst, p.t)
    le_w = count_le(p.dst, p.t, p.dst, p.t - W)
    first = np.full(p.n_nodes, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, p.dst, p.t)
    in_window = le_t - le_w - 1  # minus the scored row
    hist = le_w
    ok = hist > 0
    exposure_windows = np.maximum(((p.t - W) - first[p.dst]) / W, 1.0)
    lam = hist[ok] / exposure_windows[ok]
    out[ok] = (in_window[ok] - lam) / np.sqrt(lam)
    return out


def burst_z(events: pd.DataFrame, window_days: int = 365) -> np.ndarray:
    """Citation-burst z-score of dst at t (array, input row order).

    With W = window_days:
      C = #citations of dst with t - W < ts <= t, e excluded (the trailing window);
      H = #citations of dst with ts <= t - W (history before the window);
      t0 = time of dst's first citation; exposure = max(((t - W) - t0) / W, 1) windows;
      lambda = H / exposure (expected citations per window); z = (C - lambda) / sqrt(lambda).
    No history (H = 0, i.e. every citation of dst so far lies inside the window) gives NaN: there is no own rate
    to compare against. The exposure floor of one window keeps a history shorter than W from inflating lambda.

    Leak argument: C and H count events into dst with ts <= t only, and C removes e itself; t0 is the earliest
    citation of dst, which is <= t - W whenever H > 0, so it lies in the visible past. Nothing after t is read.
    """
    return _burst_z(_prepare(events), window_days)


# ---------------------------------------------------------------- (d) reference-list overlap


def _ref_overlap(p: _Prepared, max_candidates: int) -> np.ndarray:
    K = int(max_candidates)
    if K <= 0:
        raise ValueError("max_candidates must be positive")
    n = len(p.event_id)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    # references of every node, sorted by t, as Python lists for fast slicing and bisect
    by_src = np.lexsort((p.t, p.src))
    ref_src = p.src[by_src]
    ref_t = p.t[by_src].tolist()
    ref_dst = p.dst[by_src].tolist()
    ref_start = np.searchsorted(ref_src, np.arange(p.n_nodes), side="left").tolist()
    ref_end = np.searchsorted(ref_src, np.arange(p.n_nodes), side="right").tolist()
    cache: dict[tuple[int, int], frozenset] = {}

    def refset(node: int, t: int) -> frozenset:
        lo, hi = ref_start[node], ref_end[node]
        cut = hi if ref_t[hi - 1] <= t else bisect_right(ref_t, t, lo, hi)
        key = (node, cut)
        got = cache.get(key)
        if got is None:
            got = frozenset(ref_dst[lo:cut])
            cache[key] = got
        return got

    # citers of every dst, sorted by (t, event_id)
    by_dst = np.lexsort((p.event_id, p.t, p.dst))
    d_s = p.dst[by_dst]
    starts = np.flatnonzero(np.r_[True, d_s[1:] != d_s[:-1]])
    ends = np.r_[starts[1:], n]
    for lo, hi in zip(starts.tolist(), ends.tolist()):
        pos = by_dst[lo:hi]
        g_t = p.t[pos].tolist()
        g_src = p.src[pos].tolist()
        for i in range(hi - lo):
            t_e, s = g_t[i], g_src[i]
            j = bisect_right(g_t, t_e)
            a = None
            best = -1.0
            seen: set[int] = set()
            for q in range(j - 1, -1, -1):
                c = g_src[q]
                if q == i or c == s or c in seen:
                    continue
                seen.add(c)
                if a is None:
                    a = refset(s, t_e)
                b = refset(c, t_e)
                inter = len(a & b) - 1  # dst is in both lists: remove it
                union = (len(a) - 1) + (len(b) - 1) - inter
                jac = inter / union if union > 0 else 0.0
                if jac > best:
                    best = jac
                if len(seen) == K:
                    break
            if seen:
                out[pos[i]] = best
    return out


def ref_overlap(events: pd.DataFrame, max_candidates: int = 20) -> np.ndarray:
    """Max Jaccard between src's references and those of earlier citers of dst (array, input row order).

    For e = (src -> dst at t): A = {x : src cites x at ts <= t} minus dst; candidates are the distinct papers
    p != src citing dst at ts <= t (rows other than e), ranked by their latest such row (t, event_id), and only
    the ``max_candidates`` most recent are kept (cost cap: O(max_candidates) set intersections per event);
    B_p = {x : p cites x at ts <= t} minus dst. Score = max_p |A & B_p| / |A | B_p| (0 when both lists are
    empty after removing dst). NaN when dst has no other citer at ts <= t. Coordinated citing, where several
    papers that cite dst also cite the same other papers, raises the value; this is the synchrony idea of
    CATCHSYNC carried to reference lists.

    Leak argument: A and every B_p are built from rows with ts <= t only (a paper's references share its
    publication date, so for a normal paper the whole list is visible from that date on); candidates are rows
    into dst with ts <= t, excluding e. Removing dst from both lists means the scored link itself never
    contributes to the overlap. No later citer of dst and no later reference is read.
    """
    return _ref_overlap(_prepare(events), max_candidates)


# ---------------------------------------------------------------- all features + scoring rules


def compute_features(
    events: pd.DataFrame,
    works: pd.DataFrame | None = None,
    authorships: pd.DataFrame | None = None,
    *,
    window_days: int = 365,
    min_venue_citers: int = 3,
    max_ref_candidates: int = 20,
) -> pd.DataFrame:
    """All tier-1 features: DataFrame(event_id, *FEATURES), rows in the input order of ``events``.

    events: (event_id, src_id, dst_id, t); works: (work_id, venue_id, ...) or None; authorships:
    (work_id, author_id) or None. Leak arguments are in each feature function's docstring.
    """
    p = _prepare(events)
    return pd.DataFrame({
        "event_id": p.event_id,
        "self_citation": _self_citation(p, authorships),
        "venue_gini": _venue_gini(p, works, min_venue_citers),
        "burst_z": _burst_z(p, window_days),
        "ref_overlap": _ref_overlap(p, max_ref_candidates),
    })


def ecdf_mid(values, reference) -> np.ndarray:
    """Mid-rank ECDF of ``values`` against ``reference``: (#ref < v + 0.5 #ref == v) / #ref; NaN -> 0.5.

    0.5 is the neutral value: a missing feature neither raises nor lowers an event's position.
    """
    values = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    ref = np.sort(ref[~np.isnan(ref)])
    out = np.full(len(values), 0.5)
    if len(ref) == 0:
        return out
    ok = ~np.isnan(values)
    lo = np.searchsorted(ref, values[ok], side="left")
    hi = np.searchsorted(ref, values[ok], side="right")
    out[ok] = (lo + 0.5 * (hi - lo)) / len(ref)
    return out


def rank_average(
    features: pd.DataFrame, columns=FEATURES, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Combined tier-1 score: mean over ``columns`` of the mid-rank ECDF; DataFrame(event_id, score).

    reference: rows whose feature distribution defines each ECDF. Pass the TRAIN rows so the transform is frozen
    (the score of one test event then does not depend on any other test event, the same rule the combiner
    follows with ZStats). None uses ``features`` itself, a transductive normalisation over the scored set.
    """
    ref = features if reference is None else reference
    parts = [ecdf_mid(features[c].to_numpy(float), ref[c].to_numpy(float)) for c in columns]
    score = np.mean(np.vstack(parts), axis=0) if parts else np.zeros(len(features))
    return pd.DataFrame({"event_id": features["event_id"].to_numpy(), "score": score})


def rule_scores(features: pd.DataFrame, column: str, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """One feature as a scoring rule: the raw value, NaN replaced by the reference median (0.0 if none).

    The raw value keeps the full resolution of the feature; the median fill places events with a missing
    feature in the middle of the ranking, matching the 0.5 used by ``rank_average``.
    """
    ref = (features if reference is None else reference)[column].to_numpy(float)
    ref = ref[~np.isnan(ref)]
    fill = float(np.median(ref)) if len(ref) else 0.0
    vals = features[column].to_numpy(float)
    score = np.where(np.isnan(vals), fill, vals)
    return pd.DataFrame({"event_id": features["event_id"].to_numpy(), "score": score})

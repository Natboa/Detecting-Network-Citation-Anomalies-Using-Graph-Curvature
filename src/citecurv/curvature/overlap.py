"""Reference overlap of a citation event at its own time (bibliographic coupling share).

For every citation event e = (u -> v, t), on the graph formed by every OTHER row with
ts <= t (the same graph state as the Forman screen and EventStore.subgraph_at: rows of other
events at the same t are included, the scored row itself is not):

    n_refs_dst(e) = |succ_t(v)|                 distinct references of the cited work
    n_shared(e)   = |succ_t(u) & succ_t(v)|     references the two works have in common
    overlap(e)    = n_shared(e) / n_refs_dst(e)

n_refs_dst is out_deg_t(dst) of the Forman screen, so the two modules agree on which events
have a value. overlap is the share of the cited work's reference list that the citing work
also cites. When the citing work has no citer at t, the directed Ollivier-Ricci curvature of
the event is a function of this share alone (curvature formulation study, Proposition 2;
citecurv.curvature.variants.orc_directed_uncited): kappa = -(1 - alpha) * (1 - overlap).

Unknown values. overlap is NaN when n_refs_dst = 0: the event table cannot tell a cited work
without references from one whose reference list is not known (the same reading as the
screen's unknown_without_refs rule, ledger G13). It is NaN for a self-loop row (u == v) as
well, where there is no pair of works to compare; n_shared is 0 on such a row.

Edge cases, handled explicitly:

- Duplicate (x, y) rows count once, from their first occurrence on (the successor set of a
  simple DiGraph).
- The scored row u -> v is removed. It could only enter n_shared through w = v, that is when
  the cited work cites itself (a row v -> v with ts <= t); v is then a shared reference only
  if another u -> v row with ts <= t exists. For a self-loop row, n_refs_dst leaves the row
  itself out and counts v only if another v -> v row with ts <= t exists.

Context rows (`context`, e.g. citecurv.data.cited.reference_events: the reference lists of
the cited works) take part in the successor sets exactly as event rows do and are not scored.

Leak argument (NFR3): both counts read only rows with ts <= t(e) and never the scored row;
nothing after t(e) enters. A reference list dated after t(e) is not visible at t(e).

Cost: one factorisation, then one binary search per (event, reference of the cited work)
pair, done in chunks of at most `chunk_pairs` pairs so the working arrays stay bounded.
"""

import numpy as np
import pandas as pd

from citecurv.curvature.screen import _canonical, _with_context

OVERLAP_COLUMNS = ["event_id", "src_id", "dst_id", "t", "n_refs_dst", "n_shared", "overlap"]
_NEVER = np.iinfo(np.int64).max


def _other_row_upto(s_all, d_all, tt_all, has_loop, n_nodes, s, d, tt):
    """For the query rows (s, d, tt): is there a second row of the pair (s, d) with ts <= tt?
    With the query row itself among the rows, that is "another row of the pair at or before tt".
    Only pairs whose head has a self-loop are looked at (the caller asks about no other)."""
    sub = np.flatnonzero(has_loop[d_all])
    rows = pd.DataFrame({"k": s_all[sub] * n_nodes + d_all[sub], "t": tt_all[sub]})  # time order kept
    second = rows[rows.groupby("k", sort=False).cumcount() == 1].set_index("k")["t"]
    t2 = pd.Series(s * n_nodes + d).map(second).to_numpy(dtype=float)  # NaN: the pair has one row
    return t2 <= tt


def reference_overlap(events, *, context=None, chunk_pairs=5_000_000) -> pd.DataFrame:
    """Reference overlap of every event at its own time (module docstring).

    events: DataFrame with event_id (unique), src_id, dst_id, t (int days).
    context: optional frame with the same four columns and event ids of its own; its rows
    enter the successor sets and are not scored.
    chunk_pairs: upper bound on the (event, reference) pairs expanded at once; it changes
    memory use, never the result.

    Returns OVERLAP_COLUMNS for the rows of `events` in canonical order (t, event_id):
    n_refs_dst and n_shared int64, overlap float (NaN = unknown).
    """
    if chunk_pairs < 1:
        raise ValueError("chunk_pairs must be at least 1")
    ev = _canonical(events)
    n = len(ev)
    n_refs = np.zeros(n, dtype=np.int64)
    n_shared = np.zeros(n, dtype=np.int64)
    share = np.full(n, np.nan)
    if n:
        rows, scored = _with_context(ev, context)
        m = len(rows)
        codes, uniques = pd.factorize(pd.concat([rows["src_id"], rows["dst_id"]], ignore_index=True))
        n_nodes = len(uniques)
        if n_nodes * n_nodes >= 2**62:
            raise ValueError("node count too large for int64 pair keys")
        s_all = codes[:m].astype(np.int64)
        d_all = codes[m:].astype(np.int64)
        tt_all = rows["t"].to_numpy(np.int64)

        # distinct pairs with the time of their first row (rows are time-sorted)
        first = ~pd.DataFrame({"s": s_all, "d": d_all}).duplicated(keep="first").to_numpy()
        ps, pdst, pt = s_all[first], d_all[first], tt_all[first]
        # references of every node, stored side by side per citing node
        order = np.argsort(ps, kind="stable")
        ref_w, ref_t = pdst[order], pt[order]
        indptr = np.searchsorted(ps[order], np.arange(n_nodes + 1), "left")

        s, d, tt = s_all[scored], d_all[scored], tt_all[scored]
        # lookup table "does x cite w, and since when" for the citing works of the scored rows
        is_src = np.zeros(n_nodes, dtype=bool)
        is_src[s] = True
        keep = is_src[ps]
        key = ps[keep] * n_nodes + pdst[keep]
        key_order = np.argsort(key, kind="stable")
        key_sorted, key_t = key[key_order], pt[keep][key_order]
        del first, order, keep, key, key_order

        deg = indptr[d + 1] - indptr[d]  # references of the cited work over all time
        cum = np.cumsum(deg)
        start = 0
        while start < n:
            done = int(cum[start - 1]) if start else 0
            stop = int(np.searchsorted(cum, done + chunk_pairs, "right"))
            stop = min(max(stop, start + 1), n)
            cnt = deg[start:stop]
            total = int(cnt.sum())
            if total:
                k = stop - start
                ev_of = np.repeat(np.arange(k), cnt)
                run_start = np.cumsum(cnt) - cnt
                pos = np.repeat(indptr[d[start:stop]] - run_start, cnt) + np.arange(total)
                t_ev = tt[start:stop][ev_of]
                visible = ref_t[pos] <= t_ev
                query = s[start:stop][ev_of] * n_nodes + ref_w[pos]
                j = np.minimum(np.searchsorted(key_sorted, query, "left"), len(key_sorted) - 1)
                hit = visible & (key_sorted[j] == query) & (key_t[j] <= t_ev)
                n_refs[start:stop] = np.bincount(ev_of[visible], minlength=k)
                n_shared[start:stop] = np.bincount(ev_of[hit], minlength=k)
            start = stop

        # the scored row is not part of the graph: undo what it alone contributed
        loop_t = np.full(n_nodes, _NEVER, dtype=np.int64)
        is_loop_pair = ps == pdst
        loop_t[ps[is_loop_pair]] = pt[is_loop_pair]
        touched = np.flatnonzero(loop_t[d] <= tt)  # the cited work cites itself at or before t
        if touched.size:
            other = _other_row_upto(s_all, d_all, tt_all, loop_t < _NEVER, n_nodes,
                                    s[touched], d[touched], tt[touched])
            own_loop = s[touched] == d[touched]
            n_shared[touched[~own_loop & ~other]] -= 1  # v was "shared" through the scored row only
            n_refs[touched[own_loop & ~other]] -= 1     # the row was its own reference
        self_loop = s == d
        n_shared[self_loop] = 0
        has_value = (n_refs > 0) & ~self_loop
        share[has_value] = n_shared[has_value] / n_refs[has_value]

    out = ev.copy()
    out["n_refs_dst"] = n_refs
    out["n_shared"] = n_shared
    out["overlap"] = share
    return out[OVERLAP_COLUMNS]

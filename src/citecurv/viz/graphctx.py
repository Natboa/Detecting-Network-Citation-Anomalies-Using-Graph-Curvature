"""Graph context for the explorer's detail view: the local subgraph at time t and the cited paper's timeline.

`TemporalIndex` stores every citation event twice (by citing node and by cited node) as CSR arrays sorted by
(node, t, event_id), so "neighbours of x with ts <= t" is one binary search. It is the display-side twin of
`citecurv.curvature.timeslice.EventStore`: same graph-at-t semantics, plus a node cap (a hub can have thousands of
citers, too many to draw) and event ids, so the scored row is excluded by identity.

Leak argument (NFR3, applied to what the page shows): `local_subgraph` and the degrees behind `forman_t` read only
events with ts <= t and drop the scored event's own row (by event_id), the same-timestamp policy in CONTRACTS.md.
`received_timeline` also reads only ts <= t; it keeps the scored citation itself, because the timeline exists to show
the burst the event belongs to (the page says so). Nothing after t is ever read.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    raise ValueError(f"events need one of the columns {names}")


class TemporalIndex:
    """CSR adjacency over a citation event stream (src cites dst at t)."""

    def __init__(self, events: pd.DataFrame):
        src = _col(events, "src_id", "src").astype(str).to_numpy()
        dst = _col(events, "dst_id", "dst").astype(str).to_numpy()
        t = _col(events, "t").to_numpy(dtype=np.int64)
        if "event_id" in events.columns:
            eid = events["event_id"].to_numpy(dtype=np.int64)
        else:
            eid = np.arange(len(events), dtype=np.int64)
        n = len(src)
        codes, uniques = pd.factorize(np.concatenate([src, dst]))
        self.ids = np.asarray(uniques, dtype=object)
        self.code = {x: i for i, x in enumerate(self.ids)}
        s_code, d_code = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
        n_nodes = len(self.ids)

        def csr(key, other):
            order = np.lexsort((eid, t, key))
            ptr = np.searchsorted(key[order], np.arange(n_nodes + 1))
            return ptr, t[order], other[order], eid[order]

        self.out_ptr, self.out_t, self.out_nbr, self.out_eid = csr(s_code, d_code)
        self.in_ptr, self.in_t, self.in_nbr, self.in_eid = csr(d_code, s_code)

    def _upto(self, ptr, ts, node, t):
        lo, hi = ptr[node], ptr[node + 1]
        return lo, lo + int(np.searchsorted(ts[lo:hi], t, side="right"))

    def out_upto(self, node: int, t: int):
        lo, hi = self._upto(self.out_ptr, self.out_t, node, t)
        return self.out_nbr[lo:hi], self.out_t[lo:hi], self.out_eid[lo:hi]

    def in_upto(self, node: int, t: int):
        lo, hi = self._upto(self.in_ptr, self.in_t, node, t)
        return self.in_nbr[lo:hi], self.in_t[lo:hi], self.in_eid[lo:hi]

    def degree_upto(self, node: int, t: int, exclude_eid: int | None = None) -> tuple[int, int]:
        """(in, out) event counts with ts <= t, not counting event `exclude_eid`."""
        _, _, ie = self.in_upto(node, t)
        _, _, oe = self.out_upto(node, t)
        if exclude_eid is None:
            return len(ie), len(oe)
        return int((ie != exclude_eid).sum()), int((oe != exclude_eid).sum())


def _ring(idx: TemporalIndex, seeds, t, eid, exclude: set[int]) -> pd.DataFrame:
    """Neighbours of `seeds` (either direction, ts <= t, scored row dropped) outside `exclude`, with priority stats."""
    parts = []
    for s in seeds:
        for nbr, ts, ev in (idx.out_upto(s, t), idx.in_upto(s, t)):
            if len(nbr):
                parts.append(pd.DataFrame({"node": nbr, "ts": ts, "eid": ev, "seed": s}))
    if not parts:
        return pd.DataFrame(columns=["node", "n_seeds", "links", "last_ts"])
    df = pd.concat(parts, ignore_index=True)
    df = df[(df.eid != eid) & ~df.node.isin(list(exclude))]
    if df.empty:
        return pd.DataFrame(columns=["node", "n_seeds", "links", "last_ts"])
    g = df.groupby("node").agg(n_seeds=("seed", "nunique"), links=("eid", "size"), last_ts=("ts", "max"))
    g = g.reset_index()
    g["id"] = idx.ids[g.node.to_numpy()].astype(str)
    # priority: touches more seeds, more links, more recent, then id (deterministic)
    g = g.sort_values(["n_seeds", "links", "last_ts", "id"], ascending=[False, False, False, True], kind="mergesort")
    return g


def local_subgraph(idx: TemporalIndex, event_id: int, u: str, v: str, t: int, *, radius: int = 1,
                   max_nodes: int | None = 40) -> dict:
    """Radius-hop neighbourhood of the scored citation u -> v as it stood at time t.

    Hops traverse edges in either direction; edges keep their direction. Events with ts > t and the scored row
    itself are never read (module docstring). With `max_nodes`, the core (u, v) is always kept; ring-1 neighbours
    are ranked by (number of core nodes they touch, links, most recent link, id) and ring 2 by the same statistics
    against the kept ring-1 nodes. With radius 2, ring 1 gets at most half the budget so ring 2 stays visible.

    Returns {"nodes": [{id, hop, in_deg, out_deg}], "edges": [{a, b, n, first_t, last_t, forman_t, scored}],
    "ring1_total", "ring1_shown", "ring2_total", "ring2_shown"}. `forman_t` is the unweighted directed Forman
    screen F(a -> b) = 2 - in_deg_t(a) - out_deg_t(b) with event counts at t (scored row excluded), the formula in
    CONTRACTS.md; the edge list has one entry per (a, b) pair plus the scored edge flagged `scored`.
    """
    core = list(dict.fromkeys(c for c in (idx.code.get(u), idx.code.get(v)) if c is not None))
    budget = None if max_nodes is None else max(0, max_nodes - 2)

    ring1 = _ring(idx, core, t, event_id, set(core))
    cap1 = budget if (budget is None or radius < 2) else budget // 2
    keep1 = ring1 if cap1 is None else ring1.head(cap1)
    ring2_total, keep2 = 0, ring1.iloc[0:0]
    if radius >= 2:
        ring2 = _ring(idx, keep1.node.tolist(), t, event_id, set(core) | set(ring1.node.tolist()))
        ring2_total = len(ring2)
        cap2 = None if budget is None else budget - len(keep1)
        if cap2 is not None and len(ring2) < cap2 and len(ring1) > len(keep1):
            # ring 2 is small: give the spare budget back to ring 1 (extra ring-1 nodes do not change ring 2's
            # candidate set semantics, which is always "neighbours of the shown ring-1 nodes")
            spare = cap2 - len(ring2)
            keep1 = ring1.head(len(keep1) + spare)
            cap2 = budget - len(keep1)
        keep2 = ring2 if cap2 is None else ring2.head(cap2)

    shown = core + keep1.node.tolist() + keep2.node.tolist()
    hop = {c: 0 for c in core}
    hop.update({c: 1 for c in keep1.node.tolist()})
    hop.update({c: 2 for c in keep2.node.tolist()})
    shown_set = set(shown)

    deg = {c: idx.degree_upto(c, t, exclude_eid=event_id) for c in shown}
    nodes = []
    for name in dict.fromkeys((str(u), str(v))):  # core first; endpoints unknown to the stream still appear
        c = idx.code.get(name)
        nodes.append({"id": name, "hop": 0, "in_deg": deg[c][0] if c is not None else 0,
                      "out_deg": deg[c][1] if c is not None else 0})
    nodes += [{"id": str(idx.ids[c]), "hop": hop[c], "in_deg": deg[c][0], "out_deg": deg[c][1]}
              for c in shown if hop[c] > 0]

    pairs: dict[tuple[int, int], list] = {}
    for a in shown:
        nbr, ts, ev = idx.out_upto(a, t)
        for b, tb, e in zip(nbr.tolist(), ts.tolist(), ev.tolist()):
            if e == event_id or b not in shown_set:
                continue
            p = pairs.get((a, b))
            if p is None:
                pairs[(a, b)] = [1, tb, tb]
            else:
                p[0] += 1
                p[1] = min(p[1], tb)
                p[2] = max(p[2], tb)

    def forman(a_name: str, b_name: str) -> int:
        a, b = idx.code.get(a_name), idx.code.get(b_name)
        in_a = deg[a][0] if a is not None else 0
        out_b = deg[b][1] if b is not None else 0
        return 2 - in_a - out_b

    pos = {c: i for i, c in enumerate(shown)}
    edges = []
    for (a, b) in sorted(pairs, key=lambda k: (pos[k[0]], pos[k[1]])):
        n, first, last = pairs[(a, b)]
        an, bn = str(idx.ids[a]), str(idx.ids[b])
        edges.append({"a": an, "b": bn, "n": n, "first_t": int(first), "last_t": int(last),
                      "forman_t": forman(an, bn), "scored": False})
    edges.append({"a": str(u), "b": str(v), "n": 1, "first_t": int(t), "last_t": int(t),
                  "forman_t": forman(str(u), str(v)), "scored": True})
    return {
        "nodes": nodes,
        "edges": edges,
        "ring1_total": int(len(ring1)),
        "ring1_shown": int(len(keep1)),
        "ring2_total": int(ring2_total),
        "ring2_shown": int(len(keep2)),
    }


def received_timeline(idx: TemporalIndex, v: str, t: int, *, max_bins: int = 48) -> dict:
    """Citations received by `v` with ts <= t, binned so the last bin ends on day t.

    Returns {"w": bin width (days), "end": t, "c": counts oldest -> newest, "n": total, "same_day": count at ts == t,
    "last30": count with t - 30 < ts <= t}. Includes the scored citation itself (module docstring).
    """
    c = idx.code.get(v)
    ts = idx.in_upto(c, t)[1] if c is not None else np.array([], dtype=np.int64)
    out = {"w": 1, "end": int(t), "c": [], "n": int(len(ts)), "same_day": int((ts == t).sum()),
           "last30": int((ts > t - 30).sum())}
    if len(ts) == 0:
        return out
    span_days = int(t - ts.min()) + 1
    w = max(1, math.ceil(span_days / max_bins))
    nb = math.ceil(span_days / w)
    k = (t - ts) // w  # 0 = bin ending at t
    counts = np.bincount(k, minlength=nb)[:nb][::-1]
    out.update({"w": int(w), "c": [int(x) for x in counts]})
    return out

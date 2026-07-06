"""EventStore: reconstruct graph state as it existed at time t (task S2.1).

No library computes curvature on "the graph as of time t" (verified
2026-07-02), so this module provides the two primitives the curvature stages
need:

- subgraph_at(u, v, t, radius): the union of radius-hop neighborhoods of u and
  v over events with ts <= t. Hops traverse edges in either direction; the
  returned graph keeps edge directions. The scored event's own instance —
  (u, v) at exactly ts == t — is excluded, so an edge never sees itself in its
  context (NFR3); an earlier (u, v) instance survives.
- degrees_at(t): in/out event counts with ts <= t, advanced incrementally.
  Calls must be non-decreasing in t (the Forman screen sweeps events in time
  order), giving O(new events) per advance.

Construction sorts events once; per-node adjacency is kept as parallel arrays
sorted by t, so time cuts are binary searches.
"""

from bisect import bisect_right
from collections import Counter

import networkx as nx


class EventStore:
    def __init__(self, events):
        """events: iterable of (src, dst, t) or a DataFrame with src/dst/t columns."""
        if hasattr(events, "itertuples"):
            rows = [(r.src, r.dst, int(r.t)) for r in events.itertuples(index=False)]
        else:
            rows = [(s, d, int(t)) for s, d, t in events]
        rows.sort(key=lambda e: e[2])
        self._events = rows

        # node -> ([ts...], [neighbor...]), each pair sorted by ts (build order)
        self._out: dict = {}
        self._in: dict = {}
        for s, d, t in rows:
            self._out.setdefault(s, ([], []))
            self._out[s][0].append(t)
            self._out[s][1].append(d)
            self._in.setdefault(d, ([], []))
            self._in[d][0].append(t)
            self._in[d][1].append(s)

        self._cursor = 0
        self._cursor_t = None
        self._in_deg = Counter()
        self._out_deg = Counter()

    def _neighbors_upto(self, node, t):
        for adj in (self._out, self._in):
            ts_list, nbrs = adj.get(node, ((), ()))
            yield from nbrs[: bisect_right(ts_list, t)]

    def subgraph_at(self, u, v, t, radius) -> nx.DiGraph:
        nodes = {u, v}
        frontier = {u, v}
        for _ in range(radius):
            nxt = set()
            for n in frontier:
                for nbr in self._neighbors_upto(n, t):
                    if nbr not in nodes:
                        nxt.add(nbr)
            nodes |= nxt
            frontier = nxt
            if not frontier:
                break

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for a in nodes:
            ts_list, nbrs = self._out.get(a, ((), ()))
            cut = bisect_right(ts_list, t)
            for ts_e, b in zip(ts_list[:cut], nbrs[:cut]):
                if b not in nodes:
                    continue
                if a == u and b == v and ts_e == t:
                    continue  # the scored instance itself
                if not G.has_edge(a, b):
                    G.add_edge(a, b, t=ts_e)
        return G

    def degrees_at(self, t):
        """(in_deg, out_deg) Counters over events with ts <= t. Monotonic in t."""
        if self._cursor_t is not None and t < self._cursor_t:
            raise ValueError(f"degrees_at is monotonic: got t={t} after t={self._cursor_t}")
        while self._cursor < len(self._events) and self._events[self._cursor][2] <= t:
            s, d, _ = self._events[self._cursor]
            self._out_deg[s] += 1
            self._in_deg[d] += 1
            self._cursor += 1
        self._cursor_t = t
        return self._in_deg, self._out_deg

"""Experimental curvature formulations for the curvature formulation study (agent W2c).

These functions exist to answer one question on SYNTHETIC data (docs/curvature_formulation_study.md):
which curvature formulations carry anomaly signal beyond trivial controls such as the cited
paper's age, its reference count or bibliographic coupling. They are reference
implementations, one event at a time, on small graphs; scripts/curvature_study.py has
vectorised fast paths that are cross-checked against them. Nothing here is wired into the
pipeline.

Input convention (event-time functions). For an event e = (u -> v, t), G is the graph state
at t with the scored row removed, as EventStore.subgraph_at(u, v, t, radius) returns it
(time_graph below). Radius 2 is exact for every function in this module: each formula reads
only nodes within two hops of u or v and the edges among them. A function that needs the
scored edge for its geometry (Ollivier-Ricci transport, Forman degrees) adds it to a copy.

Leak argument shared by all event-time functions (NFR3): G holds only rows with ts <= t,
same-t rows of other events included (CONTRACTS.md), the scored row excluded. Where a
formula adds the scored edge back, it adds exactly the event being scored (its endpoints and
its time), which every score component already uses; nothing with ts > t enters. The audit
functions (S7) break this rule on purpose and say so.

Signals (names as in the study):
  S2  orc_directed_plan / orc_directed_uncited: the plan's directed ORC (GraphRicciCurvature
      convention), general form and the closed form for an uncited citing paper.
  S3a orc_reversed, S3b forman_reversed: reversed orientation (u's references, v's citers).
  S4a orc_undirected, S4b forman_augmented: undirected ORC; augmented Forman with triangles.
  S5a coupling_forman, S5b cocitation_forman: augmented Forman of the pair (u, v) in the
      bibliographic-coupling / co-citation graph derived from G.
  S6a forman_dynamics; S6b/S6c arrival_curvature_shift (+ inedge_arrivals): curvature change
      around the cited paper over a trailing window.
  S7  audit_time_graph / audit_curvatures: AUDIT MODE, curvature at t + delta.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from citecurv.curvature.forman_directed import forman_edge

try:  # POT (exact network simplex) is optional; scipy's HiGHS LP is the fallback
    import ot as _ot
except ImportError:  # pragma: no cover - depends on the environment
    _ot = None

ALPHA = 0.5
DIST_CAP = 3  # every support pair of the ORC variants is joined by a path of length <= 3


# ----------------------------------------------------------------- graph state and transport


def time_graph(store, u, v, t, radius=2) -> nx.DiGraph:
    """Graph state at t around the event, scored row removed (EventStore.subgraph_at).

    Leak argument: subgraph_at keeps only rows with ts <= t and drops the scored instance
    (u, v, t); same-t rows of other events stay, as the contract allows.
    """
    return store.subgraph_at(u, v, t, radius)


def _with_edge(G, u, v):
    H = G.copy()
    if not H.has_edge(u, v):
        H.add_edge(u, v)
    return H


def wasserstein1(a, b, cost) -> float:
    """Exact optimal-transport cost between histograms a (n,) and b (m,) under cost (n, m).

    Uses POT's network simplex when POT is installed, otherwise scipy's HiGHS linear program;
    both are exact, so the value is the same within solver tolerance (about 1e-12).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cost = np.asarray(cost, dtype=float)
    if _ot is not None:
        return float(_ot.emd2(a, b, cost, numItermax=10_000_000))
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    n, m = cost.shape
    rows = np.r_[np.repeat(np.arange(n), m), n + np.tile(np.arange(m), n)]
    cols = np.r_[np.arange(n * m), np.arange(n * m)]
    A = coo_matrix((np.ones(2 * n * m), (rows, cols)), shape=(n + m, n * m))
    res = linprog(cost.ravel(), A_eq=A, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"transport LP failed: {res.message}")
    return float(res.fun)


def hop_distances(G, sources, targets, *, directed, cap=DIST_CAP) -> np.ndarray:
    """Matrix of min(hop distance, cap) from each source to each target.

    directed=True follows edge direction; False uses the underlying undirected graph. The
    cap is exact for the ORC supports of this module, where u - v gives every pair a path of
    length <= 3.
    """
    H = G if directed else G.to_undirected(as_view=True)
    D = np.full((len(sources), len(targets)), float(cap))
    for i, s in enumerate(sources):
        dist = nx.single_source_shortest_path_length(H, s, cutoff=cap)
        for j, x in enumerate(targets):
            if x in dist:
                D[i, j] = min(dist[x], cap)
    return D


def _measure(center, nbrs, alpha):
    """alpha on center, (1 - alpha) uniform on nbrs (deduplicated, center removed); a node
    without neighbours keeps all its mass (as in GraphRicciCurvature)."""
    nb = [x for x in dict.fromkeys(nbrs) if x != center]
    if not nb:
        return [center], np.array([1.0])
    return [center] + nb, np.r_[alpha, np.full(len(nb), (1.0 - alpha) / len(nb))]


def _orc(H, u, v, nbrs_u, nbrs_v, alpha, directed):
    su, a = _measure(u, nbrs_u, alpha)
    sv, b = _measure(v, nbrs_v, alpha)
    D = hop_distances(H, su, sv, directed=directed)
    return 1.0 - wasserstein1(a, b, D) / D[0, 0]  # D[0, 0] = d(u, v) = 1 (u -> v is in H)


def _check_pair(u, v):
    if u == v:
        raise ValueError(f"self-loop {(u, v)}: curvature of a self-loop is not defined")


# ----------------------------------------------------------------- S2: the plan's directed ORC


def orc_directed_plan(G, u, v, alpha=ALPHA) -> float:
    """S2. Directed Ollivier-Ricci curvature in the GraphRicciCurvature convention.

    m_u = alpha on u + (1 - alpha) uniform on pred(u) (papers citing u); m_v = alpha on v +
    (1 - alpha) uniform on succ(v) (v's references); directed hop metric on G + (u -> v);
    kappa = 1 - W1(m_u, m_v). Equals orc.orc_directed (the library) on the same graph; this
    own-transport version lets the study evaluate it cheaply and at t + delta (S7).
    Leak argument: module docstring (G at t, scored edge added for the geometry only; it is
    neither in pred(u) nor in succ(v)).
    """
    _check_pair(u, v)
    H = _with_edge(G, u, v)
    return _orc(H, u, v, list(H.predecessors(u)), list(H.successors(v)), alpha, directed=True)


def orc_directed_uncited(n_shared, n_refs_v, alpha=ALPHA):
    """S2 closed form (Proposition 2 of the study), vectorised.

    When the citing paper u has no citer at t (pred(u) empty), m_u is a Dirac at u, a
    reference w of v is at distance 1 from u if u also cites w and 2 otherwise (via v), so
        kappa(u, v) = -(1 - alpha) * (1 - n_shared / n_refs_v),   and 0 when n_refs_v = 0,
    with n_shared = |refs(u) & refs(v)| (bibliographic coupling) and n_refs_v = |refs(v)|.
    Leak argument: both counts are read at t without the scored row (v is not a reference
    of itself, so the scored row cannot enter n_shared).
    """
    s = np.asarray(n_shared, dtype=float)
    r = np.asarray(n_refs_v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(r > 0, -(1.0 - alpha) * (1.0 - s / np.where(r > 0, r, 1.0)), 0.0)
    return float(val) if val.ndim == 0 else val


# ----------------------------------------------------------------- S3: reversed orientation


def orc_reversed(G, u, v, alpha=ALPHA) -> float:
    """S3a. ORC with the opposite neighbourhoods: m_u on succ(u) (u's references, v among
    them), m_v on pred(v) (v's citers up to t, u among them).

    The metric is the hop distance of the underlying undirected graph of G + (u -> v): in a
    citation DAG a directed path from a reference w of u to a citer x of v exists only when w
    is newer than x and reaches it backwards in time, so for most support pairs the directed
    distance is infinite and W1 undefined. Positive values mean that v's earlier citers
    sit close to u's other references (they cite them, or are cited by them), the closure a
    citation group produces.
    Leak argument: module docstring. The scored edge enters both measures (v in succ(u), u in
    pred(v)); it carries only the event itself.
    """
    _check_pair(u, v)
    H = _with_edge(G, u, v)
    return _orc(H, u, v, list(H.successors(u)), list(H.predecessors(v)), alpha, directed=False)


def forman_reversed(G, u, v) -> float:
    """S3b. Reversed directed Forman: 2 - (#out-edges of u other than e) - (#in-edges of v
    other than e), i.e. 2 - (refs(u) - 1) - (citers_t(v) - 1) on G + (u -> v).

    It is a sum of two degree controls by construction (reported to show exactly that).
    Leak argument: module docstring.
    """
    _check_pair(u, v)
    H = _with_edge(G, u, v)
    return 2.0 - (H.out_degree(u) - 1) - (H.in_degree(v) - 1)


# ----------------------------------------------------------------- S4: undirected variants


def _undirected_nbrs(H, x):
    return (set(H.successors(x)) | set(H.predecessors(x))) - {x}


def orc_undirected(G, u, v, alpha=ALPHA) -> float:
    """S4a. Undirected Ollivier-Ricci curvature (Ollivier, 2009; Ni et al., 2015) of (u, v)
    on the underlying undirected graph of G + (u -> v): uniform (1 - alpha) mass on all
    neighbours, undirected hop metric. Equals GraphRicciCurvature on the undirected graph.
    Leak argument: module docstring.
    """
    _check_pair(u, v)
    H = _with_edge(G, u, v)
    nu = sorted(_undirected_nbrs(H, u), key=str)
    nv = sorted(_undirected_nbrs(H, v), key=str)
    return _orc(H, u, v, nu, nv, alpha, directed=False)


def forman_augmented(G, u, v) -> float:
    """S4b. Augmented Forman-Ricci curvature with triangles as 2-faces (Samal et al., 2018),
    unweighted: F = 4 - deg(u) - deg(v) + 3 * #triangles(u, v), on the underlying undirected
    graph of G + (u -> v). Equals GraphRicciCurvature's FormanRicci(method="augmented").
    In a citation graph the triangles of (u, v) are papers that u and v both cite (coupling),
    papers u cites that cite v, and papers that cite both (co-citation).
    Leak argument: module docstring.
    """
    _check_pair(u, v)
    H = _with_edge(G, u, v)
    nu, nv = _undirected_nbrs(H, u), _undirected_nbrs(H, v)
    tri = len((nu & nv) - {u, v})
    return 4.0 - len(nu) - len(nv) + 3.0 * tri


# ----------------------------------------------------------------- S5: derived graphs


def _coupling_nbrs(G, x):
    """Papers sharing at least one reference with x in G."""
    out = set()
    for r in G.successors(x):
        out.update(G.predecessors(r))
    out.discard(x)
    return out


def _cocitation_nbrs(G, x):
    """Papers cited together with x by at least one paper in G."""
    out = set()
    for z in G.predecessors(x):
        out.update(G.successors(z))
    out.discard(x)
    return out


def _pair_forman(nu, nv, u, v):
    """Augmented Forman of the virtual pair (u, v) given neighbour sets in a derived graph."""
    deg_u = len(nu | {v})
    deg_v = len(nv | {u})
    tri = len((nu & nv) - {u, v})
    return 4.0 - deg_u - deg_v + 3.0 * tri


def coupling_forman(G, u, v) -> float:
    """S5a. Augmented Forman curvature of the pair (u, v) in the bibliographic-coupling graph
    of G (papers linked when they share a reference; Kessler, 1963), unweighted, with (u, v)
    added as a virtual edge when they share no reference (the same convention as adding the
    scored edge for ORC):
        F = 4 - |N_BC(u) + {v}| - |N_BC(v) + {u}| + 3 |N_BC(u) & N_BC(v)|.
    Leak argument: G has no scored row, so u is never coupled to v's citers through v; every
    coupling link comes from reference lists with ts <= t. Radius 2 suffices (a coupled
    paper is two hops from u or v).
    """
    _check_pair(u, v)
    return _pair_forman(_coupling_nbrs(G, u), _coupling_nbrs(G, v), u, v)


def cocitation_forman(G, u, v) -> float:
    """S5b. Augmented Forman curvature of the pair (u, v) in the co-citation graph of G
    (papers linked when a third paper cites both; Small, 1973), same virtual-edge
    convention as S5a. At its own publication time the citing paper u is almost never cited,
    so N_CC(u) is empty and F = 2 - deg_CC(v): a property of v alone (Proposition 3).
    Leak argument: as S5a. Co-citation links come from citing papers with ts <= t; the scored
    row is not in G, so u's own reference to v never co-cites v with u's other references.
    """
    _check_pair(u, v)
    return _pair_forman(_cocitation_nbrs(G, u), _cocitation_nbrs(G, v), u, v)


# ----------------------------------------------------------------- S6: curvature dynamics


def _edge_t(G, a, b):
    return G.edges[a, b].get("t", -math.inf)


def _plan_forman_mean(G, v, s):
    """Mean plan-Forman 2 - in_deg_s(a) - out_deg_s(b) over v's incident edges with t <= s."""
    def in_deg(x):
        return sum(1 for p in G.predecessors(x) if _edge_t(G, p, x) <= s)

    def out_deg(x):
        return sum(1 for r in G.successors(x) if _edge_t(G, x, r) <= s)

    vals = [2.0 - in_deg(x) - out_deg(v) for x in G.predecessors(v) if _edge_t(G, x, v) <= s]
    vals += [2.0 - in_deg(v) - out_deg(r) for r in G.successors(v) if _edge_t(G, v, r) <= s]
    return float(np.mean(vals)) if vals else math.nan


def forman_dynamics(G, v, t, window) -> float:
    """S6a. Change of the mean plan-Forman curvature (2 - in_deg(a) - out_deg(b), Saucan et
    al., 2019) over the edges incident to the cited paper v, between the baseline time
    s = max(t - window, time of v's first incident edge) and t, each edge evaluated on the
    graph as it stood at that time:
        S6a = mean_{edges of v at t} F_t - mean_{edges of v at s} F_s.
    NaN when v has no incident edge in G. G must be time_graph(store, u, v, t, radius=2)
    (edge attribute "t" = first time of the edge): the degrees of v's neighbours are one hop
    further out.
    Leak argument: module docstring; every edge and degree is filtered to t' <= t, the scored
    row is not in G, and the baseline lies before t.
    """
    times = [_edge_t(G, x, v) for x in G.predecessors(v)] + [_edge_t(G, v, r) for r in G.successors(v)]
    times = [x for x in times if x <= t]
    if not times:
        return math.nan
    s = max(t - window, min(times))
    return _plan_forman_mean(G, v, t) - _plan_forman_mean(G, v, s)


def inedge_arrivals(store, u, v, t, kappa_fn, radius=2):
    """[(ts, kappa)] for every in-edge x -> v with ts <= t other than the scored row (u, v, t),
    kappa = kappa_fn(time_graph(store, x, v, ts, radius), x, v): each in-edge's curvature at
    its own arrival time, as an online system would have stored it.
    Leak argument: every value is computed on a graph with ts <= its arrival time <= t.
    """
    G1 = store.subgraph_at(u, v, t, 1)  # v's in-edges up to t, scored row excluded
    out = []
    for x in G1.predecessors(v):
        ts = int(G1.edges[x, v]["t"])
        out.append((ts, float(kappa_fn(time_graph(store, x, v, ts, radius), x, v))))
    return sorted(out)


def arrival_curvature_shift(arrivals, t, window) -> float:
    """S6b / S6c. Mean arrival-time curvature of v's in-edges that arrived in (t - window, t]
    minus the mean over in-edges that arrived at or before t - window; NaN when either side
    is empty. arrivals: [(ts, kappa)] from inedge_arrivals (scored row already excluded).
    Leak argument: reads only arrivals with ts <= t.
    """
    rec = [k for ts, k in arrivals if t - window < ts <= t]
    old = [k for ts, k in arrivals if ts <= t - window]
    if not rec or not old:
        return math.nan
    return float(np.mean(rec) - np.mean(old))


# ----------------------------------------------------------------- S7: AUDIT MODE


def audit_time_graph(store, u, v, t, delta, radius=2) -> nx.DiGraph:
    """AUDIT MODE - VIOLATES NFR3 BY DESIGN. Graph state at t + delta around the event, with
    the scored edge present (it happened at t). Valid only for retrospective auditing of
    citations that are at least `delta` old; never use it to score an event at its own time.
    """
    G = store.subgraph_at(u, v, t + delta, radius)
    if not G.has_edge(u, v):  # delta = 0: subgraph_at removed the scored instance itself
        G.add_edge(u, v, t=t)
    return G


def audit_curvatures(store, u, v, t, delta, alpha=ALPHA, radius=2) -> dict:
    """S7, AUDIT MODE - VIOLATES NFR3 BY DESIGN. The plan's Forman (2 - in_deg(u) -
    out_deg(v)) and directed ORC (orc_directed_plan) of (u, v) on the graph at t + delta.

    Leak caveat: at t + delta the citing paper u has been cited by later papers, among them,
    for a planted cartel, other cartel members; the value therefore uses information that did
    not exist at t. It answers "does this citation look anomalous in hindsight", not "is it
    anomalous when it happens". delta = 0 reproduces the event-time values (S1, S2).
    """
    G = audit_time_graph(store, u, v, t, delta, radius)
    return {"forman": float(forman_edge(G, u, v)), "orc_directed": orc_directed_plan(G, u, v, alpha)}

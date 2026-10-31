"""W2c curvature formulation study: experimental signals S2 (closed form), S3-S7.

Spec under test (src/citecurv/curvature/variants.py; docs/curvature_formulation_study.md):
- Every event-time function takes G = the graph state at t with the scored row removed
  (EventStore.subgraph_at(u, v, t, 2)), adds the scored edge itself only where its formula
  needs it for the transport geometry or the virtual pair, and reads nothing with ts > t.
- Ollivier-Ricci variants use alpha mass on the node and (1 - alpha) spread uniformly over its
  neighbour set; kappa = 1 - W1(m_u, m_v) / d(u, v), hop metric, d(u, v) = 1.
    S2  plan convention (GraphRicciCurvature): m_u on pred(u), m_v on succ(v), directed metric.
    S3a reversed: m_u on succ(u) (references), m_v on pred(v) (citers), undirected metric.
    S4a undirected: m_u, m_v on undirected neighbours, undirected metric.
- Forman variants: S3b 2 - (out(u) - 1) - (in(v) - 1); S4b augmented, 4 - deg(u) - deg(v)
  + 3 * triangles (Samal et al., 2018); S5a/S5b augmented Forman of the virtual pair (u, v) in
  the bibliographic-coupling / co-citation graph derived from G.
- S6a change of mean plan-Forman over v's incident edges between max(t - W, first edge of v)
  and t; S6b/c recent minus earlier mean of arrival-time curvature of v's in-edges.
- S7 (AUDIT MODE): the plan's Forman and directed ORC on the graph at t + delta.
Hand-computed values are derived in the comments of each test; real-library cross-checks use
pytest.importorskip("GraphRicciCurvature").
"""

import math
import random

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from citecurv.curvature import variants as V
from citecurv.curvature.timeslice import EventStore

ALPHA = 0.5


def dg(edges):
    """DiGraph from (a, b) or (a, b, t) tuples; t defaults to 0."""
    G = nx.DiGraph()
    for e in edges:
        a, b = e[0], e[1]
        G.add_edge(a, b, t=e[2] if len(e) > 2 else 0)
    return G


def store_of(rows):
    return EventStore(pd.DataFrame(rows, columns=["src_id", "dst_id", "t"]))


# ------------------------------------------------------------------ transport

def test_wasserstein1_hand_case():
    # two points at distance 2, all mass moves: W = 2; identical measures: W = 0
    assert V.wasserstein1(np.array([1.0]), np.array([1.0]), np.array([[2.0]])) == pytest.approx(2.0)
    a = np.array([0.5, 0.5])
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert V.wasserstein1(a, a, D) == pytest.approx(0.0, abs=1e-12)
    # mass 0.5 must cross a cost-3 link, the rest stays: W = 1.5
    assert V.wasserstein1(np.array([1.0, 0.0]), a, np.array([[0.0, 3.0], [3.0, 0.0]])) == pytest.approx(1.5)


def test_wasserstein1_scipy_path_matches(monkeypatch):
    rng = np.random.default_rng(0)
    a = rng.random(5); a /= a.sum()
    b = rng.random(7); b /= b.sum()
    D = rng.integers(0, 4, size=(5, 7)).astype(float)
    ref = V.wasserstein1(a, b, D)
    monkeypatch.setattr(V, "_ot", None)  # force the scipy LP
    assert V.wasserstein1(a, b, D) == pytest.approx(ref, abs=1e-9)


def test_hop_distances_directed_and_undirected():
    G = dg([("a", "b"), ("b", "c"), ("d", "c")])
    D = V.hop_distances(G, ["a", "d"], ["c", "b"], directed=True, cap=3)
    # a->b->c = 2, a->b = 1; d->c = 1, d cannot reach b along edges -> cap 3
    assert D.tolist() == [[2.0, 1.0], [1.0, 3.0]]
    U = V.hop_distances(G, ["a", "d"], ["c", "b"], directed=False, cap=3)
    assert U.tolist() == [[2.0, 1.0], [1.0, 2.0]]


# ------------------------------------------------------------------ S2

def uncited_case():
    # v cites w1..w4; u cites v (scored, not in G), w1, w2, z, w4  (test_orc.py hand case)
    return dg([("v", "w1"), ("v", "w2"), ("v", "w3"), ("v", "w4"),
               ("u", "w1"), ("u", "w2"), ("u", "z"), ("u", "w4")])


def test_s2_closed_form_hand_values():
    # Proposition 2: kappa = -(1 - alpha) (1 - shared / refs(v)); 0 when v cites nothing
    got = V.orc_directed_uncited(np.array([3, 0, 4, 0]), np.array([4, 4, 4, 0]), alpha=0.5)
    assert np.allclose(got, [-0.125, -0.5, 0.0, 0.0])
    assert V.orc_directed_uncited(1, 2, alpha=0.3) == pytest.approx(-0.35)


def test_s2_general_equals_closed_form_when_source_uncited():
    G = uncited_case()
    assert V.orc_directed_plan(G, "u", "v", ALPHA) == pytest.approx(-0.125, abs=1e-12)
    assert V.orc_directed_plan(G, "u", "v", 0.3) == pytest.approx(-0.7 * 0.25, abs=1e-12)
    G2 = dg([("u", "a")])
    G2.add_node("v")
    assert V.orc_directed_plan(G2, "u", "v", ALPHA) == pytest.approx(0.0, abs=1e-12)  # v cites nothing


def test_s2_general_with_cited_source():
    # p -> u (p cites u), u -> v scored, v -> w. m_u = .5 u + .5 p, m_v = .5 v + .5 w.
    # directed d: u->v 1, u->w 2, p->v 2 (via u), p->w 3. Best plan W = .5*1 + .5*3 = 2 = .5*2 + .5*2
    # kappa = 1 - 2 = -1. If p also cites w: p->w = 1, W = .5 + .5 = 1, kappa = 0.
    G = dg([("p", "u"), ("v", "w")])
    assert V.orc_directed_plan(G, "u", "v", ALPHA) == pytest.approx(-1.0)
    G.add_edge("p", "w", t=0)
    assert V.orc_directed_plan(G, "u", "v", ALPHA) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ S3

def reversed_case(closed):
    # u -> v (scored), u -> a; x -> v; optionally x -> a (u's reference cited by v's citer)
    edges = [("u", "a"), ("x", "v")] + ([("x", "a")] if closed else [])
    G = dg(edges)
    G.add_node("v")
    return G


def test_s3a_reversed_orc_hand_values():
    # m_u = .5 u + .25 v + .25 a ; m_v = .5 v + .25 u + .25 x (undirected hop metric)
    # closed (x - a adjacent): u->u .25, u->v .25 (1), v->v .25, a->x .25 (1): W = .5, kappa = .5
    # open: d(a, x) = 3 (a-u-v-x); dual f(a)=1, f(u)=0, f(v)=-1, f(x)=-2 certifies W = 1, kappa = 0
    assert V.orc_reversed(reversed_case(True), "u", "v", ALPHA) == pytest.approx(0.5)
    assert V.orc_reversed(reversed_case(False), "u", "v", ALPHA) == pytest.approx(0.0, abs=1e-12)


def test_s3b_reversed_forman_hand_value():
    # out(u) = {v, a}, in(v) = {u, x}: 2 - (2 - 1) - (2 - 1) = 0; adding u -> b gives -1
    G = reversed_case(True)
    assert V.forman_reversed(G, "u", "v") == 0.0
    G.add_edge("u", "b", t=0)
    assert V.forman_reversed(G, "u", "v") == -1.0


# ------------------------------------------------------------------ S4

def tree_case():
    # u -> v (scored), u -> a, v -> b, x -> v : a tree, N(u) = {v, a}, N(v) = {u, b, x}
    G = dg([("u", "a"), ("v", "b"), ("x", "v")])
    return G


def test_s4a_undirected_orc_hand_values():
    # tree: m_u = .5 u + .25 v + .25 a ; m_v = .5 v + 1/6 (u, b, x).
    # dual f(a)=1, f(u)=0, f(v)=-1, f(b)=f(x)=-2 is 1-Lipschitz and attains 7/6, and the plan
    # v->v .25, a->u 1/6, a->v 1/12, u->v 1/6, u->b 1/6, u->x 1/6 costs 7/6: kappa = -1/6.
    assert V.orc_undirected(tree_case(), "u", "v", ALPHA) == pytest.approx(-1.0 / 6.0)
    # triangle u -> v, u -> a, a -> v: m_u = .5u+.25v+.25a, m_v = .5v+.25u+.25a, W = .25
    tri = dg([("u", "a"), ("a", "v")])
    assert V.orc_undirected(tri, "u", "v", ALPHA) == pytest.approx(0.75)


def test_s4b_augmented_forman_hand_values():
    # tree: 4 - 2 - 3 + 0 = -1 ; triangle: 4 - 2 - 2 + 3 = 3
    assert V.forman_augmented(tree_case(), "u", "v") == -1.0
    assert V.forman_augmented(dg([("u", "a"), ("a", "v")]), "u", "v") == 3.0
    # reciprocal pair a <-> b counts once in the undirected view
    G = dg([("u", "a"), ("a", "v"), ("v", "a")])
    assert V.forman_augmented(G, "u", "v") == 3.0


# ------------------------------------------------------------------ S5

def coupling_case():
    # u cites r1, r2 (+ v, scored); v cites r1, r3; y cites r1; z cites r2; w cites r3
    return dg([("u", "r1"), ("u", "r2"), ("v", "r1"), ("v", "r3"),
               ("y", "r1"), ("z", "r2"), ("w", "r3")])


def test_s5a_coupling_forman_hand_value():
    # N_BC(u) = {v, y, z}, N_BC(v) = {u, y, w}; with the virtual pair deg 3 and 3, one
    # common neighbour y: 4 - 3 - 3 + 3*1 = 1
    G = coupling_case()
    assert V.coupling_forman(G, "u", "v") == 1.0
    # a citer x of v couples with u only through the scored row, which G does not hold
    G.add_edge("x", "v", t=0)
    assert V.coupling_forman(G, "u", "v") == 1.0


def test_s5b_cocitation_forman_hand_value():
    # u uncited: N_CC(u) = {}; x cites v, a, b: N_CC(v) = {a, b}. 4 - 1 - 3 + 0 = 0 = 2 - deg_CC(v)
    G = dg([("x", "v"), ("x", "a"), ("x", "b"), ("u", "a")])
    assert V.cocitation_forman(G, "u", "v") == 0.0
    # once u is cited together with a (by q), N_CC(u) = {a}: common {a}; deg 2 and 3:
    # 4 - 2 - 3 + 3 = 2
    G.add_edge("q", "u", t=0)
    G.add_edge("q", "a", t=0)
    assert V.cocitation_forman(G, "u", "v") == 2.0


# ------------------------------------------------------------------ S6

def dynamics_case():
    # times: r published -1 (cites nothing), v at 0 cites r, x1 at 1 and x2 at 5 cite v,
    # y at 4 cites x1, u at 6 cites v (scored; not in G)
    return dg([("v", "r", 0), ("x1", "v", 1), ("y", "x1", 4), ("x2", "v", 5)])


def test_s6a_forman_dynamics_hand_value():
    # t = 6, W = 3 -> baseline s = max(3, first edge of v = 0) = 3
    # at 6: F(x1->v) = 2-1-1 = 0, F(x2->v) = 2-0-1 = 1, F(v->r) = 2-2-0 = 0 -> mean 1/3
    # at 3: F(x1->v) = 2-0-1 = 1, F(v->r) = 2-1-0 = 1 -> mean 1 ; change = -2/3
    assert V.forman_dynamics(dynamics_case(), "v", 6, 3) == pytest.approx(-2.0 / 3.0)
    # window reaching before v's first edge: baseline is v's first edge time 0:
    # at 0: F(v->r) = 2 - 0 - 0 = 2 -> change 1/3 - 2
    assert V.forman_dynamics(dynamics_case(), "v", 6, 100) == pytest.approx(1.0 / 3.0 - 2.0)
    G = nx.DiGraph()
    G.add_node("v")
    assert math.isnan(V.forman_dynamics(G, "v", 6, 3))


def test_s6b_arrival_shift_hand_value():
    arr = [(1, 2.0), (4, 0.0), (5, 1.0)]
    assert V.arrival_curvature_shift(arr, 6, 3) == pytest.approx(0.5 - 2.0)
    assert math.isnan(V.arrival_curvature_shift(arr, 6, 10))   # no earlier arrival
    assert math.isnan(V.arrival_curvature_shift([(1, 2.0)], 6, 3))  # no recent arrival


def test_s6_inedge_arrivals_exclude_scored_row_and_future():
    rows = [("v", "r", 0), ("x1", "v", 1), ("y", "x1", 4), ("x2", "v", 5), ("u", "v", 6),
            ("late", "v", 9)]
    st = store_of(rows)
    arr = V.inedge_arrivals(st, "u", "v", 6, V.forman_augmented, radius=1)
    # x1->v at 1: deg(x1)=1, deg(v)=2 (r, x1), no triangle: 4-1-2 = 1
    # x2->v at 5: deg(x2)=1, deg(v)=3 (r, x1, x2): 4-1-3 = 0 ; u (scored) and late excluded
    assert sorted(arr) == [(1, 1.0), (5, 0.0)]


# ------------------------------------------------------------------ S7 (audit mode)

def test_s7_audit_mode_reads_the_future_on_purpose():
    # u at 10 cites v; v cites w (t 5); a cites u at 11, b cites u at 13.
    rows = [("v", "w", 5), ("u", "v", 10), ("a", "u", 11), ("b", "u", 13)]
    st = store_of(rows)
    out = V.audit_curvatures(st, "u", "v", 10, 2, alpha=ALPHA)
    # Forman at 12: 2 - in(u) {a} - out(v) {w} = 0 (event time: 2 - 0 - 1 = 1)
    assert out["forman"] == 0.0
    # m_u = .5 u + .5 a, m_v = .5 v + .5 w; d(u,v)=1, d(u,w)=2, d(a,v)=2, d(a,w)=3 -> W = 2
    assert out["orc_directed"] == pytest.approx(-1.0)
    # if a also cites w (at 11): d(a, w) = 1, W = .5 + .5 -> kappa 0
    st2 = store_of(rows + [("a", "w", 11)])
    assert V.audit_curvatures(st2, "u", "v", 10, 2, alpha=ALPHA)["orc_directed"] == pytest.approx(0.0, abs=1e-12)
    # delta = 0 is event time (the scored row stays in, nothing later enters)
    assert V.audit_curvatures(st, "u", "v", 10, 0, alpha=ALPHA)["forman"] == 1.0


# ------------------------------------------------------------------ leakage

def test_event_time_functions_ignore_later_rows():
    base = [("v", "r1", 1), ("v", "r3", 1), ("y", "r1", 2), ("z", "r2", 2), ("w", "r3", 3),
            ("x", "v", 4), ("u", "r1", 5), ("u", "r2", 5), ("u", "v", 5)]
    later = [("q", "u", 7), ("q", "v", 7), ("q", "r2", 7), ("s", "v", 8), ("s", "r1", 8)]
    vals = []
    for rows in (base, base + later):
        st = store_of(rows)
        G = V.time_graph(st, "u", "v", 5)
        assert not G.has_edge("u", "v")  # scored row removed
        vals.append([V.orc_directed_plan(G, "u", "v"), V.orc_reversed(G, "u", "v"),
                     V.forman_reversed(G, "u", "v"), V.orc_undirected(G, "u", "v"),
                     V.forman_augmented(G, "u", "v"), V.coupling_forman(G, "u", "v"),
                     V.cocitation_forman(G, "u", "v"), V.forman_dynamics(G, "v", 5, 2)])
    assert np.allclose(vals[0], vals[1], equal_nan=True)


# ------------------------------------------------------------------ real library cross-checks

def random_dag(seed, n=24, p=0.18):
    rng = random.Random(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i):
            if rng.random() < p:
                G.add_edge(i, j, t=i)  # later node cites earlier node
    return G


def test_generic_orc_matches_graphricci_directed_and_undirected():
    pytest.importorskip("GraphRicciCurvature")
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    from citecurv.curvature.orc import orc_directed

    for seed in range(3):
        G = random_dag(seed)
        edges = list(G.edges)[:15]
        lib = orc_directed(G, edges, alpha=ALPHA, method="OTD")
        U = G.to_undirected()
        orc_u = OllivierRicci(U, alpha=ALPHA, method="OTD", proc=1, verbose="ERROR")
        lib_u = orc_u.compute_ricci_curvature_edges(edges)
        for (a, b) in edges:
            H = G.copy()
            H.remove_edge(a, b)  # the functions take the graph without the scored row
            assert V.orc_directed_plan(H, a, b, ALPHA) == pytest.approx(lib[(a, b)], abs=1e-9)
            assert V.orc_undirected(H, a, b, ALPHA) == pytest.approx(lib_u[(a, b)], abs=1e-9)


def test_augmented_forman_matches_graphricci():
    pytest.importorskip("GraphRicciCurvature")
    from GraphRicciCurvature.FormanRicci import FormanRicci

    G = random_dag(5)
    frc = FormanRicci(G.to_undirected(), method="augmented", verbose="ERROR")
    frc.compute_ricci_curvature()
    for (a, b) in list(G.edges)[:30]:
        H = G.copy()
        H.remove_edge(a, b)
        assert V.forman_augmented(H, a, b) == pytest.approx(frc.G[a][b]["formanCurvature"])

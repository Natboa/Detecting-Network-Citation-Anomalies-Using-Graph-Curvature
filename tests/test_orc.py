"""S1.5 / S2.3: directed Ollivier-Ricci wrapper, escalation of flagged events, budget table.

Spec under test (IMPLEMENTATION_PLAN.md S1.5, S2.3; CONTRACTS.md "Curvature cache"):
- orc_directed(G, edges, ...) wraps GraphRicciCurvature.OllivierRicci on a DiGraph and
  returns {(u, v): kappa} for exactly the requested edges; GraphRicciCurvature is
  imported lazily with an actionable ImportError.
- escalate(...) fills `orc` only for escalated rows, on EventStore.subgraph_at(u, v, t,
  radius) plus the scored edge itself, enforcing edge_cap by the deterministic tiered
  truncation of cap_subgraph(), and reports stats.
- Directed ORC (alpha = 0.5, unweighted, OTD): the source measure sits on u and its
  predecessors, the target measure on v and its successors (GraphRicciCurvature
  0.5.3.2 source). Hand-computed values for the S1.4 toy graphs:
    isolated u->v: both measures are Diracs, W = d(u,v) = 1          -> 0
    chain a->b (b->c): m_a = d_a, m_b = .5 d_b + .5 d_c, W = .5 + 1     -> -0.5
    chain b->c (a->b): m_b = .5 d_b + .5 d_a, m_c = d_c, W = .5 + 1     -> -0.5
    3 in-edges at u: m_u = .5 d_u + 1/6 each x_i, m_v = d_v, W = 1.5   -> -0.5
    3-cycle a->b: m_a = .5 d_a + .5 d_c, m_b = .5 d_b + .5 d_c, W = .5  -> +0.5
    v cites 5: m_u = d_u, m_v = .5 d_v + .1 each r_i, W = .5 + 1        -> -0.5
  Flipping the tested edge leaves both endpoints without the relevant neighbours in
  every case, so the flipped edge has curvature 0 (changes in 5 of 6 cases).
Pure-python parts run in the main venv with an injected fake ORC function; real-ORC
tests use pytest.importorskip("GraphRicciCurvature") (curv venv only).
"""

import math
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from citecurv.curvature.orc import (
    BUDGET_COLUMNS,
    cap_subgraph,
    edge_tiers,
    escalate,
    escalation_budget,
    orc_directed,
)
from citecurv.curvature.screen import forman_screen
from citecurv.curvature.timeslice import EventStore
from test_forman_directed import CASES

ORC_EXPECTED = [0.0, -0.5, -0.5, -0.5, 0.5, -0.5]
ORC_FLIPPED_EXPECTED = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class FakeORC:
    """Records every call; returns the number of edges of the graph it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, G, edges, **knobs):
        self.calls.append((G.copy(), list(edges), dict(knobs)))
        return {tuple(e): float(G.number_of_edges()) for e in edges}


def make_events(rows):
    return pd.DataFrame(rows, columns=["event_id", "src_id", "dst_id", "t"])


HAND = make_events([
    (1, "a", "b", 1), (2, "c", "b", 2), (3, "b", "d", 3), (4, "e", "a", 4), (5, "a", "d", 5),
    (6, "d", "f", 6), (7, "c", "a", 7), (8, "f", "b", 8), (9, "e", "d", 9), (10, "a", "f", 10),
    (11, "g", "g", 10),
])


def flagged(curv, event_ids):
    out = curv.copy()
    out["escalated"] = out["event_id"].isin(event_ids)
    return out


# ------------------------------------------------------------- orc_directed

def test_missing_library_gives_actionable_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "GraphRicciCurvature", None)
    monkeypatch.setitem(sys.modules, "GraphRicciCurvature.OllivierRicci", None)
    G = nx.DiGraph([("u", "v")])
    with pytest.raises(ImportError, match=r"GraphRicciCurvature.*pins"):
        orc_directed(G, [("u", "v")])


def test_input_validation_happens_before_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "GraphRicciCurvature", None)
    monkeypatch.setitem(sys.modules, "GraphRicciCurvature.OllivierRicci", None)
    with pytest.raises(ValueError, match="DiGraph"):
        orc_directed(nx.Graph([("u", "v")]), [("u", "v")])
    G = nx.DiGraph([("u", "v"), ("w", "w")])
    with pytest.raises(ValueError, match="not in G"):
        orc_directed(G, [("v", "u")])
    with pytest.raises(ValueError, match="self-loop"):
        orc_directed(G, [("w", "w")])
    assert orc_directed(G, []) == {}


# ------------------------------------------------------------ tiers and cap

def tier_graph():
    G = nx.DiGraph()
    E = [
        ("u", "v", 9),    # scored edge                         tier 0
        ("p1", "u", 3), ("p2", "u", 4), ("v", "w1", 1), ("v", "w2", 2),  # measures -> tier 1
        ("u", "x", 9),    # out of S                            tier 2
        ("p1", "y", 5),   # out of S                            tier 2
        ("x", "w1", 6),   # succ(S) -> T                        tier 2
        ("y", "z", 7),    # irrelevant                          tier 3
        ("q", "v", 8),    # into v from outside succ(S)         tier 3
        ("w2", "z", 8),   # leaves T, not from S or succ(S)        tier 3
    ]
    for a, b, t in E:
        G.add_edge(a, b, t=t)
    return G


def test_edge_tiers_hand_case():
    tiers = edge_tiers(tier_graph(), "u", "v")
    assert tiers == {
        ("u", "v"): 0,
        ("p1", "u"): 1, ("p2", "u"): 1, ("v", "w1"): 1, ("v", "w2"): 1,
        ("u", "x"): 2, ("p1", "y"): 2, ("x", "w1"): 2,
        ("y", "z"): 3, ("q", "v"): 3, ("w2", "z"): 3,
    }


def test_cap_subgraph_keeps_tiers_then_recency_then_labels():
    G = tier_graph()
    H, info = cap_subgraph(G, "u", "v", edge_cap=3)
    # tier 0, then tier-1 edges by most recent t first: p2->u (4), p1->u (3)
    assert set(H.edges) == {("u", "v"), ("p2", "u"), ("p1", "u")}
    assert info == {"n_edges": 11, "n_kept": 3, "capped": True, "lossy": True}
    H, info = cap_subgraph(G, "u", "v", edge_cap=8)
    assert set(H.edges) == {e for e, k in edge_tiers(G, "u", "v").items() if k <= 2}
    assert info["capped"] and not info["lossy"]  # only tier-3 edges were dropped
    H, info = cap_subgraph(G, "u", "v", edge_cap=100)
    assert set(H.edges) == set(G.edges) and not info["capped"]
    assert all(H.edges[e] == G.edges[e] for e in H.edges)  # attributes carried over


def test_cap_subgraph_prune_drops_tier3_even_under_cap():
    G = tier_graph()
    H, info = cap_subgraph(G, "u", "v", edge_cap=100, prune=True)
    assert set(H.edges) == {e for e, k in edge_tiers(G, "u", "v").items() if k <= 2}
    assert not info["capped"] and not info["lossy"] and info["n_kept"] == 8


def test_cap_subgraph_ties_break_by_label():
    G = nx.DiGraph()
    G.add_edge("u", "v", t=5)
    for n in ("pc", "pa", "pb"):
        G.add_edge(n, "u", t=1)
    H, _ = cap_subgraph(G, "u", "v", edge_cap=3)
    assert set(H.edges) == {("u", "v"), ("pa", "u"), ("pb", "u")}


# ------------------------------------------------------------------ escalate

def test_escalate_fills_only_escalated_rows_on_subgraph_plus_scored_edge():
    curv = flagged(forman_screen(HAND), [5, 9])
    store = EventStore(HAND)
    fake = FakeORC()
    out, stats = escalate(HAND, curv, store, radius=1, orc_fn=fake, alpha=0.3, method="OTD")
    assert [c[1] for c in fake.calls] == [[("a", "d")], [("e", "d")]]
    for (G, [(u, v)], knobs), (eid, t) in zip(fake.calls, [(5, 5), (9, 9)]):
        expected = store.subgraph_at(u, v, t, 1)
        assert not expected.has_edge(u, v)  # excluded from its own context ...
        assert set(G.edges) == set(expected.edges) | {(u, v)}  # ... then added back
        assert set(G.nodes) == set(expected.nodes)
        assert knobs == {"alpha": 0.3, "method": "OTD"}
    esc = out[out["escalated"]]
    assert list(esc["orc"]) == [4.0, 7.0]  # |E| of the two graphs handed to ORC
    assert (esc["radius"] == 1).all() and (esc["computed_at_stage"] == "escalate").all()
    rest = out[~out["escalated"]]
    assert rest["orc"].isna().all() and rest["radius"].isna().all()
    assert (rest["computed_at_stage"] == "screen").all()
    pd.testing.assert_frame_equal(out[["event_id", "forman"]], curv[["event_id", "forman"]])
    assert curv["orc"].isna().all()  # input untouched
    assert stats["n_escalated"] == 2 and stats["n_computed"] == 2
    assert stats["mean_subgraph_edges"] == 5.5 and stats["max_subgraph_edges"] == 7
    assert stats["n_capped"] == 0 and stats["n_self_loops"] == 0


def test_escalate_never_sees_future_events():
    curv = flagged(forman_screen(HAND), [3])  # b->d at t=3
    fake = FakeORC()
    escalate(HAND, curv, None, radius=2, orc_fn=fake)  # store built from events
    G = fake.calls[0][0]
    assert all(d["t"] <= 3 for _, _, d in G.edges(data=True))
    assert set(G.edges) == {("a", "b"), ("c", "b"), ("b", "d")}


def test_escalate_skips_self_loops():
    curv = flagged(forman_screen(HAND), [11, 10])
    fake = FakeORC()
    out, stats = escalate(HAND, curv, EventStore(HAND), radius=1, orc_fn=fake)
    assert len(fake.calls) == 1 and stats["n_self_loops"] == 1
    assert math.isnan(out.loc[out["event_id"] == 11, "orc"].item())


def test_escalate_resumes_and_respects_max_events():
    curv = flagged(forman_screen(HAND), [2, 5, 9])
    store = EventStore(HAND)
    fake = FakeORC()
    out, stats = escalate(HAND, curv, store, radius=1, orc_fn=fake, max_events=2)
    assert stats["n_computed"] == 2 and stats["n_pending"] == 3
    assert out.loc[out["event_id"] == 9, "orc"].isna().item()
    out2, stats2 = escalate(HAND, out, store, radius=1, orc_fn=fake)
    assert stats2["n_pending"] == 1 and stats2["n_computed"] == 1
    assert len(fake.calls) == 3  # rows 2 and 5 were not recomputed
    assert out2.loc[out2["escalated"], "orc"].notna().all()


def test_escalate_enforces_edge_cap_and_counts():
    curv = flagged(forman_screen(HAND), [9, 10])
    fake = FakeORC()
    out, stats = escalate(HAND, curv, EventStore(HAND), radius=2, edge_cap=3, orc_fn=fake)
    assert all(G.number_of_edges() <= 3 for G, _, _ in fake.calls)
    assert all(G.has_edge(*e[0]) for G, e, _ in fake.calls)  # the scored edge always survives
    assert stats["n_capped"] == 2 and stats["max_subgraph_edges"] > 3
    assert stats["max_kept_edges"] == 3


def test_escalate_is_deterministic():
    curv = flagged(forman_screen(HAND), [5, 7, 9, 10])
    a, _ = escalate(HAND, curv, EventStore(HAND), radius=2, edge_cap=4, orc_fn=FakeORC())
    b, _ = escalate(HAND, curv, EventStore(HAND), radius=2, edge_cap=4, orc_fn=FakeORC())
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- budget table

def random_stream(seed=0, n_papers=60, refs=4):
    rng = random.Random(seed)
    rows, eid = [], 0
    for p in range(1, n_papers):
        for q in rng.sample(range(p), min(refs, p)):
            rows.append((eid, f"w{p}", f"w{q}", p // 3))  # 3 papers per day -> same-t ties
            eid += 1
    return make_events(rows)


def test_escalation_budget_table_with_fake_orc():
    ev = random_stream()
    curv = forman_screen(ev)
    store = EventStore(ev)
    tab = escalation_budget(curv, store, radii=(1, 2), pcts=(0.05, 0.2), sample=5, orc_fn=FakeORC())
    assert list(tab.columns) == BUDGET_COLUMNS
    assert len(tab) == 4 and list(tab["radius"]) == [1, 2, 1, 2]
    assert (tab["n_sampled"] <= 5).all() and (tab["n_events"] == len(curv)).all()
    assert (tab["escalation_rate"] == tab["n_flagged"] / tab["n_events"]).all()
    assert np.allclose(tab["escalation_rate"], tab["rate_low"] + tab["rate_high"])
    r1, r2 = tab.iloc[0], tab.iloc[1]
    assert r2["mean_subgraph_edges"] >= r1["mean_subgraph_edges"]  # same sample, larger radius
    again = escalation_budget(curv, store, radii=(1, 2), pcts=(0.05, 0.2), sample=5, orc_fn=FakeORC())
    timing = [c for c in BUDGET_COLUMNS if c.endswith("_s_per_event") or c.startswith("est_")]
    pd.testing.assert_frame_equal(tab.drop(columns=timing), again.drop(columns=timing))


def test_escalation_budget_uses_reference_rows_for_cutoffs():
    ev = random_stream()
    curv = forman_screen(ev)
    ref = (curv["t"] < 10).to_numpy()
    tab = escalation_budget(curv, EventStore(ev), radii=(1,), pcts=(0.05,), sample=3,
                            orc_fn=FakeORC(), ref_mask=ref)
    v = curv.loc[ref, "forman"]
    assert tab["lo"].item() == pytest.approx(np.quantile(v, 0.05))
    assert tab["hi"].item() == pytest.approx(np.quantile(v, 0.95))


# --------------------------------------------------------------- real ORC

def test_real_orc_toy_cases_match_hand_values():
    pytest.importorskip("GraphRicciCurvature")
    for (edges, e, _), expected in zip(CASES, ORC_EXPECTED):
        for method in ("OTD", "OTDSinkhornMix"):
            got = orc_directed(nx.DiGraph(edges), [e], method=method)
            assert set(got) == {e}
            assert got[e] == pytest.approx(expected, abs=1e-9), (edges, e, method)


def test_real_orc_is_direction_sensitive():
    pytest.importorskip("GraphRicciCurvature")
    changed = 0
    for (edges, (u, v), _), before, after in zip(CASES, ORC_EXPECTED, ORC_FLIPPED_EXPECTED):
        G = nx.DiGraph(edges)
        G.remove_edge(u, v)
        G.add_edge(v, u)
        got = orc_directed(G, [(v, u)], method="OTD")[(v, u)]
        assert got == pytest.approx(after, abs=1e-9), edges
        changed += abs(got - before) > 1e-9
    assert changed == 5


def test_real_orc_upstream_directed_example():
    """Directed example from GraphRicciCurvature's own test-suite (same in 0.5.3.2 and 0.6.1)."""
    pytest.importorskip("GraphRicciCurvature")
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (1, 3), (3, 1)])
    got = orc_directed(G, list(G.edges), method="OTD")
    exp = {(0, 1): -0.5, (1, 2): 0.0, (1, 3): 0.5, (2, 3): 0.5, (3, 1): 0.75}
    assert got == pytest.approx(exp, abs=1e-6)


@pytest.mark.parametrize("prune", [False, True])
def test_real_orc_radius2_subgraph_equals_full_snapshot(prune):
    """Unweighted directed ORC of (u, v) only needs S = {u} + pred(u), T = {v} + succ(v)
    and S->T paths of length <= 2 (the path p->u->v->w caps every needed distance at 3),
    so the radius-2 subgraph, pruned or not, reproduces ORC on the whole time-t graph."""
    pytest.importorskip("GraphRicciCurvature")
    ev = random_stream(seed=1, n_papers=40, refs=3)
    # add some citations into older papers' citers so pred(u) is not always empty
    extra = [(1000 + i, f"w{a}", f"w{b}", a // 3) for i, (a, b) in enumerate([(30, 32), (20, 25), (35, 36)])]
    ev = pd.concat([ev, make_events(extra)], ignore_index=True)
    curv = flagged(forman_screen(ev), ev["event_id"].iloc[::7])
    store = EventStore(ev)
    out, _ = escalate(ev, curv, store, radius=2, prune=prune, method="OTD")
    for r in out[out["escalated"]].itertuples(index=False):
        G = nx.DiGraph()
        for s, d, t in zip(ev["src_id"], ev["dst_id"], ev["t"]):
            if t <= r.t and (s, d) != (r.src_id, r.dst_id):
                G.add_edge(s, d)
        G.add_edge(r.src_id, r.dst_id)
        full = orc_directed(G, [(r.src_id, r.dst_id)], method="OTD")[(r.src_id, r.dst_id)]
        assert r.orc == pytest.approx(full, abs=1e-9), r.event_id

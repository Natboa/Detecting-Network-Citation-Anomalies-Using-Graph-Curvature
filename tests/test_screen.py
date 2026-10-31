"""S2.2: Forman screen at event times, thresholds, and the distribution report.

Spec under test (CONTRACTS.md "Curvature cache", IMPLEMENTATION_PLAN.md S2.2):
- forman_screen(events) returns one row per event with
  F = 2 - in_deg_t(src) - out_deg_t(dst), where degrees are taken on the graph
  of all OTHER rows with ts <= t (same-t rows of other events included, the
  scored row itself excluded). The scored row src->dst is an out-edge of src
  and an in-edge of dst, so it can only enter in(src) / out(dst) when it is a
  self-loop; that case is excluded explicitly.
- Default degree semantics = distinct neighbours (duplicate (x, y) rows count
  once, as in a simple DiGraph); count_duplicates=True counts rows.
- orc NaN, escalated False, radius NaN, computed_at_stage "screen".
- screen_thresholds uses reference rows only; apply_screen flags the tails.

The brute-force reference below recomputes every value from its definition
with plain Python sets/counters and is cross-checked on random streams with
many same-t ties, self-loops and duplicate pairs.
"""

import random

import numpy as np
import pandas as pd
import pytest

from citecurv.curvature.forman_directed import forman_edge
from citecurv.curvature.screen import (
    CURVATURE_COLUMNS,
    apply_screen,
    forman_screen,
    screen_report,
    screen_thresholds,
    spot_check,
)
from citecurv.curvature.timeslice import EventStore


def make_events(rows):
    """rows: list of (event_id, src, dst, t)."""
    return pd.DataFrame(rows, columns=["event_id", "src_id", "dst_id", "t"]).astype(
        {"event_id": "int64", "t": "int64"}
    )


def brute_forman(rows, count_duplicates=False):
    """Definition-level reference: event_id -> F."""
    out = {}
    for eid, s, d, t in rows:
        ctx = [(s2, d2) for e2, s2, d2, t2 in rows if e2 != eid and t2 <= t]
        if count_duplicates:
            in_s = sum(1 for _, d2 in ctx if d2 == s)
            out_d = sum(1 for s2, _ in ctx if s2 == d)
        else:
            in_s = len({s2 for s2, d2 in ctx if d2 == s})
            out_d = len({d2 for s2, d2 in ctx if s2 == d})
        out[eid] = 2.0 - in_s - out_d
    return out


def screen_dict(rows, **kw):
    curv = forman_screen(make_events(rows), **kw)
    return dict(zip(curv["event_id"], curv["forman"]))


# ---------------------------------------------------------------- hand cases

def test_single_event_is_isolated_edge():
    assert screen_dict([(7, "a", "b", 1)]) == {7: 2.0}


def test_later_events_never_enter_earlier_rows():
    rows = [(1, "x", "a", 1), (2, "a", "b", 2)]
    # row 1: in(x)=0, out(a) at t=1 = 0 (a->b is at t=2, the future) -> 2
    # row 2: in(a)=1 (x->a), out(b)=0 -> 1
    assert screen_dict(rows) == {1: 2.0, 2: 1.0}


def test_same_t_rows_of_other_events_are_included():
    rows = [
        (1, "q", "z", 3),
        (2, "p", "q", 5),  # paper p published at t=5 cites q and r
        (3, "p", "r", 5),
        (4, "w", "p", 5),  # w, published the same day, already cites p
    ]
    # row 2: in(p)={w} (same t), out(q)={z} -> 0
    # row 3: in(p)={w}, out(r)={} -> 1
    # row 4: in(w)={}, out(p)={q, r} (same t) -> 0
    # row 1: in(q) at t=3 = {} (p->q is t=5), out(z)={} -> 2
    assert screen_dict(rows) == {1: 2.0, 2: 0.0, 3: 1.0, 4: 0.0}


def test_self_loop_excludes_itself_but_sees_other_self_loops():
    assert screen_dict([(1, "a", "a", 1)]) == {1: 2.0}
    rows = [(1, "a", "a", 1), (2, "a", "a", 2)]
    # row 2: the earlier a->a is both an in-edge and an out-edge of a -> 2-1-1
    assert screen_dict(rows) == {1: 2.0, 2: 0.0}
    rows = [(1, "x", "a", 1), (2, "a", "a", 2), (3, "a", "y", 2)]
    # row 2: in(a)={x}, out(a)={y} (same t); its own row is excluded -> 0
    assert screen_dict(rows)[2] == 0.0


def test_same_t_duplicate_self_loops_see_each_other():
    rows = [(1, "a", "a", 1), (2, "a", "a", 1)]
    assert screen_dict(rows) == {1: 0.0, 2: 0.0}


def test_duplicate_pairs_collapse_by_default_and_count_on_request():
    rows = [(1, "x", "a", 1), (2, "x", "a", 2), (3, "a", "b", 3)]
    assert screen_dict(rows)[3] == 1.0
    assert screen_dict(rows, count_duplicates=True)[3] == 0.0
    rows = [(1, "a", "a", 1), (2, "a", "a", 2), (3, "a", "a", 3)]
    assert screen_dict(rows)[3] == 0.0
    assert screen_dict(rows, count_duplicates=True)[3] == -2.0


def test_duplicate_of_the_scored_pair_does_not_change_its_value():
    rows = [(1, "a", "b", 1), (2, "a", "b", 1), (3, "a", "b", 4)]
    assert screen_dict(rows) == {1: 2.0, 2: 2.0, 3: 2.0}


# ------------------------------------------------------- schema and ordering

def test_output_schema_and_canonical_order():
    rows = [(30, "a", "b", 2), (10, "b", "c", 2), (20, "c", "a", 1)]
    curv = forman_screen(make_events(rows))
    assert list(curv.columns) == CURVATURE_COLUMNS
    assert list(curv["event_id"]) == [20, 10, 30]  # sorted by (t, event_id)
    assert curv["event_id"].dtype == np.int64 and curv["t"].dtype == np.int64
    assert curv["forman"].dtype == np.float64
    assert curv["orc"].isna().all() and curv["orc"].dtype == np.float64
    assert curv["radius"].isna().all()
    assert curv["escalated"].dtype == bool and not curv["escalated"].any()
    assert (curv["computed_at_stage"] == "screen").all()


def test_empty_input_gives_empty_frame():
    curv = forman_screen(make_events([]))
    assert list(curv.columns) == CURVATURE_COLUMNS and len(curv) == 0


def test_rejects_duplicate_event_ids_and_missing_columns():
    with pytest.raises(ValueError, match="event_id"):
        forman_screen(make_events([(1, "a", "b", 1), (1, "b", "c", 2)]))
    with pytest.raises(ValueError, match="column"):
        forman_screen(pd.DataFrame({"event_id": [1], "src": ["a"], "dst": ["b"], "t": [1]}))


# ----------------------------------------------------- brute-force sweeps

def random_rows(seed, n_events=300, n_nodes=15, t_max=8):
    """Many same-t ties (t in 1..t_max), self-loops and duplicate pairs."""
    rng = random.Random(seed)
    ids = rng.sample(range(10_000, 99_999), n_events)  # event_id is not a position
    rows = []
    for eid in ids:
        s = rng.randrange(n_nodes)
        d = s if rng.random() < 0.05 else rng.randrange(n_nodes)
        rows.append((eid, f"n{s}", f"n{d}", rng.randint(1, t_max)))
    rng.shuffle(rows)
    return rows


SHAPES = {
    "dense": {},                                             # degrees up to ~25, 40+ rows per t
    "sparse": {"n_nodes": 80, "n_events": 200, "t_max": 20},  # many first-time self-loops
}


@pytest.mark.parametrize("count_duplicates", [False, True])
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_brute_force_on_random_streams(seed, shape, count_duplicates):
    rows = random_rows(seed, **SHAPES[shape])
    got = screen_dict(rows, count_duplicates=count_duplicates)
    exp = brute_forman(rows, count_duplicates=count_duplicates)
    assert got == exp


def test_count_duplicates_matches_eventstore_degrees():
    rows = random_rows(5)
    curv = forman_screen(make_events(rows), count_duplicates=True)
    store = EventStore([(s, d, t) for _, s, d, t in rows])
    for r in curv.itertuples(index=False):
        if r.src_id == r.dst_id:
            continue
        in_deg, out_deg = store.degrees_at(r.t)  # rows are in canonical (monotonic) order
        assert r.forman == 2.0 - in_deg[r.src_id] - out_deg[r.dst_id], r.event_id


def test_distinct_matches_forman_edge_on_radius1_subgraph():
    """Screen value == directed Forman (S1.4) on the time-t radius-1 subgraph plus the scored edge."""
    rows = random_rows(6)
    curv = forman_screen(make_events(rows))
    store = EventStore([(s, d, t) for _, s, d, t in rows])
    for r in curv.itertuples(index=False):
        if r.src_id == r.dst_id:
            continue
        G = store.subgraph_at(r.src_id, r.dst_id, r.t, radius=1)
        G.add_edge(r.src_id, r.dst_id)
        assert r.forman == forman_edge(G, r.src_id, r.dst_id), r.event_id


def test_input_order_does_not_matter():
    rows = random_rows(7)
    a = forman_screen(make_events(rows))
    b = forman_screen(make_events(list(reversed(rows))))
    pd.testing.assert_frame_equal(a, b)


def test_negative_times_are_supported():
    rows = [(1, "x", "a", -400), (2, "a", "b", -3), (3, "b", "c", 0)]
    assert screen_dict(rows) == {1: 2.0, 2: 1.0, 3: 1.0}


def test_spot_check_agrees_with_screen():
    rows = random_rows(8)
    ev = make_events(rows)
    curv = forman_screen(ev)
    chk = spot_check(ev, curv, n=20, seed=0)
    assert len(chk) == 20 and chk["match"].all()
    bad = curv.copy()
    bad.loc[bad["event_id"].isin(chk["event_id"]), "forman"] += 1
    assert not spot_check(ev, bad, n=20, seed=0)["match"].any()


# ----------------------------------------------------- thresholds / report

def toy_curv():
    vals = np.arange(100, dtype=float)  # forman 0..99
    return pd.DataFrame({
        "event_id": np.arange(100, dtype=np.int64),
        "src_id": ["s"] * 100, "dst_id": ["d"] * 100,
        "t": np.arange(100, dtype=np.int64),
        "forman": vals, "orc": np.nan, "escalated": False, "radius": np.nan,
        "computed_at_stage": "screen",
        "grp": ["a"] * 50 + ["b"] * 50,
    })


def test_thresholds_use_reference_rows_only():
    curv = toy_curv()
    ref = curv["t"] < 50  # reference = forman 0..49
    th = screen_thresholds(curv, ref, pct_low=0.1, pct_high=0.9)
    assert th["global"]["lo"] == pytest.approx(np.quantile(np.arange(50.0), 0.1))
    assert th["global"]["hi"] == pytest.approx(np.quantile(np.arange(50.0), 0.9))
    assert th["global"]["n_ref"] == 50
    # values outside the reference range do not move the cut-offs
    curv2 = curv.copy()
    curv2.loc[~ref, "forman"] = 1e9
    assert screen_thresholds(curv2, ref, 0.1, 0.9)["global"] == th["global"]


def test_apply_screen_flags_strict_tails():
    curv = toy_curv()
    th = screen_thresholds(curv, np.ones(100, bool), pct_low=0.1, pct_high=0.9)
    lo, hi = th["global"]["lo"], th["global"]["hi"]
    out = apply_screen(curv, th)
    exp = (curv["forman"] < lo) | (curv["forman"] > hi)
    assert (out["escalated"] == exp).all()
    assert out["escalated"].sum() == 20
    assert not curv["escalated"].any()  # input untouched


def test_inclusive_ties_flag_more_than_strict():
    curv = toy_curv()
    curv["forman"] = np.where(curv["event_id"] < 70, 2.0, -1.0)  # heavy ties at the top
    strict = apply_screen(curv, screen_thresholds(curv, np.ones(100, bool), 0.05, 0.95))
    incl = apply_screen(curv, screen_thresholds(curv, np.ones(100, bool), 0.05, 0.95, inclusive=True))
    assert strict["escalated"].sum() == 0  # both tails are tied at the extremes
    assert incl["escalated"].sum() == 100


def test_grouped_thresholds_with_global_fallback():
    curv = toy_curv()
    ref = curv["grp"] == "a"  # group b has no reference rows -> global fallback
    th = screen_thresholds(curv, ref, 0.1, 0.9, group_col="grp")
    assert set(th["groups"]) == {"a"}
    out = apply_screen(curv, th)
    lo, hi = th["groups"]["a"]["lo"], th["groups"]["a"]["hi"]
    a = out[out["grp"] == "a"]
    assert (a["escalated"] == ((a["forman"] < lo) | (a["forman"] > hi))).all()
    b = out[out["grp"] == "b"]
    g = th["global"]
    assert (b["escalated"] == ((b["forman"] < g["lo"]) | (b["forman"] > g["hi"]))).all()


def test_thresholds_are_json_serialisable():
    import json

    curv = toy_curv()
    th = screen_thresholds(curv, np.ones(100, bool), 0.01, 0.99, group_col="grp")
    assert json.loads(json.dumps(th)) == th


def test_screen_report_by_group_and_year():
    curv = toy_curv()
    rep = screen_report(curv, by="grp", percentiles=(0.0, 0.5, 1.0))
    assert list(rep.index) == ["a", "b"]
    assert rep.loc["a", "n"] == 50 and rep.loc["a", "p50"] == pytest.approx(24.5)
    assert rep.loc["b", "p100"] == 99.0
    yr = screen_report(curv, by="year")  # derived from t (days since 1970-01-01)
    assert list(yr.index) == [1970] and yr.loc[1970, "n"] == 100
    allrep = screen_report(curv)
    assert list(allrep.index) == ["all"] and allrep.loc["all", "mean"] == pytest.approx(49.5)

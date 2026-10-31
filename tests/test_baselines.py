"""S4.4 baselines: statistical tier, XGBoost tier, PyGOD tier (L17).

Spec under test (IMPLEMENTATION_PLAN.md S4.4, CONTRACTS.md):
- Every baseline returns a DataFrame (event_id, score), higher = more anomalous.
- Tier 1 (statistical, CATCHSYNC-inspired) features per event use only graph state with ts <= t, the scored row
  excluded: self_citation (shared author), venue_gini (Gini of citing venues of dst over the venue universe),
  burst_z (Poisson z of dst's trailing-window count vs its own earlier rate), ref_overlap (max Jaccard between
  src's reference list and those of earlier citers of dst, dst itself removed, candidate set capped).
  Each feature is a scoring rule on its own; rank_average combines them through a reference ECDF.
- Tier 2 (XGBoost) trains on TRAIN-split labels only, is deterministic under a seed, and fails loudly when the
  train split has no positives.
- PyGOD tier: static snapshot at the cutoff, node scores mapped to edges by max(score(u), score(v)); the pygod 1.1.0
  adjustments in pygod_tier (isolated highest-index nodes, GADNR keyword clash and preprocessing) are regression-
  tested. Tests needing torch_geometric/pygod skip in the main venv and run in the PyGOD venv.

Hand-computed fixture F (src cites dst at t):
    e1 A->X 10 | e2 B->X 20 | e3 C->X 20 | e4 D->X 30 | e5 A->Y 10 | e6 B->Y 20 | e7 C->Z 20 | e8 D->Y 30 | e9 D->Z 30
venues: A,B,D = v1; C = v2; X,Y,Z = v3 (universe V = 3)
authors: A={a1,a2} B={a3} C={a4} D={a1} X={a1} Y={a5} Z=(none)
"""

import math
import random

import numpy as np
import pandas as pd
import pytest

from citecurv.eval.baselines import tier1_stats as t1
from citecurv.eval.baselines import tier2_xgb as t2
from citecurv.eval.baselines import pygod_tier as pg

NAN = float("nan")


def fixture_events():
    rows = [
        (1, "A", "X", 10), (2, "B", "X", 20), (3, "C", "X", 20), (4, "D", "X", 30), (5, "A", "Y", 10),
        (6, "B", "Y", 20), (7, "C", "Z", 20), (8, "D", "Y", 30), (9, "D", "Z", 30),
    ]
    return pd.DataFrame(rows, columns=["event_id", "src_id", "dst_id", "t"])


def fixture_works():
    venues = {"A": "v1", "B": "v1", "C": "v2", "D": "v1", "X": "v3", "Y": "v3", "Z": "v3"}
    return pd.DataFrame({"work_id": list(venues), "venue_id": list(venues.values())})


def fixture_authorships():
    pairs = [("A", "a1"), ("A", "a2"), ("B", "a3"), ("C", "a4"), ("D", "a1"), ("X", "a1"), ("Y", "a5")]
    return pd.DataFrame(pairs, columns=["work_id", "author_id"])


def assert_close_nan(actual, expected, tol=1e-12):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), (actual, expected)
    mask = ~np.isnan(expected)
    assert np.allclose(actual[mask], expected[mask], atol=tol, rtol=0), (actual, expected)


# ---------------------------------------------------------------- tier 1: hand-computed cases


def test_self_citation_hand_fixture():
    got = t1.self_citation(fixture_events(), fixture_authorships())
    assert_close_nan(got, [1, 0, 0, 1, 0, 0, NAN, 0, NAN])


def test_self_citation_is_nan_without_authorships():
    got = t1.self_citation(fixture_events(), None)
    assert np.isnan(got).all() and len(got) == 9


def test_venue_gini_hand_fixture():
    # e2: leave-one-out state {A:v1, C:v2} -> counts [1,1,0], Gini over V=3 venues = 1/3
    # e3: {A:v1, B:v1} -> [2,0,0] -> 2/3 ; e4: {A,B:v1, C:v2} -> [2,1,0] -> 4/9 ; single citer -> 1 - 1/3
    got = t1.venue_gini(fixture_events(), fixture_works(), min_citers=1)
    assert_close_nan(got, [NAN, 1 / 3, 2 / 3, 4 / 9, NAN, 2 / 3, NAN, 2 / 3, 2 / 3])


def test_venue_gini_min_citers_masks_thin_history():
    got = t1.venue_gini(fixture_events(), fixture_works(), min_citers=2)
    assert_close_nan(got, [NAN, 1 / 3, 2 / 3, 4 / 9, NAN, NAN, NAN, 2 / 3, NAN])


def test_venue_gini_nan_without_works():
    assert np.isnan(t1.venue_gini(fixture_events(), None)).all()


def test_gini_closed_form_matches_textbook_formula():
    # Gini over the zero-padded venue vector: sum_ij |x_i - x_j| / (2 V^2 mean)
    rng = np.random.default_rng(0)
    for _ in range(50):
        V = int(rng.integers(1, 8))
        x = rng.integers(0, 5, size=V).astype(float)
        if x.sum() == 0:
            continue
        textbook = np.abs(x[:, None] - x[None, :]).sum() / (2 * V * V * x.mean())
        assert math.isclose(t1.gini_from_counts(x[x > 0], V), textbook, abs_tol=1e-12)


def test_burst_z_hand_fixture():
    # X times [10,20,20,30], W=10. e2: C = 3-1-1 = 1, H = 1, exposure 0 -> one window, lambda 1 -> z 0.
    # e4: C = 0, H = 3, exposure 10 -> one window, lambda 3 -> z = -3/sqrt(3).
    got = t1.burst_z(fixture_events(), window_days=10)
    expected = [NAN, 0.0, 0.0, -math.sqrt(3), NAN, -1.0, NAN, -math.sqrt(2), -1.0]
    assert_close_nan(got, expected)


def test_burst_z_detects_a_burst_after_quiet_history():
    rows = [(i, f"s{i}", "W", t) for i, t in enumerate([0, 100, 105, 106, 107])]
    ev = pd.DataFrame(rows, columns=["event_id", "src_id", "dst_id", "t"])
    got = t1.burst_z(ev, window_days=10)
    # last event: C = 3 (100,105,106), H = 1 (t=0), exposure 97 days = 9.7 windows -> lambda = 1/9.7
    lam = 1 / 9.7
    assert math.isclose(got[4], (3 - lam) / math.sqrt(lam), rel_tol=1e-12)
    assert np.isnan(got[0])


def test_ref_overlap_hand_fixture():
    got = t1.ref_overlap(fixture_events(), max_candidates=20)
    assert_close_nan(got, [NAN, 1.0, 0.0, 0.5, NAN, 1.0, NAN, 0.5, 0.5])


def test_ref_overlap_candidate_cap_takes_most_recent_citer():
    # e2 (B->X @20): eligible citers of X are A (e1 @10) and C (e3 @20, same t allowed).
    # Cap 1 keeps the most recent by (t, event_id): C, whose other references {Z} share nothing with B's {Y}.
    got = t1.ref_overlap(fixture_events(), max_candidates=1)
    assert got[1] == 0.0
    assert got[3] == 0.5


def test_ref_overlap_ignores_duplicate_rows_of_the_same_citer():
    ev = fixture_events()
    dup = pd.DataFrame([(10, "B", "X", 20)], columns=ev.columns)  # B cites X twice
    got = t1.ref_overlap(pd.concat([ev, dup], ignore_index=True), max_candidates=20)
    # e2 must not compare B with itself (would give Jaccard 1 trivially): still 1.0 via A, and e3 still 0.0
    assert got[1] == 1.0 and got[2] == 0.0 and got[9] == 1.0


def test_compute_features_frame_and_row_order():
    ev = fixture_events().sample(frac=1.0, random_state=3).reset_index(drop=True)
    feats = t1.compute_features(ev, fixture_works(), fixture_authorships(), window_days=10, min_venue_citers=1)
    assert list(feats.columns) == ["event_id", *t1.FEATURES]
    assert feats["event_id"].tolist() == ev["event_id"].tolist()
    by_id = feats.set_index("event_id")
    assert by_id.loc[4, "venue_gini"] == pytest.approx(4 / 9)
    assert by_id.loc[4, "ref_overlap"] == 0.5
    assert by_id.loc[1, "self_citation"] == 1.0


def test_rank_average_hand_values():
    feats = pd.DataFrame({
        "event_id": [1, 2, 3, 4],
        "f1": [1.0, 2.0, 3.0, NAN],
        "f2": [4.0, 3.0, NAN, 1.0],
    })
    got = t1.rank_average(feats, columns=["f1", "f2"])
    assert list(got.columns) == ["event_id", "score"]
    assert_close_nan(got["score"], [0.5, 0.5, 2 / 3, 1 / 3])


def test_rank_average_uses_frozen_reference_ecdf():
    ref = pd.DataFrame({"event_id": [1, 2, 3, 4], "f": [0.0, 1.0, 2.0, 3.0]})
    test = pd.DataFrame({"event_id": [10, 11, 12], "f": [1.0, 2.5, 9.0]})
    got = t1.rank_average(test, columns=["f"], reference=ref)
    # mid-rank ECDF against the reference: 1.0 -> (1 + 0.5)/4, 2.5 -> 3/4, 9.0 -> 4/4
    assert_close_nan(got["score"], [0.375, 0.75, 1.0])
    # scores of one test event do not depend on the other test events
    alone = t1.rank_average(test.iloc[[1]], columns=["f"], reference=ref)
    assert alone["score"].iloc[0] == 0.75


def test_rule_scores_fill_missing_with_reference_median():
    feats = pd.DataFrame({"event_id": [1, 2, 3, 4], "f": [1.0, NAN, 3.0, 5.0]})
    got = t1.rule_scores(feats, "f")
    assert got["score"].tolist() == [1.0, 3.0, 3.0, 5.0]


# ---------------------------------------------------------------- tier 1: brute force + leak checks


def random_stream(seed, n_events=160, n_nodes=30, t_max=60, per_event_time=False):
    """Random stream with many shared timestamps; each src paper has one publication time unless
    per_event_time (injected events, e.g. synthetic cartels, can give one src several timestamps)."""
    rng = random.Random(seed)
    nodes = [f"p{i}" for i in range(n_nodes)]
    pub = {n: rng.randrange(t_max) for n in nodes}
    rows, eid = [], 1000
    while len(rows) < n_events:
        s, d = rng.choice(nodes), rng.choice(nodes)
        if s == d:
            continue
        rows.append((eid, s, d, rng.randrange(t_max) if per_event_time else pub[s]))
        eid += rng.choice([1, 2, 7])
    ev = pd.DataFrame(rows, columns=["event_id", "src_id", "dst_id", "t"])
    works = pd.DataFrame({
        "work_id": nodes,
        "venue_id": [rng.choice(["v1", "v2", "v3", "v4", None]) for _ in nodes],
    })
    auth = pd.DataFrame(
        [(n, f"a{rng.randrange(12)}") for n in nodes for _ in range(rng.randrange(0, 3))],
        columns=["work_id", "author_id"],
    )
    return ev, works, auth


def brute_features(ev, works, auth, window_days, min_citers, max_candidates):
    """O(E^2) reference implementation written straight from the definitions."""
    venue = {w: v for w, v in zip(works.work_id, works.venue_id) if pd.notna(v)}
    V = len(set(venue.values()))
    authors = auth.groupby("work_id")["author_id"].agg(set).to_dict()
    rows = list(ev.itertuples(index=False))
    out = []
    for e in rows:
        s, d, t = e.src_id, e.dst_id, e.t
        prior = [r for r in rows if r.t <= t and r.event_id != e.event_id]
        # self-citation
        sc = NAN if (s not in authors or d not in authors) else float(bool(authors[s] & authors[d]))
        # venue gini
        cit_venues = [venue[r.src_id] for r in prior if r.dst_id == d and r.src_id in venue]
        if len(cit_venues) >= min_citers and len(cit_venues) > 0:
            counts = pd.Series(cit_venues).value_counts().reindex(sorted(set(venue.values())), fill_value=0)
            x = counts.to_numpy(float)
            vg = np.abs(x[:, None] - x[None, :]).sum() / (2 * V * V * x.mean())
        else:
            vg = NAN
        # burst z
        into = [r.t for r in prior if r.dst_id == d] + [t]
        in_win = sum(1 for ts in into if t - window_days < ts <= t) - 1
        hist = [ts for ts in into if ts <= t - window_days]
        if hist:
            lam = len(hist) / max(((t - window_days) - min(hist)) / window_days, 1.0)
            bz = (in_win - lam) / math.sqrt(lam)
        else:
            bz = NAN
        # reference overlap
        def refs(p):
            return {r.dst_id for r in rows if r.src_id == p and r.t <= t} - {d}
        cands = sorted(
            {(r.t, r.event_id, r.src_id) for r in prior if r.dst_id == d and r.src_id != s}, reverse=True
        )
        seen, chosen = set(), []
        for _, _, p in cands:
            if p not in seen:
                seen.add(p)
                chosen.append(p)
            if len(chosen) == max_candidates:
                break
        if chosen:
            a = refs(s)
            js = []
            for p in chosen:
                b = refs(p)
                u = a | b
                js.append(len(a & b) / len(u) if u else 0.0)
            ro = max(js)
        else:
            ro = NAN
        out.append((sc, vg, bz, ro))
    return np.array(out, dtype=float)


@pytest.mark.parametrize("seed,per_event_time", [(0, False), (1, False), (2, False), (3, True), (4, True)])
def test_tier1_matches_brute_force(seed, per_event_time):
    ev, works, auth = random_stream(seed, per_event_time=per_event_time)
    got = t1.compute_features(
        ev, works, auth, window_days=7, min_venue_citers=2, max_ref_candidates=3
    )
    want = brute_features(ev, works, auth, window_days=7, min_citers=2, max_candidates=3)
    for j, name in enumerate(t1.FEATURES):
        assert_close_nan(got[name].to_numpy(), want[:, j], tol=1e-9)


def test_tier1_prefix_invariance_no_future_leak():
    """Appending later events must not change any feature of an earlier event (NFR3)."""
    ev, works, auth = random_stream(5, n_events=200)
    cut = int(ev["t"].median())
    early = ev[ev["t"] <= cut].reset_index(drop=True)
    full = t1.compute_features(ev, works, auth, window_days=5, min_venue_citers=1, max_ref_candidates=4)
    part = t1.compute_features(early, works, auth, window_days=5, min_venue_citers=1, max_ref_candidates=4)
    merged = part.merge(full, on="event_id", suffixes=("_p", "_f"))
    assert len(merged) == len(early)
    for name in t1.FEATURES:
        assert_close_nan(merged[f"{name}_p"], merged[f"{name}_f"])


# ---------------------------------------------------------------- tier 2


def test_degree_features_hand_fixture():
    deg = t2.degree_features(fixture_events())
    # e4 (D->X @30): X has 4 citations <= 30, minus the scored row -> 3; D has 3 references <= 30 minus own -> 2
    assert deg.loc[3, ["in_deg_src", "out_deg_src", "in_deg_dst", "out_deg_dst"]].tolist() == [0, 2, 3, 0]
    # e2 (B->X @20): X cited by A, B, C at <= 20 -> 2 others; B has 2 references at 20 -> 1 other
    assert deg.loc[1, ["in_deg_src", "out_deg_src", "in_deg_dst", "out_deg_dst"]].tolist() == [0, 1, 2, 0]


def test_degree_features_count_src_citations_up_to_t():
    ev = pd.DataFrame(
        [(1, "Q", "P", 5), (2, "P", "R", 9), (3, "S", "P", 9), (4, "T", "P", 12)],
        columns=["event_id", "src_id", "dst_id", "t"],
    )
    deg = t2.degree_features(ev)
    # e2 (P->R @9): P cited by Q@5 and S@9 (same t allowed), not by T@12
    assert deg.loc[1, "in_deg_src"] == 2
    # e3 (S->P @9): P has one reference at 9 -> out_deg_dst 1
    assert deg.loc[2, "out_deg_dst"] == 1


def test_embedding_cosine_hand_values_and_missing():
    ev = fixture_events().iloc[:3]
    ids = np.array(["A", "X", "B"])
    vecs = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    cos = t2.embedding_cosine(ev, (ids, vecs))
    assert cos[0] == pytest.approx(1 / math.sqrt(2))   # A . X
    assert cos[1] == pytest.approx(1 / math.sqrt(2))   # B . X
    assert np.isnan(cos[2])                             # C has no embedding


def planted_dataset(seed=0, n=600):
    """Synthetic (labelled synthetic) feature table where the label depends on two features."""
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame({"event_id": np.arange(n) * 3 + 7})
    for name in t2.TIER2_FEATURES:
        feats[name] = rng.normal(size=n)
    logit = 2.5 * feats["ref_overlap"] + 2.0 * feats["burst_z"] - 2.0
    label = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    labels = pd.DataFrame({"event_id": feats["event_id"], "label": label})
    split = pd.DataFrame({
        "event_id": feats["event_id"],
        "split": np.where(np.arange(n) < 400, "train", np.where(np.arange(n) < 450, "val", "test")),
    })
    return feats, labels, split


def test_xgb_learns_planted_signal_and_scores_only_test_rows():
    feats, labels, split = planted_dataset()
    scores, model = t2.fit_predict(feats, labels, split, seed=0, nthread=1, num_boost_round=60)
    test_ids = split.loc[split.split == "test", "event_id"]
    assert list(scores.columns) == ["event_id", "score"]
    assert scores["event_id"].tolist() == test_ids.tolist()
    from sklearn.metrics import roc_auc_score

    y = labels.set_index("event_id").loc[scores.event_id, "label"]
    assert roc_auc_score(y, scores["score"]) > 0.75
    assert model is not None


def test_xgb_is_deterministic_and_ignores_test_labels():
    feats, labels, split = planted_dataset(seed=1)
    a, _ = t2.fit_predict(feats, labels, split, seed=3, nthread=1, num_boost_round=40)
    flipped = labels.copy()
    test_mask = flipped.event_id.isin(split.loc[split.split != "train", "event_id"])
    flipped.loc[test_mask, "label"] = 1 - flipped.loc[test_mask, "label"]
    b, _ = t2.fit_predict(feats, flipped, split, seed=3, nthread=1, num_boost_round=40)
    assert np.allclose(a["score"], b["score"], atol=1e-12)


def test_xgb_raises_when_train_has_no_positives():
    feats, labels, split = planted_dataset(seed=2)
    train_ids = split.loc[split.split == "train", "event_id"]
    labels.loc[labels.event_id.isin(train_ids), "label"] = 0
    with pytest.raises(ValueError, match="no positive"):
        t2.fit_predict(feats, labels, split, seed=0, nthread=1, num_boost_round=5)


def test_build_features_has_documented_columns():
    ev = fixture_events()
    ids = np.array(["A", "B", "C", "D", "X", "Y", "Z"])
    vecs = np.eye(7, dtype=np.float32)
    feats = t2.build_features(ev, fixture_works(), fixture_authorships(), (ids, vecs), window_days=10)
    assert list(feats.columns) == ["event_id", *t2.TIER2_FEATURES]
    assert feats["event_id"].tolist() == ev["event_id"].tolist()
    assert (feats["emb_cosine"] == 0).all()


# ---------------------------------------------------------------- PyGOD tier (pure parts run everywhere)


def test_snapshot_edges_cut_at_cutoff_and_deduplicated():
    ev = pd.concat([fixture_events(), pd.DataFrame([(20, "A", "X", 15)], columns=fixture_events().columns)])
    node_ids = ["A", "B", "C", "D", "X", "Y", "Z", "NEW"]
    edges, index, stats = pg.snapshot_edges(ev, cutoff=20, node_ids=node_ids, undirected=False)
    pairs = {(node_ids[a], node_ids[b]) for a, b in edges.T}
    assert pairs == {("A", "X"), ("B", "X"), ("C", "X"), ("A", "Y"), ("B", "Y"), ("C", "Z")}
    assert edges.shape[1] == 6  # the duplicate A->X collapses
    assert index["NEW"] == 7
    und, _, _ = pg.snapshot_edges(ev, cutoff=20, node_ids=node_ids, undirected=True)
    assert und.shape[1] == 12


def test_snapshot_edges_counts_events_with_unknown_endpoints():
    _, _, stats = pg.snapshot_edges(fixture_events(), cutoff=30, node_ids=["A", "B", "X"], undirected=False)
    assert stats["n_events_dropped"] == 7 and stats["n_events_used"] == 2


def test_edge_scores_use_max_of_endpoint_scores():
    node_index = {"A": 0, "B": 1, "X": 2, "Y": 3}
    node_scores = np.array([0.1, 0.9, 0.5, 0.2])
    ev = pd.DataFrame([(7, "A", "X", 40), (8, "B", "Y", 41), (9, "A", "Y", 41)],
                      columns=["event_id", "src_id", "dst_id", "t"])
    got = pg.edge_scores_from_nodes(node_scores, node_index, ev)
    assert got["event_id"].tolist() == [7, 8, 9]
    assert got["score"].tolist() == [0.5, 0.9, 0.2]


def test_edge_scores_refuse_endpoints_outside_the_snapshot():
    with pytest.raises(ValueError, match="not in the snapshot"):
        pg.edge_scores_from_nodes(np.zeros(1), {"A": 0},
                                  pd.DataFrame([(1, "A", "B", 3)], columns=["event_id", "src_id", "dst_id", "t"]))


# ---------------------------------------------------------------- PyGOD tier (needs pygod + torch_geometric)


def small_snapshot(isolated_last=False):
    import torch

    torch.set_num_threads(1)  # determinism and speed: several torch threads on a busy shared CPU spin-wait
    ev, _, _ = random_stream(11, n_events=150, n_nodes=40, t_max=50)
    cutoff = 35
    node_ids = sorted(set(ev.src_id) | set(ev.dst_id))
    if isolated_last:  # papers with no event by the cutoff get the highest indices, as in a real snapshot
        past = ev[ev.t <= cutoff]
        linked = set(past.src_id) | set(past.dst_id)
        node_ids = [n for n in node_ids if n in linked] + [n for n in node_ids if n not in linked]
    feats = np.random.default_rng(0).normal(size=(len(node_ids), 8)).astype(np.float32)
    snap = pg.snapshot_graph(ev, cutoff, node_ids, feats)
    return ev, snap, cutoff


@pytest.mark.parametrize("isolated_last", [False, True])
@pytest.mark.parametrize("name", ["DOMINANT", "CoLA", "GADNR"])
def test_score_pygod_shapes_and_determinism(name, isolated_last):
    pytest.importorskip("pygod")
    ev, snap, cutoff = small_snapshot(isolated_last)
    test_ev = ev[ev.t > cutoff]
    a = pg.score_pygod(name, snap, test_ev, seed=0, epoch=5)
    b = pg.score_pygod(name, snap, test_ev, seed=0, epoch=5)
    assert a["event_id"].tolist() == test_ev["event_id"].tolist()
    assert np.isfinite(a["score"]).all()
    assert np.allclose(a["score"], b["score"], atol=1e-6)


def test_snapshot_graph_builds_pyg_data():
    pytest.importorskip("torch_geometric")
    ev, snap, cutoff = small_snapshot()
    assert snap.data.num_nodes == len(snap.node_ids)
    assert snap.data.x.shape == (len(snap.node_ids), 8)
    src = snap.data.edge_index[0].numpy()
    assert src.max() < len(snap.node_ids)
    assert snap.cutoff == cutoff


def test_score_pygod_rejects_unknown_detector():
    with pytest.raises(ValueError, match="unknown detector"):
        pg.score_pygod("SCAN", None, fixture_events(), seed=0)


def test_gadnr_backbone_wrapper_drops_only_tot_nodes():
    build = pg.gadnr_compatible_backbone(lambda *a, **kw: (a, kw))
    assert build(1, tot_nodes=5, hidden_channels=8) == ((1,), {"hidden_channels": 8})


def test_small_snapshot_has_isolated_test_period_nodes():
    pytest.importorskip("torch_geometric")
    _, snap, _ = small_snapshot()
    touched = set(snap.data.edge_index.flatten().tolist())
    assert len(touched) < len(snap.node_ids)  # GADNR must cope with isolated nodes


def test_gadnr_process_graph_matches_upstream():
    pytest.importorskip("pygod")
    import torch
    from pygod.nn.gadnr import GADNRBase

    _, snap, _ = small_snapshot()
    base = pg.add_isolated_self_loops(snap.data.clone())
    up = GADNRBase.process_graph(base.clone())
    ours = pg.gadnr_process_graph(base.clone())
    assert torch.equal(up[0].x, ours[0].x) and torch.equal(up[0].edge_index, ours[0].edge_index)
    assert {k: list(v) for k, v in up[1].items()} == ours[1]
    assert torch.equal(up[2], ours[2]) and up[3] == ours[3]
    # mini-batch form: the graph is a sampled subgraph with global ids in n_id and centre nodes input_id
    sub = base.clone()
    sub.n_id = torch.randperm(sub.num_nodes, generator=torch.Generator().manual_seed(0))
    centres = sub.n_id[:10].tolist()
    up = GADNRBase.process_graph(sub.clone(), centres)
    ours = pg.gadnr_process_graph(sub.clone(), centres)
    assert {int(k): [int(x) for x in v] for k, v in up[1].items()} == ours[1]
    assert torch.equal(up[2], ours[2]) and {int(k): v for k, v in up[3].items()} == ours[3]


@pytest.mark.parametrize("name", ["DOMINANT", "GADNR"])
def test_score_pygod_leaves_snapshot_and_pygod_unchanged(name):
    pytest.importorskip("pygod")
    from pygod.nn.dominant import DOMINANTBase
    from pygod.nn.gadnr import GADNRBase

    before = (DOMINANTBase.__dict__["process_graph"], GADNRBase.__dict__["process_graph"])
    ev, snap, cutoff = small_snapshot(isolated_last=True)
    x0, e0 = snap.data.x.clone(), snap.data.edge_index.clone()
    pg.score_pygod(name, snap, ev[ev.t > cutoff], seed=0, epoch=2)
    assert snap.data.x.equal(x0) and snap.data.edge_index.equal(e0)
    assert (DOMINANTBase.__dict__["process_graph"], GADNRBase.__dict__["process_graph"]) == before

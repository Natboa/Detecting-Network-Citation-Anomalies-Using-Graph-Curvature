"""S1.7: to_jodie — citation events to DyGLib's JODIE-format files.

Spec under test (IMPLEMENTATION_PLAN.md S1.7 + CONTRACTS.md "Events"/"Node embeddings"):
- rows ordered by (t, event_id); edge index idx = 1..M in that order (DyGLib +1 shift, 0 = padding);
- node index 1..N by first appearance in that order (src before dst), 0 = padding;
- ml_{name}.csv has DyGLib's columns u, i, ts, label, idx (label always 0: labels never enter training);
- ml_{name}.npy is (M + 1, 1): row 0 zero padding, then the citation-direction indicator +1 ("u cites i");
- ml_{name}_node.npy is (N + 1, d): row 0 zeros, row k = embedding of the node with index k
  (missing embeddings become zero rows and are counted, or raise when asked);
- the id map round-trips index <-> work_id and idx <-> event_id.
All expected values below are worked out by hand from the 5-event fixture.
"""

import json

import numpy as np
import pandas as pd
import pytest

from citecurv.tgn.convert import DIRECTION_FEAT_INDEX, load_idmap, to_jodie

# deliberately shuffled, non-positional event ids, ties on t
EVENTS = pd.DataFrame(
    {
        "event_id": [30, 10, 50, 20, 40],
        "src_id": ["W3", "W2", "W3", "W4", "W4"],
        "dst_id": ["W1", "W1", "W2", "W3", "W9"],
        "t": [200, 100, 200, 300, 300],
    }
)
EMB_IDS = np.array(["W1", "W2", "W3", "W4", "W7"])
EMB_VEC = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [9, 9, 9]], dtype=np.float32
)

# canonical order (t, event_id): 10, 30, 50, 20, 40
EXP_EVENT_ORDER = [10, 30, 50, 20, 40]
# first appearance, src before dst: W2=1, W1=2, W3=3, W4=4, W9=5
EXP_WORK_ORDER = ["W2", "W1", "W3", "W4", "W9"]
EXP_U = [1, 3, 3, 4, 4]
EXP_I = [2, 2, 1, 3, 5]
EXP_TS = [100, 200, 200, 300, 300]


@pytest.fixture
def converted(tmp_path):
    meta = to_jodie(EVENTS, (EMB_IDS, EMB_VEC), tmp_path, "fx")
    return tmp_path, meta


def test_csv_rows_and_plus_one_shift(converted):
    out, _ = converted
    df = pd.read_csv(out / "ml_fx.csv")
    for col in ("u", "i", "ts", "label", "idx"):
        assert col in df.columns
    assert df.idx.tolist() == [1, 2, 3, 4, 5]
    assert df.u.tolist() == EXP_U
    assert df.i.tolist() == EXP_I
    assert df.ts.tolist() == EXP_TS
    assert (df.label == 0).all()
    assert min(df.u.min(), df.i.min()) == 1  # index 0 is reserved for padding


def test_edge_features_padding_row_and_direction(converted):
    out, meta = converted
    e = np.load(out / "ml_fx.npy")
    assert e.shape == (6, 1)
    assert e.dtype == np.float32
    assert DIRECTION_FEAT_INDEX == 0
    assert meta["direction_feat_index"] == 0
    np.testing.assert_array_equal(e[0], [0.0])
    np.testing.assert_array_equal(e[1:, DIRECTION_FEAT_INDEX], np.ones(5))


def test_node_features_aligned_to_shift(converted):
    out, meta = converted
    n = np.load(out / "ml_fx_node.npy")
    assert n.shape == (6, 3)
    assert n.dtype == np.float32
    expected = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=np.float32
    )
    np.testing.assert_array_equal(n, expected)
    assert meta["n_missing_embeddings"] == 1  # W9
    assert meta["n_nodes"] == 5 and meta["n_events"] == 5
    assert meta["node_feat_dim"] == 3 and meta["edge_feat_dim"] == 1


def test_idmap_round_trip(converted):
    out, _ = converted
    nodes, events = load_idmap(out, "fx")
    assert nodes.node_idx.tolist() == [1, 2, 3, 4, 5]
    assert nodes.work_id.tolist() == EXP_WORK_ORDER
    assert events.idx.tolist() == [1, 2, 3, 4, 5]
    assert events.event_id.tolist() == EXP_EVENT_ORDER

    # join the JODIE rows back to the original events
    df = pd.read_csv(out / "ml_fx.csv")
    w = dict(zip(nodes.node_idx, nodes.work_id))
    back = pd.DataFrame(
        {
            "event_id": df.idx.map(dict(zip(events.idx, events.event_id))),
            "src_id": df.u.map(w),
            "dst_id": df.i.map(w),
            "t": df.ts,
        }
    )
    orig = EVENTS.sort_values(["t", "event_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(back, orig, check_dtype=False)


def test_meta_file_written(converted):
    out, meta = converted
    on_disk = json.loads((out / "ml_fx_meta.json").read_text())
    assert on_disk["name"] == "fx"
    assert on_disk["n_missing_embeddings"] == meta["n_missing_embeddings"]


def test_missing_embedding_can_raise(tmp_path):
    with pytest.raises(KeyError, match="W9"):
        to_jodie(EVENTS, (EMB_IDS, EMB_VEC), tmp_path, "fx", on_missing="raise")


def test_npz_style_mapping_accepted(tmp_path):
    meta = to_jodie(EVENTS, {"ids": EMB_IDS, "vectors": EMB_VEC}, tmp_path, "fx")
    assert meta["node_feat_dim"] == 3


def test_no_embeddings_gives_zero_features(tmp_path):
    meta = to_jodie(EVENTS, None, tmp_path, "fx", node_feat_dim=4)
    n = np.load(tmp_path / "ml_fx_node.npy")
    assert n.shape == (6, 4) and not n.any()
    assert meta["n_missing_embeddings"] == 0


def test_dim_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="node_feat_dim"):
        to_jodie(EVENTS, (EMB_IDS, EMB_VEC), tmp_path, "fx", node_feat_dim=64)


def test_duplicate_event_id_raises(tmp_path):
    bad = pd.concat([EVENTS, EVENTS.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="event_id"):
        to_jodie(bad, (EMB_IDS, EMB_VEC), tmp_path, "fx")


def test_deterministic_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    to_jodie(EVENTS, (EMB_IDS, EMB_VEC), a, "fx")
    to_jodie(EVENTS.sample(frac=1, random_state=3), (EMB_IDS, EMB_VEC), b, "fx")
    for f in ("ml_fx.csv", "ml_fx.npy", "ml_fx_node.npy"):
        assert (a / f).read_bytes() == (b / f).read_bytes()


# --- reduce_to: shrink node features before DyGLib (S1.7 pilot OOM, 2026-11-11) ---------------
# DyGLib sets the TGN memory width to the node-feature width (MemoryModel.py:53), so 768-d SPECTER2
# vectors give every node a 768-d memory and the pilot ran out of T4 memory. reduce_to projects
# the vectors onto their top principal components before they are written.

def _planar_fixture():
    """Six works whose 6-d vectors lie on a 2-d plane (plus an offset): PCA to 2 is lossless."""
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(6, 2)))[0].T  # 2 orthonormal rows in R^6
    coeff = np.array([[0, 0], [3, 1], [-2, 4], [5, -3], [1, 1], [-4, -2]], dtype=np.float64)
    vecs = (coeff @ basis + 7.0).astype(np.float32)
    ids = np.array([f"P{k}" for k in range(6)])
    events = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6],
            "src_id": ["P1", "P2", "P3", "P4", "P5", "P0"],
            "dst_id": ["P0", "P1", "P2", "P3", "P4", "MISSING"],
            "t": [10, 20, 30, 40, 50, 60],
        }
    )
    return events, ids, vecs


def test_reduce_to_sets_width_and_keeps_missing_rows_zero(tmp_path):
    events, ids, vecs = _planar_fixture()
    meta = to_jodie(events, (ids, vecs), tmp_path, "rd", reduce_to=2)
    n = np.load(tmp_path / "ml_rd_node.npy")
    assert n.shape == (8, 2) and n.dtype == np.float32
    nodes, _ = load_idmap(tmp_path, "rd")
    missing_idx = int(nodes.loc[nodes.work_id == "MISSING", "node_idx"].iloc[0])
    assert not n[0].any() and not n[missing_idx].any()
    assert meta["node_feat_dim"] == 2
    red = meta["node_feat_reduction"]
    assert red["method"] == "pca" and red["from_dim"] == 6 and red["to_dim"] == 2 and red["n_fit"] == 6


def test_reduce_to_preserves_distances_when_lossless(tmp_path):
    events, ids, vecs = _planar_fixture()
    meta = to_jodie(events, (ids, vecs), tmp_path, "rd", reduce_to=2)
    n = np.load(tmp_path / "ml_rd_node.npy")
    nodes, _ = load_idmap(tmp_path, "rd")
    row = {w: vecs[k] for k, w in enumerate(ids)}
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            wa, wb = nodes.work_id[a], nodes.work_id[b]
            if "MISSING" in (wa, wb):
                continue
            d_in = np.linalg.norm(row[wa].astype(np.float64) - row[wb])
            d_out = np.linalg.norm(n[a + 1].astype(np.float64) - n[b + 1])
            assert d_out == pytest.approx(d_in, rel=1e-4, abs=1e-4)
    assert meta["node_feat_reduction"]["explained_variance"] == pytest.approx(1.0, abs=1e-6)


def test_reduce_to_fit_ids_limits_the_fit(tmp_path):
    events, ids, vecs = _planar_fixture()
    meta = to_jodie(events, (ids, vecs), tmp_path, "rd", reduce_to=2, fit_ids=["P0", "P1", "P2", "NOPE"])
    assert meta["node_feat_reduction"]["n_fit"] == 3


def test_reduce_to_not_smaller_than_width_raises(tmp_path):
    with pytest.raises(ValueError, match="reduce_to"):
        to_jodie(EVENTS, (EMB_IDS, EMB_VEC), tmp_path, "fx", reduce_to=3)


def test_reduce_to_more_than_fit_rows_raises(tmp_path):
    events, ids, vecs = _planar_fixture()
    with pytest.raises(ValueError, match="reduce_to"):
        to_jodie(events, (ids, vecs), tmp_path, "rd", reduce_to=4, fit_ids=["P0", "P1", "P2"])


def test_reduce_to_deterministic_bytes(tmp_path):
    events, ids, vecs = _planar_fixture()
    a, b = tmp_path / "a", tmp_path / "b"
    to_jodie(events, (ids, vecs), a, "rd", reduce_to=2)
    to_jodie(events.sample(frac=1, random_state=5), (ids[::-1], vecs[::-1]), b, "rd", reduce_to=2)
    assert (a / "ml_rd_node.npy").read_bytes() == (b / "ml_rd_node.npy").read_bytes()


def test_no_reduction_recorded_by_default(converted):
    _, meta = converted
    assert meta["node_feat_reduction"] is None


def test_reduce_to_without_embeddings_raises(tmp_path):
    with pytest.raises(ValueError, match="reduce_to"):
        to_jodie(EVENTS, None, tmp_path, "fx", node_feat_dim=4, reduce_to=2)


# --- reduce_to on millions of vectors (2026-11-16) ---------------------------------------------
# The first version held two float64 copies of every fit row (the rows and the centred rows) and
# again of every projected row: 16 bytes per stored float32 value, on top of the array itself. The
# fit now reads the rows in chunks (two passes: mean, then centred covariance) and the projection
# is written chunk by chunk.

def _structured_fixture(n=40, d=8, seed=1):
    """n works with d-dim vectors whose variance falls off per direction (clear eigen-gaps), chained events."""
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    scale = 1.0 / np.arange(1, d + 1)
    vecs = (rng.normal(size=(n, d)) * scale @ q + 3.0).astype(np.float32)
    ids = np.array([f"S{k:03d}" for k in range(n)])
    events = pd.DataFrame({"event_id": np.arange(n - 1), "src_id": ids[1:], "dst_id": ids[:-1],
                           "t": np.arange(n - 1) * 10})
    return events, ids, vecs


def test_pca_fit_reads_bounded_chunks_and_matches_the_textbook_fit():
    from citecurv.tgn.convert import _pca_fit

    _, _, x = _structured_fixture(n=50)
    asked = []

    def get_rows(lo, hi):
        asked.append((lo, hi))
        return x[lo:hi]

    mean, comps, explained = _pca_fit(get_rows, len(x), 3, chunk_rows=16)
    assert max(hi - lo for lo, hi in asked) <= 16                       # never more than a chunk in hand
    assert sorted(asked) == sorted([(0, 16), (16, 32), (32, 48), (48, 50)] * 2)   # two passes over every row
    x64 = x.astype(np.float64)
    w, v = np.linalg.eigh(np.cov(x64, rowvar=False))
    top = np.argsort(w)[::-1][:3]
    np.testing.assert_allclose(mean, x64.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(np.abs(comps.T @ v[:, top]), np.eye(3), atol=1e-9)   # same axes, up to sign
    assert explained == pytest.approx(w[top].sum() / w.sum(), rel=1e-9)
    assert (comps[np.abs(comps).argmax(axis=0), np.arange(3)] > 0).all()            # the sign rule still holds


def test_reduce_to_in_small_chunks_gives_the_same_features(tmp_path):
    events, ids, vecs = _structured_fixture()
    a, b = tmp_path / "a", tmp_path / "b"
    meta_a = to_jodie(events, (ids, vecs), a, "ch", reduce_to=3)
    meta_b = to_jodie(events, (ids, vecs), b, "ch", reduce_to=3, chunk_rows=7)
    na, nb = np.load(a / "ml_ch_node.npy"), np.load(b / "ml_ch_node.npy")
    assert na.shape == nb.shape == (41, 3)
    # only the order of float64 additions differs between chunk sizes: a few float32 rounding steps at most
    tol = 8 * np.finfo(np.float32).eps * float(np.abs(na).max())
    np.testing.assert_allclose(nb, na, rtol=0, atol=tol)
    ra, rb = meta_a["node_feat_reduction"], meta_b["node_feat_reduction"]
    assert ra["n_fit"] == rb["n_fit"] == 40
    assert rb["explained_variance"] == pytest.approx(ra["explained_variance"], rel=1e-9)


def test_reduce_to_one_chunk_equals_the_one_shot_formula(tmp_path):
    # with every row in one chunk the bytes are those of the first version: centre, covariance, eigh, project
    events, ids, vecs = _structured_fixture()
    to_jodie(events, (ids, vecs), tmp_path, "os", reduce_to=3)
    nodes, _ = load_idmap(tmp_path, "os")
    x = vecs[pd.Series(np.arange(len(ids)), index=ids).loc[nodes.work_id].to_numpy()].astype(np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    w, v = np.linalg.eigh(xc.T @ xc / (len(x) - 1))
    comps = v[:, np.argsort(w)[::-1][:3]]
    flip = np.sign(comps[np.abs(comps).argmax(axis=0), np.arange(3)])
    expected = ((x - mean) @ (comps * flip)).astype(np.float32)
    np.testing.assert_array_equal(np.load(tmp_path / "ml_os_node.npy")[1:], expected)


def test_chunk_rows_must_be_positive(tmp_path):
    events, ids, vecs = _structured_fixture()
    with pytest.raises(ValueError, match="chunk_rows"):
        to_jodie(events, (ids, vecs), tmp_path, "ch", reduce_to=3, chunk_rows=0)


# --- which works may shape the PCA on a dated split (2026-11-16) --------------------------------
# Notebook 40_tgn fitted on the citing works of training-period events only. With cited works'
# text in the graph (decided 2026-11-12) most nodes are cited-only works, so the fit set is both
# endpoints of every training-period event: a cited work exists before the event that cites it.

def test_pca_fit_ids_are_both_endpoints_of_training_period_events():
    from citecurv.tgn.convert import pca_fit_ids

    ev = pd.DataFrame({"event_id": [1, 2, 3, 4], "src_id": ["A", "B", "C", "D"],
                       "dst_id": ["X", "A", "Y", "X"], "t": [10, 20, 30, 40]})
    assert sorted(pca_fit_ids(ev, 20)) == ["A", "B", "X"]        # the event on the boundary day is training
    assert sorted(pca_fit_ids(ev, 30)) == ["A", "B", "C", "X", "Y"]
    assert len(pca_fit_ids(ev, 5)) == 0                          # nothing before the first event
    assert pca_fit_ids(ev, None) is None                         # no split dates: to_jodie fits on every work
    shuffled = ev.sample(frac=1, random_state=2)
    assert list(pca_fit_ids(shuffled, 30)) == list(pca_fit_ids(ev, 30))   # same order whatever the row order


def test_pca_fit_ids_let_cited_only_works_into_the_fit(tmp_path):
    from citecurv.tgn.convert import pca_fit_ids

    events, ids, vecs = _planar_fixture()      # P1->P0 (t=10), P2->P1 (20), P3->P2 (30), P4->P3, P5->P4, P0->MISSING
    fit = pca_fit_ids(events, 30)              # training events: the first three; P0 is cited only in them
    assert sorted(fit) == ["P0", "P1", "P2", "P3"]
    meta = to_jodie(events, (ids, vecs), tmp_path, "rd", reduce_to=2, fit_ids=fit)
    assert meta["node_feat_reduction"]["n_fit"] == 4 and meta["node_feat_reduction"]["fit_restricted"] is True


# --- the fit-set lookup at full size (2026-11-23) ------------------------------------------------
# The full run's rehearsal sat for over ten minutes at one line: the lookup of the fit ids among the
# graph's works. The works come from pd.unique (an object array), and numpy's isin with an object
# array on either side compares every fit id against every work in a Python-level loop, so 1.4
# million fit ids against 1.8 million works never finished. The lookup now runs on two str arrays.

def test_fit_ids_lookup_is_not_quadratic(tmp_path):
    import time

    n = 60_000                     # the old lookup took 6.7 s at 30,000 works on the laptop and grows with n squared
    events, ids, vecs = _structured_fixture(n=n, d=4)
    from citecurv.tgn.convert import pca_fit_ids

    fit = pca_fit_ids(events, int(events["t"].quantile(0.7)))      # an object array, as in the notebook
    t0 = time.perf_counter()
    meta = to_jodie(events, (ids, vecs), tmp_path, "big", reduce_to=2, fit_ids=fit)
    elapsed = time.perf_counter() - t0
    assert meta["node_feat_reduction"]["n_fit"] == len(set(fit))
    assert elapsed < 5.0, f"to_jodie took {elapsed:.1f} s on {n} works"


def test_fit_ids_as_list_or_object_array_give_the_same_bytes(tmp_path):
    events, ids, vecs = _structured_fixture(n=60)
    from citecurv.tgn.convert import pca_fit_ids

    fit = pca_fit_ids(events, 300)
    a, b = tmp_path / "a", tmp_path / "b"
    to_jodie(events, (ids, vecs), a, "fx", reduce_to=3, fit_ids=fit)
    to_jodie(events, (ids, vecs), b, "fx", reduce_to=3, fit_ids=[str(x) for x in fit])
    assert (a / "ml_fx_node.npy").read_bytes() == (b / "ml_fx_node.npy").read_bytes()

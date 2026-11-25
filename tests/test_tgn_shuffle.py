"""S4.3 Task 4: a DyGLib dataset with shuffled test-period times for the timestamp-shuffle check.

Spec under test (docs/superpowers/plans/2026-11-25-s43-ablation-matrix.md, Task 4): shuffle_dyglib_dataset keeps the
node feature file and the work-id map byte for byte (the trained checkpoint indexes node features by node index),
gives every event its new time, re-sorts by (t, event_id) with ties in event-id order, renumbers idx, moves edge
feature rows with their events and refuses a different event set. The fixture is built so that re-running to_jodie
on the shuffled events would write a different node file, so the node-file assertions can fail.
"""

import numpy as np
import pandas as pd
import pytest

from citecurv.tgn.convert import load_idmap, pca_fit_ids, shuffle_dyglib_dataset, to_jodie

CUTOFF = 10
EVENTS10 = pd.DataFrame({
    "event_id": np.arange(1, 11, dtype=np.int64),
    "src_id": ["A", "C", "A", "D", "E", "F", "G", "H", "X", "I"],
    "dst_id": ["B", "B", "C", "A", "C", "A", "B", "D", "E", "X"],
    "t": np.array([1, 2, 5, 5, 8, 12, 12, 15, 20, 25], dtype=np.int64),
})
# X first appears in event 9 (t=20); the shuffle moves event 9 to t=12, ahead of H and F
NEW_T = {6: 20, 7: 12, 8: 15, 9: 12, 10: 25}
IDS = np.array(sorted(set(EVENTS10["src_id"]) | set(EVENTS10["dst_id"])))
VECS = np.random.default_rng(3).normal(size=(len(IDS), 4)).astype(np.float32)


def shuffled_times():
    t = EVENTS10.set_index("event_id")["t"].copy()
    for e, v in NEW_T.items():
        t[e] = v
    return pd.DataFrame({"event_id": t.index.to_numpy(), "t": t.to_numpy()})


def shuffled_events():
    ev = EVENTS10.copy()
    ev["t"] = ev["event_id"].map(shuffled_times().set_index("event_id")["t"])
    return ev


def build(tmp_path, **kw):
    to_jodie(EVENTS10, (IDS, VECS), tmp_path / "orig", "toy", **kw)
    to_jodie(shuffled_events(), (IDS, VECS), tmp_path / "rerun", "toy", **kw)
    return tmp_path / "orig", tmp_path / "rerun"


def read_csv(d, name):
    return pd.read_csv(d / f"ml_{name}.csv", index_col=0)


@pytest.mark.parametrize("kw", [{}, {"reduce_to": 2, "fit_ids": pca_fit_ids(EVENTS10, CUTOFF)}])
def test_moved_rows_take_new_times_and_keep_the_node_files(tmp_path, kw):
    orig, rerun = build(tmp_path, **kw)
    node_orig = (orig / "ml_toy_node.npy").read_bytes()
    assert (rerun / "ml_toy_node.npy").read_bytes() != node_orig  # a re-run of to_jodie would drift
    meta = shuffle_dyglib_dataset(orig, "toy", shuffled_times(), tmp_path / "sh", "toy_shuf")
    df = read_csv(tmp_path / "sh", "toy_shuf")
    assert (np.diff(df["ts"].to_numpy()) >= 0).all()
    nodes, evmap = load_idmap(tmp_path / "sh", "toy_shuf")
    eid = evmap["event_id"].to_numpy()
    want_t = shuffled_times().set_index("event_id")["t"]
    np.testing.assert_array_equal(df["ts"].to_numpy(), want_t.reindex(eid).to_numpy())
    # ties in event-id order: t=12 holds events 7 and 9, in that order
    assert list(eid[df["ts"].to_numpy() == 12]) == [7, 9]
    np.testing.assert_array_equal(df["idx"].to_numpy(), np.arange(1, 11))
    pd.testing.assert_frame_equal(df.iloc[:5].reset_index(drop=True), read_csv(orig, "toy").iloc[:5].reset_index(drop=True))
    assert (tmp_path / "sh" / "ml_toy_shuf_node.npy").read_bytes() == node_orig
    onodes, oevmap = load_idmap(orig, "toy")
    pd.testing.assert_frame_equal(nodes, onodes)
    # each event keeps its own endpoints (u, i are node indices of the original map)
    o = read_csv(orig, "toy").assign(event_id=oevmap["event_id"].to_numpy()).set_index("event_id")
    s = df.assign(event_id=eid).set_index("event_id")
    pd.testing.assert_frame_equal(o.loc[eid, ["u", "i"]], s.loc[eid, ["u", "i"]])
    assert meta["n_moved"] == sum(1 for e, v in NEW_T.items() if v != EVENTS10.set_index("event_id").at[e, "t"])
    assert meta["source_name"] == "toy" and meta["n_events"] == 10


def test_identity_shuffle_reproduces_the_files(tmp_path):
    orig, _ = build(tmp_path)
    same = EVENTS10[["event_id", "t"]]
    shuffle_dyglib_dataset(orig, "toy", same, tmp_path / "sh", "toy")
    pd.testing.assert_frame_equal(read_csv(tmp_path / "sh", "toy"), read_csv(orig, "toy"))
    for f in ("ml_toy_node.npy", "ml_toy.npy"):
        assert (tmp_path / "sh" / f).read_bytes() == (orig / f).read_bytes()
    a, b = load_idmap(orig, "toy"), load_idmap(tmp_path / "sh", "toy")
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_rejects_a_different_event_set(tmp_path):
    orig, _ = build(tmp_path)
    with pytest.raises(ValueError, match="event"):
        shuffle_dyglib_dataset(orig, "toy", shuffled_times().iloc[:-1], tmp_path / "sh", "toy_shuf")

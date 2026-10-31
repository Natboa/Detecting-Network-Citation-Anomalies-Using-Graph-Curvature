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

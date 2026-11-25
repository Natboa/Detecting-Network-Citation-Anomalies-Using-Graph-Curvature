"""S4.3 Task 5: TGN outputs of a shuffled stream merged with the original ones.

Spec under test (docs/superpowers/plans/2026-11-25-s43-ablation-matrix.md, Task 5): events with original
t <= cutoff keep the original scores and embeddings (they must reproduce them; the drift is reported), the others take
the shuffled values, aligned by event_id, in the original row order; different event sets raise.
"""

import numpy as np
import pandas as pd
import pytest

from citecurv.tgn.infer import merge_shuffled_tgn

CUTOFF = 50


def outputs(seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 110, dtype=np.int64)
    scores = pd.DataFrame({"event_id": ids, "p_link": rng.uniform(size=10), "neg_p_link": rng.uniform(size=10)})
    emb = {"event_id": ids.copy(), "z_src": rng.normal(size=(10, 3)).astype(np.float32),
           "z_dst": rng.normal(size=(10, 3)).astype(np.float32)}
    events = pd.DataFrame({"event_id": ids, "src_id": [f"s{i}" for i in range(10)],
                           "dst_id": [f"d{i}" for i in range(10)], "t": np.arange(10, dtype=np.int64) * 10 + 5})
    return scores, emb, events


def test_merge_takes_pre_cutoff_rows_from_the_original():
    orig_s, orig_e, events = outputs(0)
    shuf_s, shuf_e, _ = outputs(1)
    pre = (events["t"] <= CUTOFF).to_numpy()
    # the shuffled scoring reproduces pre-cutoff rows up to a small drift, in another row order
    shuf_s.loc[pre, ["p_link", "neg_p_link"]] = orig_s.loc[pre, ["p_link", "neg_p_link"]].to_numpy() + 1e-7
    shuf_e["z_src"][pre] = orig_e["z_src"][pre]
    shuf_e["z_dst"][pre] = orig_e["z_dst"][pre]
    perm = np.random.default_rng(5).permutation(10)
    shuf_s = shuf_s.iloc[perm].reset_index(drop=True)
    shuf_e = {k: v[perm] for k, v in shuf_e.items()}
    want_test_p = dict(zip(shuf_s["event_id"], shuf_s["p_link"]))
    want_test_z = dict(zip(shuf_e["event_id"], shuf_e["z_dst"]))
    keep_s, keep_e = orig_s.copy(), {k: v.copy() for k, v in orig_e.items()}
    merged_s, merged_e, report = merge_shuffled_tgn(orig_s, orig_e, shuf_s, shuf_e, events, CUTOFF)
    np.testing.assert_array_equal(merged_s["event_id"].to_numpy(), keep_s["event_id"].to_numpy())
    np.testing.assert_array_equal(merged_s.loc[pre, "p_link"].to_numpy(), keep_s.loc[pre, "p_link"].to_numpy())
    np.testing.assert_array_equal(merged_s.loc[~pre, "p_link"].to_numpy(),
                                  [want_test_p[e] for e in keep_s.loc[~pre, "event_id"]])
    np.testing.assert_array_equal(merged_e["z_src"][pre], keep_e["z_src"][pre])
    np.testing.assert_array_equal(merged_e["z_dst"][~pre], np.stack([want_test_z[e] for e in keep_s.loc[~pre, "event_id"]]))
    np.testing.assert_array_equal(merged_e["event_id"], keep_e["event_id"])
    assert report["n_pre"] == int(pre.sum()) and report["n_test"] == int((~pre).sum())
    assert report["pre_max_abs_diff_p_link"] == pytest.approx(1e-7, rel=1e-3)
    assert report["pre_max_abs_diff_neg_p_link"] == pytest.approx(1e-7, rel=1e-3)


def test_merge_rejects_different_event_sets():
    orig_s, orig_e, events = outputs(0)
    shuf_s, shuf_e, _ = outputs(1)
    shuf_s = shuf_s.iloc[:-1]
    shuf_e = {k: v[:-1] for k, v in shuf_e.items()}
    with pytest.raises(ValueError, match="event"):
        merge_shuffled_tgn(orig_s, orig_e, shuf_s, shuf_e, events, CUTOFF)

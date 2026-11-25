"""S4.3 Task 7: derived records for the tables (paper-level intervals, label views, counting controls, ORC coverage).

Spec under test (docs/superpowers/plans/2026-11-25-s43-ablation-matrix.md, Task 7):
- the weighted ranking metrics equal eval.metrics.ranking_metrics with unit weights (ties included);
- the cluster bootstrap is wider when positives sit in one cited paper and clusters follow it; the pooled point is
  the mean of per-seed deltas, and rows of different seeds are never ranked together;
- the masks drop exactly their rows; unseen_cited is defined on the cited work;
- a control's direction comes from validation rows only;
- report records leave main, ablation and label_views unchanged and feed the new tables.
"""

import json

import numpy as np
import pandas as pd
import pytest

from citecurv.data.synthetic import load_fixture
from citecurv.eval import ablation as A
from citecurv.eval import report_s43 as R
from citecurv.eval.metrics import ranking_metrics


def scores_frame(ids, total, split="test", cold=False, total_cold=None):
    n = len(ids)
    return pd.DataFrame({"event_id": np.asarray(ids, dtype=np.int64), "total": np.asarray(total, dtype=float),
                         "total_cold": np.full(n, np.nan) if total_cold is None else np.asarray(total_cold, float),
                         "cold_start": np.full(n, cold), "split": np.full(n, split, dtype=object)})


def concentrated(seed=0, shift=0.0):
    """60 test events: 40 cite paper V (all positive), 20 others (2 positive). Run a ranks the V events high,
    run b does not, so the delta rides on one cited paper."""
    rng = np.random.default_rng(seed)
    ids = np.arange(60)
    src = [f"u{i}" for i in ids]
    dst = ["V"] * 40 + [f"w{i}" for i in range(20)]
    y = np.r_[np.ones(40), np.zeros(18), np.ones(2)]
    a = np.r_[rng.uniform(0.6, 1.0, 40), rng.uniform(0, 0.7, 20)] + shift
    b = rng.uniform(0, 1, 60)
    events = pd.DataFrame({"event_id": ids, "src_id": src, "dst_id": dst, "t": np.zeros(60, dtype=np.int64)})
    labels = pd.DataFrame({"event_id": ids, "label": y.astype(int), "via_src": np.zeros(60, int),
                           "via_dst": y.astype(int)})
    return events, labels, scores_frame(ids, a), scores_frame(ids, b)


def test_weighted_metrics_equal_ranking_metrics():
    rng = np.random.default_rng(1)
    s = np.round(rng.uniform(size=300), 1)  # many ties
    y = (rng.uniform(size=300) < 0.2).astype(int)
    got = R._Ranked(s, y).metrics()
    want = ranking_metrics(s, y)
    assert got["ap"] == pytest.approx(want["ap"], abs=1e-12)
    assert got["auroc"] == pytest.approx(want["auroc"], abs=1e-12)
    w = rng.integers(0, 3, 300)
    rep = np.repeat(np.arange(300), w)
    assert R._Ranked(s, y).metrics(w)["ap"] == pytest.approx(ranking_metrics(s[rep], y[rep])["ap"], abs=1e-12)


def test_clusters_on_the_cited_paper_widen_the_interval():
    events, labels, a, b = concentrated()
    keys = R.cluster_keys(events, labels)
    res = R.paired_paper_bootstrap([(a, b)], labels, keys, metric="ap", n_draws=300, seed=0)
    w = {k: v["hi"] - v["lo"] for k, v in res["per_key"].items()}
    assert w["dst"] > w["src"]
    assert res["hi"] - res["lo"] == pytest.approx(max(w.values()))
    assert res["interval_method"] == "cluster:" + max(w, key=w.get)


def test_pooled_point_is_the_mean_of_per_seed_deltas_and_seeds_stay_apart():
    events, labels, a0, b0 = concentrated(0)
    _, _, a1, b1 = concentrated(1, shift=100.0)  # seed 1's totals live on another scale
    keys = R.cluster_keys(events, labels)
    pooled = R.paired_paper_bootstrap([(a0, b0), (a1, b1)], labels, keys, n_draws=50)
    one = [R.paired_paper_bootstrap([p], labels, keys, n_draws=10)["delta"] for p in ((a0, b0), (a1, b1))]
    assert pooled["delta"] == pytest.approx(np.mean(one))
    assert pooled["per_seed"] == pytest.approx(one)


def test_batch_key_follows_the_labelled_side():
    events, labels, _, _ = concentrated()
    keys = R.cluster_keys(events, labels, batches={"V": "pubX|2024-01-01"})
    assert set(keys["batch"].iloc[:40]) == {"batch:pubX|2024-01-01"}
    assert keys["batch"].iloc[45] == "u45"


def test_masks_drop_exactly_their_rows(tmp_path):
    events, labels, a, _ = concentrated()
    keep = R.mask_without_batches(events, labels, {"V"})
    assert (~keep).sum() == 40 and not keep.iloc[:40].any()
    rec = R.mask_without_recordless(events, ["w3", "w4"])
    assert list(np.flatnonzero(~rec.to_numpy())) == [43, 44]
    run = tmp_path / "runs" / "full_0"
    run.mkdir(parents=True)
    a.to_parquet(run / "scores.parquet")
    (run / "metrics.json").write_text(json.dumps({"kind": "model", "method": "full", "seed": 0}))
    recs = R.masked_view_records(run, labels, {"without_top_batches": keep})
    assert recs[0]["n_dropped"] == 40 and recs[0]["n_pos_dropped"] == 40
    assert recs[0]["kind"] == "report" and recs[0]["tags"]["group"] == "masked_view"


def test_unseen_cited_differs_from_the_total_when_every_citing_work_is_new():
    ev = pd.DataFrame({"event_id": np.arange(6), "src_id": ["a", "b", "c", "d", "e", "f"],
                       "dst_id": ["x", "y", "x", "z", "y", "q"], "t": [1, 2, 3, 10, 11, 12]})
    split = np.array(["train", "train", "train", "test", "test", "test"])
    keep, src_seen = R.mask_unseen_cited(ev, split)
    assert list(keep.to_numpy()) == [False, False, False, True, False, True]  # y is seen in training, z and q not
    assert src_seen == 0.0


def test_control_direction_comes_from_validation_rows(tmp_path):
    ids = np.arange(40)
    y = np.r_[np.ones(10), np.zeros(30)].astype(int)
    sc = pd.concat([scores_frame(ids[:20], np.linspace(1, 0, 20), split="val"),
                    scores_frame(ids[20:], np.linspace(1, 0, 20), split="test")], ignore_index=True)
    labels = pd.DataFrame({"event_id": ids, "label": np.r_[y[:10], y[10:20], y[:10], y[10:20]]})
    run = tmp_path / "runs" / "full_0"
    run.mkdir(parents=True)
    sc.to_parquet(run / "scores.parquet")
    (run / "metrics.json").write_text(json.dumps({"kind": "model", "method": "full", "seed": 0}))
    low_is_bad = pd.DataFrame({"event_id": ids, "score": -np.r_[np.linspace(1, 0, 20), np.linspace(1, 0, 20)]})
    recs = R.control_records(run, {"overlap": low_is_bad}, labels)
    assert recs[0]["direction"] == -1 and recs[0]["test"]["auroc"] > 0.5
    assert recs[0]["kind"] == "control" and recs[0]["method"] == "overlap"


def test_tables_unchanged_by_report_records_and_new_tables_render(tmp_path):
    from test_pipeline_integration import FakeORC, cfg_small

    fx = load_fixture()
    data = {"events": fx["events"], "works": fx["works"], "embeddings": fx["embeddings"],
            "event_labels": fx["event_labels"], "cutoff": 19180}
    A.run_matrix(["full", "no_curvature", "forman_only"], [0, 1], data, tmp_path, cfg=cfg_small(), orc_fn=FakeORC())
    before = A.tables(tmp_path)
    fx["events"].to_parquet(tmp_path / "events.parquet")
    lab = fx["event_labels"].assign(via_src=fx["event_labels"]["label"], via_dst=0)
    lab.to_parquet(tmp_path / "labels.parquet")
    recs = R.run_report({"runs": str(tmp_path), "labels": str(tmp_path / "labels.parquet"),
                         "events": str(tmp_path / "events.parquet"), "seeds": [0, 1], "n_draws": 20,
                         "comparisons": [["full", "no_curvature"], ["full", "forman_only"]]})
    assert recs and all(r["kind"] == "report" for r in recs)
    after = A.tables(tmp_path)
    for name in ("main", "ablation", "label_views", "curvature_delta"):
        cols = [c for c in before[name].columns]
        pd.testing.assert_frame_equal(before[name][cols].reset_index(drop=True),
                                      after[name][cols].reset_index(drop=True))
    iv = after["ablation_intervals"]
    assert {"full - no_curvature", "full - forman_only"} <= set(iv["comparison"])
    assert "excludes 0" in iv.columns and "stale" in iv.columns and not iv["stale"].any()
    assert "interval verdict" in after["curvature_delta"].columns
    assert "seed varies" in after["ablation"].columns

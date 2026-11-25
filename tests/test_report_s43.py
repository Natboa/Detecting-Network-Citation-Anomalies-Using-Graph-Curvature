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
    A.run_matrix(["full", "no_curvature", "forman_only"], [0, 1], data, tmp_path, cfg=cfg_small(), orc_fn=FakeORC(),
                 tags={"level": "real"})
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
        cols = [c for c in before[name].columns if c != "interval verdict"]  # the verdict is the intended change
        pd.testing.assert_frame_equal(before[name][cols].reset_index(drop=True),
                                      after[name][cols].reset_index(drop=True))
    iv = after["ablation_intervals"]
    assert {"full - no_curvature", "full - forman_only"} <= set(iv["comparison"])
    assert "excludes 0" in iv.columns and "stale" in iv.columns and not iv["stale"].any()
    assert "interval verdict" in after["curvature_delta"].columns
    cd = after["curvature_delta"].set_index("metric")["interval verdict"]
    assert cd["AP"].startswith("interval") and cd["AUROC"].startswith("interval")
    assert cd["P@100"] == "no interval"  # intervals cover AP and AUROC only
    assert "seed varies" in after["ablation"].columns


# ------------------------------------------------------------------ final review fixes (I3, I4, I6, I9, I10)

def test_all_nan_side_gives_no_interval():
    events, labels, a, b = concentrated()
    b["total"] = np.nan
    res = R.paired_paper_bootstrap([(a, b)], labels, R.cluster_keys(events, labels), n_draws=20)
    assert np.isnan(res["delta"]) and np.isnan(res["lo"]) and np.isnan(res["hi"])
    rec = R.interval_record(("x", "y"), "scored:ap", res, -1, [])
    assert rec["excludes_0"] is False


def test_partly_nan_rows_are_dropped_on_both_sides():
    events, labels, a, b = concentrated()
    b.loc[:9, "total"] = np.nan
    res = R.paired_paper_bootstrap([(a, b)], labels, R.cluster_keys(events, labels), n_draws=5)
    want = (R._Ranked(a["total"].iloc[10:], labels["label"].iloc[10:]).metrics()["ap"]
            - R._Ranked(b["total"].iloc[10:], labels["label"].iloc[10:]).metrics()["ap"])
    assert res["delta"] == pytest.approx(want)


def test_draws_keep_seeds_apart():
    events, labels, a0, b0 = concentrated(0)
    _, _, a1, b1 = concentrated(1)
    keys = R.cluster_keys(events, labels)
    plain = R.paired_paper_bootstrap([(a0, b0), (a1, b1)], labels, keys, n_draws=40, seed=3)
    a1s = a1.assign(total=a1["total"] + 100.0)
    shifted = R.paired_paper_bootstrap([(a0, b0), (a1s, b1)], labels, keys, n_draws=40, seed=3)
    assert shifted["lo"] == pytest.approx(plain["lo"]) and shifted["hi"] == pytest.approx(plain["hi"])


def test_both_metrics_come_from_one_draw_loop():
    events, labels, a, b = concentrated()
    keys = R.cluster_keys(events, labels)
    both = R.paired_paper_bootstrap_both([(a, b)], labels, keys, n_draws=30, seed=2)
    for m in ("ap", "auroc"):
        one = R.paired_paper_bootstrap([(a, b)], labels, keys, metric=m, n_draws=30, seed=2)
        assert both[m]["lo"] == pytest.approx(one["lo"]) and both[m]["hi"] == pytest.approx(one["hi"])


def test_row_mask_restricts_the_interval_rows():
    events, labels, a, b = concentrated()
    keys = R.cluster_keys(events, labels)
    keep = pd.Series(np.r_[np.zeros(40, bool), np.ones(20, bool)], index=events["event_id"])
    res = R.paired_paper_bootstrap([(a, b)], labels, keys, n_draws=5, row_mask=keep)
    assert res["n_pos"] == 2


def _run_folder(root, name, scores, level="real", escalated=None):
    d = root / "runs" / name
    d.mkdir(parents=True)
    scores.to_parquet(d / "scores.parquet")
    method, seed = name.rsplit("_", 1)
    (d / "metrics.json").write_text(json.dumps({"kind": "model", "method": method, "seed": int(seed),
                                                "tags": {"level": level}}))
    if escalated is not None:
        np.save(d / "escalated.npy", np.asarray(escalated, dtype=np.int64))
    return d


def test_orc_coverage_counts(tmp_path):
    ids = np.arange(10)
    sc = scores_frame(ids, np.linspace(1, 0, 10))
    d = _run_folder(tmp_path, "full_0", sc, escalated=[1, 2, 3, 4])
    labels = pd.DataFrame({"event_id": ids, "label": [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                           "via_src": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0], "via_dst": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]})
    rows = pd.DataFrame({"event_id": [1, 2, 3, 4], "orc": [-0.5, np.nan, -0.2, -0.1],
                         "orc_lossy": [True, False, False, False], "orc_src_in_deg": [0, 0, 2, 0],
                         "orc_error": ["", "boom", "", ""]})
    cov = R.orc_coverage(d, rows, labels)
    assert cov["n_escalated_test"] == 4 and cov["n_pos_with_orc"] == 2  # events 1 and 4 (event 2 has no value)
    assert cov["n_pos_with_orc_via_src"] == 2 and cov["n_pos_with_orc_via_dst"] == 0
    assert cov["lossy_share"] == 0.25 and cov["src_cited_share"] == 0.25 and cov["n_errors"] == 1


def test_shuffle_components_record_shape():
    value = {"matched": {"n_rows": 50, "base": {"total": {"auroc": 0.7, "ap": 0.2}, "s_temp": {"auroc": 0.6, "ap": 0.1}},
                         "shuffled": {"total": {"auroc": 0.55, "ap": 0.12}, "s_temp": {"auroc": 0.5, "ap": 0.05}}}}
    rec = R.shuffle_components_record(value, {"delta": 0.15, "lo": 0.05, "hi": 0.25}, 0, [])
    assert rec["components"]["s_temp"]["shuffled"]["auroc"] == 0.5 and rec["n_rows"] == 50
    t = A.tables([{"kind": "model", "method": "full", "seed": 0, "tags": {"level": "real"}, "test": {"ap": 0.1}}, rec])
    assert set(t["shuffle_components"]["component"]) == {"total", "s_temp"}


def test_control_readings_on_known_and_cold_rows(tmp_path):
    ids = np.arange(30)
    sc = pd.concat([scores_frame(ids[:10], np.linspace(1, 0, 10), split="val"),
                    scores_frame(ids[10:20], np.linspace(1, 0, 10)),
                    scores_frame(ids[20:], np.full(10, np.nan), cold=True, total_cold=np.linspace(1, 0, 10))],
                   ignore_index=True)
    d = _run_folder(tmp_path, "full_0", sc)
    labels = pd.DataFrame({"event_id": ids, "label": ([1, 1] + [0] * 8) * 3})
    score = np.r_[np.linspace(1, 0, 10), np.r_[np.nan, np.linspace(1, 0, 9)], np.linspace(1, 0, 10)]
    ctl = pd.DataFrame({"event_id": ids, "score": score})
    known = pd.Series(np.r_[np.ones(15, bool), np.zeros(15, bool)], index=ids)
    rec = R.control_records(d, {"c": ctl}, labels, known=known)[0]
    assert rec["direction"] == 1 and rec["n_nan_test"] == 1
    assert rec["on_known"]["control"]["n"] == 5 and "full" in rec["on_known"]
    assert rec["on_cold"]["control"]["n"] == 10 and rec["on_cold"]["full"]["auroc"] == pytest.approx(1.0)


def test_run_report_branches_aliases_and_checkpoints(tmp_path, monkeypatch):
    events, labels, a, b = concentrated()
    real = tmp_path / "real"
    for s in (0, 1):
        _run_folder(real, f"full_{s}", a, escalated=[0, 1, 2])
        _run_folder(real, f"no_curvature_{s}", b, escalated=[])
        _run_folder(real, f"no_temporal_{s}", b, escalated=[])
        _run_folder(tmp_path / "keep", f"no_temporal_{s}", a, escalated=[])
    events.to_parquet(tmp_path / "events.parquet")
    labels.to_parquet(tmp_path / "labels.parquet")
    pd.DataFrame({"event_id": events["event_id"], "keep": np.r_[np.zeros(40, bool), np.ones(20, bool)]}).to_parquet(
        tmp_path / "mask_x.parquet")
    pd.DataFrame({"event_id": events["event_id"], "score": np.arange(60, dtype=float)}).to_parquet(tmp_path / "c.parquet")
    pd.DataFrame({"event_id": events["event_id"], "ref_known": True}).to_parquet(tmp_path / "known.parquet")
    pd.DataFrame({"event_id": [0, 1, 2], "orc": [-0.1, -0.2, np.nan], "orc_lossy": False, "orc_src_in_deg": 0,
                  "orc_error": ""}).to_parquet(tmp_path / "orc.parquet")
    spec = {"runs": str(real), "labels": str(tmp_path / "labels.parquet"), "events": str(tmp_path / "events.parquet"),
            "seeds": [0, 1], "n_draws": 10, "aliases": {"no_temporal_keep_temp": [str(tmp_path / "keep"), "no_temporal"]},
            "comparisons": [["full", "no_curvature"], ["full", "no_temporal_keep_temp"],
                            ["no_temporal_keep_temp", "no_temporal"]],
            "masks": {"x": str(tmp_path / "mask_x.parquet")}, "mask_notes": {"x": "drops the V citations"},
            "controls": {"c": str(tmp_path / "c.parquet")}, "known": str(tmp_path / "known.parquet"),
            "orc": str(tmp_path / "orc.parquet")}
    recs = R.run_report(spec)
    comps = {tuple(r["comparison"]) for r in recs if r["tags"]["group"] == "interval"}
    assert ("full", "no_temporal_keep_temp") in comps and ("no_temporal_keep_temp", "no_temporal") in comps
    assert any(r.get("view", "").startswith("mask:x:") for r in recs if r["tags"]["group"] == "interval")
    assert any(r.get("note") == "drops the V citations" for r in recs if r["tags"]["group"] == "masked_view")
    assert any(r["tags"]["group"] == "orc_coverage" for r in recs)
    assert any(r["kind"] == "control" and r["method"] == "c" for r in recs)
    assert all(r["tags"]["level"] == "real" for r in recs)

    def boom(*a, **k):
        raise AssertionError("recomputed")

    monkeypatch.setattr(R, "_bootstrap", boom)
    again = R.run_report(spec)  # every interval record exists: nothing is recomputed
    assert len([r for r in again if r["tags"]["group"] == "interval"]) == len(
        [r for r in recs if r["tags"]["group"] == "interval"])

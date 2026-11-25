"""S4.3 Task 2: the curvature stage never builds an EventStore behind the caller's back; a strict ORC memo; run
files written before metrics.json and checked on resume.

Spec under test (docs/superpowers/plans/2026-11-25-s43-ablation-matrix.md, Task 2):
- curvature_stage builds no store and calls no escalate when every escalated non-self-loop row has a memo key; a
  NaN memo value is a hit; with require_orc_memo=True a miss raises OrcMemoMiss before any store exists.
- computing and memo cells return the same curvature columns (CURVATURE_COLUMNS).
- run_matrix prebuilds a store only without a memo and not strict, from events plus context rows.
- run_pipeline writes metrics.json last, with the bytes and sha256 of the other six run files; resume reruns a cell
  with a missing or changed file and raises on another configuration, data or commit.
"""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

import citecurv.curvature.orc as orc_mod
import citecurv.curvature.screen as screen_mod
import citecurv.curvature.timeslice as ts_mod
from citecurv.curvature.screen import CURVATURE_COLUMNS
from citecurv.data.synthetic import load_fixture
from citecurv.eval import ablation as A
from citecurv.eval.split import split_events
from citecurv.pipeline import OrcMemoMiss, _memo_key, curvature_stage, orc_memo_from_cache, resolve_config
from test_pipeline_integration import FakeORC, cfg_small

RUN_FILES = ("config.json", "scores.parquet", "weights.json", "zstats.json", "thresholds.json", "escalated.npy")


class Boom:
    def __init__(self, *a, **k):
        raise AssertionError("an EventStore was built")


@pytest.fixture()
def boom(monkeypatch):
    def arm():
        monkeypatch.setattr(ts_mod, "EventStore", Boom)
        monkeypatch.setattr(orc_mod, "EventStore", Boom)
    return arm


@pytest.fixture(scope="module")
def fx():
    return load_fixture()


@pytest.fixture()
def stream(fx, monkeypatch):
    """Fixture events plus one self-loop row that the screen is forced to escalate (the real data keeps
    self-loops); split and resolved config of a small `full` cell."""
    ev = fx["events"].copy()
    first = ev.iloc[5]
    loop = pd.DataFrame({"event_id": [int(ev["event_id"].max()) + 1], "src_id": [first["src_id"]],
                         "dst_id": [first["src_id"]], "t": [int(first["t"])]})
    ev = pd.concat([ev, loop], ignore_index=True).sort_values(["t", "event_id"], kind="mergesort")
    ev = ev.reset_index(drop=True)
    orig = screen_mod.apply_screen

    def forced(curv, th):
        out = orig(curv, th)
        out.loc[out["src_id"] == out["dst_id"], "escalated"] = True
        return out

    monkeypatch.setattr(screen_mod, "apply_screen", forced)
    split = split_events(ev, fx["meta"]["t_min"] - 1).to_numpy(dtype=object)
    rcfg = resolve_config({"pipeline": {"screen_percentile_low": 0.1, "screen_percentile_high": 0.9}},
                          variant="full")
    return ev, split, rcfg


def test_curvature_stage_builds_no_store_when_the_memo_covers_every_row(stream, boom):
    ev, split, rcfg = stream
    memo = {}
    first, _, s1 = curvature_stage(ev, split, rcfg, orc_fn=FakeORC(), orc_memo=memo)
    assert s1["n_self_loops_escalated"] == 1
    boom()
    fake = FakeORC()
    again, _, s2 = curvature_stage(ev, split, rcfg, orc_fn=fake, orc_memo=memo)
    pd.testing.assert_frame_equal(first, again)
    assert list(again.columns) == list(CURVATURE_COLUMNS)
    assert fake.calls == 0 and s2["n_orc_computed"] == 0
    assert s2["n_orc_memo_hits"] == s2["n_escalated"] - 1 > 0
    for k in ("orc_s_per_computed", "mean_subgraph_edges", "n_capped", "n_capped_lossy"):
        assert s2[k] is None


def test_strict_memo_raises_before_building_a_store(stream, boom):
    ev, split, rcfg = stream
    memo = {}
    curvature_stage(ev, split, rcfg, orc_fn=FakeORC(), orc_memo=memo)
    memo.pop(next(iter(memo)))
    boom()
    with pytest.raises(OrcMemoMiss, match="1 escalated row"):
        curvature_stage(ev, split, rcfg, orc_fn=FakeORC(), orc_memo=memo, require_orc_memo=True)


def test_nan_memo_value_counts_as_a_hit(stream, boom):
    ev, split, rcfg = stream
    memo = {}
    curvature_stage(ev, split, rcfg, orc_fn=FakeORC(), orc_memo=memo)
    key = next(iter(memo))
    memo[key] = math.nan
    boom()
    curv, _, stats = curvature_stage(ev, split, rcfg, orc_fn=FakeORC(), orc_memo=memo, require_orc_memo=True)
    row = curv[curv["event_id"] == key[0]]
    assert row["orc"].isna().item() and row["radius"].item() == key[4]
    assert stats["n_orc_computed"] == 0


def test_one_nan_hit_and_one_miss_compute_only_the_miss(stream):
    ev, split, rcfg = stream
    memo = {}
    curvature_stage(ev, split, rcfg, orc_fn=FakeORC(), orc_memo=memo)
    keys = iter(memo)
    nan_key, miss_key = next(keys), next(keys)
    memo[nan_key] = math.nan
    memo.pop(miss_key)
    fake = FakeORC()
    curv, _, stats = curvature_stage(ev, split, rcfg, orc_fn=fake, orc_memo=memo)
    assert fake.calls == 1 and stats["n_orc_computed"] == 1


def test_curvature_stage_builds_the_store_once_when_one_row_is_missing(stream, monkeypatch):
    ev, split, rcfg = stream
    memo = {}
    curvature_stage(ev, split, rcfg, orc_fn=FakeORC(), orc_memo=memo)
    memo.pop(next(iter(memo)))
    built = []
    real = ts_mod.EventStore

    class Counting(real):
        def __init__(self, *a, **k):
            built.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(ts_mod, "EventStore", Counting)
    monkeypatch.setattr(orc_mod, "EventStore", Counting)
    fake = FakeORC()
    curvature_stage(ev, split, rcfg, orc_fn=fake, orc_memo=memo)
    assert len(built) == 1 and fake.calls == 1


def _data(fx):
    return {"events": fx["events"], "works": fx["works"], "embeddings": fx["embeddings"],
            "event_labels": fx["event_labels"], "cutoff": 19180}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip(m):
    return {k: v for k, v in m.items() if k not in ("timings", "curvature", "files")}


def test_run_matrix_prebuilds_no_store_when_given_a_memo(fx, tmp_path, boom):
    memo = {}
    cells = ["full", "forman_only"]
    A.run_matrix(cells, [0], _data(fx), tmp_path / "a", cfg=cfg_small(), orc_fn=FakeORC(), orc_memo=memo)
    boom()
    second = A.run_matrix(cells, [0], _data(fx), tmp_path / "b", cfg=cfg_small(), orc_fn=FakeORC(), orc_memo=memo,
                          require_orc_memo=True)
    for c in cells:
        a, b = tmp_path / "a" / "runs" / f"{c}_0", tmp_path / "b" / "runs" / f"{c}_0"
        assert _sha(a / "scores.parquet") == _sha(b / "scores.parquet")
        ma, mb = (json.loads((d / "metrics.json").read_text()) for d in (a, b))
        assert _strip(ma) == _strip(mb)
    assert second[0]["curvature"]["n_orc_computed"] == 0


def test_run_matrix_store_holds_the_context_rows(fx, tmp_path, monkeypatch):
    ev = fx["events"]
    ctx = pd.DataFrame({"event_id": np.arange(3, dtype=np.int64) + int(ev["event_id"].max()) + 100,
                        "src_id": list(ev["dst_id"].iloc[:3]), "dst_id": ["ctx_r1", "ctx_r2", "ctx_r3"],
                        "t": np.array([int(ev["t"].min())] * 3, dtype=np.int64)})
    sizes = []
    real = ts_mod.EventStore

    class Recording(real):
        def __init__(self, rows, *a, **k):
            sizes.append(len(rows))
            super().__init__(rows, *a, **k)

    monkeypatch.setattr(ts_mod, "EventStore", Recording)
    monkeypatch.setattr(orc_mod, "EventStore", Recording)
    A.run_matrix(["full"], [0], _data(fx), tmp_path, cfg=cfg_small(), orc_fn=FakeORC(), context=ctx)
    assert sizes == [len(ev) + len(ctx)]


def test_resume_checks_files_and_configuration(fx, tmp_path):
    runs = tmp_path / "runs"
    A.run_matrix(["full", "no_curvature"], [0], _data(fx), tmp_path, cfg=cfg_small(), orc_fn=FakeORC(),
                 tags={"commit": "abc"})
    m = json.loads((runs / "full_0" / "metrics.json").read_text())
    assert set(m["files"]) == set(RUN_FILES)
    assert all(m["files"][f]["sha256"] == _sha(runs / "full_0" / f) for f in RUN_FILES)
    (runs / "full_0" / "thresholds.json").unlink()
    (runs / "no_curvature_0" / "weights.json").write_text("{")
    rerun = []
    A.run_matrix(["full", "no_curvature"], [0], _data(fx), tmp_path, cfg=cfg_small(), orc_fn=FakeORC(),
                 tags={"commit": "abc"}, resume=True, callback=lambda r: rerun.append(r.variant))
    assert rerun == ["full", "no_curvature"]
    skipped = []
    A.run_matrix(["full", "no_curvature"], [0], _data(fx), tmp_path, cfg=cfg_small(), orc_fn=FakeORC(),
                 tags={"commit": "abc"}, resume=True, callback=lambda r: skipped.append(r.variant))
    assert skipped == []
    with pytest.raises(ValueError, match="commit"):
        A.run_matrix(["full"], [0], _data(fx), tmp_path, cfg=cfg_small(), orc_fn=FakeORC(),
                     tags={"commit": "def"}, resume=True)
    with pytest.raises(ValueError, match="config"):
        A.run_matrix(["full"], [0], _data(fx), tmp_path, cfg=cfg_small(calibrate_weights=False), orc_fn=FakeORC(),
                     tags={"commit": "abc"}, resume=True)


def test_memo_from_cache_can_include_attempted_nan_rows():
    cache = pd.DataFrame({"event_id": [1, 2, 3], "src_id": ["a", "b", "c"], "dst_id": ["x", "y", "z"],
                          "t": [5, 5, 6], "orc": [-0.5, np.nan, np.nan], "radius": [1.0, 1.0, np.nan],
                          "orc_attempted": [True, True, False]})
    assert len(orc_memo_from_cache(cache, 1)) == 1
    memo = orc_memo_from_cache(cache, 1, include_attempted=True)
    assert set(memo) == {_memo_key("a", "x", 1, 5, 1), _memo_key("b", "y", 2, 5, 1)}
    assert math.isnan(memo[_memo_key("b", "y", 2, 5, 1)])
    plain = cache.drop(columns="orc_attempted")
    assert len(orc_memo_from_cache(plain, 1, include_attempted=True)) == 1


def test_resume_refuses_other_context_rows(fx, tmp_path):
    ev = fx["events"]
    ctx = pd.DataFrame({"event_id": np.arange(2, dtype=np.int64) + int(ev["event_id"].max()) + 100,
                        "src_id": list(ev["dst_id"].iloc[:2]), "dst_id": ["ctx_a", "ctx_b"],
                        "t": np.array([int(ev["t"].min())] * 2, dtype=np.int64)})
    A.run_matrix(["no_curvature"], [0], _data(fx), tmp_path, cfg=cfg_small(), orc_fn=FakeORC(), context=ctx)
    m = json.loads((tmp_path / "runs" / "no_curvature_0" / "metrics.json").read_text())
    assert m["context_hash"] != "none"
    A.run_matrix(["no_curvature"], [0], _data(fx), tmp_path, cfg=cfg_small(), orc_fn=FakeORC(), context=ctx,
                 resume=True)
    with pytest.raises(ValueError, match="context"):
        A.run_matrix(["no_curvature"], [0], _data(fx), tmp_path, cfg=cfg_small(), orc_fn=FakeORC(),
                     context=ctx.iloc[:1], resume=True)


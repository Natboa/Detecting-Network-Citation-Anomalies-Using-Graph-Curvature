"""S4.3 Task 3: one selection code path for cells and precompute, the ollivier_only eligibility rule, the resumable
ORC precompute and its job runner.

Spec under test (docs/superpowers/plans/2026-11-25-s43-ablation-matrix.md, Task 3):
- screen_stage + select_escalation reproduce today's curvature_stage selection (golden file of the unmodified code).
- forman_screen's ref_known = out_deg_t(dst) > 0 after the self-loop rule; with G13 on it equals ~isnan(forman).
- ollivier_only draws only among ref_known rows, with the screen's escalated count per split, and reads no Forman
  value beyond which rows are known.
- precompute_matrix_orc covers every cell's lookups, accepts unsorted events, resumes, grows and shrinks its seed
  list without losing rows, reuses `known` values only up to known_max_t, writes atomically with a sidecar it checks,
  refuses an ORC cap, marks rows in flight at a kill, and feeds strict cells through load_orc_memo.
- the job runner writes a heartbeat at each setup phase and before every ORC call.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

import citecurv.curvature.screen as screen_mod
from citecurv.curvature.screen import forman_screen
from citecurv.data.synthetic import load_fixture
from citecurv.eval.split import split_events
from citecurv import pipeline as P
from citecurv.pipeline import (
    OrcMemoMiss,
    _canonical_events,
    _memo_key,
    context_hash,
    curvature_stage,
    data_hash,
    frame_sha256,
    load_orc_memo,
    orc_settings_hash,
    precompute_matrix_orc,
    resolve_config,
    screen_stage,
    select_escalation,
    shuffle_test_timestamps,
)
from test_pipeline_integration import FakeORC

V1 = ["event_id", "src_id", "dst_id", "t", "forman", "orc", "escalated", "radius", "computed_at_stage"]
MODES = {"screen+orc": "full", "random+orc": "ollivier_only", "screen": "forman_only"}
GOLDEN = json.loads((__import__("pathlib").Path(__file__).parent / "fixtures" / "golden_escalation.json").read_text())


class Recorder:
    """Counts calls and remembers the scored keys; returns -0.25 for every edge, NaN or an error for chosen pairs."""

    def __init__(self, nan_for=(), raise_for=()):
        self.calls, self.edges = 0, []
        self.nan_for, self.raise_for = set(nan_for), set(raise_for)

    def __call__(self, G, edges, **knobs):
        self.calls += 1
        out = {}
        for e in edges:
            e = tuple(e)
            self.edges.append(e)
            if e in self.raise_for:
                raise RuntimeError(f"fake failure on {e}")
            out[e] = math.nan if e in self.nan_for else -0.25
        return out


def raising(G, edges, **knobs):
    raise AssertionError("ORC was computed")


@pytest.fixture(scope="module")
def fx():
    return load_fixture()


def cfg(**curv):
    return {"pipeline": {"screen_percentile_low": 0.1, "screen_percentile_high": 0.9},
            "curvature": {"screen_unknown_without_refs": True, **curv}}


def rc(c=None, variant="full", seed=0):
    return resolve_config(c or cfg(), variant=variant, seed=seed)


def cut(fx):
    return fx["meta"]["t_min"] - 1


def split_of(ev, cutoff, rcfg):
    return split_events(_canonical_events(ev), cutoff, val_frac=rcfg["split"]["val_frac"]).to_numpy(dtype=object)


def memo_for(out_path, ev, rcfg, context=None):
    return load_orc_memo(out_path, data_hash=data_hash(_canonical_events(ev)), context_hash=context_hash(context),
                         settings_hash=orc_settings_hash(rcfg), radius=rcfg["curvature"]["orc_radius"])


# ------------------------------------------------------------------ selection

def test_selection_matches_the_golden_file(fx):
    ev = fx["events"]
    split = split_events(ev, cut(fx)).to_numpy(dtype=object)
    for key, want in GOLDEN.items():
        mode, seed, cap = key.split("|")
        rcfg = resolve_config({"pipeline": {"screen_percentile_low": 0.1, "screen_percentile_high": 0.9,
                                            "orc_max_events": None if cap == "None" else int(cap)},
                               "curvature": {"screen_unknown_without_refs": False}},
                              variant=MODES[mode], seed=int(seed))
        curv, _, _ = curvature_stage(ev, split, rcfg, seed=int(seed), orc_fn=FakeORC())
        assert sorted(int(x) for x in curv.loc[curv["escalated"], "event_id"]) == want["ids"], key
        assert frame_sha256(curv[V1]) == want["sha"], key


def test_ref_known_matches_the_unknown_rule(fx):
    ev = fx["events"]
    loop = pd.DataFrame({"event_id": [10_000], "src_id": ["zz_alone"], "dst_id": ["zz_alone"],
                         "t": [int(ev["t"].max())]})
    ev2 = pd.concat([ev, loop], ignore_index=True)
    known = forman_screen(ev2, unknown_without_refs=True)
    plain = forman_screen(ev2, unknown_without_refs=False)
    assert "ref_known" in screen_mod.CURVATURE_COLUMNS
    np.testing.assert_array_equal(known["ref_known"].to_numpy(), ~np.isnan(known["forman"].to_numpy()))
    np.testing.assert_array_equal(known["ref_known"].to_numpy(), plain["ref_known"].to_numpy())
    assert not known.loc[known["event_id"] == 10_000, "ref_known"].item()
    assert (~known["ref_known"]).sum() >= 2


def test_ollivier_only_never_escalates_a_row_without_known_references(fx):
    ev = fx["events"]
    rcfg = rc(variant="ollivier_only")
    split = split_of(ev, cut(fx), rcfg)
    curv, _, _ = curvature_stage(_canonical_events(ev), split, rcfg, orc_fn=FakeORC())
    assert curv.loc[curv["escalated"], "ref_known"].all()
    screen, _, _ = screen_stage(_canonical_events(ev), split, rcfg)
    for part in ("train", "val", "test"):
        assert curv["escalated"][split == part].sum() == screen["escalated"][split == part].sum()


def test_ollivier_only_ignores_forman_values(fx, monkeypatch):
    ev = _canonical_events(fx["events"])
    rcfg = rc(variant="ollivier_only")
    split = split_of(ev, cut(fx), rcfg)
    before, _, _ = curvature_stage(ev, split, rcfg, orc_fn=FakeORC())
    orig = screen_mod.forman_screen

    def affine(*a, **k):
        out = orig(*a, **k)
        out["forman"] = 3.0 * out["forman"] - 7.0  # NaN stays NaN; every quantile flag is kept
        return out

    monkeypatch.setattr(screen_mod, "forman_screen", affine)
    after, _, _ = curvature_stage(ev, split, rcfg, orc_fn=FakeORC())
    np.testing.assert_array_equal(before["escalated"].to_numpy(), after["escalated"].to_numpy())


def test_select_escalation_reproduces_each_cell(fx):
    ev = _canonical_events(fx["events"])
    for mode, variant in MODES.items():
        rcfg = rc(variant=variant, seed=1)
        split = split_of(ev, cut(fx), rcfg)
        cell, _, _ = curvature_stage(ev, split, rcfg, seed=1, orc_fn=FakeORC())
        screen, _, _ = screen_stage(ev, split, rcfg)
        sel = select_escalation(screen, split, mode, 1, None, unknown_without_refs=True)
        np.testing.assert_array_equal(sel["escalated"].to_numpy(), cell["escalated"].to_numpy())


# ------------------------------------------------------------------ precompute

def test_precompute_covers_every_cell(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    out_path = tmp_path / "orc.parquet"
    res = precompute_matrix_orc(ev, cut(fx), rcfg, [0, 1], out_path=out_path, orc_fn=Recorder())
    assert res["n_pending"] == 0 and res["n_targets"] > 0
    memo = memo_for(out_path, ev, rcfg)
    cev = _canonical_events(ev)
    split = split_of(ev, cut(fx), rcfg)
    for variant, seed in (("full", 0), ("ollivier_only", 0), ("ollivier_only", 1)):
        r = rc(variant=variant, seed=seed)
        _, _, stats = curvature_stage(cev, split, r, seed=seed, orc_fn=raising, orc_memo=memo,
                                      require_orc_memo=True)
        assert stats["n_orc_computed"] == 0 and stats["n_escalated"] > 0


def test_precompute_accepts_unsorted_shuffled_events(fx, tmp_path):
    sh = shuffle_test_timestamps(fx["events"], cut(fx), seed=777)
    rcfg = rc()
    res = precompute_matrix_orc(sh, cut(fx), rcfg, [], out_path=tmp_path / "o.parquet", orc_fn=Recorder())
    run = P.run_pipeline(sh, fx["works"], embeddings=fx["embeddings"], event_labels=fx["event_labels"],
                         cutoff=cut(fx), variant="no_curvature", cfg=cfg())
    for k in ("n_train", "n_val", "n_test"):
        assert res["split_counts"][k] == run.metrics["split_counts"][k]


def test_precompute_resumes_without_recomputing(fx, tmp_path):
    ev, rcfg, rec = fx["events"], rc(), Recorder()
    out_path = tmp_path / "o.parquet"
    first = precompute_matrix_orc(ev, cut(fx), rcfg, [0], out_path=out_path, orc_fn=rec, chunk=3, max_chunks=1)
    assert first["n_pending"] > 0 and first["n_chunks_run"] == 1
    second = precompute_matrix_orc(ev, cut(fx), rcfg, [0], out_path=out_path, orc_fn=rec, chunk=3)
    assert second["n_pending"] == 0 and rec.calls == second["n_targets"]


def test_precompute_with_a_nan_row_finishes(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    probe = Recorder()
    precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=tmp_path / "p.parquet", orc_fn=probe)
    rec = Recorder(nan_for=[probe.edges[0]])
    res = precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=tmp_path / "o.parquet", orc_fn=rec)
    assert res["n_pending"] == 0 and res["n_attempted_nan"] == 1 and rec.calls == res["n_targets"]


def test_precompute_rerun_with_more_seeds_computes_only_new_targets(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    out_path = tmp_path / "o.parquet"
    one = precompute_matrix_orc(ev, cut(fx), rcfg, [0], out_path=out_path, orc_fn=Recorder())
    rec = Recorder()
    two = precompute_matrix_orc(ev, cut(fx), rcfg, [0, 1], out_path=out_path, orc_fn=rec)
    assert two["n_targets"] > one["n_targets"] and rec.calls == two["n_targets"] - one["n_targets"]


def test_rerun_with_fewer_seeds_keeps_computed_rows(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    out_path = tmp_path / "o.parquet"
    precompute_matrix_orc(ev, cut(fx), rcfg, [0, 1], out_path=out_path, orc_fn=Recorder())
    n_rows = len(pd.read_parquet(out_path))
    rec = Recorder()
    precompute_matrix_orc(ev, cut(fx), rcfg, [0], out_path=out_path, orc_fn=rec)
    assert rec.calls == 0 and len(pd.read_parquet(out_path)) == n_rows
    assert json.loads(out_path.with_suffix(".json").read_text())["seeds"] == [0, 1]


def swapped_stream(fx):
    """Two test events swap their times; every other event keeps its own t."""
    ev = _canonical_events(fx["events"]).copy()
    test = np.flatnonzero(ev["t"].to_numpy() > cut(fx))
    a = test[0]
    b = next(j for j in test[::-1] if ev.at[j, "t"] != ev.at[a, "t"])
    ev.loc[a, "t"], ev.loc[b, "t"] = ev.at[b, "t"], ev.at[a, "t"]
    return ev


def test_known_values_after_cutoff_are_recomputed(fx, tmp_path):
    rcfg = rc()
    orig_path = tmp_path / "orig.parquet"
    precompute_matrix_orc(fx["events"], cut(fx), rcfg, [], out_path=orig_path, orc_fn=Recorder())
    sh = swapped_stream(fx)
    fresh = Recorder()
    fres = precompute_matrix_orc(sh, cut(fx), rcfg, [], out_path=tmp_path / "fresh.parquet", orc_fn=fresh)
    rec = Recorder()
    res = precompute_matrix_orc(sh, cut(fx), rcfg, [], out_path=tmp_path / "sh.parquet", orc_fn=rec,
                                known=orig_path, known_max_t=cut(fx))
    ck = pd.read_parquet(tmp_path / "sh.parquet")
    n_test_targets = int((ck["t"] > cut(fx)).sum())
    assert n_test_targets > 0 and rec.calls == n_test_targets  # every test-period target computed, none reused
    assert res["n_known"] == fres["n_targets"] - n_test_targets
    with pytest.raises(ValueError, match="known_max_t"):
        precompute_matrix_orc(sh, cut(fx), rcfg, [], out_path=tmp_path / "x.parquet", orc_fn=rec, known=orig_path)


def test_known_nan_value_reaches_the_strict_cell(fx, tmp_path):
    rcfg = rc()
    probe = Recorder()
    precompute_matrix_orc(fx["events"], cut(fx), rcfg, [], out_path=tmp_path / "probe.parquet", orc_fn=probe)
    ck = pd.read_parquet(tmp_path / "probe.parquet")
    pre = ck[ck["t"] <= cut(fx)].iloc[0]
    orig_path = tmp_path / "orig.parquet"
    precompute_matrix_orc(fx["events"], cut(fx), rcfg, [], out_path=orig_path,
                          orc_fn=Recorder(nan_for=[(pre["src_id"], pre["dst_id"])]))
    sh = swapped_stream(fx)
    out_path = tmp_path / "sh.parquet"
    precompute_matrix_orc(sh, cut(fx), rcfg, [], out_path=out_path, orc_fn=Recorder(), known=orig_path,
                          known_max_t=cut(fx))
    memo = memo_for(out_path, sh, rcfg)
    key = _memo_key(pre["src_id"], pre["dst_id"], pre["event_id"], pre["t"], 1)
    assert key in memo and math.isnan(memo[key])
    curv, _, stats = curvature_stage(_canonical_events(sh), split_of(sh, cut(fx), rcfg), rcfg, orc_fn=raising,
                                     orc_memo=memo, require_orc_memo=True)
    assert stats["n_orc_computed"] == 0


def test_checkpoint_survives_a_truncated_temp_file(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    out_path = tmp_path / "o.parquet"
    precompute_matrix_orc(ev, cut(fx), rcfg, [0], out_path=out_path, orc_fn=Recorder())
    out_path.with_suffix(".tmp").write_bytes(b"garbage, not parquet")
    rec = Recorder()
    res = precompute_matrix_orc(ev, cut(fx), rcfg, [0], out_path=out_path, orc_fn=rec)
    assert rec.calls == 0 and res["n_pending"] == 0


def test_precompute_refuses_other_settings(fx, tmp_path):
    ev = fx["events"]
    out_path = tmp_path / "o.parquet"
    precompute_matrix_orc(ev, cut(fx), rc(), [0], out_path=out_path, orc_fn=Recorder())
    other = resolve_config({"pipeline": {"screen_percentile_low": 0.2, "screen_percentile_high": 0.9},
                            "curvature": {"screen_unknown_without_refs": True}}, variant="full")
    with pytest.raises(ValueError, match="settings"):
        precompute_matrix_orc(ev, cut(fx), other, [0], out_path=out_path, orc_fn=Recorder())
    with pytest.raises(ValueError, match="cutoff"):
        precompute_matrix_orc(ev, cut(fx) - 30, rc(), [0], out_path=out_path, orc_fn=Recorder())


def test_memo_loader_refuses_another_context(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    out_path = tmp_path / "o.parquet"
    precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=out_path, orc_fn=Recorder())
    other = pd.DataFrame({"event_id": [99_999], "src_id": ["c1"], "dst_id": ["c2"], "t": [int(ev["t"].min())]})
    with pytest.raises(ValueError, match="context"):
        memo_for(out_path, ev, rcfg, context=other)


def test_precompute_refuses_an_orc_cap(fx, tmp_path):
    capped = resolve_config({"pipeline": {"screen_percentile_low": 0.1, "screen_percentile_high": 0.9,
                                          "orc_max_events": 5}}, variant="full")
    with pytest.raises(ValueError, match="orc_max_events"):
        precompute_matrix_orc(fx["events"], cut(fx), capped, [], out_path=tmp_path / "o.parquet", orc_fn=Recorder())


def test_known_checkpoint_from_other_settings_is_refused(fx, tmp_path):
    orig_path = tmp_path / "orig.parquet"
    other = resolve_config({"pipeline": {"screen_percentile_low": 0.2, "screen_percentile_high": 0.9},
                            "curvature": {"screen_unknown_without_refs": True}}, variant="full")
    precompute_matrix_orc(fx["events"], cut(fx), other, [], out_path=orig_path, orc_fn=Recorder())
    with pytest.raises(ValueError, match="known"):
        precompute_matrix_orc(swapped_stream(fx), cut(fx), rc(), [], out_path=tmp_path / "s.parquet",
                              orc_fn=Recorder(), known=orig_path, known_max_t=cut(fx))


def test_precompute_error_row_reaches_the_strict_cell(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    probe = Recorder()
    precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=tmp_path / "p.parquet", orc_fn=probe)
    bad = probe.edges[1]
    out_path = tmp_path / "o.parquet"
    res = precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=out_path, orc_fn=Recorder(raise_for=[bad]),
                                max_error_share=0.5)
    assert res["n_errors"] == 1 and res["n_pending"] == 0
    memo = memo_for(out_path, ev, rcfg)
    curv, _, stats = curvature_stage(_canonical_events(ev), split_of(ev, cut(fx), rcfg), rcfg, orc_fn=raising,
                                     orc_memo=memo, require_orc_memo=True)
    row = curv[(curv["src_id"] == bad[0]) & (curv["dst_id"] == bad[1]) & curv["escalated"]]
    assert len(row) and row["orc"].isna().all() and stats["n_orc_computed"] == 0


def test_precompute_stops_on_too_many_errors(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    probe = Recorder()
    precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=tmp_path / "p.parquet", orc_fn=probe)
    with pytest.raises(RuntimeError, match="error"):
        precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=tmp_path / "o.parquet",
                              orc_fn=Recorder(raise_for=probe.edges[:3]))


def test_rows_in_flight_at_a_kill_are_marked_on_restart(fx, tmp_path):
    ev, rcfg = fx["events"], rc()
    probe = Recorder()
    precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=tmp_path / "p.parquet", orc_fn=probe)
    ck = pd.read_parquet(tmp_path / "p.parquet")
    victim = int(ck["event_id"].iloc[0])
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"phase": "orc", "time": 0, "in_flight": [victim], "clean": False}))
    rec = Recorder()
    out_path = tmp_path / "o.parquet"
    res = precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=out_path, orc_fn=rec, heartbeat=hb,
                                max_error_share=0.5)
    got = pd.read_parquet(out_path).set_index("event_id")
    assert got.loc[victim, "orc_error"] == "killed: timeout" and math.isnan(got.loc[victim, "orc"])
    assert rec.calls == res["n_targets"] - 1
    again = Recorder()
    precompute_matrix_orc(ev, cut(fx), rcfg, [], out_path=out_path, orc_fn=again, retry_errors=True, heartbeat=hb)
    assert again.calls == 1
    assert pd.read_parquet(out_path).set_index("event_id").loc[victim, "orc"] == -0.25
    assert json.loads(hb.read_text())["clean"] is True


# ------------------------------------------------------------------ job runner

def test_orc_job_writes_a_heartbeat_per_batch(fx, tmp_path, monkeypatch):
    from citecurv.eval import orc_job

    ev, works = fx["events"], fx["works"]
    ev.to_parquet(tmp_path / "events.parquet")
    works[["work_id"]].to_parquet(tmp_path / "works.parquet")
    refs = pd.DataFrame({"work_id": ["cited_x"], "pub_date": ["2018-01-05"], "ref_ids": [["r1", "r2"]],
                         "title": ["t"]})
    refs.to_parquet(tmp_path / "cited_joint.parquet")
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg()))
    seen = []
    real = P._write_heartbeat

    def spy(path, payload):
        seen.append(dict(payload))
        real(path, payload)

    monkeypatch.setattr(P, "_write_heartbeat", spy)
    job = {"events": str(tmp_path / "events.parquet"), "works": str(tmp_path / "works.parquet"),
           "cited_joint": str(tmp_path / "cited_joint.parquet"), "shuffle": None, "config": str(tmp_path / "cfg.yaml"),
           "cutoff": cut(fx), "seeds": [0], "out_path": str(tmp_path / "orc.parquet"), "known": None,
           "known_max_t": None, "batch": 1, "chunk": 4, "worker": False, "order_seed": 0, "max_chunks": None,
           "heartbeat": str(tmp_path / "hb.json"), "result": str(tmp_path / "result.json")}
    rec = Recorder()
    result = orc_job.run_job(job, orc_fn=rec)
    phases = [p["phase"] for p in seen]
    assert phases[0] == "load" and "screen" in phases and phases[-1] == "done"
    assert sum(p == "orc" for p in phases) == rec.calls == result["n_done"] > 0
    assert json.loads((tmp_path / "result.json").read_text())["n_pending"] == 0
    assert json.loads((tmp_path / "hb.json").read_text())["clean"] is True

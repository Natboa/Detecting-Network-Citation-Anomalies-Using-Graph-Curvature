"""Option A (plan docs/superpowers/plans/2026-11-26-orc-dir-closed-form.md, Task 1): the curvature stage's closed mode.

Mode "closed" computes, for every event, the directed Ollivier-Ricci curvature by its closed form
(docs/curvature_formulation_study.md, Proposition 2): orc_dir = -(1 - alpha) * (1 - n_shared / n_refs_dst), NaN where
the cited work has no known reference (G13) and on self-loop rows, with the reference overlap computed on the same
graph state and context as the Forman screen. citing_cited_at_t flags rows where the premise of Proposition 2 fails
(the citing work already has a citer at t). A passed coupling frame is accepted only when it belongs to these events.
"""

import copy

import numpy as np
import pandas as pd
import pytest

import citecurv.curvature.overlap as overlap_mod
from citecurv.curvature.overlap import reference_overlap
from citecurv.curvature.screen import CURVATURE_COLUMNS, ORC_DIR_COLUMNS, forman_screen
from citecurv.curvature.variants import orc_directed_uncited
from citecurv.pipeline import (_canonical_events, curvature_stage, orc_settings_hash, resolve_config,
                               run_config_hash, shuffle_test_timestamps)
from test_overlap import _frame, _stream
from test_pipeline_integration import cfg_small


def _rcfg(**curv):
    cfg = cfg_small()
    cfg["curvature"] = {**(cfg.get("curvature") or {}), **curv}
    rcfg = resolve_config(cfg, variant="forman_only")
    rcfg["variant_spec"]["curvature"] = "closed"  # no variant uses the mode before Task 3
    return rcfg


def _stage(events, rcfg, context=None, coupling=None):
    ev = _canonical_events(events)
    split = np.full(len(ev), "train", dtype=object)
    return curvature_stage(ev, split, rcfg, context=context, coupling=coupling)


def test_closed_mode_orc_dir_equals_closed_form():
    events, ctx = _stream(3, 30, 400, 300, 40)
    curv, _, stats = _stage(events, _rcfg(), context=ctx)
    assert list(curv.columns) == CURVATURE_COLUMNS + ORC_DIR_COLUMNS
    ov = reference_overlap(_canonical_events(events), context=ctx)
    assert curv["event_id"].tolist() == ov["event_id"].tolist()
    assert (curv["n_refs_dst"].to_numpy() == ov["n_refs_dst"].to_numpy()).all()
    assert (curv["n_shared"].to_numpy() == ov["n_shared"].to_numpy()).all()
    loop = (curv["src_id"] == curv["dst_id"]).to_numpy()
    k = curv["n_refs_dst"].to_numpy()
    want = np.where((k > 0) & ~loop, orc_directed_uncited(curv["n_shared"].to_numpy(), k, 0.5), np.nan)
    got = curv["orc_dir"].to_numpy(dtype=float)
    assert loop.any() and (k == 0).any()  # the stream exercises both NaN cases
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=0, atol=1e-12)
    assert not curv["escalated"].any() and curv["orc"].isna().all()
    assert stats["mode"] == "closed" and stats["n_orc_computed"] == 0
    assert stats["n_orc_dir_known"] == int(np.isfinite(got).sum())
    for key in ("n_selected", "n_eligible", "orc_subsampled"):
        assert key in stats


@pytest.mark.parametrize("count_duplicates", [False, True])
def test_premise_flag_from_in_degree(count_duplicates):
    rows = [
        ("v", "a", 1), ("v", "a", 1), ("v", "b", 1),   # v's references, one duplicated
        ("x", "u", 5), ("x", "u", 5),                   # u has a citer, the row duplicated
        ("w", "u", 10),                                 # and a citer at the same t as u's own citation
        ("u", "v", 10), ("u", "a", 10),                 # u cites v (and a, a shared reference)
        ("y", "z", 10),                                 # z has no known reference: NaN F, NaN orc_dir
        ("q", "v", 10),                                 # q is uncited
    ]
    events = _frame(rows)
    rcfg = _rcfg(screen_count_duplicates=count_duplicates)
    curv, _, stats = _stage(events, rcfg)
    degrees = forman_screen(_canonical_events(events), count_duplicates=count_duplicates,
                            unknown_without_refs=True, return_degrees=True)
    assert (curv["citing_cited_at_t"].to_numpy() == (degrees["in_deg_src"].to_numpy() > 0)).all()
    flagged = set(curv.loc[curv["citing_cited_at_t"], "src_id"])
    assert flagged == {"u"}
    yz = curv.set_index("event_id").loc[8]
    assert np.isnan(yz["orc_dir"]) and yz["n_refs_dst"] == 0
    uv = curv.set_index("event_id").loc[6]
    assert uv["n_refs_dst"] == 2 and uv["n_shared"] == 1 and uv["orc_dir"] == pytest.approx(-0.25)
    finite = np.isfinite(curv["orc_dir"].to_numpy(dtype=float))
    n_flag = int((curv["citing_cited_at_t"].to_numpy() & finite).sum())
    assert stats["n_citing_cited_at_t"] == n_flag
    assert stats["share_citing_cited_at_t"] == pytest.approx(n_flag / finite.sum())
    # the coupling computed outside is accepted under both duplicate rules
    passed, _, _ = _stage(events, rcfg, coupling=reference_overlap(_canonical_events(events)))
    pd.testing.assert_frame_equal(passed, curv)


def test_context_rows_enter_the_coupling():
    # ("t0", "c", 1) is an event row, so the screen has a training value with or without the context
    events = _frame([("u", "v", 10), ("u", "a", 10), ("s", "t0", 2), ("t0", "c", 1)])
    ctx = _frame([("v", "a", 1), ("v", "b", 1)], start_id=100)
    curv, _, _ = _stage(events, _rcfg(), context=ctx)
    uv = curv.set_index("event_id").loc[0]
    assert uv["n_refs_dst"] == 2 and uv["n_shared"] == 1 and uv["orc_dir"] == pytest.approx(-0.25)
    alone, _, _ = _stage(events, _rcfg(), context=None)
    assert np.isnan(alone.set_index("event_id").loc[0, "orc_dir"])  # without the context v cites nothing


def test_coupling_from_other_stream_raises(monkeypatch):
    events, ctx = _stream(5, 25, 300, 200, 30)
    rcfg = _rcfg()
    ev = _canonical_events(events)
    good = reference_overlap(ev, context=ctx)
    shuffled = reference_overlap(_canonical_events(shuffle_test_timestamps(events, 15, seed=1)), context=ctx)
    with pytest.raises(ValueError):
        _stage(events, rcfg, context=ctx, coupling=shuffled)
    with pytest.raises(ValueError):
        _stage(events, rcfg, context=ctx, coupling=good.iloc[1:])
    with pytest.raises(ValueError):
        _stage(events, rcfg, context=ctx, coupling=good.drop(columns=["t"]))
    with pytest.raises(ValueError):
        _stage(events, rcfg, context=None, coupling=good)  # computed with another context
    computed, _, _ = _stage(events, rcfg, context=ctx)
    calls = []
    real = overlap_mod.reference_overlap
    monkeypatch.setattr(overlap_mod, "reference_overlap", lambda *a, **k: calls.append(1) or real(*a, **k))
    passed, _, _ = _stage(events, rcfg, context=ctx, coupling=good.sample(frac=1.0, random_state=0))
    assert calls == []
    pd.testing.assert_frame_equal(passed, computed)


def test_formulation_changes_config_hash():
    rc = resolve_config(cfg_small(), variant="full")
    assert rc["scoring"]["components"]["formulation"] == "orc_dir_closed_v1"
    old = copy.deepcopy(rc)
    del old["scoring"]["components"]["formulation"]
    assert run_config_hash(rc) != run_config_hash(old)
    assert orc_settings_hash(rc) == orc_settings_hash(old)


def test_resume_refuses_old_formulation_folder(tmp_path, monkeypatch):
    import citecurv.pipeline as P
    from citecurv.data.synthetic import load_fixture
    from citecurv.eval import ablation as A
    from test_pipeline_integration import FakeORC

    fx = load_fixture()
    data = {"events": fx["events"], "works": fx["works"], "embeddings": fx["embeddings"],
            "event_labels": fx["event_labels"], "cutoff": 19180}
    real = P.resolve_config

    def old_config(*a, **k):
        rc = real(*a, **k)
        rc["scoring"]["components"].pop("formulation", None)
        return rc

    monkeypatch.setattr(P, "resolve_config", old_config)
    A.run_matrix(["no_curvature"], [0], data, tmp_path, cfg=cfg_small(), orc_fn=FakeORC())
    monkeypatch.setattr(P, "resolve_config", real)
    with pytest.raises(ValueError, match="another config"):
        A.run_matrix(["no_curvature"], [0], data, tmp_path, cfg=cfg_small(), orc_fn=FakeORC(), resume=True)

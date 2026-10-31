"""Integration: the full pipeline on the committed mini fixture (plan 9.4, book 4.2.2 (ii) and (iii)).

Spec under test (citecurv.pipeline, CONTRACTS.md "Scores file", IMPLEMENTATION_PLAN.md S3.4/S4.3):
- run_pipeline on tests/fixtures/mini_stream (197 events, one planted cartel, all positives on or after
  meta["t_min"]) with a fake ORC (closed form for an uncited citing paper, no GraphRicciCurvature) and static
  embeddings returns a scores frame whose columns start with exactly the scores.parquet contract columns, in
  order, with the contract dtypes, one row per event.
- Value ranges: total finite on scored rows and NaN on cold-start rows; cold_start equals a brute-force
  recount (fewer than k citations of dst with ts < t); s_geom and s_curvdyn are >= 0; weights lie on the
  simplex; ORC values only on escalated rows; screen thresholds are the train-row quantiles.
- Every ablation cell runs; disabled components are all-NaN with weight 0; no_temporal ignores TGN outputs
  and (by default) switches s_temp off; forman_only computes no ORC; ollivier_only escalates as many rows as
  the screen flags.
- Seeded double run -> byte-identical scores.parquet (sha256), the NFR4 hash compare.
- Train-only fitting: shuffling test-period timestamps changes no threshold, z-statistic or weight.
- Metrics come from eval.metrics.evaluate on non-cold-start test rows.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from citecurv.config import load_config
from citecurv.data.synthetic import load_fixture
from citecurv.eval.ablation import VARIANTS
from citecurv.eval.metrics import evaluate
from citecurv.pipeline import (
    baseline_record,
    evaluate_baseline,
    frame_sha256,
    gate_cold_start,
    gate_fairness,
    gate_reproducibility,
    gate_shuffle,
    gate_top_breakdown,
    orc_memo_from_cache,
    run_pipeline,
    scores_sha256,
    shuffle_test_timestamps,
)
from citecurv.scoring.aggregate import SCORES_COLUMNS, read_scores
from citecurv.scoring.combiner import COMPONENTS, ZStats, calibrate_weights


class FakeORC:
    """kappa(u, v) = -(1 - alpha) * (1 - |succ(u) & succ(v)| / |succ(v)|), 0 when v cites nothing: the exact
    directed ORC of an uncited citing paper (curvature/orc.py docstring). Counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, G, edges, alpha=0.5, **knobs):
        out = {}
        for u, v in edges:
            self.calls += 1
            sv = set(G.successors(v)) - {v}
            su = set(G.successors(u)) - {u}
            out[(u, v)] = 0.0 if not sv else -(1 - alpha) * (1 - len(su & sv) / len(sv))
        return out


@pytest.fixture(scope="module")
def fx():
    return load_fixture()


def cfg_small(**pipeline):
    cfg = load_config()
    cfg["scoring"]["components"]["min_cell"] = 5
    cfg["pipeline"] = {"screen_percentile_low": 0.1, "screen_percentile_high": 0.9, **pipeline}
    return cfg


def run(fx, variant="full", seed=0, cfg=None, orc=None, **kw):
    kw.setdefault("cutoff", fx["meta"]["t_min"] - 1)
    kw.setdefault("event_labels", fx["event_labels"])
    return run_pipeline(fx["events"], fx["works"], authorships=fx["authorships"], embeddings=fx["embeddings"],
                        variant=variant, seed=seed, cfg=cfg or cfg_small(), orc_fn=orc or FakeORC(), **kw)


def test_scores_schema_is_exactly_the_contract(fx, tmp_path):
    res = run(fx, out_dir=tmp_path / "full_0")
    s = res.scores
    assert list(s.columns[: len(SCORES_COLUMNS)]) == SCORES_COLUMNS
    assert len(s) == len(fx["events"])
    assert s["event_id"].is_unique and set(s["event_id"]) == set(fx["events"]["event_id"])
    assert list(s["event_id"]) == list(fx["events"].sort_values(["t", "event_id"])["event_id"])
    for col, dt in {"event_id": "int64", "t": "int64", "seed": "int64", "cold_start": "bool",
                    "total": "float64", **{c: "float64" for c in COMPONENTS}}.items():
        assert str(s[col].dtype) == dt, col
    assert (s["model_variant"] == "full").all() and (s["seed"] == 0).all()
    assert s["config_hash"].nunique() == 1 and s["config_hash"].iloc[0] == res.config_hash
    back = read_scores(tmp_path / "full_0" / "scores.parquet")
    assert list(back.columns) == list(s.columns)
    pd.testing.assert_frame_equal(back, s.reset_index(drop=True))
    for name in ("metrics.json", "config.json", "weights.json", "zstats.json", "thresholds.json"):
        assert (tmp_path / "full_0" / name).is_file(), name
    cfg_file = json.loads((tmp_path / "full_0" / "config.json").read_text())
    assert cfg_file["config_hash"] == res.config_hash and cfg_file["variant"] == "full"
    zs = ZStats.from_dict(json.loads((tmp_path / "full_0" / "zstats.json").read_text()))
    assert zs.to_dict() == res.zstats.to_dict()


def test_value_ranges_and_cold_start_rule(fx):
    res = run(fx)
    s, ev = res.scores, fx["events"]
    k = res.config["scoring"]["cold_start_k"]
    prior = [int(((ev["dst_id"] == d) & (ev["t"] < t)).sum()) for d, t in zip(s["dst_id"], s["t"])]
    assert s["cold_start"].tolist() == [p < k for p in prior]
    scored = ~s["cold_start"]
    assert np.isfinite(s.loc[scored, "total"]).all()
    assert s.loc[~scored, "total"].isna().all()
    for c in ("s_geom", "s_curvdyn"):
        v = s[c].dropna()
        assert len(v) and (v >= 0).all(), c
    assert np.isfinite(s[["s_embed", "s_geom", "s_temp", "s_curvdyn"]].to_numpy()[scored.to_numpy()]).any()
    w = np.array([res.weights[c] for c in COMPONENTS])
    assert (w >= 0).all() and math.isclose(w.sum(), 1.0, abs_tol=1e-9)
    assert res.weights_source == "fixed_config"  # the fixture's val period has no positive
    cur = res.curvature
    assert cur["orc"].notna().sum() > 0
    assert cur.loc[~cur["escalated"], "orc"].isna().all()
    assert cur.loc[cur["escalated"] & (cur["src_id"] != cur["dst_id"]), "orc"].between(-1, 1).all()
    train = (res.split == "train").to_numpy()
    f = cur["forman"].to_numpy()[train]
    assert res.thresholds["global"]["lo"] == pytest.approx(np.quantile(f, 0.1))
    assert res.thresholds["global"]["hi"] == pytest.approx(np.quantile(f, 0.9))
    assert set(res.split) == {"train", "val", "test"}
    assert (s.loc[res.split.to_numpy() == "test", "t"] >= fx["meta"]["t_min"]).all()


def test_metrics_use_the_shared_metric_path_on_scored_test_rows(fx):
    res = run(fx)
    s = res.scores
    test = s[(res.split == "test").to_numpy()]
    want = evaluate(test[["event_id", "total", "cold_start"]], fx["event_labels"], ks=(100, 1000))
    assert res.metrics["test"] == want
    assert res.metrics["test"]["n"] == int((~test["cold_start"]).sum())
    lab = fx["event_labels"].set_index("event_id")["label"]
    assert res.metrics["test_n_pos_all"] == int(lab.reindex(test["event_id"]).sum())
    assert res.metrics["test_n_pos_all"] == fx["meta"]["n_positive"]  # every positive is post-cutoff
    assert set(res.metrics["test_components"]) == set(COMPONENTS)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_runs(fx, variant):
    orc = FakeORC()
    res = run(fx, variant=variant, orc=orc)
    s = res.scores
    assert list(s.columns[: len(SCORES_COLUMNS)]) == SCORES_COLUMNS
    assert (s["model_variant"] == variant).all()
    disabled = set(res.metrics["disabled_components"])
    for c in disabled:
        assert s[c].isna().all(), c
        assert res.weights[c] == 0.0
    if VARIANTS[variant].combiner == "calibrated":
        w = np.array([res.weights[c] for c in COMPONENTS])
        assert (w >= 0).all() and math.isclose(w.sum(), 1.0, abs_tol=1e-9)
    assert np.isfinite(s.loc[~s["cold_start"], "total"]).all()
    if variant == "no_curvature":
        assert disabled == {"s_geom", "s_curvdyn"} and res.curvature is None and orc.calls == 0
    if variant == "forman_only":
        assert orc.calls == 0 and not res.curvature["escalated"].any()
        assert s["s_geom"].notna().any()
    if variant == "ollivier_only":
        n_screen = res.metrics["curvature"]["n_screen_flagged"]
        assert int(res.curvature["escalated"].sum()) == n_screen
        assert s.loc[~res.curvature["escalated"].to_numpy(), "s_geom"].isna().all()
    if variant == "no_temporal":
        assert disabled == {"s_temp"} and res.metrics["embed_mode"] == "static_cosine"
    if variant == "full_learned":
        assert res.weights_source == "fixed_config"  # no val positive in the fixture -> documented fallback
        assert any("learned head not fitted" in n for n in res.metrics["notes"])


@pytest.mark.parametrize("shuffle_seed", [0, 1, 3])
def test_ollivier_only_fitted_objects_ignore_test_period(fx, shuffle_seed):
    """Regression (NFR3): the matched random escalation of ollivier_only must not let test-period events choose
    which pre-cutoff rows get ORC. Shuffling test timestamps changes the number of test-period screen flags;
    the pre-cutoff escalated set, thresholds, ZStats and weights must stay identical."""
    cut = fx["meta"]["t_min"] - 1
    a = run(fx, variant="ollivier_only")
    ev = shuffle_test_timestamps(fx["events"], cut, seed=shuffle_seed)
    b = run_pipeline(ev, fx["works"], embeddings=fx["embeddings"], event_labels=fx["event_labels"], cutoff=cut,
                     variant="ollivier_only", seed=0, cfg=cfg_small(), orc_fn=FakeORC())
    for r in (a, b):
        assert int(r.curvature["escalated"].sum()) == r.metrics["curvature"]["n_screen_flagged"]

    def pre_escalated(r):
        pre = (r.split != "test").to_numpy() & r.curvature["escalated"].to_numpy()
        return set(r.curvature.loc[pre, "event_id"])

    assert pre_escalated(a) == pre_escalated(b)
    assert a.thresholds == b.thresholds
    assert a.zstats.to_dict() == b.zstats.to_dict()
    assert a.weights == b.weights


def test_no_temporal_flag_keeps_s_temp_when_off(fx):
    on = run(fx, variant="no_temporal")
    off = run(fx, variant="no_temporal", cfg=cfg_small(no_temporal_drop_s_temp=False))
    full = run(fx, variant="full")
    assert on.scores["s_temp"].isna().all()
    assert off.metrics["disabled_components"] == []
    # with static embeddings the encoder-only reading has nothing left to ablate: identical to full
    np.testing.assert_array_equal(off.scores["total"].to_numpy(), full.scores["total"].to_numpy())


def test_temporal_stat_config_reaches_the_components(fx):
    from citecurv.scoring.components import score_events

    base = run(fx)
    assert base.config["scoring"]["components"]["temporal_stat"] == "rate_ratio"  # default unchanged
    cfg = cfg_small()
    cfg["scoring"]["components"]["temporal_stat"] = "poisson_tail"
    tail = run(fx, cfg=cfg)
    assert tail.config["scoring"]["components"]["temporal_stat"] == "poisson_tail"
    assert tail.config_hash != base.config_hash
    direct = score_events(fx["events"], fx["works"], None, embeddings=fx["embeddings"], min_cell=5,
                          temporal_stat="poisson_tail", variant="no_curvature")
    np.testing.assert_allclose(tail.components["s_temp"].to_numpy(), direct["s_temp"].to_numpy(),
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    assert not np.allclose(tail.components["s_temp"].to_numpy(), base.components["s_temp"].to_numpy(),
                           equal_nan=True)


def test_tgn_outputs_reach_full_but_not_no_temporal(fx):
    ev = fx["events"]
    p = np.linspace(0.05, 0.95, len(ev))
    tgn = pd.DataFrame({"event_id": ev["event_id"].to_numpy(), "p_link": p})
    full = run(fx, variant="full", tgn_scores=tgn)
    nt = run(fx, variant="no_temporal", tgn_scores=tgn)
    want = pd.Series(-np.log(p), index=ev["event_id"].to_numpy())
    got = full.scores.set_index("event_id")["s_embed"]
    np.testing.assert_allclose(got.to_numpy(), want.reindex(got.index).to_numpy())
    assert full.metrics["embed_mode"] == "tgn" and nt.metrics["embed_mode"] == "static_cosine"


def test_seeded_double_run_is_byte_identical(fx, tmp_path):
    a = run(fx, out_dir=tmp_path / "a")
    b = run(fx, out_dir=tmp_path / "b")
    assert scores_sha256(tmp_path / "a" / "scores.parquet") == scores_sha256(tmp_path / "b" / "scores.parquet")
    assert frame_sha256(a.scores) == frame_sha256(b.scores)
    assert a.config_hash == b.config_hash and a.data_hash == b.data_hash


def test_orc_memo_reuse_leaves_output_unchanged(fx):
    memo = {}
    first = run(fx, orc_memo=memo)
    assert len(memo) == first.metrics["curvature"]["n_orc_computed"] > 0
    orc = FakeORC()
    again = run(fx, orc_memo=memo, orc=orc)
    assert orc.calls == 0
    assert frame_sha256(first.scores) == frame_sha256(again.scores)
    pd.testing.assert_frame_equal(first.curvature, again.curvature)


def test_fitted_objects_do_not_see_the_test_period(fx):
    base = run(fx)
    ev = fx["events"].copy()
    test = ev["t"] > fx["meta"]["t_min"] - 1
    rng = np.random.default_rng(0)
    ev.loc[test, "t"] = rng.permutation(ev.loc[test, "t"].to_numpy())
    shuffled = run_pipeline(ev, fx["works"], embeddings=fx["embeddings"], event_labels=fx["event_labels"],
                            cutoff=fx["meta"]["t_min"] - 1, variant="full", seed=0, cfg=cfg_small(),
                            orc_fn=FakeORC())
    assert shuffled.thresholds == base.thresholds
    assert shuffled.zstats.to_dict() == base.zstats.to_dict()
    assert shuffled.weights == base.weights
    pre = (base.split != "test").to_numpy()
    a = base.scores.loc[pre].reset_index(drop=True)
    b = shuffled.scores.loc[(shuffled.split != "test").to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(a.drop(columns="config_hash"), b.drop(columns="config_hash"))


def test_calibrated_and_learned_weights_when_val_has_positives(fx):
    # cutoff inside the cartel window: 5 positives fall in val (one of them past cold start), 3 in test
    cutoff = 19180
    res = run(fx, cutoff=cutoff)
    assert res.metrics["split_counts"]["n_pos_val"] == 5
    assert res.metrics["calibration"]["n_val_pos"] == 1
    assert res.weights_source == "calibrated_val"
    w = np.array([res.weights[c] for c in COMPONENTS])
    assert (w >= 0).all() and math.isclose(w.sum(), 1.0, abs_tol=1e-9)
    val = ((res.split == "val") & ~res.scores["cold_start"]).to_numpy()
    lab = fx["event_labels"].set_index("event_id")["label"]
    y = lab.reindex(res.components.loc[val, "event_id"]).to_numpy()
    want, _, _ = calibrate_weights(res.components.loc[val], y, res.zstats, step=0.1, metric="ap")
    assert res.weights == pytest.approx(want)
    learned = run(fx, variant="full_learned", cutoff=cutoff)
    assert learned.weights_source == "learned_val" and learned.learned is not None
    scored = ~learned.scores["cold_start"]
    np.testing.assert_allclose(learned.scores.loc[scored, "total"],
                               learned.learned.predict_score(learned.components.loc[scored.to_numpy()]))


def test_unknown_variant_raises(fx):
    with pytest.raises(ValueError, match="unknown variant"):
        run(fx, variant="no_such_cell")


def test_calibration_can_be_switched_off(fx):
    res = run(fx, cutoff=19180, cfg=cfg_small(calibrate_weights=False))
    assert res.weights_source == "fixed_config"
    assert res.weights == {c: 0.25 for c in COMPONENTS}


def test_orc_memo_from_cache_matches_the_run_memo(fx):
    memo = {}
    res = run(fx, orc_memo=memo)
    assert orc_memo_from_cache(res.curvature, radius=1) == memo
    assert orc_memo_from_cache(res.curvature, radius=2) == {}


def test_sanity_gates_on_fixture(fx, tmp_path):
    a = run(fx, out_dir=tmp_path / "a")
    b = run(fx, out_dir=tmp_path / "b")
    g = gate_reproducibility(tmp_path / "a" / "scores.parquet", tmp_path / "b" / "scores.parquet")
    assert g["status"] == "PASS"
    cs = gate_cold_start(a, [b.metrics])
    assert cs["status"] == "PASS" and all(cs["value"][k] for k in ("flag_matches_recount", "total_nan_iff_cold"))
    broken = run(fx)
    broken.scores.loc[broken.scores.index[-1], "cold_start"] = not broken.scores["cold_start"].iloc[-1]
    assert gate_cold_start(broken)["status"] == "FAIL"
    fair = gate_fairness(a, fx["works"], flag_share=0.5, max_share=1.0)
    assert fair["status"] == "PASS" and fair["value"]["n_flags"] == math.ceil(0.5 * a.metrics["test"]["n"])
    assert gate_fairness(a, fx["works"], flag_share=0.5, max_share=0.0)["status"] == "FAIL"
    top = gate_top_breakdown(a, k=3)
    assert top["value"]["k"] == 3 and top["status"] in ("PASS", "FAIL")
    ev = shuffle_test_timestamps(fx["events"], fx["meta"]["t_min"] - 1, seed=1)
    pre = ev["t"] <= fx["meta"]["t_min"] - 1
    assert (ev.loc[pre, "t"] == fx["events"].loc[pre, "t"]).all()
    assert sorted(ev.loc[~pre, "t"]) == sorted(fx["events"].loc[~pre, "t"])
    shuf = run_pipeline(
        ev, fx["works"], embeddings=fx["embeddings"], event_labels=fx["event_labels"], cutoff=fx["meta"]["t_min"] - 1,
        variant="full", seed=0, cfg=cfg_small(), orc_fn=FakeORC())
    sg = gate_shuffle(a, shuf)
    assert sg["value"]["fitted_objects_identical"] is True
    assert sg["status"] == ("PASS" if abs(shuf.metrics["test"]["auroc"] - 0.5) < abs(a.metrics["test"]["auroc"] - 0.5)
                            else "FAIL")


def test_evaluate_baseline_uses_the_models_scored_test_events(fx):
    res = run(fx)
    test = res.scores[(res.split == "test").to_numpy()]
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"event_id": test["event_id"].to_numpy()[::-1], "score": rng.random(len(test))})
    got = evaluate_baseline(res, base, fx["event_labels"])
    merged = test[["event_id", "cold_start"]].merge(base, on="event_id")
    assert got == evaluate(merged, fx["event_labels"], score_col="score")
    assert got["n"] == res.metrics["test"]["n"]
    rec = baseline_record(res, "baseline", "rand", got, tags={"level": "fixture"})
    assert rec["kind"] == "baseline" and rec["data_hash"] == res.data_hash
    scored_first = test.loc[~test["cold_start"], "event_id"].iloc[0]
    with pytest.raises(ValueError, match="no score"):
        evaluate_baseline(res, base[base["event_id"] != scored_first], fx["event_labels"])

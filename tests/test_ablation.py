"""S4.3/S4.6: variant specs, run artifacts, and results tables regenerated from run artifacts only.

Spec under test (citecurv.eval.ablation, IMPLEMENTATION_PLAN.md S4.3 and S4.6):
- VARIANTS covers the five ablation cells of book 4.2.1 plus the learned-combiner cell (L13), and each maps to
  a scoring.components variant.
- write_run writes runs/<method>_<seed>/metrics.json (+ config.json); tables(src) reads every metrics.json under
  a folder (or a .jsonl bundle) and rebuilds: main (mean ± sample sd over seeds), ablation (paired deltas vs
  full, matched by seed), curvature_delta (full minus no_curvature with the seed-noise verdict),
  obviousness (recall@k = found@k / n_pos), shuffle, gates.
- The --tables CLI writes tables.md + CSVs; two invocations give identical bytes; a bundle gives the same
  tables as the folder.
- run_matrix on the mini fixture writes one run folder per (variant, seed) with metrics.json and a config.json
  carrying the config_hash; all cells of one configuration share that hash.
"""

import json
import math

import pandas as pd
import pytest

from citecurv.config import load_config
from citecurv.data.synthetic import load_fixture
from citecurv.eval import ablation as A
from citecurv.scoring import components as C


def test_variant_specs_are_consistent():
    assert set(A.ABLATION_VARIANTS) == {"full", "no_curvature", "forman_only", "ollivier_only", "no_temporal"}
    assert set(A.VARIANTS) == set(A.ABLATION_VARIANTS) | set(A.EXTRA_CELLS) | set(A.RECORD_CELLS)
    for name, spec in A.VARIANTS.items():
        assert spec.name == name
        assert spec.component_variant in C.VARIANTS
        assert spec.curvature in ("screen+orc", "screen", "random+orc", "none", "closed")
        assert spec.combiner in ("calibrated", "learned")
    assert A.VARIANTS["no_temporal"].temporal_encoder is False
    assert A.VARIANTS["no_curvature"].curvature == "none"


def rec(method, seed, ap, auroc, *, kind="model", level="moderate", n=1000, n_pos=20, found100=5, found1000=10,
        group=None, **extra):
    tags = {"level": level, "synthetic": True}
    if group:
        tags["group"] = group
    test = {"n": n, "n_pos": n_pos, "auroc": auroc, "ap": ap, "p_at_100": found100 / 100, "found_at_100": found100,
            "p_at_1000": found1000 / 1000, "found_at_1000": found1000}
    return {"kind": kind, "method": method, "seed": seed, "tags": tags, "test": test, "config_hash": "abc",
            "data_hash": f"d{seed}", "test_n_pos_all": n_pos + 5, **extra}


RECORDS = [
    rec("full", 0, 0.30, 0.90, found1000=12), rec("full", 1, 0.40, 0.92, found1000=16),
    rec("no_curvature", 0, 0.20, 0.88), rec("no_curvature", 1, 0.35, 0.91),
    rec("forman_only", 0, 0.31, 0.90), rec("forman_only", 1, 0.39, 0.92),
    rec("xgb", 0, 0.10, 0.70, kind="baseline"), rec("xgb", 1, 0.12, 0.72, kind="baseline"),
    rec("dst_age", 0, 0.05, 0.95, kind="control", found1000=18), rec("dst_age", 1, 0.07, 0.96, kind="control",
                                                                       found1000=20),
    rec("shuffle_full", 0, 0.02, 0.55, kind="sanity"),
    rec("full", 0, 0.50, 0.97, level="blatant"),
    rec("full", 0, 0.25, 0.80, level="moderate", group="tgn_extra"),
    {"kind": "sanity", "method": "gates", "seed": 0, "tags": {"level": "moderate", "synthetic": True},
     "test": None, "gates": {"reproducibility": {"status": "PASS", "value": "same sha256", "rule": "identical"},
                             "d4_fairness": {"status": "FAIL", "value": 0.62, "rule": "max cell share <= 0.5"}}},
]


@pytest.fixture()
def runs_dir(tmp_path):
    for r in RECORDS:
        level = r["tags"]["level"]
        group = r["tags"].get("group", "matrix")
        A.write_run(tmp_path / group / level / "runs" / A.run_dir_name(r["method"], r["seed"]), r)
    return tmp_path


def row(df, **match):
    sel = df
    for k, v in match.items():
        sel = sel[sel[k] == v]
    assert len(sel) == 1, match
    return sel.iloc[0]


def test_tables_hand_computed_values(runs_dir):
    t = A.tables(runs_dir)
    sd = math.sqrt(2 * 0.05 ** 2)  # sample sd of {0.30, 0.40}
    main = t["main"]
    assert row(main, level="moderate", method="full")["AP"] == f"0.350 ± {sd:.3f}"
    assert list(main[main["level"] == "moderate"]["method"])[:2] == ["full", "no_curvature"]
    assert row(main, level="moderate", method="xgb")["kind"] == "baseline"
    assert row(main, level="blatant", method="full")["AP"] == "0.500"  # one seed: no spread shown
    assert "shuffle_full" not in set(main["method"]) and "gates" not in set(main["method"])

    abl = t["ablation"]
    assert row(abl, level="moderate", method="no_curvature")["ΔAP vs full"] == "-0.075 ± 0.035"
    assert row(abl, level="moderate", method="forman_only")["ΔAP vs full"] == "0.000 ± 0.014"
    assert row(abl, level="moderate", method="full")["ΔAP vs full"] == ""

    d = row(t["curvature_delta"], level="moderate", metric="AP")
    assert d["delta (full - no_curv)"] == "0.075 ± 0.035"
    assert d["per-seed deltas"] == "s0: +0.1000, s1: +0.0500"
    assert d["verdict"] == "within seed noise"  # same sign, but 0.075 < max seed sd (0.106)
    assert row(t["curvature_delta"], level="moderate", metric="AUROC")["verdict"] == "within seed noise"

    obv = row(t["obviousness"], level="moderate")
    assert obv["recall@1000"] == f"0.700 ± {math.sqrt(2 * 0.1 ** 2):.3f}"  # 12/20 and 16/20
    assert obv["recall@1000 dst_age"] == f"0.950 ± {math.sqrt(2 * 0.05 ** 2):.3f}"
    assert obv["positives incl. cold start"] == 25.0

    sh = row(t["shuffle"], level="moderate")
    assert sh["AP shuffled"] == "0.020" and sh["AP full"] == "0.300"  # only the shuffled seed is compared
    assert sh["chance AP (prevalence)"] == "0.0200"

    g = t["gates"]
    assert list(g["gate"]) == ["d4_fairness", "reproducibility"]
    assert row(g, gate="d4_fairness")["status"] == "FAIL"
    assert row(t["extra"], group="tgn_extra", method="full")["AP"] == "0.250"
    assert len(t["runs"]) == len(RECORDS)


def test_cli_writes_identical_tables_and_bundle_round_trips(runs_dir, tmp_path):
    out1, out2 = tmp_path / "t1", tmp_path / "t2"
    bundle = tmp_path / "bundle.jsonl"
    assert A.main(["--tables", str(runs_dir), "--out", str(out1), "--bundle", str(bundle)]) == 0
    assert A.main(["--tables", str(bundle), "--out", str(out2)]) == 0
    for name in ("tables.md", "main.csv", "ablation.csv", "curvature_delta.csv", "obviousness.csv", "shuffle.csv",
                 "gates.csv", "runs.csv"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
    md = (out1 / "tables.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in md and "## Ablation" in md
    assert len(bundle.read_text().splitlines()) == len(RECORDS)


def test_write_run_validates_kind(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        A.write_run(tmp_path / "x", {"kind": "other", "method": "m", "seed": 0})
    with pytest.raises(ValueError, match="method"):
        A.write_run(tmp_path / "x", {"kind": "model", "seed": 0})


def test_run_matrix_on_fixture_writes_runs_and_tables(tmp_path):
    from test_pipeline_integration import FakeORC, cfg_small

    fx = load_fixture()
    data = {"events": fx["events"], "works": fx["works"], "embeddings": fx["embeddings"],
            "event_labels": fx["event_labels"], "cutoff": 19180}
    orc = FakeORC()
    seen = []
    metrics = A.run_matrix(list(A.VARIANTS), [0, 1], data, tmp_path, cfg=cfg_small(), orc_fn=orc,
                           tags={"level": "fixture", "synthetic": True}, callback=lambda r: seen.append(r.variant))
    assert len(metrics) == 2 * len(A.VARIANTS) == len(seen)
    hashes = set()
    for v in A.VARIANTS:
        for s in (0, 1):
            d = tmp_path / "runs" / f"{v}_{s}"
            m = json.loads((d / "metrics.json").read_text())
            c = json.loads((d / "config.json").read_text())
            assert m["variant"] == v and m["seed"] == s and c["config_hash"] == m["config_hash"]
            assert (d / "scores.parquet").is_file()
            hashes.add(c["config_hash"])
    assert len(hashes) == 1  # one configuration, many cells
    t = A.tables(tmp_path)
    assert set(t["ablation"]["method"]) == set(A.VARIANTS)
    assert set(t["learned_vs_fixed"]["cell"]) == {"full_learned"}
    # option A (2026-11-26): the closed cells compute no ORC; plan_full (the old full) still does
    closed = [m for m in metrics if m["variant"] in ("full", "no_temporal", "full_learned")]
    assert all(m["curvature"]["mode"] == "closed" and m["curvature"]["n_orc_computed"] == 0 for m in closed)
    assert all(m["curvature"]["n_orc_computed"] > 0 for m in metrics
               if m["variant"] == "plan_full" and m["seed"] == 0)
    assert isinstance(pd.read_parquet(tmp_path / "runs" / "full_0" / "scores.parquet"), pd.DataFrame)
    assert load_config()  # default config untouched


def test_run_matrix_resume_skips_finished_cells(tmp_path):
    from test_pipeline_integration import FakeORC, cfg_small

    fx = load_fixture()
    data = {"events": fx["events"], "works": fx["works"], "embeddings": fx["embeddings"],
            "event_labels": fx["event_labels"], "cutoff": 19180}
    first = A.run_matrix(["full", "no_curvature"], [0], data, tmp_path, cfg=cfg_small(), orc_fn=FakeORC())
    orc, calls = FakeORC(), []
    again = A.run_matrix(["full", "no_curvature"], [0], data, tmp_path, cfg=cfg_small(), orc_fn=orc, resume=True,
                         callback=calls.append)
    assert orc.calls == 0 and calls == []
    assert [m["test"] for m in again] == [m["test"] for m in first]


def test_bundle_slims_nested_metrics(tmp_path):
    r = rec("full", 0, 0.3, 0.9, test_components={"s_geom": {"auroc": 0.6, "ap": 0.1, "n": 5, "p_at_100": 0.0}},
            test_top100_contribution={"k": 100, "n_pos": 3, "mean_total": 1.0, "mean_contribution": {"s_geom": 0.2}})
    A.write_bundle([r], tmp_path / "b.jsonl")
    back = A.load_records(tmp_path / "b.jsonl")[0]
    assert back["test_components"] == {"s_geom": {"auroc": 0.6, "ap": 0.1}}
    assert back["test_top100_contribution"] == {"k": 100, "n_pos": 3, "mean_contribution": {"s_geom": 0.2}}
    comp = A.tables([r])["components"]
    assert row(comp, component="s_geom")["AUROC alone"] == "0.600"


# ---------------------------------------------------------------- label sources, per paper, cold-start (G10, G11)

def _m(ap, auroc, n=100, n_pos=4, found=2):
    return {"n": n, "n_pos": n_pos, "auroc": auroc, "ap": ap, "p_at_100": found / 100, "found_at_100": found,
            "p_at_1000": found / 1000, "found_at_1000": found}


def _views(ap):
    return {"citing_side": _m(ap, 0.8, n_pos=3), "cited_side": _m(ap / 2, 0.6, n_pos=1),
            "per_citing_paper": _m(ap + 0.1, 0.9, n=40, n_pos=2), "per_cited_paper": _m(ap, 0.7, n=60, n_pos=1)}


VIEW_RECORDS = [
    rec("full", 0, 0.30, 0.90, test_views=_views(0.2), test_cold=_m(0.10, 0.65, n=500, n_pos=9),
        test_cold_views=_views(0.1)),
    rec("full", 1, 0.40, 0.92, test_views=_views(0.4), test_cold=_m(0.20, 0.75, n=520, n_pos=11),
        test_cold_views=_views(0.3)),
    rec("xgb", 0, 0.10, 0.70, kind="baseline", test_views=_views(0.05)),
    rec("dst_age", 0, 0.05, 0.95, kind="control", test_views=None),      # labels without source columns
    rec("shuffle_full", 0, 0.02, 0.55, kind="sanity", test_views=_views(0.01)),
]


def test_label_views_table_shows_the_sources_and_papers_next_to_the_total():
    t = A.tables(VIEW_RECORDS)["label_views"]
    sd = math.sqrt(2 * 0.1 ** 2)  # sample sd of two values 0.2 apart
    full = t[(t["method"] == "full") & (t["rows"] == "scored")]
    assert list(full["view"]) == ["all", "citing_side", "cited_side", "per_citing_paper", "per_cited_paper"]
    assert row(full, view="all")["AP"] == f"0.350 ± {math.sqrt(2 * 0.05 ** 2):.3f}"       # the `test` numbers
    assert row(full, view="citing_side")["AP"] == f"0.300 ± {sd:.3f}"
    assert row(full, view="cited_side")["AP"] == f"0.150 ± {sd / 2:.3f}"
    assert row(full, view="per_citing_paper")["n"] == 40.0 and row(full, view="per_citing_paper")["n_pos"] == 2.0
    assert row(full, view="citing_side")["n_seeds"] == 2 and row(full, view="citing_side")["AUROC"] == "0.800 ± 0.000"
    assert row(t, method="xgb", rows="scored", view="cited_side")["AP"] == "0.025"
    assert row(t, method="xgb", rows="scored", view="all")["kind"] == "baseline"


def test_label_views_table_gives_cold_start_rows_their_own_lines():
    t = A.tables(VIEW_RECORDS)["label_views"]
    cold = t[(t["method"] == "full") & (t["rows"] == "cold-start")]
    assert list(cold["view"]) == ["all", "citing_side", "cited_side", "per_citing_paper", "per_cited_paper"]
    assert row(cold, view="all")["AP"] == f"0.150 ± {math.sqrt(2 * 0.05 ** 2):.3f}"      # the `test_cold` numbers
    assert row(cold, view="all")["n_pos"] == 10.0 and row(cold, view="all")["n"] == 510.0
    assert row(cold, view="citing_side")["AP"] == f"0.200 ± {math.sqrt(2 * 0.1 ** 2):.3f}"
    assert t[(t["method"] == "xgb") & (t["rows"] == "cold-start")].empty                # baselines have no cold score


def test_label_views_table_skips_runs_without_views_and_sanity_runs():
    t = A.tables(VIEW_RECORDS)["label_views"]
    assert "dst_age" not in set(t["method"]) and "shuffle_full" not in set(t["method"])
    assert A.tables(RECORDS)["label_views"].empty      # older records: no views, no cold-start metrics


def test_label_views_table_survives_the_bundle(tmp_path):
    A.write_bundle(VIEW_RECORDS, tmp_path / "b.jsonl")
    back = A.tables(tmp_path / "b.jsonl")["label_views"]
    pd.testing.assert_frame_equal(back, A.tables(VIEW_RECORDS)["label_views"])
    out = tmp_path / "t"
    A.write_tables(A.tables(VIEW_RECORDS), out)
    md = (out / "tables.md").read_text(encoding="utf-8")
    assert "label source" in md and "cold-start" in md and (out / "label_views.csv").is_file()


def test_label_views_table_from_pipeline_runs_on_the_fixture(tmp_path):
    """End to end without the learned cell (it needs sklearn): pipeline -> metrics.json -> tables -> files."""
    from test_pipeline_integration import FakeORC, _with_sources, cfg_small

    fx = load_fixture()
    data = {"events": fx["events"], "works": fx["works"], "embeddings": fx["embeddings"],
            "event_labels": _with_sources(fx["event_labels"]), "cutoff": fx["meta"]["t_min"] - 1}
    metrics = A.run_matrix(["full", "no_curvature"], [0, 1], data, tmp_path, cfg=cfg_small(), orc_fn=FakeORC(),
                           tags={"level": "fixture", "synthetic": True})
    t = A.tables(tmp_path)["label_views"]
    full = t[t["method"] == "full"]
    assert list(full["rows"]) == ["scored"] * 5 + ["cold-start"] * 5
    assert list(full[full["rows"] == "scored"]["view"]) == list(A.VIEW_ORDER)
    m0 = next(m for m in metrics if m["variant"] == "full" and m["seed"] == 0)
    m1 = next(m for m in metrics if m["variant"] == "full" and m["seed"] == 1)
    for rows, total, views in (("scored", "test", "test_views"), ("cold-start", "test_cold", "test_cold_views")):
        got = row(full, rows=rows, view="all")
        assert got["n"] == round((m0[total]["n"] + m1[total]["n"]) / 2, 1) and got["n_seeds"] == 2
        side = row(full, rows=rows, view="citing_side")
        assert side["n_pos"] == round((m0[views]["citing_side"]["n_pos"] + m1[views]["citing_side"]["n_pos"]) / 2, 1)
        assert row(full, rows=rows, view="cited_side")["n_pos"] + side["n_pos"] == got["n_pos"]
    # no_curvature switches s_geom off, so it has no cold-start score and no cold-start lines
    assert set(t[t["method"] == "no_curvature"]["rows"]) == {"scored"}
    out = tmp_path / "tables"
    assert A.main(["--tables", str(tmp_path), "--out", str(out), "--bundle", str(tmp_path / "b.jsonl")]) == 0
    assert A.main(["--tables", str(tmp_path / "b.jsonl"), "--out", str(tmp_path / "tables2")]) == 0
    assert (out / "label_views.csv").read_bytes() == (tmp_path / "tables2" / "label_views.csv").read_bytes()
    assert (out / "tables.md").read_bytes() == (tmp_path / "tables2" / "tables.md").read_bytes()


# ---------------------------------------------------------------- option A cells (plan 2026-11-26, Task 3)


def _fixture_data():
    fx = load_fixture()
    return fx, {"events": fx["events"], "works": fx["works"], "embeddings": fx["embeddings"],
                "event_labels": fx["event_labels"], "cutoff": 19180}


def test_full_cell_runs_closed_mode_end_to_end(tmp_path):
    from test_pipeline_integration import FakeORC, cfg_small

    _, data = _fixture_data()
    ms = A.run_matrix(["full", "forman_only", "no_curvature"], [0], data, tmp_path, cfg=cfg_small(), orc_fn=FakeORC())
    by = {m["variant"]: m for m in ms}
    assert by["full"]["curvature"]["mode"] == "closed" and by["full"]["curvature"]["n_orc_computed"] == 0
    assert by["full"]["curvature"]["share_citing_cited_at_t"] is not None
    assert by["forman_only"]["curvature"]["mode"] == "screen"
    assert by["no_curvature"]["curvature"]["mode"] == "none"


def test_coupling_passed_once_gives_same_scores(tmp_path, monkeypatch):
    from citecurv.curvature import overlap as overlap_mod
    from citecurv.pipeline import _canonical_events
    from test_pipeline_integration import FakeORC, cfg_small

    fx, data = _fixture_data()
    A.run_matrix(["full"], [0], data, tmp_path / "a", cfg=cfg_small(), orc_fn=FakeORC())
    coupling = overlap_mod.reference_overlap(_canonical_events(fx["events"]))
    calls, real = [], overlap_mod.reference_overlap
    monkeypatch.setattr(overlap_mod, "reference_overlap", lambda *a, **k: calls.append(1) or real(*a, **k))
    m = A.run_matrix(["full"], [0], data, tmp_path / "b", cfg=cfg_small(), orc_fn=FakeORC(), coupling=coupling)
    assert calls == [] and m[0]["curvature"]["mode"] == "closed"
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / "runs" / "full_0" / "scores.parquet"),
                                  pd.read_parquet(tmp_path / "b" / "runs" / "full_0" / "scores.parquet"))


def test_plan_full_keeps_the_old_path(tmp_path):
    from test_pipeline_integration import FakeORC, cfg_small

    _, data = _fixture_data()
    m = A.run_matrix(["plan_full"], [0], data, tmp_path, cfg=cfg_small(), orc_fn=FakeORC())[0]
    assert m["curvature"]["mode"] == "screen+orc" and m["curvature"]["n_orc_computed"] > 0
    assert "plan_full" not in A.ABLATION_VARIANTS

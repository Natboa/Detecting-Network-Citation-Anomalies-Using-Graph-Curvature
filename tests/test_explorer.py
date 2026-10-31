"""Results explorer (citecurv.viz): one self-contained HTML page over the pipeline outputs.

Spec under test (agent assignment A10; CONTRACTS.md; IMPLEMENTATION_PLAN.md FR7-FR9, NFR7, NFR10, S3.3, S3.4a / D4):
- build_explorer(out_html, scores, ...) writes ONE HTML file that loads nothing over the network (no <script src>,
  no <link href>, no url(http...)), embeds one JSON blob that parses, and is byte-identical for the same inputs + seed.
- Ranking: total descending, ties by the canonical order (t, event_id); cold-start rows get no rank.
- Per-event details exist for the top_n ranked events only; contributions w_c * z_c with frozen z statistics.
- Optional inputs missing -> their panel is omitted and the reason recorded.
- Local subgraph at time t uses only events with ts <= t, never the scored row itself (NFR3 for the display), and
  matches EventStore.subgraph_at when uncapped (brute-force cross-check).
- AUROC / AP match scikit-learn; the D4 heatmap holds the share of top-1% flags per (field, year) cell.
- Paper rollup = mean of the top-5 incident edge totals + count of incident edges at or above the threshold (S3.3).
"""

import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from citecurv.curvature.timeslice import EventStore
from citecurv.viz import build_explorer, explorer_payload, iframe_html, make_demo_data
from citecurv.viz.graphctx import TemporalIndex, local_subgraph, received_timeline
from citecurv.viz.precompute import (
    coerce_weights,
    coerce_zstats,
    flag_heatmap,
    paper_rollup,
    rank_events,
    ranking_curves,
)

REPO = Path(__file__).resolve().parents[1]
COMPONENTS = ["s_embed", "s_geom", "s_temp", "s_curvdyn"]


@pytest.fixture(scope="module")
def demo():
    return make_demo_data(seed=0, n_papers=600)


@pytest.fixture(scope="module")
def demo_html(tmp_path_factory, demo):
    out = tmp_path_factory.mktemp("explorer") / "demo.html"
    build_explorer(out, top_n=60, seed=0, **demo)
    return out


def embedded_json(html: str) -> dict:
    m = re.search(r'<script type="application/json" id="[^"]+-data">(.*?)</script>', html, re.S)
    assert m, "embedded JSON blob not found"
    return json.loads(m.group(1))


def tiny_scores():
    # event 3 and 1 tie on total; canonical order is (t, event_id) -> event 1 (t=5) before event 3 (t=7).
    return pd.DataFrame(
        {
            "event_id": [10, 1, 3, 7, 4],
            "src_id": ["a", "b", "c", "d", "e"],
            "dst_id": ["x", "x", "y", "y", "z"],
            "t": [1, 5, 7, 2, 3],
            "s_embed": [0.0, 1.0, 2.0, 3.0, 4.0],
            "s_geom": [1.0, 1.0, 1.0, 1.0, 1.0],
            "s_temp": [0.5, 0.5, 0.5, 0.5, 0.5],
            "s_curvdyn": [0.0, 0.0, 0.0, 0.0, 0.0],
            "total": [0.2, 0.9, 0.9, 1.5, 3.0],
            "cold_start": [False, False, False, False, True],
        }
    )


# ---------------------------------------------------------------- self-contained output


def test_html_is_written_and_loads_nothing_from_the_network(demo_html):
    html = demo_html.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert not re.search(r"<script[^>]*\bsrc\s*=", html, re.I)
    assert not re.search(r"<link[^>]*\bhref\s*=", html, re.I)
    assert not re.search(r"url\(\s*['\"]?\s*https?:", html, re.I)
    assert not re.search(r"@import", html, re.I)
    assert not re.search(r"<(img|iframe|embed|object|video|audio|source)\b[^>]*\bsrc\s*=", html, re.I)
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    # the only absolute URLs allowed are XML namespace identifiers (never fetched)
    urls = set(re.findall(r"https?://[^\s\"'<>)]+", html))
    assert urls <= {"http://www.w3.org/2000/svg"}, urls


def test_embedded_json_parses_with_expected_keys(demo_html):
    data = embedded_json(demo_html.read_text(encoding="utf-8"))
    for key in ("meta", "nodes", "fields", "venues", "rows", "details", "hist", "curves", "heatmap",
                "tables", "metrics", "papers", "authors", "cartels", "component_text", "missing"):
        assert key in data, key
    meta = data["meta"]
    assert meta["components"] == COMPONENTS
    assert meta["synthetic"] is True and "synthetic" in meta["data_note"]
    assert set(data["component_text"]) == set(COMPONENTS)


def test_banner_and_title_are_in_the_page(demo_html):
    html = demo_html.read_text(encoding="utf-8")
    assert "<title>Citation anomaly explorer</title>" in html
    assert "synthetic demo data" in html


def test_top_n_respected(demo, tmp_path):
    payload = explorer_payload(top_n=25, **demo)
    assert len(payload["details"]) == 25
    ranked = [r for r in payload["rows"]["rank"] if r is not None]
    assert ranked == list(range(1, 26))
    # every detail belongs to a ranked row inside the top 25
    ids = payload["rows"]["event_id"]
    ranks = dict(zip(ids, payload["rows"]["rank"]))
    assert all(ranks[int(k)] is not None and ranks[int(k)] <= 25 for k in payload["details"])


def test_deterministic_bytes_for_same_seed(demo, tmp_path):
    a = build_explorer(tmp_path / "a.html", top_n=30, seed=3, **demo).read_bytes()
    b = build_explorer(tmp_path / "b.html", top_n=30, seed=3, **demo).read_bytes()
    c = build_explorer(tmp_path / "c.html", top_n=30, seed=4, **demo).read_bytes()
    assert a == b
    assert a != c  # layouts depend on the seed


def test_size_bound_for_top_500(tmp_path):
    data = make_demo_data(seed=1)
    out = build_explorer(tmp_path / "big.html", top_n=500, seed=0, **data)
    assert out.stat().st_size < 8 * 1024 * 1024
    payload = embedded_json(out.read_text(encoding="utf-8"))
    assert len(payload["details"]) == 500


def test_optional_panels_omitted_cleanly(demo, tmp_path):
    out = build_explorer(tmp_path / "bare.html", demo["scores"], top_n=20)
    data = embedded_json(out.read_text(encoding="utf-8"))
    assert data["curves"] is None and data["heatmap"] is None and data["cartels"] is None
    assert data["authors"] is None and data["tables"] == [] and data["metrics"] == []
    for panel in ("labels", "events", "works", "authorships", "cartels", "curvature", "tables", "metrics"):
        assert data["missing"][panel], panel
    d = next(iter(data["details"].values()))
    assert d["graph"] is None and d["timeline"] is None
    # fallbacks are disclosed
    assert data["meta"]["zstats_source"] == "fallback" and data["meta"]["weights_source"] == "fallback"
    assert any("z statistics" in n for n in data["meta"]["notes"])


def test_title_text_cannot_break_out_of_the_script_tag(demo, tmp_path):
    works = demo["works"].copy()
    evil = "</script><script>alert(1)</script> & <b>"
    top_src = demo["scores"].sort_values("total", ascending=False).iloc[0]["src_id"]
    works.loc[works.work_id == top_src, "title"] = evil
    kw = dict(demo, works=works)
    out = build_explorer(tmp_path / "x.html", top_n=10, title="A <b>&</b> title", **kw)
    html = out.read_text(encoding="utf-8")
    assert "<script>alert(1)" not in html
    assert "<title>A &lt;b&gt;&amp;&lt;/b&gt; title</title>" in html
    data = embedded_json(html)
    assert evil in data["nodes"]["title"]


def test_multiple_runs_in_one_scores_frame_are_rejected(demo):
    s = demo["scores"].copy()
    s.loc[s.index[:5], "seed"] = 99
    with pytest.raises(ValueError, match="one run"):
        explorer_payload(s)


def test_missing_required_columns_rejected():
    with pytest.raises(ValueError, match="total"):
        explorer_payload(pd.DataFrame({"event_id": [1], "src_id": ["a"], "dst_id": ["b"], "t": [0]}))


# ---------------------------------------------------------------- ranking and contributions


def test_rank_order_ties_and_cold_start():
    ranked = rank_events(tiny_scores())
    # display order: total desc, ties by (t, event_id); cold-start rows keep their place but get no rank
    assert ranked["event_id"].tolist() == [4, 7, 1, 3, 10]
    r = ranked.set_index("event_id")["rank"]
    assert pd.isna(r.loc[4])  # cold start: largest total, still unranked
    assert r.drop(4).astype(int).to_dict() == {7: 1, 1: 2, 3: 3, 10: 4}


def test_contributions_sum_to_total_with_frozen_stats():
    s = tiny_scores()
    stats = {"s_embed": (1.0, 2.0), "s_geom": (0.5, 0.5), "s_temp": (0.0, 1.0), "s_curvdyn": (0.0, 1.0)}
    w = {"s_embed": 0.4, "s_geom": 0.3, "s_temp": 0.2, "s_curvdyn": 0.1}
    z = {c: (s[c] - m) / sd for c, (m, sd) in stats.items()}
    s["total"] = sum(w[c] * z[c] for c in COMPONENTS)
    payload = explorer_payload(s, zstats=stats, weights=w, top_n=4)
    assert payload["meta"]["zstats_source"] == "given" and payload["meta"]["weights_source"] == "given"
    for eid, d in payload["details"].items():
        row = s.set_index("event_id").loc[int(eid)]
        for c in COMPONENTS:
            m, sd = stats[c]
            assert d["z"][c] == pytest.approx((row[c] - m) / sd, abs=1e-4)
            assert d["contrib"][c] == pytest.approx(w[c] * (row[c] - m) / sd, abs=1e-4)
        assert sum(d["contrib"].values()) == pytest.approx(row["total"], abs=1e-3)
        assert abs(d["sum_gap"]) < 1e-6


def test_zstats_and_weights_accept_common_shapes():
    expect = {"s_embed": (1.0, 2.0), "s_geom": (0.0, 1.0), "s_temp": (3.0, 4.0), "s_curvdyn": (5.0, 6.0)}
    shapes = [
        expect,
        {c: {"mean": m, "std": s} for c, (m, s) in expect.items()},
        {"mean": {c: m for c, (m, _) in expect.items()}, "std": {c: s for c, (_, s) in expect.items()}},
        pd.DataFrame({"mean": [m for m, _ in expect.values()], "std": [s for _, s in expect.values()]},
                     index=list(expect)),
        type("ZStats", (), {"mean": {c: m for c, (m, _) in expect.items()},
                            "std": {c: s for c, (_, s) in expect.items()}})(),
    ]
    for shape in shapes:
        assert coerce_zstats(shape) == expect
    w = {"s_embed": 0.1, "s_geom": 0.2, "s_temp": 0.3, "s_curvdyn": 0.4}
    assert coerce_weights(w) == w
    assert coerce_weights(pd.Series(w)) == w
    assert coerce_weights([0.1, 0.2, 0.3, 0.4]) == w
    assert coerce_weights(type("W", (), {"weights": w})()) == w
    with pytest.raises(TypeError):
        coerce_zstats(42)


# ---------------------------------------------------------------- metrics, heatmap, rollups


def test_auroc_ap_match_sklearn_with_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    y = (rng.random(400) < 0.15).astype(int)
    s = np.round(rng.normal(size=400) + y, 1)  # rounding creates ties
    out = ranking_curves(s, y)
    assert out["auroc"] == pytest.approx(metrics.roc_auc_score(y, s), abs=1e-9)
    assert out["ap"] == pytest.approx(metrics.average_precision_score(y, s), abs=1e-9)
    assert out["roc"][0] == [0.0, 0.0] and out["roc"][-1] == [1.0, 1.0]
    assert len(out["pr"]) <= 320 and len(out["roc"]) <= 320


def test_flag_heatmap_d4_share_and_gate():
    # 200 scored events, top 1% = 2 flags, both in (bio, 2020) -> that cell holds 100% > 50% -> FAIL
    rows = []
    for i in range(200):
        field = "bio" if i < 100 else "chem"
        year = 2020 if i % 2 == 0 else 2021
        total = 100.0 - i if field == "bio" and year == 2020 else float(i % 7)
        rows.append({"event_id": i, "t": int((pd.Timestamp(f"{year}-06-01") - pd.Timestamp("1970-01-01")).days),
                     "total": total, "field": field})
    df = pd.DataFrame(rows)
    hm = flag_heatmap(df, flag_share=0.01)
    assert hm["n_flags"] == 2
    i, j = hm["fields"].index("bio"), hm["years"].index(2020)
    assert hm["flags"][i][j] == 2 and hm["share"][i][j] == pytest.approx(1.0)
    assert hm["max_share"] == pytest.approx(1.0) and hm["passed"] is False
    assert sum(map(sum, hm["events"])) == 200
    # spread flags: top 4 % over 4 cells -> max share 25 % -> PASS
    df2 = df.copy()
    df2["total"] = np.arange(200, 0, -1, dtype=float)
    df2["field"] = ["bio", "chem"] * 100
    df2["t"] = [df["t"].iloc[0] if (i // 2) % 2 == 0 else df["t"].iloc[1] for i in range(200)]
    hm2 = flag_heatmap(df2, flag_share=0.02)
    assert hm2["n_flags"] == 4 and hm2["max_share"] == pytest.approx(0.25) and hm2["passed"] is True


def test_paper_rollup_mean_top5_and_count_above_threshold():
    totals = [9, 8, 7, 6, 5, 4, 1]
    s = pd.DataFrame({
        "event_id": range(7), "src_id": ["p"] * 4 + ["q", "r", "s"], "dst_id": ["a", "b", "c", "d", "p", "p", "p"],
        "t": range(7), "total": [float(x) for x in totals], "cold_start": [False] * 7,
    })
    roll = paper_rollup(s, threshold=6.0).set_index("paper_id")
    assert roll.loc["p", "score"] == pytest.approx((9 + 8 + 7 + 6 + 5) / 5)
    assert roll.loc["p", "n_above"] == 4
    assert roll.loc["p", "n_events"] == 7 and roll.loc["p", "n_as_src"] == 4 and roll.loc["p", "n_as_dst"] == 3
    assert roll.loc["a", "score"] == pytest.approx(9.0) and roll.loc["a", "n_above"] == 1
    assert roll.index[0] == "a"  # best paper first (score 9)


# ---------------------------------------------------------------- local graph at time t


def random_stream(n_nodes=25, n_events=160, seed=0):
    rnd = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n_events:
        s, d = rnd.sample(range(n_nodes), 2)
        t = rnd.randint(0, 40)
        if (s, d, t) in seen:
            continue
        seen.add((s, d, t))
        rows.append((f"n{s}", f"n{d}", t))
    return pd.DataFrame(rows, columns=["src_id", "dst_id", "t"]).assign(event_id=lambda d: np.arange(len(d)) * 3 + 11)


@pytest.mark.parametrize("radius", [1, 2])
def test_uncapped_subgraph_matches_eventstore(radius):
    ev = random_stream()
    idx = TemporalIndex(ev)
    store = EventStore([(r.src_id, r.dst_id, r.t) for r in ev.itertuples()])
    for r in ev.sample(40, random_state=1).itertuples():
        g = local_subgraph(idx, r.event_id, r.src_id, r.dst_id, r.t, radius=radius, max_nodes=None)
        ref = store.subgraph_at(r.src_id, r.dst_id, r.t, radius)
        assert {n["id"] for n in g["nodes"]} == set(ref.nodes)
        context = {(e["a"], e["b"]) for e in g["edges"] if not e["scored"]}
        assert context == set(ref.edges)
        assert [(e["a"], e["b"]) for e in g["edges"] if e["scored"]] == [(r.src_id, r.dst_id)]


def test_subgraph_is_leak_free_and_forman_matches_brute_force():
    ev = random_stream(n_nodes=15, n_events=120, seed=3)
    idx = TemporalIndex(ev)
    for r in ev.sample(30, random_state=2).itertuples():
        g = local_subgraph(idx, r.event_id, r.src_id, r.dst_id, r.t, radius=2, max_nodes=None)
        past = ev[(ev.t <= r.t) & (ev.event_id != r.event_id)]
        in_deg = past.groupby("dst_id").size()
        out_deg = past.groupby("src_id").size()
        for e in g["edges"]:
            assert e["last_t"] <= r.t
            expect = 2 - in_deg.get(e["a"], 0) - out_deg.get(e["b"], 0)
            assert e["forman_t"] == expect
        hops = {n["id"]: n["hop"] for n in g["nodes"]}
        assert hops[r.src_id] == 0 and hops[r.dst_id] == 0


def test_capped_subgraph_keeps_core_and_reports_totals():
    # a hub with 30 citers before t: cap 10 nodes -> core + 8 neighbours, total ring-1 reported
    rows = [(f"c{i}", "hub", i) for i in range(30)] + [("me", "hub", 50)]
    ev = pd.DataFrame(rows, columns=["src_id", "dst_id", "t"]).assign(event_id=lambda d: np.arange(len(d)))
    idx = TemporalIndex(ev)
    g = local_subgraph(idx, 30, "me", "hub", 50, radius=1, max_nodes=10)
    ids = [n["id"] for n in g["nodes"]]
    assert ids[:2] == ["me", "hub"] and len(ids) == 10
    assert g["ring1_total"] == 30 and g["ring1_shown"] == 8
    # most recent citers are kept first
    assert set(ids[2:]) == {f"c{i}" for i in range(22, 30)}


def test_received_timeline_counts_only_up_to_t():
    ev = pd.DataFrame({"src_id": list("abcdef"), "dst_id": ["v"] * 6, "t": [0, 10, 10, 20, 30, 99]})
    ev["event_id"] = range(6)
    idx = TemporalIndex(ev)
    tl = received_timeline(idx, "v", 30, max_bins=4)
    assert sum(tl["c"]) == 5 and tl["n"] == 5 and tl["same_day"] == 1
    assert tl["end"] == 30 and tl["w"] * len(tl["c"]) >= 31


# ---------------------------------------------------------------- cartels, helpers, script


def test_cartel_recall_per_level(demo):
    payload = explorer_payload(top_n=50, **demo)
    cart = payload["cartels"]
    assert cart is not None and cart["cartels"]
    ranks = [e["rank"] for c in cart["cartels"] for e in c["events"]]
    levels = [c["level"] for c in cart["cartels"] for _ in c["events"]]
    for k in cart["ks"]:
        for lvl in set(levels):
            hits = sum(1 for r, l in zip(ranks, levels) if l == lvl and r is not None and r <= k)
            n = sum(1 for l in levels if l == lvl)
            assert cart["recall"][lvl][str(k)] == pytest.approx(hits / n, abs=1e-6)  # stored at 6 significant digits


def test_iframe_helper_escapes_the_page(demo_html):
    html = iframe_html(demo_html, height=700)
    assert html.startswith("<iframe") and 'height="700"' in html
    assert "srcdoc=" in html and "<script" not in html  # the page is attribute-escaped


def test_build_script_writes_demo_page(tmp_path):
    out = tmp_path / "demo_script.html"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_explorer.py"), "--demo", "--top-n", "20",
         "--demo-papers", "400", "--out", str(out)],
        capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and "synthetic demo data" in out.read_text(encoding="utf-8")

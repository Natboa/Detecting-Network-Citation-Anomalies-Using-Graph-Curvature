"""S3.1 / S2.4: the four score components and the cold-start rule.

Spec under test: docs/score_components.md (PROVISIONAL, S3.1 gate).
- s_embed: -log p_link (mode "tgn"); static_cosine mode = (mean - cos) / sd of
  earlier events' endpoint cosine in the event's reference cell.
- s_geom: |z| of Forman against the reference cell; escalated events take
  max(|z_forman|, |z_orc|); forman_only ignores ORC; ollivier_only uses only ORC.
- s_temp: z of x = log((n_recent+1)/(n_before+1)) + log(T_before/W) against the cell
  (temporal_stat "rate_ratio", default); option "poisson_tail": x = -ln P(N >= n_recent),
  N negative binomial = Gamma(a0 + n_before, b0 + T_hist)-Poisson predictive count in W days
  (default a0 = 1, b0 = W),
  T_hist = T_before if n_before > 0 else 0; x = 0 when n_recent = 0.
- s_curvdyn: |asinh(observed) - predicted| / rmse_train, ridge fitted on TRAIN rows only.
- Reference cell = (field of src, calendar year of t), fallback field -> global,
  expanding window over strictly earlier events (ts < t), min_cell rows and sd > 0.
- cold_start: dst has fewer than k events with ts < t.
- score_event (single-event path) agrees with score_events (vectorised path).
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from citecurv.scoring.components import (
    COMPONENT_COLUMNS,
    ScoreBreakdown,
    build_context,
    fit_curvdyn_model,
    matched_random_escalation,
    score_event,
    score_events,
)

YEAR = 365  # day offset of 1971-01-01 is 365; 1972-01-01 is 730


def make_events(rows, start_id=0):
    """rows: list of (src, dst, t). event_id = start_id + position."""
    return pd.DataFrame(
        {
            "event_id": np.arange(start_id, start_id + len(rows), dtype=np.int64),
            "src_id": [r[0] for r in rows],
            "dst_id": [r[1] for r in rows],
            "t": np.array([r[2] for r in rows], dtype=np.int64),
        }
    )


def make_works(events, field_of=None, default="F"):
    ids = sorted(set(events["src_id"]) | set(events["dst_id"]))
    field_of = field_of or {}
    return pd.DataFrame({"work_id": ids, "field": [field_of.get(i, default) for i in ids]})


def make_curv(events, forman, orc=None, escalated=None):
    n = len(events)
    orc = np.full(n, np.nan) if orc is None else np.asarray(orc, dtype=float)
    esc = ~np.isnan(orc) if escalated is None else np.asarray(escalated, dtype=bool)
    return pd.DataFrame(
        {
            "event_id": events["event_id"].to_numpy(),
            "src_id": events["src_id"].to_numpy(),
            "dst_id": events["dst_id"].to_numpy(),
            "t": events["t"].to_numpy(),
            "forman": np.asarray(forman, dtype=float),
            "orc": orc,
            "escalated": esc,
            "radius": np.where(esc, 1.0, np.nan),
            "computed_at_stage": "test",
        }
    )


def random_stream(n=300, n_nodes=40, seed=7, d=6, t_values=None):
    """A random stream with heavy same-timestamp sharing, three fields, partial
    escalation, a few nodes without embeddings, a few events without p_link."""
    rng = np.random.default_rng(seed)
    t_values = np.arange(0, 1100, 25) if t_values is None else t_values
    rows = []
    for _ in range(n):
        s, d_ = rng.choice(n_nodes, size=2, replace=False)
        rows.append((f"n{s}", f"n{d_}", int(rng.choice(t_values))))
    ev = make_events(rows, start_id=1000)
    ev = ev.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # input order != time order
    fields = {f"n{i}": "FGH"[i % 3] for i in range(n_nodes)}
    works = make_works(ev, fields)
    forman = rng.integers(-40, 3, size=n).astype(float)
    esc = rng.random(n) < 0.25
    orc = np.where(esc, rng.uniform(-1.0, 1.0, size=n), np.nan)
    orc[np.flatnonzero(esc)[:2]] = np.nan  # escalated but ORC failed
    curv = make_curv(ev, forman, orc, esc)
    ids = np.array([f"n{i}" for i in range(n_nodes) if i not in (3, 11)])
    vecs = rng.normal(size=(len(ids), d)).astype(np.float32)
    p = rng.uniform(0.01, 0.99, size=n)
    tgn_scores = pd.DataFrame({"event_id": ev["event_id"], "p_link": p}).iloc[5:]
    tgn_emb = {
        "event_id": ev["event_id"].to_numpy(),
        "z_src": rng.normal(size=(n, d)).astype(np.float32),
        "z_dst": rng.normal(size=(n, d)).astype(np.float32),
    }
    train_mask = ev["t"].to_numpy() < 700
    return ev, works, curv, (ids, vecs), tgn_scores, tgn_emb, train_mask


# ---------------------------------------------------------------- schema, order


def test_output_schema_and_canonical_order():
    ev, works, curv, emb, tgn_scores, tgn_emb, train = random_stream(n=60)
    out = score_events(ev, works, curv, embeddings=emb, train_mask=train, min_cell=3)
    assert list(out.columns) == COMPONENT_COLUMNS
    assert COMPONENT_COLUMNS == [
        "event_id", "src_id", "dst_id", "t",
        "s_embed", "s_geom", "s_temp", "s_curvdyn", "cold_start",
    ]
    assert out["event_id"].dtype == np.int64 and out["t"].dtype == np.int64
    assert out["cold_start"].dtype == bool
    for c in ("s_embed", "s_geom", "s_temp", "s_curvdyn"):
        assert out[c].dtype == np.float64
    order = sorted(zip(ev["t"], ev["event_id"]))
    assert list(zip(out["t"], out["event_id"])) == order
    assert set(out["event_id"]) == set(ev["event_id"])
    assert out.attrs["variant"] == "full" and out.attrs["embed_mode"] == "static_cosine"


def test_with_inputs_adds_trace_columns_only():
    ev, works, curv, emb, *_ , train = random_stream(n=40)
    base = score_events(ev, works, curv, embeddings=emb, train_mask=train, min_cell=3)
    traced = score_events(ev, works, curv, embeddings=emb, train_mask=train, min_cell=3,
                          with_inputs=True)
    pd.testing.assert_frame_equal(traced[COMPONENT_COLUMNS], base)
    extra = [c for c in traced.columns if c not in COMPONENT_COLUMNS]
    assert extra and all(c.startswith("in_") for c in extra)
    for c in ("in_field", "in_year", "in_forman", "in_z_forman", "in_temp_x",
              "in_n_recent", "in_n_before", "in_n_prior_dst", "in_cosine", "in_curv_pred"):
        assert c in traced.columns


# ---------------------------------------------------------------- cold start


def test_cold_start_flags_first_k_citations_of_a_paper():
    # P is cited at five distinct times; prior-event counts are 0,1,2,3,4.
    ev = make_events([(f"s{i}", "P", 10 * (i + 1)) for i in range(5)])
    out = score_events(ev, make_works(ev), None, k=3)
    # citations with 0..k-1 = 0,1,2 prior events are cold; the one with k = 3 priors is not.
    assert out["cold_start"].tolist() == [True, True, True, False, False]
    out1 = score_events(ev, make_works(ev), None, k=1)
    assert out1["cold_start"].tolist() == [True, False, False, False, False]


def test_cold_start_counts_strictly_earlier_events_only():
    # three citations on the same day all see 0 prior events (ts < t rule)
    ev = make_events([("a", "P", 5), ("b", "P", 5), ("c", "P", 5), ("d", "P", 9)])
    out = score_events(ev, make_works(ev), None, k=3, with_inputs=True)
    assert out["in_n_prior_dst"].tolist() == [0, 0, 0, 3]
    assert out["cold_start"].tolist() == [True, True, True, False]


# ---------------------------------------------------------------- s_geom


def _cell_fixture(scored_forman):
    rows = [("a1", "b", 10), ("a2", "b", 20), ("a3", "b", 30), ("a4", "b", 40), ("a5", "c", 50)]
    ev = make_events(rows)
    return ev, make_works(ev), make_curv(ev, [-1, -3, -5, -7, scored_forman])


def test_s_geom_is_zero_at_cell_mean():
    ev, works, curv = _cell_fixture(-4.0)
    out = score_events(ev, works, curv, min_cell=4)
    assert out["s_geom"].iloc[4] == pytest.approx(0.0, abs=1e-12)
    # the first events have no reference with >= min_cell rows anywhere: NaN by design
    assert out["s_geom"].iloc[:4].isna().all()


def test_s_geom_hand_computed_two_sd():
    sd = math.sqrt(20 / 3)  # sample sd of -1,-3,-5,-7
    ev, works, curv = _cell_fixture(-4.0 - 2 * sd)
    out = score_events(ev, works, curv, min_cell=4)
    assert out["s_geom"].iloc[4] == pytest.approx(2.0, abs=1e-9)


def test_s_geom_reference_excludes_same_timestamp_events():
    rows = [("a1", "b", 10), ("a2", "b", 20), ("a3", "b", 30), ("a4", "b", 40),
            ("x", "y", 50), ("a5", "c", 50)]
    ev = make_events(rows)
    curv = make_curv(ev, [-1, -3, -5, -7, 500.0, -4.0])
    out = score_events(ev, make_works(ev), curv, min_cell=4)
    # the F=500 event at the same t must not enter the scored event's reference
    assert out.set_index("event_id").loc[5, "s_geom"] == pytest.approx(0.0, abs=1e-12)


def test_s_geom_falls_back_to_field_then_global():
    rows = [("a1", "b", 10), ("a2", "b", 20), ("a3", "b", 30),   # field F, 1970
            ("a4", "c", YEAR + 10),                               # field F, 1971
            ("g1", "c", YEAR + 20)]                               # field G, 1971
    ev = make_events(rows)
    works = make_works(ev, {"g1": "G"})
    curv = make_curv(ev, [0.0, -2.0, -4.0, -1.0, -3.0])
    out = score_events(ev, works, curv, min_cell=3, with_inputs=True).set_index("event_id")
    # event 3: cell (F,1971) empty -> field F (all earlier): mean -2, sd 2 -> |(-1+2)/2| = 0.5
    assert out.loc[3, "in_geom_level"] == 1
    assert out.loc[3, "s_geom"] == pytest.approx(0.5)
    # event 4: (G,1971) and G empty -> global: values 0,-2,-4,-1 -> mean -1.75
    vals = np.array([0.0, -2.0, -4.0, -1.0])
    assert out.loc[4, "in_geom_level"] == 2
    assert out.loc[4, "s_geom"] == pytest.approx(abs((-3 - vals.mean()) / vals.std(ddof=1)))


def test_s_geom_level_needs_positive_spread():
    # three identical earlier values in the cell: sd = 0 -> unusable -> NaN (no coarser data)
    rows = [("a1", "b", 10), ("a2", "b", 20), ("a3", "b", 30), ("a4", "c", 40)]
    ev = make_events(rows)
    out = score_events(ev, make_works(ev), make_curv(ev, [-2, -2, -2, -9]), min_cell=3)
    assert np.isnan(out["s_geom"].iloc[3])


def _escalation_fixture():
    sd_f = math.sqrt(20 / 3)          # Forman 0,-2,-4,-6: mean -3
    sd_o = math.sqrt(0.08)            # ORC 0.1, 0.5: mean 0.3, sample sd 0.2828
    rows = [("a1", "b", 10), ("a2", "b", 20), ("a3", "b", 30), ("a4", "b", 40),
            ("a5", "c", 50), ("a6", "d", 50)]  # same t: both see only events 0..3
    ev = make_events(rows)
    forman = [0, -2, -4, -6, -3 + sd_f, -3 + 0.5 * sd_f]
    orc = [0.1, np.nan, np.nan, 0.5, 0.3 + 3 * sd_o, np.nan]
    return ev, make_works(ev), make_curv(ev, forman, orc)


@pytest.mark.parametrize(
    "variant, expected_scored, expected_plain",
    [("full", 3.0, 0.5), ("no_temporal", 3.0, 0.5), ("forman_only", 1.0, 0.5),
     ("ollivier_only", 3.0, np.nan), ("no_curvature", np.nan, np.nan)],
)
def test_s_geom_combination_and_ablation_semantics(variant, expected_scored, expected_plain):
    ev, works, curv = _escalation_fixture()
    raw = score_events(ev, works, curv, min_cell=2, variant=variant)
    out = raw.set_index("event_id")
    for eid, exp in ((4, expected_scored), (5, expected_plain)):
        if np.isnan(exp):
            assert np.isnan(out.loc[eid, "s_geom"])
        else:
            assert out.loc[eid, "s_geom"] == pytest.approx(exp, abs=1e-9)
    if variant == "no_curvature":
        assert out["s_geom"].isna().all() and out["s_curvdyn"].isna().all()
        assert set(raw.attrs["disabled_components"]) == {"s_geom", "s_curvdyn"}


def test_ollivier_only_never_reads_forman():
    ev, works, curv = _escalation_fixture()
    a = score_events(ev, works, curv, min_cell=2, variant="ollivier_only")
    curv2 = curv.assign(forman=curv["forman"] * 7 + 100)
    b = score_events(ev, works, curv2, min_cell=2, variant="ollivier_only")
    pd.testing.assert_frame_equal(a, b)


def test_full_falls_back_to_forman_when_orc_failed():
    ev, works, curv = _escalation_fixture()
    curv.loc[4, "orc"] = np.nan  # escalated but ORC missing
    out = score_events(ev, works, curv, min_cell=2).set_index("event_id")
    assert out.loc[4, "s_geom"] == pytest.approx(1.0)


# ---------------------------------------------------------------- s_temp


def _burst_scenario(burst: bool, window=90, temporal_stat="rate_ratio"):
    rows = []
    # background: 12 papers, each cited 16 times at seeded random days in [0, 1500], field F
    rng = np.random.default_rng(1)
    for j in range(12):
        for m, day in enumerate(np.sort(rng.integers(0, 1501, 16))):
            rows.append((f"bg{j}_{m}", f"Q{j}", int(day)))
    # P history: 6 citations, every 100 days
    rows += [(f"h{m}", "P", 100 * m) for m in range(6)]
    if burst:
        rows += [(f"x{m}", "P", 1000) for m in range(10)]        # 10 in one day
    else:
        rows += [(f"x{m}", "P", 600 + 100 * m) for m in range(10)]  # natural spacing
    ev = make_events(rows)
    out = score_events(ev, make_works(ev), None, window_days=window, min_cell=10,
                       with_inputs=True, temporal_stat=temporal_stat)
    return out[out["src_id"].str.startswith("x")]


@pytest.mark.parametrize("temporal_stat", ["rate_ratio", "poisson_tail"])
def test_burst_of_ten_in_one_day_raises_s_temp_above_natural_spacing(temporal_stat):
    burst = _burst_scenario(True, temporal_stat=temporal_stat)
    natural = _burst_scenario(False, temporal_stat=temporal_stat)
    assert not burst["cold_start"].any() and not natural["cold_start"].any()
    assert burst["s_temp"].min() > natural["s_temp"].max()
    assert burst["s_temp"].min() > 2.0


def test_s_temp_raw_statistic_hand_computed():
    burst = _burst_scenario(True)
    row = burst.iloc[0]
    # same-day siblings count (9), self does not; history 0..500 lies before t-W=910
    assert row["in_n_recent"] == 9 and row["in_n_before"] == 6
    expected = math.log(10 / 7) + math.log((910 - 0) / 90)
    assert row["in_temp_x"] == pytest.approx(expected)


def test_s_temp_new_paper_without_history():
    # first citations all inside the window: n_before = 0, T_before = W -> x = log(n_recent + 1)
    ev = make_events([("a", "P", 50), ("b", "P", 60), ("c", "P", 70)])
    out = score_events(ev, make_works(ev), None, window_days=30, with_inputs=True)
    assert out["in_temp_x"].tolist() == pytest.approx([0.0, math.log(2), math.log(3)])


def _nb_tail_brute(n, n_before, hist, window, a0=1.0, b0=90.0):
    """P(N >= n) of the Gamma-Poisson predictive by summing the negative binomial pmf by hand."""
    r = a0 + n_before
    q = (b0 + hist) / (b0 + hist + window)
    pmf = [math.gamma(k + r) / (math.gamma(r) * math.factorial(k)) * q ** r * (1 - q) ** k for k in range(n)]
    return 1.0 - sum(pmf)


def test_poisson_tail_new_paper_without_history_hand_computed():
    # n_before = 0 -> prior only (a0 = 1, b0 = W): r = 1, q = 1/2, geometric: P(N >= n) = 2^-n
    ev = make_events([("a", "P", 50), ("b", "P", 60), ("c", "P", 70)])
    out = score_events(ev, make_works(ev), None, window_days=30, with_inputs=True,
                       temporal_stat="poisson_tail")
    step = math.log(2)
    assert out["in_temp_x"].tolist() == pytest.approx([0.0, step, 2 * step])


def test_poisson_tail_burst_hand_computed():
    burst = _burst_scenario(True, temporal_stat="poisson_tail")
    row = burst.iloc[0]
    assert row["in_n_recent"] == 9 and row["in_n_before"] == 6 and row["in_exposure_days"] == 910
    # r = 7, q = (90 + 910) / (90 + 910 + 90)
    assert row["in_temp_x"] == pytest.approx(-math.log(_nb_tail_brute(9, 6, 910, 90)), rel=1e-9)
    assert row["in_temp_x"] > 10  # nine same-day citations against about one per 150 days


def test_poisson_tail_prior_parameters_and_validation():
    ev = make_events([("a", "P", 50), ("b", "P", 60)])
    out = score_events(ev, make_works(ev), None, window_days=30, with_inputs=True,
                       temporal_stat="poisson_tail", temporal_prior_count=2.0, temporal_prior_days=100.0)
    # r = 2, q = 100 / 130: P(N >= 1) = 1 - q^2
    assert out["in_temp_x"].iloc[1] == pytest.approx(-math.log(1 - (100 / 130) ** 2))
    with pytest.raises(ValueError):
        score_events(ev, make_works(ev), None, temporal_stat="gap")
    with pytest.raises(ValueError):
        score_events(ev, make_works(ev), None, temporal_stat="poisson_tail", temporal_prior_count=0)


def _constant_rate_stream(seed=3):
    """Papers cited by homogeneous Poisson processes (rates 1/30, 1/90, 1/365 per day), published at
    days 0 .. 3000, observed until day 6000: no burst anywhere."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(150):
        rate = (1 / 30, 1 / 90, 1 / 365)[j % 3]
        start = int(rng.integers(0, 3001))
        n = rng.poisson(rate * (6000 - start))
        for m, day in enumerate(np.sort(rng.integers(start, 6001, n))):
            rows.append((f"c{j}_{m}", f"P{j}", int(day)))
    rows.sort(key=lambda r: r[2])
    return make_events(rows)


def test_poisson_tail_constant_rate_gives_near_zero_mean_surprise_at_every_age():
    ev = _constant_rate_stream()
    works = make_works(ev)
    tail = score_events(ev, works, None, window_days=90, with_inputs=True, temporal_stat="poisson_tail")
    ratio = score_events(ev, works, None, window_days=90, with_inputs=True)
    hist = (tail["in_n_prior_dst"] >= 3).to_numpy()  # past cold start
    rate = tail["dst_id"].map(lambda d: int(d[1:]) % 3).to_numpy()  # 0: 1/30, 1: 1/90, 2: 1/365 per day
    x_tail, x_ratio = tail["in_temp_x"].to_numpy(), ratio["in_temp_x"].to_numpy()
    # -ln of a tail p-value under the null: E <= 1 (E[-ln U] = 1), near 0 for rarely cited papers
    assert (x_tail >= 0).all() and x_tail[hist].mean() < 1.0
    rare = hist & (rate == 2)
    assert x_tail[rare].mean() < 0.5
    # no drift with history length on the rarely cited papers; the rate ratio drifts upward with age
    old, young = rare & (tail["in_exposure_days"].to_numpy() > 1500), rare & (tail["in_exposure_days"].to_numpy() <= 1500)
    assert old.sum() > 100 and young.sum() > 50
    assert abs(x_tail[old].mean() - x_tail[young].mean()) < 0.15
    assert x_ratio[old].mean() - x_ratio[young].mean() > 0.3
    assert x_ratio[rare].mean() > 2 * x_tail[rare].mean()


# ---------------------------------------------------------------- s_embed


def test_s_embed_tgn_is_negative_log_p_link():
    ev = make_events([("a", "b", 1), ("c", "d", 2), ("e", "f", 3), ("g", "h", 4)])
    tgn = pd.DataFrame({"event_id": [0, 1, 2], "p_link": [0.5, 0.0, 1.0]})
    out = score_events(ev, make_works(ev), None, tgn_scores=tgn, mode="tgn", p_link_eps=1e-7)
    assert out.attrs["embed_mode"] == "tgn"
    assert out["s_embed"].iloc[0] == pytest.approx(math.log(2))
    assert out["s_embed"].iloc[1] == pytest.approx(-math.log(1e-7))
    assert out["s_embed"].iloc[2] == pytest.approx(-math.log(1 - 1e-7))
    assert np.isnan(out["s_embed"].iloc[3])  # no p_link row: genuinely missing


def test_s_embed_static_cosine_surprise():
    cosines = [0.9, 0.8, 1.0, 0.7, 0.0, 0.85]
    rows, ids, vecs = [], [], []
    for i, c in enumerate(cosines):
        rows.append((f"s{i}", f"d{i}", 10 * (i + 1)))
        ids += [f"s{i}", f"d{i}"]
        vecs += [[1.0, 0.0], [c, math.sqrt(1 - c * c)]]
    ev = make_events(rows)
    emb = (np.array(ids), np.array(vecs, dtype=np.float64))
    out = score_events(ev, make_works(ev), None, embeddings=emb, mode="static_cosine",
                       min_cell=4)
    sd = math.sqrt(0.05 / 3)
    assert out["s_embed"].iloc[4] == pytest.approx(0.85 / sd, rel=1e-6)   # orthogonal: surprising
    ref = np.array(cosines[:5])
    assert out["s_embed"].iloc[5] == pytest.approx((ref.mean() - 0.85) / ref.std(ddof=1), rel=1e-6)


def test_mode_auto_picks_tgn_when_scores_given_else_static():
    ev, works, curv, emb, tgn_scores, tgn_emb, train = random_stream(n=50)
    a = score_events(ev, works, curv, embeddings=emb, tgn_scores=tgn_scores, train_mask=train)
    b = score_events(ev, works, curv, embeddings=emb, train_mask=train)
    assert a.attrs["embed_mode"] == "tgn" and b.attrs["embed_mode"] == "static_cosine"


# ---------------------------------------------------------------- s_curvdyn


def test_curvdyn_fitted_on_train_rows_only():
    ev, works, curv, emb, tgn_scores, tgn_emb, train = random_stream(n=200)
    kw = dict(train_mask=train, pca_dim=4, ridge_alpha=1.0, seed=0)
    m1 = fit_curvdyn_model(ev, curv, tgn_embeddings=tgn_emb, **kw)

    rng = np.random.default_rng(99)
    test_rows = ~train
    curv2 = curv.copy()
    curv2.loc[test_rows, "forman"] = rng.integers(-500, 3, size=test_rows.sum())
    tgn2 = {k: np.array(v, copy=True) for k, v in tgn_emb.items()}
    tgn2["z_src"][test_rows] = rng.normal(size=(test_rows.sum(), 6)) * 10
    tgn2["z_dst"][test_rows] = rng.normal(size=(test_rows.sum(), 6)) * 10
    m2 = fit_curvdyn_model(ev, curv2, tgn_embeddings=tgn2, **kw)
    for name in ("center", "basis", "feat_mean", "feat_scale", "coef"):
        np.testing.assert_array_equal(getattr(m1, name), getattr(m2, name))
    assert (m1.intercept, m1.rmse, m1.n_fit) == (m2.intercept, m2.rmse, m2.n_fit)
    assert m1.n_fit == int((train & np.isfinite(curv["forman"])).sum())

    # train-row scores are unchanged too
    o1 = score_events(ev, works, curv, tgn_embeddings=tgn_emb, **kw)
    o2 = score_events(ev, works, curv2, tgn_embeddings=tgn2, **kw)
    tr = set(ev.loc[train, "event_id"])
    s1 = o1.set_index("event_id").loc[sorted(tr), "s_curvdyn"]
    s2 = o2.set_index("event_id").loc[sorted(tr), "s_curvdyn"]
    pd.testing.assert_series_equal(s1, s2)

    # non-vacuous: changing a TRAIN row does change the model
    curv3 = curv.copy()
    curv3.loc[np.flatnonzero(train)[0], "forman"] += 50
    m3 = fit_curvdyn_model(ev, curv3, tgn_embeddings=tgn_emb, **kw)
    assert not np.array_equal(m1.coef, m3.coef)


def test_curvdyn_static_fit_ignores_test_only_nodes():
    ev, works, curv, (ids, vecs), *_, train = random_stream(n=200)
    test_rows = ~train
    ev2 = ev.copy()
    ev2.loc[test_rows, "dst_id"] = [f"new{i}" for i in range(test_rows.sum())]
    curv2 = curv.copy()
    curv2["dst_id"] = ev2["dst_id"]
    rng = np.random.default_rng(3)
    ids2 = np.concatenate([ids, [f"new{i}" for i in range(test_rows.sum())]])
    vecs2 = np.vstack([vecs, rng.normal(size=(test_rows.sum(), vecs.shape[1])).astype(np.float32)])
    kw = dict(train_mask=train, pca_dim=4, seed=0)
    m1 = fit_curvdyn_model(ev2, curv2, embeddings=(ids2, vecs2), **kw)
    vecs3 = vecs2.copy()
    vecs3[len(ids):] *= -5
    m2 = fit_curvdyn_model(ev2, curv2, embeddings=(ids2, vecs3), **kw)
    np.testing.assert_array_equal(m1.coef, m2.coef)
    np.testing.assert_array_equal(m1.basis, m2.basis)
    assert m1.source == "static" and m1.target == "forman"


def test_curvdyn_recovers_signal_and_flags_inconsistent_event():
    rng = np.random.default_rng(5)
    n, d = 240, 3
    rows = [(f"s{i}", f"d{i}", int(t)) for i, t in enumerate(np.sort(rng.integers(0, 900, n)))]
    ev = make_events(rows)
    z_src, z_dst = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    cos = np.einsum("ij,ij->i", z_src, z_dst) / np.linalg.norm(z_src, axis=1) / np.linalg.norm(z_dst, axis=1)
    y = 0.8 * z_src[:, 0] - 0.5 * z_dst[:, 1] + 0.6 * cos + rng.normal(scale=0.05, size=n)
    forman = np.sinh(y)
    forman[-1] = np.sinh(y[-1] + 2.0)  # geometry inconsistent with the representations
    curv = make_curv(ev, forman)
    tgn_emb = {"event_id": ev["event_id"].to_numpy(), "z_src": z_src, "z_dst": z_dst}
    train = np.arange(n) < 160
    out = score_events(ev, make_works(ev), curv, tgn_embeddings=tgn_emb, train_mask=train,
                       pca_dim=3, ridge_alpha=1e-6, with_inputs=True)
    s = out["s_curvdyn"].to_numpy()
    assert np.nanmedian(s[160:-1]) < 2.0
    assert s[-1] > 20.0
    assert out["in_curv_obs"].iloc[-1] == pytest.approx(y[-1] + 2.0)


def test_curvdyn_without_train_mask_is_nan_with_warning():
    ev, works, curv, emb, *_ = random_stream(n=40)
    with pytest.warns(UserWarning, match="train_mask"):
        out = score_events(ev, works, curv, embeddings=emb)
    assert out["s_curvdyn"].isna().all()


def test_curvdyn_ollivier_only_targets_orc_on_escalated_rows():
    ev, works, curv, emb, *_, train = random_stream(n=200)
    m = fit_curvdyn_model(ev, curv, embeddings=emb, train_mask=train, target="orc", pca_dim=3)
    esc_ok = train & curv["escalated"].to_numpy() & np.isfinite(curv["orc"].to_numpy())
    assert m.target == "orc"
    out = score_events(ev, works, curv, embeddings=emb, train_mask=train,
                       variant="ollivier_only", pca_dim=3).set_index("event_id")
    not_esc = curv.loc[~curv["escalated"], "event_id"]
    assert out.loc[not_esc, "s_curvdyn"].isna().all()
    assert m.n_fit <= esc_ok.sum()


# ---------------------------------------------------------------- determinism


def test_determinism_under_seed():
    ev, works, curv, emb, tgn_scores, tgn_emb, train = random_stream(n=200)
    kw = dict(embeddings=emb, tgn_scores=tgn_scores, tgn_embeddings=tgn_emb, train_mask=train,
              max_fit_rows=40, pca_dim=4, min_cell=5)
    a = score_events(ev, works, curv, seed=0, **kw)
    b = score_events(ev, works, curv, seed=0, **kw)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    m0 = fit_curvdyn_model(ev, curv, tgn_embeddings=tgn_emb, train_mask=train, max_fit_rows=40,
                           pca_dim=4, seed=0)
    m1 = fit_curvdyn_model(ev, curv, tgn_embeddings=tgn_emb, train_mask=train, max_fit_rows=40,
                           pca_dim=4, seed=1)
    assert m0.n_fit == 40 and not np.array_equal(m0.coef, m1.coef)  # the seed is really used


def test_matched_random_escalation():
    ev, works, curv, *_ = random_stream(n=200)
    a = matched_random_escalation(curv, seed=0)
    b = matched_random_escalation(curv, seed=0)
    c = matched_random_escalation(curv, seed=1)
    assert len(a) == int(curv["escalated"].sum())
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert set(a) <= set(curv["event_id"]) and len(set(a)) == len(a)


def test_matched_random_escalation_per_stratum_ignores_other_strata():
    """Regression (NFR3): with strata, each stratum's draw has that stratum's screen count and reads only its
    own rows, so changing the test rows (their flags, their number) leaves the train selection unchanged."""
    ev, works, curv, *_ = random_stream(n=200)
    split = np.where(curv["t"].to_numpy() < 700, "train", "test")
    a = matched_random_escalation(curv, seed=0, strata=split)
    esc = curv["escalated"].to_numpy()
    for g in ("train", "test"):
        ids_g = set(curv.loc[split == g, "event_id"])
        assert len(set(a) & ids_g) == int(esc[split == g].sum())
    changed = curv.copy()
    test_rows = np.flatnonzero(split == "test")
    changed.loc[test_rows, "escalated"] = ~changed.loc[test_rows, "escalated"].to_numpy()  # other test count
    b = matched_random_escalation(changed, seed=0, strata=split)
    train_ids = set(curv.loc[split == "train", "event_id"])
    assert set(a) & train_ids == set(b) & train_ids
    assert set(a) - train_ids != set(b) - train_ids
    np.testing.assert_array_equal(a, matched_random_escalation(curv, seed=0, strata=split))


# ---------------------------------------------------------------- agreement + brute force


@pytest.mark.parametrize("variant,temporal_stat", [("full", "rate_ratio"), ("forman_only", "rate_ratio"),
                                                   ("ollivier_only", "rate_ratio"), ("no_curvature", "rate_ratio"),
                                                   ("full", "poisson_tail"), ("no_curvature", "poisson_tail")])
@pytest.mark.parametrize("mode", ["static_cosine", "tgn"])
def test_single_event_path_agrees_with_vectorised(variant, mode, temporal_stat):
    ev, works, curv, emb, tgn_scores, tgn_emb, train = random_stream(n=300)
    kw = dict(embeddings=emb, tgn_scores=tgn_scores, tgn_embeddings=tgn_emb, train_mask=train,
              mode=mode, variant=variant, k=3, window_days=60, min_cell=5, pca_dim=4,
              temporal_stat=temporal_stat)
    vec = score_events(ev, works, curv, with_inputs=True, **kw).set_index("event_id")
    ctx = build_context(ev, works, curv, **kw)
    for eid in ev["event_id"]:
        b = score_event(eid, ctx)
        assert isinstance(b, ScoreBreakdown) and b.total is None
        row = vec.loc[eid]
        assert b.cold_start == bool(row["cold_start"])
        for c in ("s_embed", "s_geom", "s_temp", "s_curvdyn"):
            got, want = getattr(b, c), row[c]
            if np.isnan(want):
                assert np.isnan(got), (eid, c)
            else:
                assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (eid, c)
        for key, val in b.inputs.items():
            col = f"in_{key}"
            if col in vec.columns and isinstance(val, (int, float, np.floating, np.integer)):
                want = row[col]
                if isinstance(want, float) and np.isnan(want):
                    assert np.isnan(val), (eid, key)
                else:
                    assert val == pytest.approx(want, rel=1e-9, abs=1e-9), (eid, key)


def _brute_z(rows, i, value_of, min_cell):
    """Reference z for row i: earlier rows (t < t_i) with a finite value, cascade
    (field, year) -> field -> global, first level with >= min_cell rows and sd > 0."""
    me = rows[i]
    x = value_of(me)
    if not np.isfinite(x):
        return np.nan
    earlier = [r for r in rows if r["t"] < me["t"] and np.isfinite(value_of(r))]
    for keep in (lambda r: r["field"] == me["field"] and r["year"] == me["year"],
                 lambda r: r["field"] == me["field"],
                 lambda r: True):
        vals = np.array([value_of(r) for r in earlier if keep(r)])
        if len(vals) >= max(min_cell, 2) and vals.std(ddof=1) > 1e-12:
            return (x - vals.mean()) / vals.std(ddof=1)
    return np.nan


@pytest.mark.filterwarnings("ignore:s_curvdyn is NaN")
def test_vectorised_matches_brute_force():
    ev, works, curv, (ids, vecs), *_ = random_stream(n=150, seed=11)
    W, K, MC = 60, 3, 5
    out = score_events(ev, works, curv, embeddings=(ids, vecs), mode="static_cosine",
                       variant="forman_only", k=K, window_days=W, min_cell=MC,
                       with_inputs=True).set_index("event_id")
    field = dict(zip(works["work_id"], works["field"]))
    vec_of = dict(zip(ids, vecs.astype(np.float64)))
    fm = dict(zip(curv["event_id"], curv["forman"]))
    rows = []
    for r in ev.itertuples(index=False):
        a, b = vec_of.get(r.src_id), vec_of.get(r.dst_id)
        cos = np.nan if a is None or b is None else float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        times = sorted(int(x) for x in ev.loc[ev["dst_id"] == r.dst_id, "t"])
        n_recent = sum(r.t - W < s <= r.t for s in times) - 1
        before = [s for s in times if s <= r.t - W]
        T = max(r.t - W - before[0], W) if before else W
        x = math.log((n_recent + 1) / (len(before) + 1)) + math.log(T / W)
        rows.append(dict(eid=r.event_id, t=int(r.t), field=field[r.src_id],
                         year=1970 + int(np.datetime64(int(r.t), "D").astype("datetime64[Y]").astype(int)),
                         cos=cos, x=x, F=fm[r.event_id],
                         prior=sum(s < r.t for s in times)))
    for i, r in enumerate(rows):
        got = out.loc[r["eid"]]
        assert got["cold_start"] == (r["prior"] < K)
        assert got["in_temp_x"] == pytest.approx(r["x"])
        for comp, val, sign, fold in (("s_temp", "x", 1, False), ("s_embed", "cos", -1, False),
                                      ("s_geom", "F", 1, True)):
            z = _brute_z(rows, i, lambda q: q[val], MC)
            want = abs(z) if fold else sign * z
            if np.isnan(want):
                assert np.isnan(got[comp]), (r["eid"], comp)
            else:
                assert got[comp] == pytest.approx(want, rel=1e-9, abs=1e-9), (r["eid"], comp)


# ---------------------------------------------------------------- misc contracts


def test_field_falls_back_to_first_concept():
    ev = make_events([("a", "b", 1), ("c", "d", 2)])
    works = pd.DataFrame({"work_id": ["a", "b", "c", "d"],
                          "concepts": [["C1", "C2"], [], "C9", None]})
    out = score_events(ev, works, None, with_inputs=True)
    assert out["in_field"].tolist() == ["C1", "C9"]
    works2 = pd.DataFrame({"work_id": ["a"], "concepts": [[]]})
    out2 = score_events(ev, works2, None, with_inputs=True)
    assert out2["in_field"].tolist() == ["unknown", "unknown"]


def test_score_breakdown_inputs_are_traceable():
    ev, works, curv, emb, tgn_scores, tgn_emb, train = random_stream(n=120)
    ctx = build_context(ev, works, curv, embeddings=emb, tgn_scores=tgn_scores,
                        tgn_embeddings=tgn_emb, train_mask=train, min_cell=5, pca_dim=4)
    eid = int(ev["event_id"].iloc[-1])
    b = score_event(eid, ctx)
    for key in ("event_id", "src_id", "dst_id", "t", "field", "year", "variant", "embed_mode",
                "p_link", "forman", "orc", "escalated", "z_forman", "geom_level", "temp_x",
                "n_recent", "n_before", "temp_level", "curv_obs", "curv_pred", "curv_source",
                "n_prior_dst", "k"):
        assert key in b.inputs, key
    assert b.inputs["event_id"] == eid and b.inputs["curv_source"] == "tgn"


def test_rejects_unknown_variant_and_mode():
    ev = make_events([("a", "b", 1)])
    with pytest.raises(ValueError):
        score_events(ev, make_works(ev), None, variant="nope")
    with pytest.raises(ValueError):
        score_events(ev, make_works(ev), None, mode="nope")


def test_curvature_rows_must_match_events():
    ev = make_events([("a", "b", 1), ("c", "d", 2)])
    curv = make_curv(ev, [0.0, -1.0])
    curv.loc[1, "t"] = 99
    with pytest.raises(ValueError, match="curvature"):
        score_events(ev, make_works(ev), curv)


def test_no_quadratic_blowup_on_a_larger_stream():
    # 60k events must score in a few seconds (3M feasible at linearithmic cost)
    import time
    rng = np.random.default_rng(0)
    n = 60_000
    ev = pd.DataFrame({"event_id": np.arange(n, dtype=np.int64),
                       "src_id": rng.integers(0, 20_000, n).astype(str),
                       "dst_id": rng.integers(0, 5_000, n).astype(str),
                       "t": np.sort(rng.integers(0, 3650, n)).astype(np.int64)})
    works = pd.DataFrame({"work_id": np.arange(20_000).astype(str),
                          "field": np.array(list("ABCDE"))[np.arange(20_000) % 5]})
    curv = make_curv(ev, rng.integers(-50, 3, n).astype(float))
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = score_events(ev, works, curv)
    assert time.perf_counter() - t0 < 20
    assert len(out) == n and out["s_temp"].notna().mean() > 0.99

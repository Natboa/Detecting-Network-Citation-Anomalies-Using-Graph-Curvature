"""S3.2: combiner - frozen z-statistics, simplex weights, calibration, learned head.

Spec under test (IMPLEMENTATION_PLAN.md S3.2, CONTRACTS.md "Combiner"):
- ZStats are fitted on TRAIN rows only (mean/std with a std floor, or
  median/MAD*1.4826 when robust) and freeze to / thaw from a plain dict.
- combine: z_c = (x_c - center_c) / scale_c; total = sum_c w_c * z_c with
  weights on the simplex; a NaN component contributes 0 and
  n_components_used counts the positive-weight components that contributed.
- With weights (1, 0, 0, 0) total == z(s_embed).
- simplex_grid returns every weight vector on the step lattice of the simplex.
- calibrate_weights maximises AP (or AUROC) on validation rows over the grid;
  ties go to the vector closest to uniform, then the lexicographically
  smallest one.
- LearnedCombiner: logistic head over the z-features, deterministic,
  round-trips through to_dict/from_dict (used only for the L13 ablation cell).
"""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from citecurv.eval.metrics import ranking_metrics
from citecurv.scoring.combiner import (
    COMPONENTS,
    LearnedCombiner,
    ZStats,
    calibrate_weights,
    combine,
    simplex_grid,
)


def frame(rows, cold=None):
    df = pd.DataFrame(rows, columns=list(COMPONENTS))
    df.insert(0, "event_id", np.arange(len(df), dtype=np.int64))
    if cold is not None:
        df["cold_start"] = cold
    return df


def test_zstats_hand_values():
    df = frame([[1, 10, 0, 5], [3, 10, 2, 5], [1, 10, 4, 5], [3, 10, 6, 5]])
    zs = ZStats.fit(df, std_floor=1e-3)
    assert zs.center["s_embed"] == pytest.approx(2.0)
    assert zs.scale["s_embed"] == pytest.approx(1.0)  # population std (ddof=0)
    assert zs.center["s_temp"] == pytest.approx(3.0)
    assert zs.scale["s_temp"] == pytest.approx(math.sqrt(5.0))
    assert zs.scale["s_geom"] == pytest.approx(1e-3)  # constant column -> floor
    assert zs.n_fit == {c: 4 for c in COMPONENTS}


def test_zstats_robust_median_mad():
    df = frame([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [100, 0, 0, 0]])
    zs = ZStats.fit(df, robust=True)
    assert zs.center["s_embed"] == pytest.approx(3.0)
    assert zs.scale["s_embed"] == pytest.approx(1.4826)  # MAD = median(|2,1,0,1,97|) = 1


def test_zstats_ignore_nan_and_cold_start_rows():
    df = frame(
        [[1, np.nan, 0, 0], [3, 2, 0, 0], [50, 4, 0, 0], [np.nan, np.nan, 0, 0]],
        cold=[False, False, True, False],
    )
    zs = ZStats.fit(df)
    assert zs.center["s_embed"] == pytest.approx(2.0)  # 50 is a cold-start row
    assert zs.n_fit["s_embed"] == 2 and zs.n_fit["s_geom"] == 1
    kept = ZStats.fit(df, exclude_cold_start=False)
    assert kept.center["s_embed"] == pytest.approx(18.0)


def _split_fixture(seed=0, n=300):
    rng = np.random.default_rng(seed)
    df = frame(rng.normal(size=(n, 4)) * [1, 2, 3, 4] + [0, 1, 2, 3])
    split = pd.Series(np.repeat(["train", "val", "test"], [n // 2, n // 4, n - n // 2 - n // 4]))
    return df, split


def test_zstats_never_touched_by_val_or_test_rows():
    df, split = _split_fixture()
    zs = ZStats.fit(df, split=split)
    perturbed = df.copy()
    held_out = (split != "train").to_numpy()
    rng = np.random.default_rng(99)
    perturbed.loc[held_out, list(COMPONENTS)] = rng.normal(size=(held_out.sum(), 4)) * 1e6
    zs_perturbed = ZStats.fit(perturbed, split=split)
    assert zs_perturbed.to_dict() == zs.to_dict()
    # the fit really depends on which rows are used
    assert ZStats.fit(perturbed).to_dict() != zs.to_dict()
    # frozen stats: train rows score identically whatever the held-out rows hold
    w = {"s_embed": 0.4, "s_geom": 0.3, "s_temp": 0.2, "s_curvdyn": 0.1}
    a = combine(df, w, zs)
    b = combine(perturbed, w, zs_perturbed)
    assert np.array_equal(a.loc[~held_out, "total"], b.loc[~held_out, "total"])


def test_zstats_split_by_column_name_and_empty_train_raises():
    df, split = _split_fixture()
    with_col = df.assign(split=split)
    assert ZStats.fit(with_col, split="split").to_dict() == ZStats.fit(df, split=split).to_dict()
    with pytest.raises(ValueError, match="train"):
        ZStats.fit(df, split=pd.Series(["test"] * len(df)))


def test_zstats_dict_round_trip_is_json_safe():
    df = frame([[1, np.nan, 0, 2], [3, np.nan, 1, 4]])
    zs = ZStats.fit(df, robust=True, std_floor=0.01)
    d = zs.to_dict()
    text = json.dumps(d, allow_nan=False)  # NaN stored as null
    back = ZStats.from_dict(json.loads(text))
    assert back.to_dict() == d
    assert math.isnan(back.center["s_geom"]) and back.robust and back.std_floor == 0.01


def test_weights_one_hot_total_equals_z_embed():
    df, split = _split_fixture(1)
    zs = ZStats.fit(df, split=split)
    out = combine(df, {"s_embed": 1.0}, zs)
    expected = (df["s_embed"] - zs.center["s_embed"]) / zs.scale["s_embed"]
    assert np.allclose(out["z_s_embed"], expected, atol=0, rtol=1e-15)
    assert np.array_equal(out["total"].to_numpy(), out["z_s_embed"].to_numpy())
    out2 = combine(df, (1.0, 0.0, 0.0, 0.0), zs)
    assert np.array_equal(out2["total"].to_numpy(), out["total"].to_numpy())


def test_combine_hand_case_and_nan_contributes_zero():
    train = frame([[0, 0, 0, 0], [2, 4, 2, 2]])  # centers 1,2,1,1; scales 1,2,1,1
    zs = ZStats.fit(train)
    df = frame([[3, 6, np.nan, 1], [np.nan, np.nan, np.nan, np.nan]], cold=[False, True])
    out = combine(df, {"s_embed": 0.5, "s_geom": 0.25, "s_temp": 0.25}, zs)
    assert list(out.columns[: len(df.columns)]) == list(df.columns)  # input columns kept
    assert out.loc[0, "z_s_embed"] == pytest.approx(2.0)
    assert out.loc[0, "z_s_geom"] == pytest.approx(2.0)
    assert math.isnan(out.loc[0, "z_s_temp"])
    assert out.loc[0, "total"] == pytest.approx(0.5 * 2 + 0.25 * 2)  # s_temp NaN -> 0
    assert out.loc[0, "n_components_used"] == 2  # s_curvdyn has weight 0
    assert out.loc[1, "total"] == 0.0 and out.loc[1, "n_components_used"] == 0


@pytest.mark.parametrize(
    "weights, match",
    [
        ({"s_embed": 1.2, "s_geom": -0.2}, "non-negative"),
        ({"s_embed": 0.5}, "sum to 1"),
        ({"s_embed": 0.5, "s_other": 0.5}, "unknown"),
        ((0.5, 0.5), "length"),
    ],
)
def test_combine_rejects_bad_weights(weights, match):
    df, _ = _split_fixture()
    zs = ZStats.fit(df)
    with pytest.raises(ValueError, match=match):
        combine(df, weights, zs)


def test_simplex_grid_small_exact():
    g = simplex_grid(n=3, step=0.5)
    assert g.tolist() == [
        [0.0, 0.0, 1.0], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0],
    ]


def test_simplex_grid_default_is_the_full_lattice():
    g = simplex_grid(n=4, step=0.1)
    assert g.shape == (math.comb(13, 3), 4)  # 286 compositions of 10 into 4 parts
    assert (g >= 0).all()
    assert np.allclose(g.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(g * 10, np.round(g * 10), atol=1e-9)
    brute = {p for p in itertools.product(range(11), repeat=4) if sum(p) == 10}
    assert {tuple(int(round(x * 10)) for x in row) for row in g} == brute
    with pytest.raises(ValueError, match="step"):
        simplex_grid(n=4, step=0.3)


def _brute_calibrate(df, y, zs, step, metric):
    best_key, best = None, None
    for w in simplex_grid(len(zs.components), step):
        total = combine(df, dict(zip(zs.components, w)), zs)["total"]
        value = ranking_metrics(total.to_numpy(), y)[metric]
        parts = np.round(w / step).astype(int)
        key = (-value, float(((parts - (1 / step) / len(parts)) ** 2).sum()), tuple(parts))
        if best_key is None or key < best_key:
            best_key, best = key, (w, value)
    return best


@pytest.mark.parametrize("metric", ["ap", "auroc"])
def test_calibrate_matches_brute_force(metric):
    rng = np.random.default_rng(5)
    n = 400
    y = (rng.random(n) < 0.15).astype(int)
    df = frame(np.c_[
        rng.normal(size=n) + 1.5 * y,     # informative
        rng.normal(size=n) + 0.7 * y,     # weakly informative
        rng.normal(size=n),               # noise
        rng.normal(size=n) - 0.5 * y,     # anti-informative
    ])
    zs = ZStats.fit(df)
    weights, value, table = calibrate_weights(df, y, zs, step=0.25, metric=metric)
    w_brute, v_brute = _brute_calibrate(df, y, zs, 0.25, metric)
    assert value == pytest.approx(v_brute, abs=1e-12)
    assert np.allclose([weights[c] for c in COMPONENTS], w_brute)
    assert weights["s_embed"] >= 0.5 and weights["s_curvdyn"] == 0.0
    assert len(table) == math.comb(4 + 3, 3) and table[metric].max() == pytest.approx(value)
    assert table["chosen"].sum() == 1


def test_calibrate_tie_break_closest_to_uniform_then_lexicographic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = (x + rng.normal(size=200) > 1).astype(int)
    df = frame(np.c_[x, x, x, x])  # every weight vector gives the same ranking
    zs = ZStats.fit(df)
    weights, _, table = calibrate_weights(df, y, zs, step=0.1)
    # (2,2,3,3)/10 and its permutations are the lattice points nearest 0.25 each
    assert [weights[c] for c in COMPONENTS] == pytest.approx([0.2, 0.2, 0.3, 0.3])


def test_calibrate_labels_by_event_id_and_cold_start_dropped():
    rng = np.random.default_rng(2)
    n = 120
    y = (rng.random(n) < 0.2).astype(int)
    df = frame(np.c_[rng.normal(size=n) + y, rng.normal(size=(n, 3))], cold=[False] * n)
    zs = ZStats.fit(df)
    expected = calibrate_weights(df, y, zs, step=0.5)
    labels = pd.DataFrame({"event_id": df["event_id"], "label": y}).iloc[::-1]
    # add cold-start rows that would wreck the ranking if they were kept
    extra = frame(np.full((10, 4), 50.0), cold=[True] * 10)
    extra["event_id"] += 1000
    both = pd.concat([df, extra], ignore_index=True)
    lab_both = pd.concat([labels, pd.DataFrame({"event_id": extra["event_id"], "label": 0})])
    got = calibrate_weights(both, lab_both, zs, step=0.5)
    assert got[0] == expected[0] and got[1] == pytest.approx(expected[1])


def test_calibrate_without_positives_raises():
    df, _ = _split_fixture()
    zs = ZStats.fit(df)
    with pytest.raises(ValueError, match="positive"):
        calibrate_weights(df, np.zeros(len(df), dtype=int), zs)


def test_learned_combiner_fits_deterministically_and_round_trips():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    n = 600
    y = (rng.random(n) < 0.2).astype(int)
    df = frame(np.c_[rng.normal(size=n) + 2 * y, rng.normal(size=(n, 3))])
    zs = ZStats.fit(df)
    a = LearnedCombiner(zs, random_state=0).fit(df, y)
    b = LearnedCombiner(zs, random_state=0).fit(df, y)
    sa, sb = a.predict_score(df), b.predict_score(df)
    assert np.array_equal(sa, sb)
    assert a.coef["s_embed"] > 0 and abs(a.coef["s_embed"]) > 3 * max(
        abs(a.coef[c]) for c in COMPONENTS[1:]
    )
    assert ranking_metrics(sa, y)["auroc"] > 0.85
    d = json.loads(json.dumps(a.to_dict(), allow_nan=False))
    back = LearnedCombiner.from_dict(d)
    assert np.allclose(back.predict_score(df), sa, atol=1e-12)


def test_learned_combiner_nan_component_is_zero_feature():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(4)
    n = 200
    y = (rng.random(n) < 0.3).astype(int)
    df = frame(np.c_[rng.normal(size=n) + y, rng.normal(size=(n, 3))])
    zs = ZStats.fit(df)
    model = LearnedCombiner(zs).fit(df, y)
    row = df.iloc[[0]].copy()
    row["s_geom"] = np.nan
    expected = model.intercept + sum(
        model.coef[c] * (row[c].iloc[0] - zs.center[c]) / zs.scale[c] for c in COMPONENTS if c != "s_geom"
    )
    assert model.predict_score(row)[0] == pytest.approx(expected)

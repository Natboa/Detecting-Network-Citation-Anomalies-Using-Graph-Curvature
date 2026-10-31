"""S4.6: ranking metrics shared by the model and every baseline.

Spec under test (IMPLEMENTATION_PLAN.md S4.6, CONTRACTS.md "Metrics"):
- ranking_metrics(scores, labels, ks) -> dict with keys n, n_pos, auroc, ap,
  p_at_{k}, found_at_{k}; higher score = more anomalous.
- AUROC is the rank (Mann-Whitney) statistic with average ranks for ties; AP
  uses the same step definition as sklearn.metrics.average_precision_score.
- top-k order: descending score, ties broken by input order (stable).
- k > n uses the whole list; precision@k divides by min(k, n).
- n_pos == 0 gives NaN auroc/ap without raising.
- evaluate() joins on event_id and drops cold_start rows unless asked not to.
- seed_summary() gives mean and sample std (ddof=1) per key.
"""

import math

import numpy as np
import pandas as pd
import pytest

from citecurv.eval.metrics import evaluate, ranking_metrics, seed_summary


def test_keys_are_the_contract_keys():
    m = ranking_metrics([0.3, 0.2, 0.1], [1, 0, 0], ks=(100, 1000))
    assert set(m) == {
        "n", "n_pos", "auroc", "ap",
        "p_at_100", "found_at_100", "p_at_1000", "found_at_1000",
    }


def test_hand_computed_no_ties():
    # descending: pos, neg, pos, neg, neg
    m = ranking_metrics([0.9, 0.8, 0.7, 0.6, 0.5], [1, 0, 1, 0, 0], ks=(1, 2, 3))
    assert m["n"] == 5 and m["n_pos"] == 2
    # positives outrank 3/3 and 2/3 of the negatives -> (3 + 2) / 6
    assert m["auroc"] == pytest.approx(5 / 6)
    # AP = 0.5 * 1 + 0.5 * 2/3
    assert m["ap"] == pytest.approx(5 / 6)
    assert (m["found_at_1"], m["p_at_1"]) == (1, 1.0)
    assert (m["found_at_2"], m["p_at_2"]) == (1, 0.5)
    assert (m["found_at_3"], m["p_at_3"]) == (2, pytest.approx(2 / 3))


def test_hand_computed_with_ties():
    # pairs (pos, neg): (1 vs 2) loses, (1 vs 1) ties -> 0.5, (0 vs 2) and (0 vs 1) lose
    scores = [2.0, 1.0, 1.0, 0.0]
    labels = [0, 1, 0, 1]
    m = ranking_metrics(scores, labels, ks=(1,))
    assert m["auroc"] == pytest.approx(0.5 / 4)
    # thresholds 2, 1, 0 -> (P, R) = (0, 0), (1/3, 1/2), (1/2, 1)
    assert m["ap"] == pytest.approx(0.5 * (1 / 3) + 0.5 * 0.5)


def test_all_tied_scores_give_chance_auroc():
    m = ranking_metrics([1.0] * 6, [1, 0, 0, 1, 0, 0], ks=(3,))
    assert m["auroc"] == pytest.approx(0.5)
    assert m["ap"] == pytest.approx(2 / 6)  # single threshold: precision = prevalence


@pytest.mark.parametrize("seed", range(25))
def test_cross_check_against_sklearn(seed):
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 400))
    labels = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(int)
    if labels.sum() in (0, n):
        labels[0], labels[-1] = 1, 0
    scores = rng.normal(size=n) + labels * rng.uniform(0, 2)
    if seed % 2:  # heavy ties
        scores = np.round(scores, 1)
    m = ranking_metrics(scores, labels)
    assert m["auroc"] == pytest.approx(sklearn_metrics.roc_auc_score(labels, scores), abs=1e-12)
    assert m["ap"] == pytest.approx(
        sklearn_metrics.average_precision_score(labels, scores), abs=1e-12
    )


def test_top_k_ties_broken_by_input_order():
    scores = [1.0, 1.0, 1.0, 0.0]
    m = ranking_metrics(scores, [0, 0, 1, 1], ks=(1, 2, 3))
    assert [m["found_at_1"], m["found_at_2"], m["found_at_3"]] == [0, 0, 1]
    m = ranking_metrics(scores, [1, 0, 0, 1], ks=(1, 2, 3))
    assert [m["found_at_1"], m["found_at_2"], m["found_at_3"]] == [1, 1, 1]


def test_top_k_brute_force():
    rng = np.random.default_rng(7)
    scores = np.round(rng.normal(size=300), 1)
    labels = (rng.random(300) < 0.2).astype(int)
    m = ranking_metrics(scores, labels, ks=(10, 50, 299))
    order = sorted(range(300), key=lambda i: (-scores[i], i))
    for k in (10, 50, 299):
        found = int(sum(labels[i] for i in order[:k]))
        assert m[f"found_at_{k}"] == found
        assert m[f"p_at_{k}"] == pytest.approx(found / k)


def test_k_larger_than_n():
    m = ranking_metrics([0.4, 0.3, 0.2, 0.1, 0.0], [0, 1, 0, 1, 0], ks=(100,))
    assert m["found_at_100"] == 2
    assert m["p_at_100"] == pytest.approx(2 / 5)


def test_no_positives_gives_nan_without_raising():
    m = ranking_metrics([0.3, 0.2, 0.1], [0, 0, 0], ks=(2,))
    assert m["n_pos"] == 0
    assert math.isnan(m["auroc"]) and math.isnan(m["ap"])
    assert m["found_at_2"] == 0 and m["p_at_2"] == 0.0


def test_all_positives():
    m = ranking_metrics([0.3, 0.2, 0.1], [1, 1, 1], ks=(2,))
    assert math.isnan(m["auroc"])  # undefined without negatives
    assert m["ap"] == pytest.approx(1.0)  # sklearn's value for this case


def test_empty_input():
    m = ranking_metrics([], [], ks=(5,))
    assert m["n"] == 0 and m["n_pos"] == 0 and m["found_at_5"] == 0
    assert math.isnan(m["auroc"]) and math.isnan(m["ap"]) and math.isnan(m["p_at_5"])


def test_accepts_series_and_bool_labels():
    s = pd.Series([0.9, 0.1, 0.5], index=[10, 11, 12])
    y = pd.Series([True, False, False], index=[10, 11, 12])
    m = ranking_metrics(s, y, ks=(1,))
    assert m["auroc"] == pytest.approx(1.0) and m["found_at_1"] == 1


def test_series_index_mismatch_raises():
    s = pd.Series([0.9, 0.1], index=[1, 2])
    y = pd.Series([1, 0], index=[2, 1])
    with pytest.raises(ValueError, match="index"):
        ranking_metrics(s, y)


@pytest.mark.parametrize(
    "scores, labels, match",
    [
        ([0.1, 0.2], [1], "length"),
        ([0.1, 0.2], [1, 2], "0/1"),
        ([0.1, float("nan")], [1, 0], "finite"),
    ],
)
def test_bad_input_raises(scores, labels, match):
    with pytest.raises(ValueError, match=match):
        ranking_metrics(scores, labels)


def test_metrics_are_json_plain_types():
    m = ranking_metrics(np.array([0.2, 0.1]), np.array([1, 0]), ks=(1,))
    assert type(m["n"]) is int and type(m["found_at_1"]) is int
    assert type(m["auroc"]) is float and type(m["p_at_1"]) is float


def _scores_df():
    return pd.DataFrame({
        "event_id": [5, 3, 9, 1, 7],
        "total": [0.9, 0.8, 0.7, 0.6, 0.5],
        "cold_start": [False, False, True, False, False],
    })


def test_evaluate_joins_on_event_id_and_drops_cold_start():
    labels = pd.DataFrame({"event_id": [1, 3, 5, 7, 9, 42], "label": [0, 0, 1, 0, 1, 1]})
    m = evaluate(_scores_df(), labels, ks=(1, 2))
    # event 9 is cold start; 42 is not scored. Kept ranking: 5(+), 3, 1, 7
    assert m["n"] == 4 and m["n_pos"] == 1
    assert m["auroc"] == pytest.approx(1.0)
    assert m["found_at_1"] == 1


def test_evaluate_can_keep_cold_start_rows():
    labels = pd.DataFrame({"event_id": [1, 3, 5, 7, 9], "label": [0, 0, 1, 0, 1]})
    m = evaluate(_scores_df(), labels, ks=(2,), drop_cold_start=False)
    assert m["n"] == 5 and m["n_pos"] == 2
    direct = ranking_metrics([0.9, 0.8, 0.7, 0.6, 0.5], [1, 0, 1, 0, 0], ks=(2,))
    assert m == direct


def test_evaluate_baseline_score_column():
    df = pd.DataFrame({"event_id": [1, 2, 3], "score": [0.1, 0.9, 0.5]})
    labels = pd.DataFrame({"event_id": [3, 2, 1], "label": [0, 1, 0]})
    m = evaluate(df, labels, score_col="score", ks=(1,))
    assert m["found_at_1"] == 1 and m["auroc"] == pytest.approx(1.0)


def test_evaluate_missing_label_raises_unless_filled():
    labels = pd.DataFrame({"event_id": [5], "label": [1]})
    with pytest.raises(ValueError, match="label"):
        evaluate(_scores_df(), labels)
    m = evaluate(_scores_df(), labels, fill_missing_label=0, ks=(1,))
    assert m["n"] == 4 and m["n_pos"] == 1 and m["found_at_1"] == 1


def test_evaluate_rejects_duplicate_event_ids():
    labels = pd.DataFrame({"event_id": [1, 1, 3, 5, 7, 9], "label": [0, 1, 0, 1, 0, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(_scores_df(), labels)


def test_seed_summary_mean_and_sample_std():
    runs = [
        {"n": 10, "auroc": 0.6, "ap": 0.2},
        {"n": 10, "auroc": 0.7, "ap": 0.4},
        {"n": 10, "auroc": 0.8, "ap": 0.3},
    ]
    s = seed_summary(runs)
    assert s["auroc"]["mean"] == pytest.approx(0.7)
    assert s["auroc"]["std"] == pytest.approx(0.1)  # ddof=1
    assert s["ap"]["mean"] == pytest.approx(0.3)
    assert s["ap"]["std"] == pytest.approx(0.1)
    assert s["n"] == {"mean": 10.0, "std": 0.0, "n_seeds": 3}


def test_seed_summary_single_seed_std_is_nan_and_keys_must_match():
    s = seed_summary([{"ap": 0.5}])
    assert s["ap"]["mean"] == 0.5 and math.isnan(s["ap"]["std"])
    with pytest.raises(ValueError, match="keys"):
        seed_summary([{"ap": 0.5}, {"auroc": 0.5}])
    with pytest.raises(ValueError):
        seed_summary([])

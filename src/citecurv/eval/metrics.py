"""Ranking metrics: one code path for the model and every baseline (task S4.6).

Higher score = more anomalous, everywhere. Labels are 0/1 (bool accepted).

- AUROC: the rank (Mann-Whitney) statistic. Scores are ranked ascending with
  average ranks for ties, so a tied (positive, negative) pair counts 1/2.
  Equal to sklearn.metrics.roc_auc_score.
- AP: the step definition of sklearn.metrics.average_precision_score,
  AP = sum_i (R_i - R_{i-1}) * P_i over the distinct score values taken as
  thresholds in descending order (tied scores enter the list together).
- precision@k / found@k: the top-k list orders events by descending score and
  breaks ties by input order (a stable sort), so the list is a deterministic
  function of the inputs. Callers that want a data-independent tie order sort
  their input by (t, event_id) first, the canonical event order. When k > n
  the whole list is used and p_at_k divides by min(k, n), the length of the
  list actually returned.
- Degenerate inputs never raise: with no positives AUROC and AP are NaN; with
  no negatives AUROC is NaN and AP is 1.0 (sklearn's value); an empty input
  gives NaN for every ratio and 0 for every count.

Leak argument: these functions read finished scores and outcome labels only;
they produce no feature and feed nothing back into scoring.
"""

import numpy as np

DEFAULT_KS = (100, 1000)


def _to_arrays(scores, labels):
    if hasattr(scores, "iloc") and hasattr(labels, "iloc"):  # two pandas objects
        if not scores.index.equals(labels.index):
            raise ValueError(
                "scores and labels are both pandas objects with different index; "
                "align them first (evaluate() joins on event_id)"
            )
    s = np.asarray(scores, dtype=float).ravel()
    y_raw = np.asarray(labels).ravel()
    if s.shape != y_raw.shape:
        raise ValueError(f"length mismatch: {s.size} scores vs {y_raw.size} labels")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite (no NaN or inf)")
    if y_raw.size and not np.isin(y_raw, (0, 1)).all():
        raise ValueError("labels must be 0/1 (or bool)")
    return s, y_raw.astype(np.int64)


def _average_ranks(x):
    """1-based ranks of x ascending; tied values share the mean of their ranks."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], xs.size]  # exclusive
    group_rank = (starts + 1 + ends) / 2.0  # mean of ranks starts+1 .. ends
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.repeat(group_rank, ends - starts)
    return ranks


def auroc(scores, labels):
    """Rank-based AUROC with average ranks for ties; NaN if a class is absent."""
    s, y = _to_arrays(scores, labels)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = _average_ranks(s)[y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores, labels):
    """Step-wise AP as in sklearn; NaN when there is no positive."""
    s, y = _to_arrays(scores, labels)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s_desc, y_desc = s[order], y[order]
    # last position of each run of equal scores = one threshold
    last = np.flatnonzero(np.r_[s_desc[1:] != s_desc[:-1], True])
    tp = np.cumsum(y_desc)[last]
    precision = tp / (last + 1.0)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def top_k_order(scores):
    """Indices by descending score, ties by input order (stable)."""
    s = np.asarray(scores, dtype=float).ravel()
    return np.argsort(-s, kind="stable")


def ranking_metrics(scores, labels, ks=DEFAULT_KS) -> dict:
    """AUROC, AP, precision@k and positives-found@k for one ranked list.

    scores, labels: equal-length array-likes or pandas Series (two Series must
    share the same index; values are matched by position otherwise).
    Returns plain Python numbers: n, n_pos (int), auroc, ap (float, NaN when
    undefined), and for each k: found_at_{k} (int, positives in the top
    min(k, n)) and p_at_{k} (float, found / min(k, n); NaN when n == 0).
    """
    s, y = _to_arrays(scores, labels)
    n = int(y.size)
    out = {
        "n": n,
        "n_pos": int(y.sum()),
        "auroc": auroc(s, y),
        "ap": average_precision(s, y),
    }
    hits = np.cumsum(y[top_k_order(s)])
    for k in ks:
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        m = min(k, n)
        found = int(hits[m - 1]) if m else 0
        out[f"p_at_{k}"] = found / m if m else float("nan")
        out[f"found_at_{k}"] = found
    return out


def evaluate(
    scores_df,
    labels_df,
    score_col="total",
    ks=DEFAULT_KS,
    drop_cold_start=True,
    fill_missing_label=None,
) -> dict:
    """ranking_metrics on a scores frame joined to event labels by event_id.

    scores_df: event_id + score_col (+ optional cold_start). Baselines pass
    their (event_id, score) frame with score_col="score".
    labels_df: event_id, label. Labelled events that were not scored are
    ignored. A scored event without a label raises, unless fill_missing_label
    gives the value to use (0 when labels_df lists positives only).
    drop_cold_start: when the cold_start column exists, those rows are removed
    first (cold-start events are excluded from scoring, CONTRACTS.md).
    Rows are ranked in scores_df order, so top-k ties follow that order.
    """
    import pandas as pd

    for name, df in (("scores_df", scores_df), ("labels_df", labels_df)):
        if df["event_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate event_id values")
    df = scores_df
    if drop_cold_start and "cold_start" in df.columns:
        df = df[~df["cold_start"].astype(bool)]
    lab = pd.Series(labels_df["label"].to_numpy(), index=labels_df["event_id"].to_numpy())
    y = lab.reindex(df["event_id"].to_numpy())
    if y.isna().any():
        if fill_missing_label is None:
            raise ValueError(
                f"{int(y.isna().sum())} scored events have no label "
                "(pass fill_missing_label=0 if labels_df lists positives only)"
            )
        y = y.fillna(fill_missing_label)
    return ranking_metrics(df[score_col].to_numpy(), y.to_numpy().astype(np.int64), ks=ks)


def seed_summary(metric_dicts) -> dict:
    """Mean and sample std (ddof=1) per key over seeds: {key: {mean, std, n_seeds}}.

    All dicts must have the same keys. NaN values propagate (a seed with an
    undefined metric makes the summary undefined rather than silently
    dropping it). With one seed std is NaN.
    """
    runs = list(metric_dicts)
    if not runs:
        raise ValueError("seed_summary needs at least one metric dict")
    keys = list(runs[0])
    if any(set(r) != set(keys) for r in runs):
        raise ValueError("all metric dicts must have the same keys")
    out = {}
    for key in keys:
        vals = np.array([float(r[key]) for r in runs])
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
        out[key] = {"mean": float(np.mean(vals)), "std": std, "n_seeds": len(runs)}
    return out


__all__ = [
    "auroc", "average_precision", "evaluate", "ranking_metrics", "seed_summary",
    "top_k_order", "DEFAULT_KS",
]

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


LABEL_VIEWS = ("citing_side", "cited_side", "per_citing_paper", "per_cited_paper")


def _paper_metrics(s, positive, paper_ids, ks):
    """ranking_metrics over papers: a paper's score is the mean score of its rows, and it is positive when
    any of its rows is. Papers keep first-appearance order, so top-k ties are deterministic."""
    import pandas as pd

    codes, uniques = pd.factorize(np.asarray(paper_ids, dtype=object))
    if (codes < 0).any():
        raise ValueError("paper ids must not be null")
    n = len(uniques)
    count = np.bincount(codes, minlength=n)
    mean = np.bincount(codes, weights=s, minlength=n) / np.maximum(count, 1)
    pos = np.bincount(codes, weights=positive.astype(float), minlength=n) > 0
    return ranking_metrics(mean, pos.astype(np.int64), ks=ks)


def label_views(scores, labels, via_src, via_dst, src_ids, dst_ids, ks=DEFAULT_KS) -> dict:
    """One ranked list seen four more ways (ledger G10): the two label sources apart, and per paper.

    Inputs are equal-length array-likes, one entry per scored event: score, label (0/1), via_src and via_dst
    (0/1: the event is positive because its citing paper, its cited paper is labelled), and the two paper ids.
    Every positive must carry at least one source flag.

    - citing_side: the positives labelled through the citing paper (via_src, including events labelled on both
      sides) ranked against the negatives; positives of the other source leave the list.
    - cited_side: the positives labelled through the cited paper only (via_dst and not via_src) ranked against
      the same negatives. The two sources are disjoint and add up to all positives.
    - per_citing_paper: one row per citing paper; score = mean score of its events in the list, positive when
      the citing paper is labelled.
    - per_cited_paper: one row per cited paper; score = mean score of the events citing it in the list, positive
      when the cited paper is labelled.

    Each value is a ranking_metrics dict (n and n_pos count events in the first two views, papers in the last
    two). A view without positives has NaN ratios, as in ranking_metrics. The mean is PROVISIONAL as the
    per-paper aggregate (it is what the pilot's paper-level check used); the choice is reporting only.
    """
    s, y = _to_arrays(scores, labels)
    cols = {"via_src": via_src, "via_dst": via_dst, "src_ids": src_ids, "dst_ids": dst_ids}
    arr = {k: np.asarray(v).ravel() for k, v in cols.items()}
    for name, a in arr.items():
        if a.shape != s.shape:
            raise ValueError(f"length mismatch: {s.size} scores vs {a.size} {name}")
    pos = y == 1
    citing = pos & (arr["via_src"].astype(np.int64) == 1)
    cited = pos & (arr["via_dst"].astype(np.int64) == 1)
    if (pos & ~citing & ~cited).any():
        raise ValueError(f"{int((pos & ~citing & ~cited).sum())} positive events have neither via_src nor via_dst")
    cited_only = cited & ~citing
    out = {}
    for name, mine in (("citing_side", citing), ("cited_side", cited_only)):
        rows = ~pos | mine
        out[name] = ranking_metrics(s[rows], mine[rows].astype(np.int64), ks=ks)
    out["per_citing_paper"] = _paper_metrics(s, citing, arr["src_ids"], ks)
    out["per_cited_paper"] = _paper_metrics(s, cited, arr["dst_ids"], ks)
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


"""Classical-ML baseline tier: XGBoost on leak-checked per-event features (task S4.4, decision L17).

Feature list (TIER2_FEATURES), all computed for event e = (src cites dst at t) from state with ts <= t, the
scored row excluded:

=============== ==========================================================================================
feature         definition and leak argument
=============== ==========================================================================================
self_citation,  the four tier-1 features; definitions and leak arguments in ``tier1_stats``.
venue_gini,
burst_z,
ref_overlap
in_deg_src      #events into src with ts <= t (minus e if src == dst). Counts only rows at or before t.
out_deg_src     #events out of src with ts <= t, minus e: the other references of src visible at t.
in_deg_dst      #events into dst with ts <= t, minus e: citations dst has received by t.
out_deg_dst     #events out of dst with ts <= t (minus e if src == dst): dst's references visible at t.
emb_cosine      cosine similarity of the endpoint text embeddings (SPECTER2 in real runs). Text is fixed at
                publication, and each paper is published by t, so no later information enters. NaN when an
                endpoint has no embedding (XGBoost routes NaN through its learned default branch).
=============== ==========================================================================================

Note for the comparison table: the unweighted Forman screen value of the curvature module is
2 - in_deg_src - out_deg_dst, an exact linear function of two tier-2 features, so this baseline can represent the
Forman screen itself. That is a property of the plan's feature list, not a leak.

Labels: the model is fitted on TRAIN-split rows only (labels of val/test rows are never read); it then scores the
requested splits (default: test). Labels are outcome data (retraction status), used only as training targets.
The rows of TIER2_FEATURES for all splits come from one pass over the full event stream, as in tier 1.
Determinism: fixed seed, ``subsample = colsample = 1`` by default, and ``nthread = 1`` in tests; the hist tree
method is deterministic on CPU for a fixed thread count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from citecurv.eval.baselines import tier1_stats as t1

DEGREE_FEATURES = ("in_deg_src", "out_deg_src", "in_deg_dst", "out_deg_dst")
TIER2_FEATURES = (*t1.FEATURES, *DEGREE_FEATURES, "emb_cosine")

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1.0,
    "lambda": 1.0,
}


def degree_features(events: pd.DataFrame) -> pd.DataFrame:
    """In/out event degrees of both endpoints at t, scored row excluded (DataFrame, input row order).

    Leak argument: each count is #rows with the node as endpoint and ts <= t (binary search on per-node sorted
    timestamps), so later rows are never counted; e itself is subtracted wherever it would be counted.
    """
    p = t1._prepare(events)
    in_src = t1.count_le(p.dst, p.t, p.src, p.t)
    out_src = t1.count_le(p.src, p.t, p.src, p.t) - 1
    in_dst = t1.count_le(p.dst, p.t, p.dst, p.t) - 1
    out_dst = t1.count_le(p.src, p.t, p.dst, p.t)
    loop = (p.src == p.dst).astype(np.int64)
    return pd.DataFrame({
        "in_deg_src": in_src - loop,
        "out_deg_src": out_src,
        "in_deg_dst": in_dst,
        "out_deg_dst": out_dst - loop,
    })


def embedding_cosine(events: pd.DataFrame, embeddings) -> np.ndarray:
    """Cosine similarity of the src and dst embedding rows; NaN when either is missing or has zero norm.

    embeddings: tuple (ids (N,), vectors (N, d)) as in CONTRACTS.md, or None (all NaN).
    Leak argument: see the module table (text embeddings are fixed at publication).
    """
    n = len(events)
    out = np.full(n, np.nan)
    if embeddings is None or n == 0:
        return out
    ids, vectors = embeddings
    vectors = np.asarray(vectors, dtype=np.float64)
    row = {k: i for i, k in enumerate(np.asarray(ids).tolist())}
    si = np.fromiter((row.get(s, -1) for s in events["src_id"]), np.int64, count=n)
    di = np.fromiter((row.get(d, -1) for d in events["dst_id"]), np.int64, count=n)
    ok = (si >= 0) & (di >= 0)
    a = vectors[si[ok]]
    b = vectors[di[ok]]
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.full(len(a), np.nan)
    nz = norm > 0
    cos[nz] = np.einsum("ij,ij->i", a[nz], b[nz]) / norm[nz]
    out[ok] = cos
    return out


def build_features(
    events: pd.DataFrame,
    works: pd.DataFrame | None = None,
    authorships: pd.DataFrame | None = None,
    embeddings=None,
    *,
    tier1: pd.DataFrame | None = None,
    window_days: int = 365,
    min_venue_citers: int = 3,
    max_ref_candidates: int = 20,
) -> pd.DataFrame:
    """DataFrame(event_id, *TIER2_FEATURES) in the input row order of ``events``.

    tier1: precomputed ``tier1_stats.compute_features`` output for the same events (reused, not recomputed).
    """
    if tier1 is None:
        tier1 = t1.compute_features(
            events, works, authorships, window_days=window_days, min_venue_citers=min_venue_citers,
            max_ref_candidates=max_ref_candidates,
        )
    else:
        tier1 = tier1.set_index("event_id").loc[events["event_id"].to_numpy()].reset_index()
    out = tier1[["event_id", *t1.FEATURES]].reset_index(drop=True)
    deg = degree_features(events)
    for c in DEGREE_FEATURES:
        out[c] = deg[c].to_numpy()
    out["emb_cosine"] = embedding_cosine(events, embeddings)
    return out


def _split_series(split) -> pd.Series:
    if isinstance(split, pd.DataFrame):
        return split.set_index("event_id")["split"]
    return split


def fit_predict(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split,
    *,
    seed: int = 0,
    nthread: int = 1,
    num_boost_round: int = 300,
    params: dict | None = None,
    predict_on=("test",),
    feature_columns=TIER2_FEATURES,
):
    """Fit XGBoost on TRAIN rows, score ``predict_on`` rows; returns (DataFrame(event_id, score), Booster).

    features: DataFrame(event_id, *feature_columns); labels: DataFrame(event_id, label 0/1);
    split: DataFrame(event_id, split) or Series event_id -> 'train'/'val'/'test'.
    Scores are predicted probabilities (higher = more anomalous), rows in ``features`` order.
    scale_pos_weight defaults to #neg / #pos of the train rows (override through ``params``).
    Raises ValueError when the train split has no positive or no negative labels.
    """
    import xgboost as xgb

    sp = _split_series(split)
    lab = labels.set_index("event_id")["label"]
    part = sp.reindex(features["event_id"]).to_numpy()
    train_mask = part == "train"
    train_ids = features["event_id"].to_numpy()[train_mask]
    y = lab.reindex(train_ids)
    has_label = y.notna().to_numpy()
    X_train = features.loc[train_mask, list(feature_columns)].to_numpy(np.float32)[has_label]
    y = y.to_numpy()[has_label].astype(np.float32)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError(
            f"tier-2 XGBoost: the train split has no positive labels ({len(y)} labelled train rows); a supervised "
            "baseline cannot be fitted. Check the label join and the cutoff choice (S4.1)."
        )
    if n_pos == len(y):
        raise ValueError(f"tier-2 XGBoost: the train split has no negative labels ({len(y)} labelled train rows).")
    cfg = {**DEFAULT_PARAMS, "seed": int(seed), "nthread": int(nthread),
           "scale_pos_weight": (len(y) - n_pos) / n_pos}
    cfg.update(params or {})
    dtrain = xgb.DMatrix(X_train, label=y, feature_names=list(feature_columns), missing=np.nan)
    booster = xgb.train(cfg, dtrain, num_boost_round=int(num_boost_round))
    pred_mask = np.isin(part.astype(object), list(predict_on))
    X_pred = features.loc[pred_mask, list(feature_columns)].to_numpy(np.float32)
    dpred = xgb.DMatrix(X_pred, feature_names=list(feature_columns), missing=np.nan)
    scores = booster.predict(dpred).astype(np.float64)
    return pd.DataFrame({"event_id": features.loc[pred_mask, "event_id"].to_numpy(), "score": scores}), booster

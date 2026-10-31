"""Four-component anomaly score with a named breakdown (FR7, NFR7, NFR10).

Definitions, leak arguments and failure modes: docs/score_components.md (task S3.1,
PROVISIONAL until Maor's review). This module implements exactly that note.

Components (raw values; higher = more anomalous; the combiner z-normalises them):

- s_embed   mode "tgn": -log p_link, p_link from the TGN decoder computed before
            the event's batch updates memory. Mode "static_cosine" (rehearsal
            fallback): (mean - c) / sd, c = cosine of the endpoint embeddings,
            mean/sd over earlier events of the reference cell.
- s_geom    |z| of the event's Forman screen value against the reference cell;
            an escalated event with ORC takes max(|z_forman|, |z_orc|), the ORC
            reference being earlier escalated events of the cell.
- s_temp    z of x = log((n_recent + 1) / (n_before + 1)) + log(T_before / W)
            against the reference cell: the cited paper's citation rate in the
            trailing W days relative to its own earlier rate.
- s_curvdyn |asinh(observed curvature) - predicted| / rmse_train, a ridge
            regressor on PCA-reduced endpoint representations fitted on TRAIN
            rows only.
- cold_start  dst has fewer than k events with ts < t.

Reference cell = (field of src, calendar year of t), expanding window over
strictly earlier events (ts < t). A level is usable with >= max(min_cell, 2)
earlier valid rows and sd > 0; otherwise fall back to (field), then (global).

Leak summary (details per function and in the note): every per-event input uses
graph state with ts <= t minus the event's own row (contract same-timestamp
policy); every reference statistic uses ts < t; the s_curvdyn regressor is fitted
on train rows only, and the train period precedes val/test. No labels enter any
component.

Two code paths compute the same numbers: score_events (vectorised, O(E log E),
for the pipeline) and build_context + score_event (one event at a time with
binary searches, for the NFR2 latency measurement and the investigator view).
tests/test_components.py checks that they agree.
"""

from __future__ import annotations

import math
import warnings
import zlib
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

COMPONENTS = ("s_embed", "s_geom", "s_temp", "s_curvdyn")
COMPONENT_COLUMNS = ["event_id", "src_id", "dst_id", "t", *COMPONENTS, "cold_start"]
VARIANTS = ("full", "no_curvature", "forman_only", "ollivier_only", "no_temporal")
EMBED_MODES = ("auto", "tgn", "static_cosine")
LEVELS = ("field_year", "field", "global")  # index stored in *_level columns; -1 = none
UNKNOWN_FIELD = "unknown"

_SD_FLOOR = 1e-12
_RMSE_FLOOR = 1e-12
_PCA_MAX_ROWS = 20_000
_CHUNK = 16_384
_REF_STATS = ("embed", "geom", "orc", "temp")  # trace-column prefixes of the four references


@dataclass
class ScoreBreakdown:
    """One event's named breakdown (FR7). `total` is filled by the combiner;
    `inputs` holds the ids and every value used (NFR10)."""

    s_embed: float
    s_geom: float
    s_temp: float
    s_curvdyn: float
    total: float | None = None
    cold_start: bool = False
    inputs: dict = dc_field(default_factory=dict)


@dataclass(frozen=True)
class CurvDynModel:
    """Frozen s_curvdyn regressor: y = asinh(curvature) ~ ridge(features).

    features(a, b) = [P(a), P(b), P(a) * P(b), cos(a, b)] with P(v) = (v - center) @ basis,
    standardised by feat_mean / feat_scale. All arrays come from train rows only.
    """

    source: str          # "tgn" (per-event z_src/z_dst) or "static" (node embeddings)
    target: str          # "forman" or "orc"
    center: np.ndarray   # (d,)
    basis: np.ndarray    # (d, r)
    feat_mean: np.ndarray
    feat_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    rmse: float
    n_fit: int

    def features(self, a, b):
        return _features(a, b, self.center, self.basis)

    def predict(self, a, b):
        z = (self.features(a, b) - self.feat_mean) / self.feat_scale
        return z @ self.coef + self.intercept


# ----------------------------------------------------------------- input adapters


def _first_concept(c):
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return UNKNOWN_FIELD
    if isinstance(c, str):
        for sep in (";", "|", ","):
            c = c.split(sep)[0]
        return c.strip() or UNKNOWN_FIELD
    try:
        items = list(c)
    except TypeError:
        return UNKNOWN_FIELD
    if not items or items[0] is None:
        return UNKNOWN_FIELD
    return str(items[0]).strip() or UNKNOWN_FIELD


def _field_of(works, src_ids, field_col):
    """Field of each src paper: works[field_col] when present, else the first
    entry of works.concepts; missing or empty -> "unknown"."""
    if works is None or len(works) == 0:
        return np.full(len(src_ids), UNKNOWN_FIELD, dtype=object)
    w = works.drop_duplicates("work_id")
    if field_col in w.columns:
        vals = [v.strip() if isinstance(v, str) and v.strip() else UNKNOWN_FIELD for v in w[field_col]]
    elif "concepts" in w.columns:
        vals = [_first_concept(c) for c in w["concepts"]]
    else:
        vals = [UNKNOWN_FIELD] * len(w)
    lookup = pd.Series(vals, index=w["work_id"].astype(str).to_numpy(), dtype=object)
    return lookup.reindex(src_ids).fillna(UNKNOWN_FIELD).to_numpy(dtype=object)


class _EmbTable:
    """Static node embeddings: ids -> matrix rows. Accepts (ids, vectors), a mapping
    with 'ids' and 'vectors' (an npz), or (dict id -> row, matrix)."""

    def __init__(self, embeddings):
        if isinstance(embeddings, tuple) and len(embeddings) == 2 and isinstance(embeddings[0], dict):
            keys = list(embeddings[0].keys())
            self.row_of = np.asarray(list(embeddings[0].values()), dtype=np.int64)
            self.matrix = np.asarray(embeddings[1])
        else:
            if isinstance(embeddings, tuple):
                ids, vectors = embeddings
            else:
                ids, vectors = embeddings["ids"], embeddings["vectors"]
            keys = [str(i) for i in np.asarray(ids)]
            self.matrix = np.asarray(vectors)
            self.row_of = np.arange(len(keys), dtype=np.int64)
        self.index = pd.Index([str(k) for k in keys])
        if not self.index.is_unique:
            raise ValueError("embedding ids must be unique")

    def rows(self, ids):
        pos = self.index.get_indexer(ids)
        return np.where(pos >= 0, self.row_of[np.maximum(pos, 0)], -1)


class _Endpoints:
    """Endpoint representations for s_curvdyn: per-event TGN z_src/z_dst when
    given, else static node embeddings of src and dst."""

    def __init__(self, ev, embeddings=None, tgn_embeddings=None):
        if tgn_embeddings is not None:
            self.source = "tgn"
            ids = np.asarray(tgn_embeddings["event_id"], dtype=np.int64)
            self._zs = np.asarray(tgn_embeddings["z_src"])
            self._zd = np.asarray(tgn_embeddings["z_dst"])
            r = pd.Index(ids).get_indexer(ev["event_id"].to_numpy())
            self._ra = self._rb = r
            self.valid = r >= 0
        else:
            self.source = "static"
            tab = embeddings if isinstance(embeddings, _EmbTable) else _EmbTable(embeddings)
            self._zs = self._zd = tab.matrix
            self._ra = tab.rows(ev["src_id"].to_numpy())
            self._rb = tab.rows(ev["dst_id"].to_numpy())
            self.valid = (self._ra >= 0) & (self._rb >= 0)

    def vectors(self, rows):
        return (np.asarray(self._zs[self._ra[rows]], dtype=np.float64),
                np.asarray(self._zd[self._rb[rows]], dtype=np.float64))


def _canonical_events(events, works, curvature, field_col):
    """Events in canonical order (t, event_id) with year, field and curvature
    columns, plus `_pos` = row position in the input frame."""
    eid = events["event_id"].to_numpy(dtype=np.int64)
    t = events["t"].to_numpy(dtype=np.int64)
    if len(np.unique(eid)) != len(eid):
        raise ValueError("event_id must be unique")
    order = np.lexsort((eid, t))
    ev = pd.DataFrame({
        "event_id": eid[order],
        "src_id": events["src_id"].astype(str).to_numpy(dtype=object)[order],
        "dst_id": events["dst_id"].astype(str).to_numpy(dtype=object)[order],
        "t": t[order],
    })
    ev["year"] = ev["t"].to_numpy().astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970
    ev["field"] = _field_of(works, ev["src_id"].to_numpy(), field_col)
    ev["_pos"] = order

    n = len(ev)
    forman, orc, esc = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool)
    if curvature is not None and len(curvature):
        cv = curvature
        if not cv["event_id"].is_unique:
            raise ValueError("curvature has duplicate event_id rows")
        idx = pd.Index(cv["event_id"].to_numpy(dtype=np.int64)).get_indexer(ev["event_id"].to_numpy())
        has = idx >= 0
        take = idx[has]
        for col in ("src_id", "dst_id", "t"):
            if col in cv.columns:
                theirs = cv[col].to_numpy()[take]
                ours = ev[col].to_numpy()[has]
                same = theirs.astype(np.int64) == ours if col == "t" else theirs.astype(str) == ours.astype(str)
                if not np.all(same):
                    raise ValueError(f"curvature rows do not match events on column {col!r}")
        forman[has] = cv["forman"].to_numpy(dtype=float)[take]
        if "orc" in cv.columns:
            orc[has] = cv["orc"].to_numpy(dtype=float)[take]
        if "escalated" in cv.columns:
            esc[has] = cv["escalated"].fillna(False).to_numpy(dtype=bool)[take]
        else:
            esc = np.isfinite(orc)
    ev["forman"], ev["orc"], ev["escalated"] = forman, orc, esc
    return ev


# ----------------------------------------------------------------- raw statistics


def _temporal_raw(dst, t, window):
    """Per event, over events citing the same dst (vectorised, O(E log E)):
    n_prior = #(ts < t)                      (cold-start count)
    n_recent = #(t - W < ts <= t) - 1        (same-day siblings count, self does not)
    n_before = #(ts <= t - W), T_before = max(t - W - first ts, W) if n_before else W
    x = log((n_recent + 1) / (n_before + 1)) + log(T_before / W)
    Leak: only events of dst with ts <= t enter, and the event's own row is removed."""
    codes, uniq = pd.factorize(dst)
    codes = codes.astype(np.int64)
    order = np.lexsort((t, codes))
    sc, st = codes[order], t[order]
    starts = np.searchsorted(sc, np.arange(len(uniq)), "left")
    tmin = int(t.min())
    off = window + 1
    span = int(t.max()) - tmin + window + 2
    comp = sc * span + (st - tmin + off)
    base = codes * span
    g0 = starts[codes]
    n_le = np.searchsorted(comp, base + (t - tmin + off), "right") - g0
    n_lt = np.searchsorted(comp, base + (t - tmin + off), "left") - g0
    n_before = np.searchsorted(comp, base + (t - window - tmin + off), "right") - g0
    n_recent = n_le - 1 - n_before
    t_first = st[g0]
    exposure = np.where(n_before > 0, np.maximum(t - window - t_first, window), window).astype(float)
    x = np.log((n_recent + 1) / (n_before + 1)) + np.log(exposure / window)
    return {"n_prior_dst": n_lt, "n_recent": n_recent, "n_before": n_before,
            "exposure_days": exposure, "temp_x": x}


def _cosine_rows(tab, src, dst):
    """Cosine of static endpoint embeddings, float64, chunked; NaN if either is
    missing or has zero norm. Leak: text embeddings of src (published at t) and
    dst (published before t); no graph state."""
    ra, rb = tab.rows(src), tab.rows(dst)
    out = np.full(len(src), np.nan)
    ok = np.flatnonzero((ra >= 0) & (rb >= 0))
    for s in range(0, len(ok), _CHUNK):
        rows = ok[s:s + _CHUNK]
        a = np.asarray(tab.matrix[ra[rows]], dtype=np.float64)
        b = np.asarray(tab.matrix[rb[rows]], dtype=np.float64)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows] = np.where(den > 0, np.einsum("ij,ij->i", a, b) / den, np.nan)
    return out


def _cosine_one(tab, src, dst):
    ra, rb = tab.rows([src])[0], tab.rows([dst])[0]
    if ra < 0 or rb < 0:
        return float("nan")
    a = np.asarray(tab.matrix[ra], dtype=np.float64)
    b = np.asarray(tab.matrix[rb], dtype=np.float64)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def _neg_log_p(p, eps):
    return -np.log(np.clip(p, eps, 1.0 - eps))


# ----------------------------------------------------------------- reference cells


def _level_frames(ev):
    return [ev[["field", "year"]].reset_index(drop=True),
            ev[["field"]].reset_index(drop=True),
            pd.DataFrame({"_g": np.zeros(len(ev), dtype=np.int64)})]


def _expanding_ref(key_frame, t, x, valid):
    """(n, mean, sd) per row over rows with the same key, ts < t and valid.

    Rows must be in canonical time order. Exclusive per-key cumulative sums are
    read at the first row of each (key, t) block, so same-timestamp rows never
    see each other: the reference is strictly earlier (ts < t). Sums are shifted
    by the key's first valid value (exact algebra, better numerics). O(E) after
    the sort, done by pandas group-wise cumsum."""
    kcols = list(key_frame.columns)
    df = key_frame.copy()
    df["_x"] = np.where(valid, x, np.nan)
    K = df.groupby(kcols, sort=False)["_x"].transform("first").to_numpy(dtype=float)
    K = np.where(np.isnan(K), 0.0, K)
    d = np.where(valid, x - K, 0.0)
    parts = np.column_stack([valid.astype(float), d, d * d])
    df[["_c0", "_c1", "_c2"]] = parts
    excl = df.groupby(kcols, sort=False)[["_c0", "_c1", "_c2"]].cumsum().to_numpy() - parts
    blk = key_frame.copy()
    blk[["_e0", "_e1", "_e2"]] = excl
    blk["_t"] = t
    first = blk.groupby(kcols + ["_t"], sort=False)[["_e0", "_e1", "_e2"]].transform("first").to_numpy()
    n, s1, s2 = first[:, 0], first[:, 1], first[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = K + s1 / n
        var = (s2 - s1 * s1 / n) / (n - 1)
    sd = np.sqrt(np.clip(var, 0.0, None))
    return n, mean, sd


def _cell_z(frames, t, x, valid, min_cell):
    """z against the first usable level (field-year, field, global).
    Returns dict with z, level, ref_n, ref_mean, ref_sd (NaN / -1 where unusable)."""
    n_rows = len(t)
    out = {"z": np.full(n_rows, np.nan), "level": np.full(n_rows, -1, dtype=np.int64),
           "ref_n": np.full(n_rows, np.nan), "ref_mean": np.full(n_rows, np.nan),
           "ref_sd": np.full(n_rows, np.nan)}
    need = max(int(min_cell), 2)
    valid = valid & np.isfinite(x)
    for li, kf in enumerate(frames):
        todo = valid & (out["level"] < 0)
        if not todo.any():
            break
        n, mean, sd = _expanding_ref(kf, t, x, valid)
        use = todo & (n >= need) & (sd > _SD_FLOOR)
        out["z"][use] = (x[use] - mean[use]) / sd[use]
        out["level"][use] = li
        out["ref_n"][use], out["ref_mean"][use], out["ref_sd"][use] = n[use], mean[use], sd[use]
    return out


@dataclass
class _Prefix:
    times: np.ndarray
    c1: np.ndarray
    c2: np.ndarray
    K: float


def _prefix_tables(frames, t, x, valid):
    """Single-event path: per level, key -> prefix sums over the key's valid rows
    in time order. A query for time t reads the prefix of rows with ts < t."""
    valid = valid & np.isfinite(x)
    vpos = np.flatnonzero(valid)
    tables = []
    for kf in frames:
        sub = kf.iloc[vpos]
        table = {}
        for key, p in sub.groupby(list(sub.columns), sort=False).indices.items():
            rows = vpos[np.sort(p)]
            xs = x[rows]
            K = float(xs[0])
            dlt = xs - K
            table[key if isinstance(key, tuple) else (key,)] = _Prefix(
                t[rows], np.concatenate([[0.0], np.cumsum(dlt)]),
                np.concatenate([[0.0], np.cumsum(dlt * dlt)]), K)
        tables.append(table)
    return tables


def _scalar_z(tables, keys, t, x, min_cell):
    nan = float("nan")
    if tables is None or not np.isfinite(x):
        return {"z": nan, "level": -1, "ref_n": nan, "ref_mean": nan, "ref_sd": nan}
    need = max(int(min_cell), 2)
    for li, key in enumerate(keys):
        p = tables[li].get(key)
        if p is None:
            continue
        n = int(np.searchsorted(p.times, t, "left"))
        if n < need:
            continue
        s1, s2 = p.c1[n], p.c2[n]
        mean = p.K + s1 / n
        sd = math.sqrt(max((s2 - s1 * s1 / n) / (n - 1), 0.0))
        if sd <= _SD_FLOOR:
            continue
        return {"z": (x - mean) / sd, "level": li, "ref_n": float(n), "ref_mean": mean, "ref_sd": sd}
    return {"z": nan, "level": -1, "ref_n": nan, "ref_mean": nan, "ref_sd": nan}


# ----------------------------------------------------------------- s_curvdyn regressor


def _features(a, b, center, basis):
    pa = (a - center) @ basis
    pb = (b - center) @ basis
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(den > 0, np.einsum("ij,ij->i", a, b) / den, 0.0)
    return np.hstack([pa, pb, pa * pb, cos[:, None]])


def _curv_target(ev, target):
    """y = asinh(curvature) and its validity. target "orc" reads ORC on escalated
    rows only; "forman" reads the screen value."""
    if target == "orc":
        raw = np.where(ev["escalated"].to_numpy(), ev["orc"].to_numpy(), np.nan)
    else:
        raw = ev["forman"].to_numpy()
    y = np.arcsinh(raw)
    return y, np.isfinite(y)


def _fit_curvdyn(ev, train, endpoints, target, pca_dim, ridge_alpha, max_fit_rows, seed):
    """Ridge fit on TRAIN rows only (PCA basis, standardisation, coefficients, rmse).
    Leak: every array is computed from rows where train is True; val/test rows
    never enter, so changing them cannot change the model (tested)."""
    y, ok = _curv_target(ev, target)
    rows = np.flatnonzero(np.asarray(train, dtype=bool) & ok & endpoints.valid)
    if len(rows) == 0:
        return None
    rng = np.random.default_rng(seed)
    if len(rows) > max_fit_rows:
        rows = np.sort(rng.choice(rows, size=int(max_fit_rows), replace=False))
    pca_rows = rows
    if len(rows) > _PCA_MAX_ROWS // 2:
        pca_rows = np.sort(rng.choice(rows, size=_PCA_MAX_ROWS // 2, replace=False))
    a, b = endpoints.vectors(pca_rows)
    X = np.vstack([a, b])
    center = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - center, full_matrices=False)
    r = min(int(pca_dim), vt.shape[0])
    basis = vt[:r].T.copy()
    signs = np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(r)])
    basis *= np.where(signs == 0, 1.0, signs)

    def chunks():
        for s in range(0, len(rows), _CHUNK):
            rr = rows[s:s + _CHUNK]
            yield _features(*endpoints.vectors(rr), center, basis), y[rr]

    s1 = s2 = 0.0
    for F, _ in chunks():
        s1 = s1 + F.sum(axis=0)
        s2 = s2 + (F * F).sum(axis=0)
    m = len(rows)
    feat_mean = s1 / m
    feat_sd = np.sqrt(np.clip(s2 / m - feat_mean ** 2, 0.0, None))
    feat_scale = np.where(feat_sd > _SD_FLOOR, feat_sd, 1.0)
    ybar = float(y[rows].mean())
    p = len(feat_mean)
    ztz, zty = np.zeros((p, p)), np.zeros(p)
    for F, yy in chunks():
        Z = (F - feat_mean) / feat_scale
        ztz += Z.T @ Z
        zty += Z.T @ (yy - ybar)
    coef = np.linalg.solve(ztz + ridge_alpha * np.eye(p), zty)
    sse = 0.0
    for F, yy in chunks():
        res = yy - (((F - feat_mean) / feat_scale) @ coef + ybar)
        sse += float(res @ res)
    rmse = max(math.sqrt(sse / m), _RMSE_FLOOR)
    return CurvDynModel(endpoints.source, target, center, basis, feat_mean, feat_scale, coef,
                        ybar, rmse, m)


# ----------------------------------------------------------------- shared preparation


@dataclass
class _Prepared:
    ev: pd.DataFrame
    variant: str
    embed_mode: str
    params: dict
    p_link: np.ndarray
    emb: _EmbTable | None
    cosine: np.ndarray
    temporal: dict
    endpoints: _Endpoints | None
    model: CurvDynModel | None
    disabled: tuple
    notes: list


def _prepare(events, works, curvature, *, embeddings, tgn_scores, tgn_embeddings, train_mask,
             mode, variant, k, window_days, min_cell, curvdyn_model, pca_dim, ridge_alpha,
             max_fit_rows, p_link_eps, seed, field_col):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if mode not in EMBED_MODES:
        raise ValueError(f"mode must be one of {EMBED_MODES}, got {mode!r}")
    if int(window_days) < 1 or int(k) < 0:
        raise ValueError("window_days must be >= 1 and k >= 0")
    if mode == "tgn" and tgn_scores is None:
        raise ValueError("mode='tgn' needs tgn_scores")
    if mode == "static_cosine" and embeddings is None:
        raise ValueError("mode='static_cosine' needs embeddings")
    notes = []
    ev = _canonical_events(events, works, curvature, field_col)
    n = len(ev)
    train = None
    if train_mask is not None:
        tm = np.asarray(train_mask, dtype=bool)
        if tm.shape != (n,):
            raise ValueError("train_mask must be a boolean array aligned with the events rows")
        train = tm[ev["_pos"].to_numpy()]

    resolved = mode
    if mode == "auto":
        resolved = "tgn" if tgn_scores is not None else ("static_cosine" if embeddings is not None else "none")
    if resolved == "none":
        notes.append("s_embed is NaN: neither tgn_scores nor embeddings were given")

    p_link = np.full(n, np.nan)
    if tgn_scores is not None and len(tgn_scores):
        s = pd.Series(tgn_scores["p_link"].to_numpy(dtype=float),
                      index=tgn_scores["event_id"].to_numpy(dtype=np.int64))
        p_link = s.reindex(ev["event_id"].to_numpy()).to_numpy(dtype=float)

    emb = _EmbTable(embeddings) if embeddings is not None else None
    cosine = (_cosine_rows(emb, ev["src_id"].to_numpy(), ev["dst_id"].to_numpy())
              if emb is not None else np.full(n, np.nan))
    temporal = (_temporal_raw(ev["dst_id"].to_numpy(), ev["t"].to_numpy(), int(window_days)) if n else
                {key: np.zeros(0) for key in ("n_prior_dst", "n_recent", "n_before", "exposure_days", "temp_x")})

    disabled = ("s_geom", "s_curvdyn") if variant == "no_curvature" else ()
    endpoints, model = None, None
    if "s_curvdyn" not in disabled and n and (tgn_embeddings is not None or emb is not None):
        endpoints = _Endpoints(ev, emb, tgn_embeddings)
        target = "orc" if variant == "ollivier_only" else "forman"
        if curvdyn_model is not None:
            if (curvdyn_model.source, curvdyn_model.target) != (endpoints.source, target):
                raise ValueError(
                    f"curvdyn_model is ({curvdyn_model.source}, {curvdyn_model.target}); "
                    f"inputs and variant need ({endpoints.source}, {target})")
            model = curvdyn_model
        elif train is not None:
            model = _fit_curvdyn(ev, train, endpoints, target, pca_dim, ridge_alpha, max_fit_rows, seed)
            if model is None:
                notes.append("s_curvdyn is NaN: no train row has both a target and endpoint vectors")
        elif curvature is not None:
            msg = "s_curvdyn is NaN: no train_mask and no curvdyn_model (the regressor is fitted on train rows only)"
            warnings.warn(msg, UserWarning, stacklevel=3)
            notes.append(msg)
    params = {"k": int(k), "window_days": int(window_days), "min_cell": int(min_cell),
              "pca_dim": int(pca_dim), "ridge_alpha": float(ridge_alpha),
              "max_fit_rows": int(max_fit_rows), "p_link_eps": float(p_link_eps), "seed": int(seed),
              "field_col": field_col}
    return _Prepared(ev, variant, resolved, params, p_link, emb, cosine, temporal, endpoints, model,
                     disabled, notes)


def _uses(variant):
    """(reads forman, reads orc) for the variant (ablation semantics, note section 7)."""
    return {"full": (True, True), "no_temporal": (True, True), "forman_only": (True, False),
            "ollivier_only": (False, True), "no_curvature": (False, False)}[variant]


# ----------------------------------------------------------------- vectorised path


def score_events(events, works, curvature, *, embeddings=None, tgn_scores=None, tgn_embeddings=None,
                 train_mask=None, mode="auto", variant="full", k=3, window_days=90, min_cell=30,
                 curvdyn_model=None, pca_dim=32, ridge_alpha=1.0, max_fit_rows=500_000,
                 p_link_eps=1e-7, seed=0, field_col="field", with_inputs=False) -> pd.DataFrame:
    """Raw components for every event (the `components` contract frame).

    events: event_id, src_id, dst_id, t. works: work_id + field (or concepts).
    curvature: the curvature cache frame (event_id, forman, orc, escalated, ...)
    or None. embeddings: static node embeddings ((ids, vectors) or npz mapping).
    tgn_scores: event_id, p_link. tgn_embeddings: mapping event_id, z_src, z_dst.
    train_mask: boolean array aligned positionally with `events` rows; the
    s_curvdyn regressor is fitted on these rows only (or pass curvdyn_model).
    mode: s_embed source ("auto" = tgn if tgn_scores else static_cosine).
    variant: ablation cell (docs/score_components.md section 7); disabled
    components are all-NaN and listed in attrs["disabled_components"].

    Returns rows in canonical order (t, event_id) with COMPONENT_COLUMNS; with
    with_inputs=True also the in_* trace columns (NFR10). Cold-start rows carry
    component values too (needed by S2.4); the combiner excludes them.

    Leak argument: see the module docstring and the note; each input is read at
    ts <= t minus the event's own row, each reference at ts < t, and the
    regressor on train rows only. Cost: O(E log E) time, O(E) memory.
    """
    P = _prepare(events, works, curvature, embeddings=embeddings, tgn_scores=tgn_scores,
                 tgn_embeddings=tgn_embeddings, train_mask=train_mask, mode=mode, variant=variant,
                 k=k, window_days=window_days, min_cell=min_cell, curvdyn_model=curvdyn_model,
                 pca_dim=pca_dim, ridge_alpha=ridge_alpha, max_fit_rows=max_fit_rows,
                 p_link_eps=p_link_eps, seed=seed, field_col=field_col)
    ev, n = P.ev, len(P.ev)
    t = ev["t"].to_numpy()
    frames = _level_frames(ev)
    nan = np.full(n, np.nan)
    trace = {}

    # s_embed
    ref_embed = None
    if P.embed_mode == "tgn":
        s_embed = _neg_log_p(P.p_link, P.params["p_link_eps"])
    elif P.embed_mode == "static_cosine":
        ref_embed = _cell_z(frames, t, P.cosine, np.isfinite(P.cosine), min_cell)
        s_embed = -ref_embed["z"]
    else:
        s_embed = nan.copy()

    # s_temp
    tx = P.temporal["temp_x"]
    ref_temp = _cell_z(frames, t, tx, np.ones(n, dtype=bool), min_cell)
    s_temp = ref_temp["z"]

    # s_geom
    reads_f, reads_o = _uses(P.variant)
    forman = ev["forman"].to_numpy() if reads_f else nan.copy()
    orc = ev["orc"].to_numpy() if reads_o else nan.copy()
    esc = ev["escalated"].to_numpy()
    ref_f = _cell_z(frames, t, forman, np.isfinite(forman), min_cell) if reads_f else None
    ref_o = _cell_z(frames, t, orc, esc & np.isfinite(orc), min_cell) if reads_o else None
    zf = ref_f["z"] if ref_f else nan
    zo = np.where(esc, ref_o["z"], np.nan) if ref_o else nan
    s_geom = np.fmax(np.abs(zf), np.abs(zo))

    # s_curvdyn
    s_curv, obs, pred = nan.copy(), nan.copy(), nan.copy()
    if P.model is not None:
        y, ok = _curv_target(ev, P.model.target)
        obs = np.where(ok, y, np.nan)
        rows = np.flatnonzero(ok & P.endpoints.valid)
        for s in range(0, len(rows), _CHUNK):
            rr = rows[s:s + _CHUNK]
            pred[rr] = P.model.predict(*P.endpoints.vectors(rr))
        s_curv = np.abs(obs - pred) / P.model.rmse

    out = pd.DataFrame({
        "event_id": ev["event_id"].to_numpy(dtype=np.int64),
        "src_id": ev["src_id"].to_numpy(),
        "dst_id": ev["dst_id"].to_numpy(),
        "t": t.astype(np.int64),
        "s_embed": np.asarray(s_embed, dtype=np.float64),
        "s_geom": np.asarray(s_geom, dtype=np.float64),
        "s_temp": np.asarray(s_temp, dtype=np.float64),
        "s_curvdyn": np.asarray(s_curv, dtype=np.float64),
        "cold_start": P.temporal["n_prior_dst"] < P.params["k"],
    })
    for c in P.disabled:
        out[c] = np.nan

    if with_inputs:
        trace["field"] = ev["field"].to_numpy()
        trace["year"] = ev["year"].to_numpy()
        trace["p_link"] = P.p_link
        trace["cosine"] = P.cosine
        trace["forman"] = forman
        trace["orc"] = orc
        trace["escalated"] = esc
        trace["z_forman"] = zf
        trace["z_orc"] = zo
        for key in ("temp_x", "n_recent", "n_before", "exposure_days", "n_prior_dst"):
            trace[key] = P.temporal[key]
        for prefix, ref in zip(_REF_STATS, (ref_embed, ref_f, ref_o, ref_temp)):
            trace[f"{prefix}_level"] = ref["level"] if ref else np.full(n, -1, dtype=np.int64)
            for part in ("ref_n", "ref_mean", "ref_sd"):
                trace[f"{prefix}_{part}"] = ref[part] if ref else nan
        trace["curv_obs"] = obs
        trace["curv_pred"] = pred
        for key, val in trace.items():
            out[f"in_{key}"] = val

    out.attrs.update({"variant": P.variant, "embed_mode": P.embed_mode,
                      "disabled_components": list(P.disabled), "params": dict(P.params),
                      "curvdyn_source": P.model.source if P.model else None,
                      "notes": list(P.notes)})
    return out


def fit_curvdyn_model(events, curvature, *, train_mask, embeddings=None, tgn_embeddings=None,
                      target="forman", pca_dim=32, ridge_alpha=1.0, max_fit_rows=500_000,
                      seed=0) -> CurvDynModel | None:
    """Fit the s_curvdyn regressor on train rows only and return it frozen.

    Endpoint vectors: tgn_embeddings (per event) when given, else static
    embeddings of src and dst. target: "forman" (screen) or "orc" (escalated
    rows; the ollivier_only cell). The seed drives the row subsample when train
    rows exceed max_fit_rows and the PCA subsample. Returns None when no train
    row has both a finite target and endpoint vectors.
    """
    if target not in ("forman", "orc"):
        raise ValueError("target must be 'forman' or 'orc'")
    if embeddings is None and tgn_embeddings is None:
        raise ValueError("need embeddings or tgn_embeddings")
    ev = _canonical_events(events, None, curvature, "field")
    tm = np.asarray(train_mask, dtype=bool)
    if tm.shape != (len(ev),):
        raise ValueError("train_mask must be a boolean array aligned with the events rows")
    endpoints = _Endpoints(ev, embeddings, tgn_embeddings)
    return _fit_curvdyn(ev, tm[ev["_pos"].to_numpy()], endpoints, target, pca_dim, ridge_alpha,
                        max_fit_rows, seed)


def matched_random_escalation(curvature, *, seed=0, strata=None) -> np.ndarray:
    """Event ids to escalate in the ollivier_only cell: a uniform random sample
    (seeded) of the same size as the screen's escalated set, so the cell spends
    the same ORC budget without any Forman signal choosing the events. The
    curvature stage computes ORC for these ids; score_events(variant=
    "ollivier_only") then reads only ORC.

    strata: optional labels aligned with the curvature rows (the pipeline passes
    the train/val/test split). The sample is then drawn inside each stratum, of
    the size of the screen's escalated set in that stratum, from a generator
    seeded by (seed, crc32(label)). Leak: without strata one global draw mixes
    all rows, so the number of test-period screen flags changes which TRAIN rows
    get an ORC value, and with it the train-fitted ZStats and s_curvdyn
    regressor. With strata the draw for a stratum reads only that stratum's
    rows, so test-period events cannot move any pre-cutoff selection."""
    ids = curvature["event_id"].to_numpy(dtype=np.int64)
    flagged = curvature["escalated"].fillna(False).astype(bool).to_numpy()
    if strata is None:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(ids, size=int(flagged.sum()), replace=False))
    lab = np.asarray(strata).astype(str)
    if lab.shape != (len(ids),):
        raise ValueError("strata must be aligned with the curvature rows")
    picked = []
    for g in np.unique(lab):
        rows = lab == g
        rng = np.random.default_rng([int(seed), zlib.crc32(g.encode("utf-8"))])
        picked.append(rng.choice(ids[rows], size=int(flagged[rows].sum()), replace=False))
    return np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)


# ----------------------------------------------------------------- single-event path


@dataclass
class ScoringContext:
    """Everything score_event needs, built once: canonical events, raw-statistic
    prefix sums per reference cell, per-dst citation times, the frozen s_curvdyn
    model. A query is a handful of binary searches (O(log E))."""

    prepared: _Prepared
    pos: pd.Index
    tables: dict
    dst_index: pd.Index
    dst_start: np.ndarray
    dst_times: np.ndarray


def build_context(events, works, curvature, *, embeddings=None, tgn_scores=None, tgn_embeddings=None,
                  train_mask=None, mode="auto", variant="full", k=3, window_days=90, min_cell=30,
                  curvdyn_model=None, pca_dim=32, ridge_alpha=1.0, max_fit_rows=500_000,
                  p_link_eps=1e-7, seed=0, field_col="field") -> ScoringContext:
    """Same inputs and meaning as score_events; returns the state for score_event."""
    P = _prepare(events, works, curvature, embeddings=embeddings, tgn_scores=tgn_scores,
                 tgn_embeddings=tgn_embeddings, train_mask=train_mask, mode=mode, variant=variant,
                 k=k, window_days=window_days, min_cell=min_cell, curvdyn_model=curvdyn_model,
                 pca_dim=pca_dim, ridge_alpha=ridge_alpha, max_fit_rows=max_fit_rows,
                 p_link_eps=p_link_eps, seed=seed, field_col=field_col)
    ev = P.ev
    t = ev["t"].to_numpy()
    frames = _level_frames(ev)
    reads_f, reads_o = _uses(P.variant)
    tables = {"temp": _prefix_tables(frames, t, P.temporal["temp_x"], np.ones(len(ev), dtype=bool))}
    if P.embed_mode == "static_cosine":
        tables["embed"] = _prefix_tables(frames, t, P.cosine, np.isfinite(P.cosine))
    if reads_f:
        f = ev["forman"].to_numpy()
        tables["geom"] = _prefix_tables(frames, t, f, np.isfinite(f))
    if reads_o:
        o = ev["orc"].to_numpy()
        tables["orc"] = _prefix_tables(frames, t, o, ev["escalated"].to_numpy() & np.isfinite(o))
    codes, uniq = pd.factorize(ev["dst_id"].to_numpy())
    order = np.lexsort((t, codes))
    starts = np.searchsorted(codes[order], np.arange(len(uniq) + 1), "left")
    return ScoringContext(P, pd.Index(ev["event_id"].to_numpy()), tables, pd.Index(uniq), starts,
                          t[order])


def score_event(event_id, ctx: ScoringContext) -> ScoreBreakdown:
    """Breakdown of one event from a built context (NFR2 latency path).

    Computes the event's own raw statistics independently of the vectorised
    path (binary searches over the cited paper's citation times; one cosine;
    one regressor prediction) and reads reference statistics as prefix sums
    over strictly earlier rows of its cell. Agrees with score_events."""
    P = ctx.prepared
    ev = P.ev
    i = ctx.pos.get_loc(event_id)
    src, dst = ev["src_id"].iat[i], ev["dst_id"].iat[i]
    t, year, fld = int(ev["t"].iat[i]), int(ev["year"].iat[i]), ev["field"].iat[i]
    keys = [(fld, year), (fld,), (0,)]
    prm = P.params
    W = prm["window_days"]
    nan = float("nan")

    # temporal raw statistics of dst, from its sorted citation times
    c = ctx.dst_index.get_loc(dst)
    times = ctx.dst_times[ctx.dst_start[c]:ctx.dst_start[c + 1]]
    n_prior = int(np.searchsorted(times, t, "left"))
    n_le = int(np.searchsorted(times, t, "right"))
    n_before = int(np.searchsorted(times, t - W, "right"))
    n_recent = n_le - 1 - n_before
    exposure = float(max(t - W - int(times[0]), W)) if n_before else float(W)
    temp_x = math.log((n_recent + 1) / (n_before + 1)) + math.log(exposure / W)
    rt = _scalar_z(ctx.tables["temp"], keys, t, temp_x, prm["min_cell"])

    # s_embed
    p_link = float(P.p_link[i])
    cosine = _cosine_one(P.emb, src, dst) if P.emb is not None else nan
    re = _scalar_z(ctx.tables.get("embed"), keys, t, cosine, prm["min_cell"])
    if P.embed_mode == "tgn":
        s_embed = float(_neg_log_p(p_link, prm["p_link_eps"]))
    elif P.embed_mode == "static_cosine":
        s_embed = -re["z"]
    else:
        s_embed = nan

    # s_geom
    reads_f, reads_o = _uses(P.variant)
    esc = bool(ev["escalated"].iat[i])
    forman = float(ev["forman"].iat[i]) if reads_f else nan
    orc = float(ev["orc"].iat[i]) if reads_o else nan
    rf = _scalar_z(ctx.tables.get("geom"), keys, t, forman, prm["min_cell"])
    ro = _scalar_z(ctx.tables.get("orc") if esc else None, keys, t, orc, prm["min_cell"])
    s_geom = float(np.fmax(abs(rf["z"]), abs(ro["z"])))

    # s_curvdyn
    obs = pred = s_curv = nan
    if P.model is not None:
        raw = (orc if esc else nan) if P.model.target == "orc" else float(ev["forman"].iat[i])
        if np.isfinite(raw):
            obs = float(np.arcsinh(raw))
            if P.endpoints.valid[i]:
                pred = float(P.model.predict(*P.endpoints.vectors(np.array([i])))[0])
                s_curv = abs(obs - pred) / P.model.rmse
    values = {"s_embed": s_embed, "s_geom": s_geom, "s_temp": rt["z"], "s_curvdyn": s_curv}
    for comp in P.disabled:
        values[comp] = nan

    inputs = {"event_id": int(event_id), "src_id": src, "dst_id": dst, "t": t, "field": fld,
              "year": year, "variant": P.variant, "embed_mode": P.embed_mode, "k": prm["k"],
              "p_link": p_link, "cosine": cosine, "forman": forman, "orc": orc, "escalated": esc,
              "z_forman": rf["z"], "z_orc": ro["z"], "temp_x": temp_x, "n_recent": n_recent,
              "n_before": n_before, "exposure_days": exposure, "n_prior_dst": n_prior,
              "curv_obs": obs, "curv_pred": pred,
              "curv_source": P.model.source if P.model else None,
              "curv_target": P.model.target if P.model else None,
              "curv_rmse": P.model.rmse if P.model else nan}
    for prefix, ref in zip(_REF_STATS, (re, rf, ro, rt)):
        if prefix == "embed" and P.embed_mode != "static_cosine":
            ref = _scalar_z(None, keys, t, nan, prm["min_cell"])
        inputs[f"{prefix}_level"] = ref["level"]
        for part in ("ref_n", "ref_mean", "ref_sd"):
            inputs[f"{prefix}_{part}"] = ref[part]
    return ScoreBreakdown(**values, total=None, cold_start=n_prior < prm["k"], inputs=inputs)

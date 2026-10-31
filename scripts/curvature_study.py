"""Curvature formulation study (agent W2c). SYNTHETIC DATA ONLY.

Question: which curvature formulations of a citation event (u -> v, t) carry anomaly signal
beyond trivial controls (cited-paper age, degrees, bibliographic coupling, co-citation, field,
embedding similarity) on the synthetic stream with planted cartels? Write-up:
docs/curvature_formulation_study.md. All numbers go to results/curvature_study.json.

One command reproduces everything (seeds fixed):

    PYTHONPATH=src nice -n 10 python scripts/curvature_study.py

Needs numpy, pandas, scipy, scikit-learn; POT makes the transport problems fast (scipy's LP is
the fallback); GraphRicciCurvature (curv env) enables the real-library validation block, which
is skipped otherwise. --quick runs a small smoke configuration.

Design (details and reasons in the doc):
- Per seed s in SEEDS: generate_citation_stream(n_works, seed=s); cutoff c = the latest event
  time with at least 20 % of events after it; fit period (c - 5 years, c].
- Per setting (OBVIOUSNESS level or clique): first plant cartels inside the fit period
  (works published in [c - 5y, c - 1], injection seed s + 1000), then plant cartels of the same
  setting in the test period (t_min = c, injection seed s), members disjoint. Positives of the
  test set = test-period injected events; fit positives = fit-period injected events.
- Samples (identical for every signal of a run): test = all positive test events + N_NEG_TEST
  random negative test events; fit = all fit-period positives + N_NEG_FIT random natural events
  of the fit period. Sampling seed = (s, setting index).
- Single-signal metrics: ranking_metrics on the test sample; the orientation (sign) of each
  signal is chosen on the fit sample (AUROC >= 0.5), never on test labels.
- Incremental value: logistic regression (and gradient boosting as a sensitivity check) fitted
  on the fit sample, evaluated on the test sample: controls alone vs controls + one signal.
  No test label is used for fitting or for any choice.

Fast-path premise (asserted for every run): every event has t = pub(src) and pub(dst) < pub(src),
so the graph at time t is the subgraph induced by papers published by t. Every fast-path signal
is cross-checked against the reference implementations in citecurv.curvature.variants on a
random subsample of each run (max abs difference recorded; > 1e-9 aborts).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # shared machine: single-threaded numerics

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from citecurv.config import load_config
from citecurv.curvature import variants as V
from citecurv.curvature.forman_directed import forman_edge
from citecurv.curvature.screen import forman_screen
from citecurv.curvature.timeslice import EventStore
from citecurv.data.synthetic import generate_citation_stream, to_days
from citecurv.eval.inject import OBVIOUSNESS, inject_cartel, inject_clique
from citecurv.eval.metrics import auroc, ranking_metrics
from citecurv.eval.split import split_events

REPO = Path(__file__).resolve().parents[1]

CONFIG = {
    "n_works": 12000,                 # synthetic stream size (about 294k natural events at seed 0)
    "seeds": [0, 1, 2],
    "settings": ["blatant", "moderate", "subtle", "clique"],
    "n_cartels": {"blatant": 10, "moderate": 10, "subtle": 20},  # subtle: 20 for ~100 positives
    "clique": {"n_cliques": 10, "clique_size": 10, "drop_prob": 0.0},
    "test_share": 0.20,               # cutoff leaves about 20 % of natural events as test
    "fit_years": 5,                   # fit period (c - 5 y, c] for fit-period cartels and negatives
    "fit_seed_offset": 1000,
    "n_neg_test": 10000,
    "n_neg_fit": 3000,
    "window_days": 365,               # S6 trailing window and the recent-citations control
    "audit_deltas": [365, 730],       # S7 audit horizons (days)
    "alpha": 0.5,                     # ORC mass kept on the node (plan default)
    "n_xcheck": 10,                   # events per run re-computed with variants.py
    "n_xcheck_inedges_max": 40,       # S6b/c cross-check only when v has <= this many in-edges
    "n_lib": 12,                      # events per setting (seed 0) checked against GraphRicciCurvature
    "lr_C": 1.0,
    "hgb": {"max_iter": 150, "learning_rate": 0.1, "max_leaf_nodes": 15, "l2_regularization": 1.0},
}
QUICK = {"n_works": 5000, "seeds": [0], "settings": ["blatant", "clique"], "n_neg_test": 1500,
         "n_neg_fit": 800, "n_xcheck": 6, "n_lib": 4, "n_cartels": {"blatant": 3, "moderate": 3, "subtle": 6},
         "clique": {"n_cliques": 4, "clique_size": 8, "drop_prob": 0.0}}

# ---------------------------------------------------------------- signal catalogue
# name -> (group, transform for regressions, short description)
CONTROLS_A = {
    "c_age": ("log1p", "cited-paper age at t (days)"),
    "c_refs_v": ("log1p", "out_deg(v): reference count of the cited paper"),
    "c_indeg_v": ("log1p", "in_deg_t(v): citations of v by t (scored row excluded)"),
    "c_recent_indeg_v": ("log1p", "citations of v in (t - W, t] (temporal control, like s_temp)"),
    "c_refs_u": ("log1p", "out_deg(u) - 1: other references of the citing paper"),
    "c_coupling": ("log1p", "|refs(u) & refs(v)|: bibliographic coupling count"),
    "c_jaccard": ("raw", "coupling Jaccard |refs(u) & refs(v)| / |refs(u) | refs(v)|"),
    "c_cocite": ("log1p", "co-citation count of (u, v) up to t"),
    "c_same_field": ("raw", "same-field indicator"),
    "c_cosine": ("raw", "embedding cosine of u and v"),
}
ADDED_CONTROLS = ("c_recent_indeg_v", "c_refs_u")  # beyond the minimal control list; base A0 omits them
CONTROLS_B_EXTRA = {
    "b_coupling_share": ("raw", "|refs(u) & refs(v)| / |refs(v)|, 1 if v cites nothing (S2's equivalent)"),
    "b_transitive": ("log1p", "|refs(u) & citers_t(v)|: papers u cites that cite v"),
}
CURV = {
    "s1_forman": "plan directed Forman screen 2 - in_deg_t(u) - out_deg_t(v)",
    "s1_forman_abs": "two-sided screen |F - median_fit(F)|",
    "s2_orc_dir": "plan directed ORC at t (GraphRicciCurvature convention), closed form",
    "s3a_orc_rev": "reversed directed ORC: m_u on refs(u), m_v on citers_t(v), undirected metric",
    "s3b_forman_rev": "reversed Forman 2 - (out(u) - 1) - (in_t(v) - 1)",
    "s4a_orc_undir": "undirected ORC on the time-t graph",
    "s4b_forman_aug": "augmented Forman 4 - deg(u) - deg(v) + 3 triangles (time-t, undirected)",
    "s5a_bc_forman": "augmented Forman of (u, v) in the bibliographic-coupling graph at t",
    "s5b_cc_forman": "augmented Forman of (u, v) in the co-citation graph at t",
    "s6a_forman_dyn": "change of mean plan-Forman of v's incident edges over (t - W, t]",
    "s6b_aug_shift": "recent minus earlier mean arrival-time augmented Forman of v's in-edges",
    "s6c_orc_shift": "recent minus earlier mean arrival-time directed ORC of v's in-edges",
}
SUSPECTED = {  # curvature signal -> the trivial quantity it is suspected to re-express
    "s1_forman": "c_refs_v", "s1_forman_abs": "c_refs_v", "s2_orc_dir": "b_coupling_share",
    "s3a_orc_rev": "b_transitive", "s3b_forman_rev": "c_indeg_v", "s4a_orc_undir": "b_transitive",
    "s4b_forman_aug": "c_indeg_v", "s5a_bc_forman": "c_coupling", "s5b_cc_forman": "c_indeg_v",
    "s6a_forman_dyn": "c_recent_indeg_v", "s6b_aug_shift": "c_recent_indeg_v",
    "s6c_orc_shift": "c_recent_indeg_v",
}


def audit_names(delta):
    return {f"s7_forman_d{delta}": f"AUDIT plan Forman at t + {delta} d",
            f"s7_orc_dir_d{delta}": f"AUDIT plan directed ORC at t + {delta} d"}


def audit_controls(delta):
    return {f"a_indeg_u_d{delta}": ("log1p", f"AUDIT in_deg(u) at t + {delta} d"),
            f"a_indeg_v_d{delta}": ("log1p", f"AUDIT in_deg(v) at t + {delta} d"),
            f"a_cocite_d{delta}": ("log1p", f"AUDIT co-citation count of (u, v) at t + {delta} d")}


# ---------------------------------------------------------------- graph arrays


class FastGraph:
    """CSR views of the final citation graph with publication-time filters.

    Premise (asserted by the caller): an edge x -> y exists at time s iff pub(x) <= s.
    """

    def __init__(self, src, dst, P):
        n = len(P)
        self.n, self.P = n, P
        self.P0 = int(P.min())
        self.span = int(P.max()) - self.P0 + 2
        o = np.lexsort((dst, src))
        self.out_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=n))]
        self.out_idx = dst[o]
        i = np.lexsort((src, P[src], dst))
        self.in_ptr = np.r_[0, np.cumsum(np.bincount(dst, minlength=n))]
        self.in_idx = src[i]
        self.in_P = P[src][i]
        self.in_keys = dst[i].astype(np.int64) * self.span + (self.in_P - self.P0)
        self.refcount = np.diff(self.out_ptr)
        self.edge_keys = np.sort(src.astype(np.int64) * n + dst)
        self.A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))

    def _tt(self, t):
        return np.clip(np.asarray(t, dtype=np.int64) - self.P0, -1, self.span - 1)

    def in_count(self, x, t):
        """Number of in-edges of x with pub(citer) <= t (vectorised)."""
        x = np.asarray(x, dtype=np.int64)
        return np.searchsorted(self.in_keys, x * self.span + self._tt(t), "right") - self.in_ptr[x]

    def refs(self, x):
        return self.out_idx[self.out_ptr[x]:self.out_ptr[x + 1]]

    def citers(self, x, t):
        lo = self.in_ptr[x]
        return self.in_idx[lo:lo + int(self.in_count(x, t))]

    def nbrs(self, x, t):
        return np.concatenate([self.refs(x), self.citers(x, t)])

    def gather_refs(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return _gather(self.out_ptr[rows], self.refcount[rows], self.out_idx)

    def gather_citers(self, rows, t):
        rows = np.asarray(rows, dtype=np.int64)
        return _gather(self.in_ptr[rows], self.in_count(rows, t), self.in_idx)

    def has_edge(self, a, b):
        k = np.asarray(a, dtype=np.int64) * self.n + np.asarray(b, dtype=np.int64)
        pos = np.searchsorted(self.edge_keys, k)
        pos = np.minimum(pos, len(self.edge_keys) - 1)
        return self.edge_keys[pos] == k


def _gather(starts, lens, idx):
    lens = np.asarray(lens, dtype=np.int64)
    total = int(lens.sum())
    if total == 0:
        return idx[:0]
    offs = np.repeat(starts - np.r_[0, np.cumsum(lens)[:-1]], lens)
    return idx[offs + np.arange(total)]


def _indicator(lists, n):
    lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    indptr = np.r_[0, np.cumsum(lens)]
    indices = np.concatenate(lists) if len(lists) else np.empty(0, dtype=np.int64)
    return sp.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(lists), n))


def _w1_compressed(a, b, D):
    """Exact W1 after merging targets (sources) with identical cost columns (rows)."""
    Dc, inv = np.unique(D, axis=1, return_inverse=True)
    bc = np.bincount(np.ravel(inv), weights=b, minlength=Dc.shape[1])
    Dr, invr = np.unique(Dc, axis=0, return_inverse=True)
    ac = np.bincount(np.ravel(invr), weights=a, minlength=Dr.shape[0])
    return V.wasserstein1(ac, bc, Dr)


def _masses(n_nbrs, alpha):
    if n_nbrs == 0:
        return np.array([1.0])
    return np.r_[alpha, np.full(n_nbrs, (1.0 - alpha) / n_nbrs)]


# ---------------------------------------------------------------- per-event fast paths


def und_distance(fg, S, T, t):
    """Undirected hop distances (capped at 3) between node arrays S and T on G_t."""
    MS = _indicator([fg.nbrs(x, t) for x in S], fg.n)
    MT = _indicator([fg.nbrs(x, t) for x in T], fg.n)
    D = np.full((len(S), len(T)), 3.0)
    D[(MS @ MT.T).toarray() > 0] = 2.0
    D[MS[:, T].toarray() > 0] = 1.0
    D[S[:, None] == T[None, :]] = 0.0
    return D


def dir_distance(fg, S, T):
    """Directed hop distances (capped at 3) from S to T on the graph at max pub(S)."""
    R1 = _indicator([fg.refs(x) for x in S], fg.n)
    R2 = R1 @ fg.A
    D = np.full((len(S), len(T)), 3.0)
    D[R2[:, T].toarray() > 0] = 2.0
    D[R1[:, T].toarray() > 0] = 1.0
    D[S[:, None] == T[None, :]] = 0.0
    return D


def expensive_signals(fg, u, v, t, cfg):
    """S3a, S4a, S5a, S5b, S6a, S7 and audit controls for one event (fast path)."""
    alpha, W = cfg["alpha"], cfg["window_days"]
    out = {}
    refs_u = fg.refs(u)                       # includes v (scored edge, geometry)
    cit_u = fg.citers(u, t)                   # empty in the synthetic stream
    refs_v = fg.refs(v)
    cit_v = fg.citers(v, t)                   # includes u (scored edge, geometry)
    Su = np.r_[u, refs_u, cit_u].astype(np.int64)
    Tv = np.r_[v, refs_v, cit_v].astype(np.int64)
    assert len(np.unique(Su)) == len(Su) and len(np.unique(Tv)) == len(Tv)
    D = und_distance(fg, Su, Tv, t)
    # S4a undirected ORC
    out["s4a_orc_undir"] = 1.0 - _w1_compressed(_masses(len(Su) - 1, alpha), _masses(len(Tv) - 1, alpha), D)
    # S3a reversed ORC: rows u + refs(u), columns v + citers_t(v)
    rows = np.arange(1 + len(refs_u))
    cols = np.r_[0, np.arange(1 + len(refs_v), len(Tv))]
    out["s3a_orc_rev"] = 1.0 - _w1_compressed(_masses(len(refs_u), alpha), _masses(len(cit_v), alpha),
                                              D[np.ix_(rows, cols)])

    # S5a coupling-graph Forman (scored row excluded: refs(u) without v)
    ru = refs_u[refs_u != v]
    nbc_u = np.setdiff1d(fg.gather_citers(ru, t), [u])
    nbc_v = np.setdiff1d(fg.gather_citers(refs_v, t), [v])
    out["s5a_bc_forman"] = _pair_forman_fast(nbc_u, nbc_v, u, v)
    # S5b co-citation-graph Forman (scored row excluded: u is not a citer of v here)
    cv = cit_v[cit_v != u]
    ncc_v = np.setdiff1d(fg.gather_refs(cv), [v])
    ncc_u = np.setdiff1d(fg.gather_refs(cit_u), [u])
    out["s5b_cc_forman"] = _pair_forman_fast(ncc_u, ncc_v, u, v)
    out["deg_cc_v"] = float(len(ncc_v))
    out["deg_bc_u"] = float(len(nbc_u))
    out["deg_bc_v"] = float(len(nbc_v))

    # S6a change of mean plan-Forman over v's incident edges
    times = ([int(fg.P[v])] if len(refs_v) else []) + ([int(fg.P[cv].min())] if len(cv) else [])
    if times:
        s = max(t - W, min(times))
        out["s6a_forman_dyn"] = _plan_forman_mean(fg, u, v, t, t) - _plan_forman_mean(fg, u, v, t, s)
    else:
        out["s6a_forman_dyn"] = math.nan

    # co-citation count of (u, v) up to t (control)
    out["c_cocite"] = float(len(np.intersect1d(cit_u, cv)))

    # S7 audit mode
    for d in cfg["audit_deltas"]:
        ta = t + d
        cu = fg.citers(u, ta)
        out[f"s7_forman_d{d}"] = 2.0 - len(cu) - len(refs_v)
        S7 = np.r_[u, cu].astype(np.int64)
        T7 = np.r_[v, refs_v].astype(np.int64)
        D7 = dir_distance(fg, S7, T7)
        out[f"s7_orc_dir_d{d}"] = 1.0 - _w1_compressed(_masses(len(cu), alpha), _masses(len(refs_v), alpha), D7)
        cva = fg.citers(v, ta)
        out[f"a_indeg_u_d{d}"] = float(len(cu))
        out[f"a_indeg_v_d{d}"] = float(len(cva))
        out[f"a_cocite_d{d}"] = float(len(np.intersect1d(cu, cva)))
    return out


def _pair_forman_fast(nu, nv, u, v):
    deg_u = len(nu) + (0 if np.any(nu == v) else 1)
    deg_v = len(nv) + (0 if np.any(nv == u) else 1)
    common = np.intersect1d(nu, nv, assume_unique=True)
    tri = int(np.sum((common != u) & (common != v)))
    return 4.0 - deg_u - deg_v + 3.0 * tri


def _plan_forman_mean(fg, u, v, t, s):
    """Mean of 2 - in_deg_s(a) - out_deg_s(b) over v's incident edges at s (scored row excluded)."""
    cit = fg.citers(v, s)
    if s >= t:
        cit = cit[cit != u]
    refs_v = fg.refs(v) if s >= fg.P[v] else fg.refs(v)[:0]
    in_v = len(cit)
    vals = np.r_[2.0 - fg.in_count(cit, s) - len(refs_v), 2.0 - in_v - fg.refcount[refs_v]]
    return float(vals.mean()) if len(vals) else math.nan


# ---------------------------------------------------------------- all-event arrays


def all_event_arrays(fg, src, dst, t, works_field, emb, cfg, chunk=40000):
    """Cheap per-event quantities for every event (vectorised)."""
    E = len(src)
    W = cfg["window_days"]
    coupling = np.zeros(E)
    transitive = np.zeros(E)
    cosine = np.zeros(E)
    embn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    for lo in range(0, E, chunk):
        hi = min(E, lo + chunk)
        rows = src[lo:hi]
        lens = fg.refcount[rows]
        y = fg.gather_refs(rows)
        ev = np.repeat(np.arange(hi - lo), lens)
        vv = dst[lo:hi][ev]
        coupling[lo:hi] = np.bincount(ev, weights=fg.has_edge(vv, y), minlength=hi - lo)
        transitive[lo:hi] = np.bincount(ev, weights=fg.has_edge(y, vv), minlength=hi - lo)
        cosine[lo:hi] = np.einsum("ij,ij->i", embn[rows], embn[dst[lo:hi]])
    indeg_u = fg.in_count(src, t)
    indeg_v = fg.in_count(dst, t) - 1  # scored row excluded
    recent_v = fg.in_count(dst, t) - fg.in_count(dst, t - W) - 1
    refs_u = fg.refcount[src] - 1
    refs_v = fg.refcount[dst].astype(float)
    union = refs_u + refs_v - coupling
    jacc = np.where(union > 0, coupling / np.where(union > 0, union, 1), 0.0)
    # share of v's references that u also cites; 1 when v cites nothing (u misses none of them)
    share = np.where(refs_v > 0, coupling / np.where(refs_v > 0, refs_v, 1), 1.0)
    a = {
        "c_age": (t - fg.P[dst]).astype(float),
        "c_refs_v": refs_v,
        "c_indeg_v": indeg_v.astype(float),
        "c_recent_indeg_v": recent_v.astype(float),
        "c_refs_u": refs_u.astype(float),
        "c_coupling": coupling,
        "c_jaccard": jacc,
        "c_same_field": (works_field[src] == works_field[dst]).astype(float),
        "c_cosine": cosine,
        "b_coupling_share": share,
        "b_transitive": transitive,
        "indeg_u": indeg_u.astype(float),
    }
    # S2 closed form (valid because indeg_u == 0 everywhere; asserted by the caller)
    a["s2_orc_dir"] = V.orc_directed_uncited(coupling, refs_v, cfg["alpha"])
    # S3b reversed Forman: 2 - (out(u) - 1) - (in_t(v) - 1), scored edge counted once each
    a["s3b_forman_rev"] = 2.0 - refs_u - indeg_v
    # S4b augmented Forman on G_t + e: deg(u) = refs(u) incl v (+ citers_t(u) = 0),
    # deg(v) = refs(v) + citers_t(v) incl u, triangles = coupling + transitive
    deg_u = fg.refcount[src] + indeg_u
    deg_v = refs_v + indeg_v + 1
    a["s4b_forman_aug"] = 4.0 - deg_u - deg_v + 3.0 * (coupling + transitive)
    return a


def arrival_shift_arrays(fg, src, dst, t, kappa, sample_pos, window):
    """S6b/S6c for the events at sample_pos from per-event arrival-time values kappa."""
    key = dst.astype(np.int64) * fg.span + (t - fg.P0)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cs = np.r_[0.0, np.cumsum(kappa[order])]
    v = dst[sample_pos].astype(np.int64)
    tt = t[sample_pos]
    lo = np.searchsorted(ks, v * fg.span, "left")
    k_t = np.searchsorted(ks, v * fg.span + fg._tt(tt), "right")
    k_w = np.searchsorted(ks, v * fg.span + fg._tt(tt - window), "right")
    n_rec = k_t - k_w - 1
    n_old = k_w - lo
    rec = (cs[k_t] - cs[k_w] - kappa[sample_pos]) / np.maximum(n_rec, 1)
    old = (cs[k_w] - cs[lo]) / np.maximum(n_old, 1)
    return np.where((n_rec > 0) & (n_old > 0), rec - old, np.nan)


# ---------------------------------------------------------------- run construction


def choose_cutoff(t, share):
    q = np.quantile(t, 1.0 - share)
    return int(t[t <= q].max())


def inject(setting, events, works, authorships, cutoff, fit_lo, seed, cfg):
    """Fit-period cartels first, then test-period cartels; returns events and truth ids."""
    pub = to_days(works["pub_date"].to_numpy())
    works_fit = works[(pub >= fit_lo) & (pub <= cutoff - 1)]
    fseed = seed + cfg["fit_seed_offset"]
    if setting == "clique":
        k = cfg["clique"]
        r1 = inject_clique(events, works_fit, k["n_cliques"], k["clique_size"], seed=fseed, t_min=fit_lo,
                           drop_prob=k["drop_prob"])
        r2 = inject_clique(r1.events, works, k["n_cliques"], k["clique_size"], seed=seed, t_min=cutoff,
                           drop_prob=k["drop_prob"], exclude_ids=set(r1.cartels["work_id"]))
    else:
        preset = OBVIOUSNESS[setting]
        n = cfg["n_cartels"][setting]
        r1 = inject_cartel(events, works_fit, n, seed=fseed, t_min=fit_lo, authorships=authorships, **preset)
        r2 = inject_cartel(r1.events, works, n, seed=seed, t_min=cutoff, authorships=authorships,
                           exclude_ids=set(r1.cartels["work_id"]), **preset)
    return r2.events, r1.truth_event_ids, r2.truth_event_ids, r1.cartels, r2.cartels


# ---------------------------------------------------------------- evaluation helpers


def transform(x, how):
    x = np.asarray(x, dtype=float)
    if how == "log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    if how == "asinh":
        return np.arcsinh(x)
    return x


def oriented(values, fit_values, fit_y):
    a = auroc(fit_values, fit_y)
    sign = 1.0 if (math.isnan(a) or a >= 0.5) else -1.0
    return sign * values, sign, a


def r4(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def design(cols, F, fit_idx, test_idx, trans):
    """Standardised design matrices (fit statistics only); constant columns dropped."""
    Xf, Xt, kept = [], [], []
    for c in cols:
        xf = transform(F[c][fit_idx], trans[c])
        xt = transform(F[c][test_idx], trans[c])
        med = np.nanmedian(xf) if np.isfinite(xf).any() else 0.0
        xf = np.where(np.isfinite(xf), xf, med)
        xt = np.where(np.isfinite(xt), xt, med)
        sd = xf.std()
        if sd <= 1e-12:
            continue
        mu = xf.mean()
        Xf.append((xf - mu) / sd)
        Xt.append((xt - mu) / sd)
        kept.append(c)
    return np.column_stack(Xf), np.column_stack(Xt), kept


def fit_score(model, Xf, yf, Xt, seed, cfg):
    if model == "lr":
        m = LogisticRegression(C=cfg["lr_C"], max_iter=5000)
    else:
        m = HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **cfg["hgb"])
    m.fit(Xf, yf)
    return m.predict_proba(Xt)[:, 1]


def r_squared(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    sst = ((y - y.mean()) ** 2).sum()
    return float(1.0 - (res ** 2).sum() / sst) if sst > 0 else float("nan")


# ---------------------------------------------------------------- cross-checks


def xcheck(store, idx2id, sample_rows, F, src, dst, t, cfg, rng):
    """Re-compute fast-path signals with citecurv.curvature.variants on a few events."""
    alpha, W = cfg["alpha"], cfg["window_days"]
    pick = rng.choice(len(sample_rows), size=min(cfg["n_xcheck"], len(sample_rows)), replace=False)
    worst = {}
    n_s6 = 0

    def note(name, fast, ref):
        if (isinstance(fast, float) and math.isnan(fast)) and math.isnan(ref):
            d = 0.0
        else:
            d = abs(float(fast) - float(ref))
        worst[name] = max(worst.get(name, 0.0), d)

    for j in pick:
        e = sample_rows[j]
        u, v, te = idx2id[src[e]], idx2id[dst[e]], int(t[e])
        G = V.time_graph(store, u, v, te, radius=2)
        note("s1_forman", F["s1_forman"][j], forman_edge(G, u, v))
        note("s2_orc_dir", F["s2_orc_dir"][j], V.orc_directed_plan(G, u, v, alpha))
        note("s3a_orc_rev", F["s3a_orc_rev"][j], V.orc_reversed(G, u, v, alpha))
        note("s3b_forman_rev", F["s3b_forman_rev"][j], V.forman_reversed(G, u, v))
        note("s4a_orc_undir", F["s4a_orc_undir"][j], V.orc_undirected(G, u, v, alpha))
        note("s4b_forman_aug", F["s4b_forman_aug"][j], V.forman_augmented(G, u, v))
        note("s5a_bc_forman", F["s5a_bc_forman"][j], V.coupling_forman(G, u, v))
        note("s5b_cc_forman", F["s5b_cc_forman"][j], V.cocitation_forman(G, u, v))
        note("s6a_forman_dyn", F["s6a_forman_dyn"][j], V.forman_dynamics(G, v, te, W))
        for d in cfg["audit_deltas"]:
            a = V.audit_curvatures(store, u, v, te, d, alpha)
            note(f"s7_forman_d{d}", F[f"s7_forman_d{d}"][j], a["forman"])
            note(f"s7_orc_dir_d{d}", F[f"s7_orc_dir_d{d}"][j], a["orc_directed"])
        if F["c_indeg_v"][j] <= cfg["n_xcheck_inedges_max"]:
            n_s6 += 1
            arr_b = V.inedge_arrivals(store, u, v, te, V.forman_augmented, radius=1)
            note("s6b_aug_shift", F["s6b_aug_shift"][j], V.arrival_curvature_shift(arr_b, te, W))
            arr_c = V.inedge_arrivals(store, u, v, te, lambda H, x, y: V.orc_directed_plan(H, x, y, alpha),
                                      radius=2)
            note("s6c_orc_shift", F["s6c_orc_shift"][j], V.arrival_curvature_shift(arr_c, te, W))
    bad = {k: x for k, x in worst.items() if x > 1e-9}
    if bad:
        raise AssertionError(f"fast path disagrees with variants.py: {bad}")
    return {"n_events": len(pick), "n_events_s6bc": n_s6, "max_abs_diff": {k: float(x) for k, x in worst.items()}}


def library_check(store, events, sample_rows, F, src, dst, t, idx2id, cfg, rng):
    """S2 / S4a / S4b / S7 against GraphRicciCurvature on a few events (curv env only)."""
    try:
        from GraphRicciCurvature.FormanRicci import FormanRicci
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError:
        return {"skipped": "GraphRicciCurvature not installed"}
    import GraphRicciCurvature

    from citecurv.curvature.orc import cap_subgraph, escalate, orc_directed

    alpha = cfg["alpha"]
    knobs = {"alpha": alpha, "method": "OTD", "proc": 1, "shortest_path": "pairwise"}
    pick = np.sort(rng.choice(len(sample_rows), size=min(cfg["n_lib"], len(sample_rows)), replace=False))
    rows = sample_rows[pick]
    ev_ids = events["event_id"].to_numpy()[rows]
    curv = forman_screen(events)
    curv["escalated"] = curv["event_id"].isin(ev_ids)
    out = {"library_version": GraphRicciCurvature.__version__, "n_events": int(len(rows)), "knobs": knobs}
    t0 = time.perf_counter()
    for radius in (1, 2):
        res, st = escalate(None, curv, store, radius=radius, edge_cap=10**9, prune=True, **knobs)
        got = res.set_index("event_id").loc[ev_ids, "orc"].to_numpy()
        out[f"s2_escalate_radius{radius}_max_abs_diff_vs_closed_form"] = float(
            np.max(np.abs(got - F["s2_orc_dir"][pick])))
        out[f"s2_escalate_radius{radius}_mean_subgraph_edges"] = st["mean_subgraph_edges"]
    d4a, d4b, d7 = [], [], []
    for j, e in zip(pick, rows):
        u, v, te = idx2id[src[e]], idx2id[dst[e]], int(t[e])
        G2 = V.time_graph(store, u, v, te, 2)
        G2.add_edge(u, v, t=te)
        H, _ = cap_subgraph(G2, u, v, 10**9, prune=False)
        U = H.to_undirected()
        orc_u = OllivierRicci(U, alpha=alpha, method="OTD", proc=1, shortest_path="pairwise", verbose="ERROR")
        d4a.append(abs(orc_u.compute_ricci_curvature_edges([(u, v)])[(u, v)] - F["s4a_orc_undir"][j]))
        G1 = V.time_graph(store, u, v, te, 1)
        G1.add_edge(u, v, t=te)
        frc = FormanRicci(G1.to_undirected(), method="augmented", verbose="ERROR")
        frc.compute_ricci_curvature()
        d4b.append(abs(frc.G[u][v]["formanCurvature"] - F["s4b_forman_aug"][j]))
        for d in cfg["audit_deltas"]:
            Ga = V.audit_time_graph(store, u, v, te, d, 2)
            Ha, _ = cap_subgraph(Ga, u, v, 10**9, prune=True)
            val = orc_directed(Ha, [(u, v)], **knobs)[(u, v)]
            d7.append(abs(val - F[f"s7_orc_dir_d{d}"][j]))
    out["s4a_max_abs_diff_vs_library_undirected"] = float(max(d4a))
    out["s4b_max_abs_diff_vs_library_augmented_forman"] = float(max(d4b))
    out["s7_orc_max_abs_diff_vs_library_directed"] = float(max(d7))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


# ---------------------------------------------------------------- one run


def run_one(seed, setting, stream, cutoff, cfg, do_library):
    t_start = time.perf_counter()
    works = stream["works"]
    fit_lo = cutoff - int(round(cfg["fit_years"] * 365.25))
    events, truth_fit, truth_test, cart_fit, cart_test = inject(
        setting, stream["events"], works, stream["authorships"], cutoff, fit_lo, seed, cfg)
    events = events.sort_values(["t", "event_id"], kind="mergesort").reset_index(drop=True)

    ids = works["work_id"].astype(str).to_numpy()
    id2idx = {w: i for i, w in enumerate(ids)}
    P = to_days(works["pub_date"].to_numpy()).astype(np.int64)
    src = events["src_id"].map(id2idx).to_numpy(np.int64)
    dst = events["dst_id"].map(id2idx).to_numpy(np.int64)
    t = events["t"].to_numpy(np.int64)
    # fast-path premise
    assert np.all(t == P[src]) and np.all(P[dst] < P[src]), "fast-path premise violated"
    assert not pd.Series(src * len(P) + dst).duplicated().any(), "duplicate (src, dst) rows"

    eid = events["event_id"].to_numpy(np.int64)
    is_fit_pos = np.isin(eid, truth_fit)
    is_test_pos = np.isin(eid, truth_test)
    split = split_events(events, cutoff).to_numpy()
    test = split == "test"
    assert is_test_pos[~test].sum() == 0, "a test-period positive fell before the cutoff"
    assert np.all(t[is_fit_pos] <= cutoff) and np.all(t[is_fit_pos] > fit_lo)

    fg = FastGraph(src, dst, P)
    field = works["field"].astype(str).to_numpy()
    emb = stream["embeddings"][1].astype(np.float64)
    assert np.array_equal(stream["embeddings"][0].astype(str), ids)

    A = all_event_arrays(fg, src, dst, t, field, emb, cfg)
    assert np.all(A["indeg_u"] == 0), "a citing paper is cited at its own time (not in this generator)"
    screen = forman_screen(events)
    assert np.array_equal(screen["event_id"].to_numpy(), eid)
    A["s1_forman"] = screen["forman"].to_numpy(float)
    assert np.allclose(A["s1_forman"], 2.0 - A["c_refs_v"]), "Proposition 1 identity failed"

    # samples
    srng = np.random.default_rng([seed, cfg["settings_all"].index(setting)])
    neg_test = np.flatnonzero(test & ~is_test_pos)
    neg_fit_pool = np.flatnonzero((t > fit_lo) & (t <= cutoff) & ~is_fit_pos & ~is_test_pos)
    test_rows = np.sort(np.r_[np.flatnonzero(is_test_pos),
                              srng.choice(neg_test, size=min(cfg["n_neg_test"], len(neg_test)), replace=False)])
    fit_rows = np.sort(np.r_[np.flatnonzero(is_fit_pos),
                             srng.choice(neg_fit_pool, size=min(cfg["n_neg_fit"], len(neg_fit_pool)),
                                         replace=False)])
    rows = np.r_[fit_rows, test_rows]
    nf = len(fit_rows)
    y_all = (is_fit_pos | is_test_pos).astype(np.int64)

    # per-row features: cheap arrays + expensive fast paths
    F = {k: A[k][rows] for k in A}
    t_exp = time.perf_counter()
    exp_rows = [expensive_signals(fg, src[e], dst[e], int(t[e]), cfg) for e in rows]
    for k in exp_rows[0]:
        F[k] = np.array([r[k] for r in exp_rows], dtype=float)
    sec_expensive = time.perf_counter() - t_exp
    F["s6b_aug_shift"] = arrival_shift_arrays(fg, src, dst, t, A["s4b_forman_aug"], rows, cfg["window_days"])
    F["s6c_orc_shift"] = arrival_shift_arrays(fg, src, dst, t, A["s2_orc_dir"], rows, cfg["window_days"])
    med_fit_F = float(np.median(F["s1_forman"][:nf]))
    F["s1_forman_abs"] = np.abs(F["s1_forman"] - med_fit_F)

    # cross-checks against variants.py (and the library on seed 0)
    t_x = time.perf_counter()
    store = EventStore(events)
    idx2id = ids
    xrng = np.random.default_rng([seed, 99, cfg["settings_all"].index(setting)])
    Ftest = {k: F[k][nf:] for k in F}
    xc = xcheck(store, idx2id, test_rows, Ftest, src, dst, t, cfg, xrng)
    lib = library_check(store, events, test_rows, Ftest, src, dst, t, idx2id, cfg, xrng) if do_library else None
    sec_check = time.perf_counter() - t_x
    del store

    yf, yt = y_all[fit_rows], y_all[test_rows]
    fit_idx, test_idx = np.arange(nf), nf + np.arange(len(test_rows))
    # age-restricted view: test events whose cited paper is no older than the oldest fit positive
    # (threshold from the fit period only), so that cited-paper recency cannot carry the ranking
    age_max = float(F["c_age"][fit_idx][yf == 1].max())
    restr = F["c_age"][test_idx] <= age_max
    yr = yt[restr]

    deltas = cfg["audit_deltas"]
    trans = {k: v[0] for k, v in {**CONTROLS_A, **CONTROLS_B_EXTRA}.items()}
    for d in deltas:
        trans.update({k: v[0] for k, v in audit_controls(d).items()})
    curv_names = list(CURV)
    audit_sig = {d: list(audit_names(d)) for d in deltas}
    for k in curv_names + [n for d in deltas for n in audit_sig[d]]:
        trans[k] = "asinh"

    def metrics(score):
        m = ranking_metrics(score, yt)
        mr = ranking_metrics(score[restr], yr)
        return {"auroc": r4(m["auroc"]), "ap": r4(m["ap"]), "p_at_100": r4(m["p_at_100"]),
                "found_at_100": m["found_at_100"], "auroc_r": r4(mr["auroc"]), "ap_r": r4(mr["ap"]),
                "found_at_100_r": mr["found_at_100"]}

    # single-signal metrics on the test sample (orientation from the fit sample)
    single = {}
    nan_share = {}
    for k in trans:
        x = F[k].astype(float)
        nan_share[k] = float(np.mean(~np.isfinite(x[test_idx])))
        med = np.nanmedian(x[fit_idx]) if np.isfinite(x[fit_idx]).any() else 0.0
        x = np.where(np.isfinite(x), x, med)
        xo, sign, fit_auc = oriented(x, x[fit_idx], yf)
        single[k] = {"sign": sign, "fit_auroc_raw": r4(fit_auc), **metrics(xo[test_idx])}

    # full test set for the cheap all-event signals
    cheap = list(CONTROLS_A) + list(CONTROLS_B_EXTRA) + ["s1_forman", "s2_orc_dir", "s3b_forman_rev",
                                                         "s4b_forman_aug"]
    full = {}
    test_all = np.flatnonzero(test)
    for k in cheap:
        if k not in A:
            continue
        m = ranking_metrics(A[k][test_all] * single[k]["sign"], y_all[test_all])
        full[k] = {"auroc": r4(m["auroc"]), "ap": r4(m["ap"]), "p_at_100": r4(m["p_at_100"])}

    # correlation with the suspected trivial equivalent, max |rho| over controls, R^2 on controls
    ctrlA = list(CONTROLS_A)
    ctrlB = ctrlA + list(CONTROLS_B_EXTRA)
    corr = {}
    for k in curv_names + [n for d in deltas for n in audit_sig[d]]:
        if k.startswith("s7"):
            d = int(k.rsplit("_d", 1)[1])
            cA, cB = ctrlA + list(audit_controls(d)), ctrlB + list(audit_controls(d))
            sus = f"a_indeg_u_d{d}"
        else:
            cA, cB, sus = ctrlA, ctrlB, SUSPECTED[k]
        x = F[k][test_idx]
        ok = np.isfinite(x)
        xs = transform(x[ok], "asinh")
        rho = {}
        for c in cB:
            cv = F[c][test_idx][ok]
            if np.std(cv) > 0 and np.std(xs) > 0:
                rho[c] = float(spearmanr(xs, cv).statistic)
        top = max(rho, key=lambda c: abs(rho[c])) if rho else None
        XA = np.column_stack([transform(F[c][test_idx][ok], trans[c]) for c in cA])
        XB = np.column_stack([transform(F[c][test_idx][ok], trans[c]) for c in cB])
        corr[k] = {
            "suspected": sus, "rho_suspected": r4(rho.get(sus)),
            "max_abs_rho_control": top, "rho_max": r4(rho[top]) if top else None,
            "r2_on_controls_A": r4(r_squared(xs, XA)) if np.std(xs) > 0 else None,
            "r2_on_controls_B": r4(r_squared(xs, XB)) if np.std(xs) > 0 else None,
            "nan_share_test": r4(nan_share[k]),
        }

    # incremental value: controls alone vs controls + signal (fit sample -> test sample)
    incr = {}
    bases = {"A0": [c for c in ctrlA if c not in ADDED_CONTROLS], "A": ctrlA, "B": ctrlB}
    for d in deltas:
        bases[f"audit_d{d}"] = ctrlA + list(audit_controls(d))
    for model in ("lr", "hgb"):
        res = {}
        for bname, bcols in bases.items():
            Xf, Xt, kept = design(bcols, F, fit_idx, test_idx, trans)
            base_m = metrics(fit_score(model, Xf, yf, Xt, seed, cfg))
            entry = {"base": base_m, "kept_controls": kept, "plus": {}}
            if bname.startswith("audit"):
                d = int(bname.split("_d")[1])
                groups = {s: [s] for s in audit_sig[d]}
                groups["all_audit"] = audit_sig[d]
            else:
                groups = {s: [s] for s in curv_names}
                groups["all_event_time"] = curv_names
            for sname, extra in groups.items():
                Xf2, Xt2, _ = design(bcols + extra, F, fit_idx, test_idx, trans)
                m2 = metrics(fit_score(model, Xf2, yf, Xt2, seed, cfg))
                entry["plus"][sname] = {
                    **m2,
                    "d_auroc": r4(m2["auroc"] - base_m["auroc"]), "d_ap": r4(m2["ap"] - base_m["ap"]),
                    "d_auroc_r": r4(m2["auroc_r"] - base_m["auroc_r"]), "d_ap_r": r4(m2["ap_r"] - base_m["ap_r"]),
                }
            res[bname] = entry
        incr[model] = res

    data_end = int(P.max())
    info = {
        "seed": seed, "setting": setting, "cutoff_t": cutoff,
        "cutoff_date": str(np.datetime64("1970-01-01") + np.timedelta64(cutoff, "D")),
        "fit_lo_date": str(np.datetime64("1970-01-01") + np.timedelta64(fit_lo, "D")),
        "n_events": int(len(events)), "n_test": int(test.sum()), "n_test_pos": int(is_test_pos.sum()),
        "n_fit_pos": int(is_fit_pos.sum()), "n_test_sample": int(len(test_rows)), "n_fit_sample": int(nf),
        "n_groups_test": int(cart_test["cartel_id"].nunique()), "n_groups_fit": int(cart_fit["cartel_id"].nunique()),
        "share_events_src_uncited_at_t": float(np.mean(A["indeg_u"] == 0)),
        "share_test_audit_horizon_beyond_data_end": {str(d): r4(float(np.mean(t[test] + d > data_end)))
                                                       for d in deltas},
        "median_fit_forman": med_fit_F,
        "age_restriction_days": age_max, "n_test_sample_age_restricted": int(restr.sum()),
        "n_test_pos_age_restricted": int(yr.sum()),
        "mean_test_sample_size_support_v": r4(float(np.mean(F["c_indeg_v"][test_idx] + F["c_refs_v"][test_idx]))),
        "seconds_expensive_signals": round(sec_expensive, 1),
        "seconds_checks": round(sec_check, 1),
        "seconds_total": round(time.perf_counter() - t_start, 1),
    }
    return {"info": info, "single": single, "single_full_test": full, "corr": corr, "incremental": incr,
            "xcheck": xc, "library_check": lib}


# ---------------------------------------------------------------- aggregation


def _ms(vals, per_seed=False):
    """[mean, sd] (+ per-seed values when per_seed); None entries dropped from mean and sd."""
    raw = [None if (x is None or not np.isfinite(x)) else x for x in vals]
    v = np.array([x for x in raw if x is not None], dtype=float)
    if len(v) == 0:
        return None
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out = [round(float(v.mean()), 4), round(sd, 4)]
    return out + [[r4(x) for x in raw]] if per_seed else out


SINGLE_KEYS = ("auroc", "ap", "p_at_100", "found_at_100", "auroc_r", "ap_r", "found_at_100_r")
SINGLE_PER_SEED = ("auroc", "ap")
INCR_KEYS = ("ap", "d_auroc", "d_ap", "ap_r", "d_auroc_r", "d_ap_r")
INCR_PER_SEED = ("d_auroc", "d_ap")


def summarise(runs, cfg):
    out = {}
    for setting in cfg["settings"]:
        rs = [runs[setting][str(s)] for s in cfg["seeds"]]
        single = {}
        for k in rs[0]["single"]:
            single[k] = {m: _ms([r["single"][k][m] for r in rs], m in SINGLE_PER_SEED) for m in SINGLE_KEYS}
            signs = sorted({r["single"][k]["sign"] for r in rs})
            single[k]["sign"] = signs[0] if len(signs) == 1 else "mixed"
        full = {k: {m: _ms([r["single_full_test"][k][m] for r in rs]) for m in ("auroc", "ap", "p_at_100")}
                for k in rs[0]["single_full_test"]}
        corr = {k: {m: _ms([r["corr"][k][m] for r in rs]) for m in
                    ("rho_suspected", "rho_max", "r2_on_controls_A", "r2_on_controls_B", "nan_share_test")}
                for k in rs[0]["corr"]}
        for k in corr:
            corr[k]["suspected"] = rs[0]["corr"][k]["suspected"]
            corr[k]["max_abs_rho_control"] = sorted({r["corr"][k]["max_abs_rho_control"] or "" for r in rs})
        incr = {}
        for model in ("lr", "hgb"):
            incr[model] = {}
            for b in rs[0]["incremental"][model]:
                base = {m: _ms([r["incremental"][model][b]["base"][m] for r in rs])
                        for m in ("auroc", "ap", "auroc_r", "ap_r")}
                plus = {s: {m: _ms([r["incremental"][model][b]["plus"][s][m] for r in rs], m in INCR_PER_SEED)
                            for m in INCR_KEYS}
                        for s in rs[0]["incremental"][model][b]["plus"]}
                incr[model][b] = {"base": base, "plus": plus,
                                  "kept_controls": rs[0]["incremental"][model][b]["kept_controls"]}
        counts = {m: _ms([r["info"][m] for r in rs]) for m in (
            "n_events", "n_test", "n_test_pos", "n_fit_pos", "n_test_sample", "n_fit_sample",
            "n_test_sample_age_restricted", "n_test_pos_age_restricted", "age_restriction_days")}
        out[setting] = {"counts": counts, "single": single, "single_full_test": full, "corr": corr,
                        "incremental": incr}
    return out


def dump_json(obj, fh, depth=0, max_depth=4):
    """Indented to max_depth, compact below (keeps the file small and diff-friendly)."""
    if depth >= max_depth or not isinstance(obj, (dict, list)) or not obj:
        fh.write(json.dumps(obj, separators=(",", ":"), default=_json_default))
        return
    pad, inner = " " * depth, " " * (depth + 1)
    if isinstance(obj, dict):
        fh.write("{\n")
        for i, (k, v) in enumerate(obj.items()):
            fh.write(f"{inner}{json.dumps(str(k))}: ")
            dump_json(v, fh, depth + 1, max_depth)
            fh.write(",\n" if i < len(obj) - 1 else "\n")
        fh.write(pad + "}")
    else:
        fh.write("[\n")
        for i, v in enumerate(obj):
            fh.write(inner)
            dump_json(v, fh, depth + 1, max_depth)
            fh.write(",\n" if i < len(obj) - 1 else "\n")
        fh.write(pad + "]")


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _f(x, nd=3):
    return "n/a" if x is None else f"{x[0]:.{nd}f} ± {x[1]:.{nd}f}"


def _fd(x, nd=3):
    return "n/a" if x is None else f"{x[0]:+.{nd}f} ± {x[1]:.{nd}f}"


def markdown_tables(path):
    """Markdown tables for docs/curvature_formulation_study.md, read from the results JSON."""
    with open(path) as fh:
        d = json.load(fh)
    S = d["summary"]
    settings = list(S)
    sig_a = list(d["signals"]["controls_A"]) + list(d["signals"]["controls_B_extra"])
    sig_c = list(d["signals"]["curvature_event_time"])
    deltas = d["config"]["audit_deltas"]
    sig_audit = [k for dd in deltas for k in (f"a_indeg_u_d{dd}", f"a_indeg_v_d{dd}", f"a_cocite_d{dd}",
                                               f"s7_forman_d{dd}", f"s7_orc_dir_d{dd}")]
    out = []

    def table(title, header, rows):
        out.append(f"\n**{title}**\n")
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "|".join(["---"] * len(header)) + "|")
        out.extend("| " + " | ".join(r) + " |" for r in rows)

    hdr = ["Setting", "test events", "test positives", "fit positives", "test sample (pos)",
           "age-restricted sample (pos)", "age limit (days)"]
    rows = []
    for st in settings:
        c = S[st]["counts"]
        rows.append([st, f"{c['n_test'][0]:.0f}", _f(c["n_test_pos"], 0), _f(c["n_fit_pos"], 0),
                     f"{c['n_test_sample'][0]:.0f} ({c['n_test_pos'][0]:.0f})",
                     f"{c['n_test_sample_age_restricted'][0]:.0f} ({c['n_test_pos_age_restricted'][0]:.0f})",
                     _f(c["age_restriction_days"], 0)])
    table("Table 1. Event counts (mean ± sd over seeds)", hdr, rows)

    for metric, title in (("auroc", "Table 2a. Single-signal AUROC on the test sample"),
                          ("ap", "Table 2b. Single-signal AP on the test sample (negatives subsampled)"),
                          ("auroc_r", "Table 2c. Single-signal AUROC on the age-restricted test sample")):
        rows = []
        for k in sig_a + sig_c + sig_audit:
            row = [f"`{k}`"]
            for st in settings:
                sg = S[st]["single"][k]["sign"]
                mark = "mixed" if sg == "mixed" else ("+" if sg > 0 else "-")
                row.append(f"{_f(S[st]['single'][k][metric])} ({mark})")
            rows.append(row)
        table(title + " (mean ± sd over seeds; in brackets the orientation chosen on the fit sample: "
              "+ means higher raw value = more anomalous)", ["Signal"] + settings, rows)

    rows = []
    for k in sig_c + [x for x in sig_audit if x.startswith("s7")]:
        c0 = S["moderate" if "moderate" in S else settings[0]]["corr"][k]
        row = [f"`{k}`", f"`{c0['suspected']}`", _f(c0["rho_suspected"]), ", ".join(c0["max_abs_rho_control"]),
               _f(c0["rho_max"])]
        row += [_f(S[st]["corr"][k]["r2_on_controls_A"], 2) for st in settings]
        rows.append(row)
    table("Table 3. Spearman correlation with the suspected trivial equivalent and with the closest control "
          "(moderate setting), and R² of the signal on tier-A controls (asinh / log1p scales, test sample)",
          ["Signal", "suspected equivalent", "rho", "closest control", "rho", *[f"R² A {st}" for st in settings]],
          rows)

    for model, mname in (("lr", "logistic regression"), ("hgb", "gradient boosting")):
        for base in ("A0", "A", "B"):
            if base not in S[settings[0]]["incremental"][model]:
                continue
            rows = [["controls only (AP)"] + [_f(S[st]["incremental"][model][base]["base"]["ap"])
                                             for st in settings]]
            for k in sig_c + ["all_event_time"]:
                rows.append([f"+ `{k}`"] + [
                    _fd(S[st]["incremental"][model][base]["plus"][k]["d_ap"]) + " / "
                    + _fd(S[st]["incremental"][model][base]["plus"][k]["d_auroc"], 4) for st in settings])
            table(f"Table 4-{model}-{base}. Incremental value, {mname}, base {base}: "
                  "delta AP / delta AUROC on the test sample (mean ± sd over seeds)", ["Added signal"] + settings,
                  rows)
    for model, mname in (("lr", "logistic regression"), ("hgb", "gradient boosting")):
        rows = [["controls only, age-restricted (AUROC)"] + [
            _f(S[st]["incremental"][model]["A"]["base"]["auroc_r"]) for st in settings]]
        for k in sig_c + ["all_event_time"]:
            rows.append([f"+ `{k}`"] + [_fd(S[st]["incremental"][model]["A"]["plus"][k]["d_auroc_r"], 4)
                                        for st in settings])
        table(f"Table 5-{model}. Incremental value on the age-restricted test sample, {mname}, base A: "
              "delta AUROC", ["Added signal"] + settings, rows)
    for model, mname in (("lr", "logistic regression"), ("hgb", "gradient boosting")):
        rows = []
        for dd in deltas:
            b = f"audit_d{dd}"
            rows.append([f"A + audit controls, delta = {dd} d (AP)"] + [
                _f(S[st]["incremental"][model][b]["base"]["ap"]) for st in settings])
            for k in (f"s7_forman_d{dd}", f"s7_orc_dir_d{dd}", "all_audit"):
                rows.append([f"+ `{k}`"] + [_fd(S[st]["incremental"][model][b]["plus"][k]["d_ap"]) + " / "
                                            + _fd(S[st]["incremental"][model][b]["plus"][k]["d_auroc"], 4)
                                            for st in settings])
        table(f"Table 6-{model}. AUDIT MODE incremental value, {mname}: delta AP / delta AUROC",
              ["Model"] + settings, rows)
    rows = []
    for st in settings:
        for sd, r in d["runs"][st].items():
            xc, lib = r["xcheck"], r["library_check"]
            rows.append([st, sd, str(xc["n_events"]), f"{max(xc['max_abs_diff'].values()):.1e}",
                         "-" if not lib or "skipped" in lib else
                         f"{lib['s2_escalate_radius1_max_abs_diff_vs_closed_form']:.1e} / "
                         f"{lib['s2_escalate_radius2_max_abs_diff_vs_closed_form']:.1e}",
                         "-" if not lib or "skipped" in lib else
                         f"{lib['s4a_max_abs_diff_vs_library_undirected']:.1e} / "
                         f"{lib['s4b_max_abs_diff_vs_library_augmented_forman']:.1e} / "
                         f"{lib['s7_orc_max_abs_diff_vs_library_directed']:.1e}",
                         f"{r['info']['share_events_src_uncited_at_t']:.3f}"])
    table("Table 7. Validation: fast path vs citecurv.curvature.variants (max abs difference over all checked "
          "signals) and vs GraphRicciCurvature (S2 closed form vs escalate radius 1 / 2; S4a / S4b / S7 ORC)",
          ["Setting", "seed", "events", "fast vs variants", "S2 vs library r1 / r2", "S4a / S4b / S7 vs library",
           "share of events with uncited source"], rows)
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO / "results" / "curvature_study.json"))
    ap.add_argument("--quick", action="store_true", help="small smoke configuration")
    ap.add_argument("--no-library", action="store_true", help="skip the GraphRicciCurvature check")
    ap.add_argument("--tables", metavar="JSON", help="print the doc's markdown tables from a results file and exit")
    ap.add_argument("--config", default=None,
                    help="YAML whose optional curvature_study block overrides CONFIG (default configs/default.yaml)")
    args = ap.parse_args(argv)
    if args.tables:
        print(markdown_tables(args.tables))
        return 0
    cfg = dict(CONFIG)
    block = (load_config(args.config) or {}).get("curvature_study") or {}
    unknown = set(block) - set(CONFIG)
    if unknown:
        raise ValueError(f"unknown curvature_study key(s): {sorted(unknown)}")
    cfg.update(block)
    if args.quick:
        cfg.update(QUICK)
    cfg["settings_all"] = list(CONFIG["settings"])
    t0, c0 = time.perf_counter(), time.process_time()
    runs = {s: {} for s in cfg["settings"]}
    stream_info = {}
    for seed in cfg["seeds"]:
        stream = generate_citation_stream(n_works=cfg["n_works"], seed=seed)
        cutoff = choose_cutoff(stream["events"]["t"].to_numpy(), cfg["test_share"])
        stream_info[str(seed)] = {"n_works": cfg["n_works"], "n_natural_events": int(len(stream["events"])),
                                  "cutoff_t": cutoff, "test_share_natural": round(float(
                                      np.mean(stream["events"]["t"].to_numpy() > cutoff)), 4)}
        for setting in cfg["settings"]:
            res = run_one(seed, setting, stream, cutoff, cfg,
                          do_library=(seed == cfg["seeds"][0] and not args.no_library))
            runs[setting][str(seed)] = res
            print(f"seed {seed} {setting}: {res['info']['n_test_pos']} test pos, "
                  f"{res['info']['seconds_total']} s (expensive {res['info']['seconds_expensive_signals']} s)",
                  flush=True)
    try:
        import ot
        pot_version = ot.__version__
    except ImportError:
        pot_version = None
    try:
        import GraphRicciCurvature
        grc_version = GraphRicciCurvature.__version__
    except ImportError:
        grc_version = None
    out = {
        "synthetic": True,
        "banner": "SYNTHETIC DATA - planted cartels on the citecurv synthetic stream; not a real-data result",
        "doc": "docs/curvature_formulation_study.md",
        "command": "PYTHONPATH=src nice -n 10 python scripts/curvature_study.py" + (" --quick" if args.quick else ""),
        "config": {k: v for k, v in cfg.items() if k != "settings_all"},
        "obviousness_presets": {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in p.items()}
                                for k, p in OBVIOUSNESS.items()},
        "signals": {"controls_A": {k: v[1] for k, v in CONTROLS_A.items()},
                    "controls_B_extra": {k: v[1] for k, v in CONTROLS_B_EXTRA.items()},
                    "curvature_event_time": CURV,
                    "audit": {**{k: v for d in cfg["audit_deltas"] for k, v in audit_names(d).items()},
                              **{k: v[1] for d in cfg["audit_deltas"] for k, v in audit_controls(d).items()}}},
        "streams": stream_info,
        "summary": summarise(runs, cfg),
        "runs": {st: {sd: {k: r[k] for k in ("info", "xcheck", "library_check")} for sd, r in v.items()}
                 for st, v in runs.items()},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "POT": pot_version,
                        "GraphRicciCurvature": grc_version},
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "cpu_seconds_main_process": round(time.process_time() - c0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        dump_json(out, fh)
        fh.write("\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB) in {out['wall_seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

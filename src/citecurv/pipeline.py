"""End-to-end scoring pipeline: split, curvature, components, combiner, scores, metrics (S3.4, S4.2, S4.3).

``run_pipeline(events, works, ...)`` performs one run of one ablation cell (``variant``) with one seed:

1. Canonical event order (t, event_id); time split (``eval.split``): test = t > cutoff, val = last
   ``val_frac`` of the pre-cutoff span, train = the rest.
2. Curvature (skipped by ``no_curvature``): Forman screen at event times (``curvature.screen``), two-sided
   percentile thresholds fitted on TRAIN rows and frozen, applied to every row; then directed ORC on the
   escalated rows (``curvature.orc.escalate``; ``orc_fn`` is injectable so tests run without
   GraphRicciCurvature). ``forman_only`` stops after the screen (no escalation). ``ollivier_only`` replaces
   the screen's escalation set by ``components.matched_random_escalation`` (same size, seeded), so no Forman
   value chooses which events get ORC.
3. Components (``scoring.components.score_events``), variant semantics in docs/score_components.md section 8.
4. ``ZStats`` fitted on train rows that are not cold-start, frozen, applied to every row.
5. Weights. ``calibrate_weights`` on VAL rows when labels are given and val holds a positive that is not
   cold-start; otherwise (or with ``pipeline.calibrate_weights`` false) the fixed weights of
   ``scoring.weights``. Which one was used is recorded (``weights_source``). Disabled components (all-NaN, or
   listed by the variant) get weight 0 and the remaining weights are renormalised; calibration then searches
   only the simplex of the enabled components. The ``full_learned`` cell (L13) replaces the weights by
   ``LearnedCombiner`` fitted on the split named by ``pipeline.learned_fit_split`` (default val, the same
   supervision the calibration uses); its log-odds coefficients are reported in ``weights``.

Sanity gates for S3.4 / S4.3 (``gate_*``) and ``orc_memo_from_cache`` (reuse ORC values cached by the curvature
stage) sit at the end of the module.
6. ``combine``: total = sum_c w_c z_c; a NaN component contributes 0. Cold-start rows get total = NaN and are
   excluded from fitting, calibration and metrics (CONTRACTS.md).
7. Scores frame per the scores.parquet contract, stamped (model_variant, seed, config_hash); written with
   ``aggregate.write_scores`` when ``out_dir`` is given, next to metrics.json, config.json, weights.json,
   zstats.json and thresholds.json.
8. Metrics on TEST rows through ``eval.metrics.evaluate`` (the one metric code path); val metrics and
   per-component test metrics for the "which component carries the signal" check.

The ``no_temporal`` cell (decision D11, PROVISIONAL until Maor's review). It removes every time-derived input,
mirroring ``no_curvature``, which removes every curvature-derived input: (a) no TGN memory, so s_embed runs in
``static_cosine`` mode and s_curvdyn reads static endpoint vectors even when TGN outputs are given, and (b) s_temp
is switched off (all-NaN, weight 0). Part (b) is the flag ``pipeline.no_temporal_drop_s_temp`` (default True).
With the flag False the cell is the note's encoder-only reading; in a run whose s_embed is already static (the
synthetic rehearsal) that reading leaves nothing to ablate and reproduces ``full`` exactly. Event-time slicing of
the graph (curvature at t, references at ts < t) is kept in every cell, because removing it would leak future
events into features.

Leak argument (NFR3). This module adds no feature of its own; it wires functions whose leak arguments are in
their docstrings and fixes what each fitted object may see: screen thresholds and ZStats read train rows only;
the s_curvdyn regressor is fitted on train rows inside ``score_events``; weights read val labels (or train/val
labels for the learned head), never test labels; test labels enter only ``evaluate``. The split is by time, so
changing any test-period event cannot change thresholds, ZStats, the regressor or the weights
(tests/test_pipeline_integration.py checks this by shuffling test timestamps). The ORC memo (``orc_memo``) only
caches values computed earlier for the same (event_id, src_id, dst_id, t, radius) on the same stream; a value
depends on graph state with ts <= t, so it is valid exactly when the stream up to t is unchanged, which the
caller guarantees (it is used across variants of one stream, and for pre-cutoff rows of a stream whose test
period was shuffled).

Determinism (NFR4): every random draw is seeded from ``seed`` (matched random escalation, optional ORC cap
subsample, s_curvdyn PCA subsample); no global RNG is used. Same inputs, config and seed give byte-identical
scores.parquet.
"""

from __future__ import annotations

import copy
import datetime as _dt
import math
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from citecurv.eval.ablation import VARIANTS, write_json, write_run
from citecurv.scoring.aggregate import SCORES_COLUMNS, config_hash as _config_hash, write_scores
from citecurv.scoring.combiner import COMPONENTS, LearnedCombiner, ZStats, calibrate_weights, combine

EVENT_COLUMNS = ["event_id", "src_id", "dst_id", "t"]
EPOCH = _dt.date(1970, 1, 1)

# Pipeline knobs that have no home in configs/default.yaml yet (proposed block `pipeline:`).
PIPELINE_DEFAULTS = {
    "screen_percentile_low": 0.01,   # used when curvature.screen_percentile_low is null (S2.3 not closed)
    "screen_percentile_high": 0.99,  # used when curvature.screen_percentile_high is null
    "orc_radius": 1,                 # used when curvature.orc_radius is null; 1 is exact for uncited citing papers
    "orc_max_events": None,          # cap on ORC computations per run (seeded subsample); None = all escalated rows
    "test_share": 0.2,               # cutoff rule when neither a cutoff nor labels are available
    "no_temporal_drop_s_temp": True,  # D11: no_temporal also switches s_temp off (PROVISIONAL)
    "learned_fit_split": "val",      # L13 learned head: "val" | "train" | "train+val"
    "calibrate_weights": True,       # False: always the fixed scoring.weights (sensitivity setting)
    "with_inputs": False,            # keep the in_* trace columns of score_events in the components frame
}


@dataclass
class RunResult:
    """Everything one pipeline run produced. ``scores`` follows the scores.parquet contract (stamp columns
    included, extra breakdown columns after them); ``split`` is aligned with the rows of ``scores``."""

    variant: str
    seed: int
    config: dict
    config_hash: str
    data_hash: str
    cutoff: int
    split: pd.Series
    curvature: pd.DataFrame | None
    thresholds: dict | None
    components: pd.DataFrame
    zstats: ZStats
    weights: dict
    weights_source: str
    learned: LearnedCombiner | None
    scores: pd.DataFrame
    metrics: dict
    out_dir: Path | None = None
    stats: dict = field(default_factory=dict)


# ----------------------------------------------------------------- configuration


def _default_cfg():
    from citecurv.config import load_config

    return load_config()


def resolve_config(cfg=None, *, variant="full", seed=0) -> dict:
    """The run configuration actually used (stored as config.json; NFR9/NFR10).

    cfg: a loaded configs/*.yaml dict (None = configs/default.yaml). Only the keys the pipeline reads are
    copied, with null values replaced by PIPELINE_DEFAULTS. The cutoff is added by run_pipeline.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    base = _default_cfg() if cfg is None else copy.deepcopy(cfg)
    pipe = {**PIPELINE_DEFAULTS, **(base.get("pipeline") or {})}
    if pipe["learned_fit_split"] not in ("val", "train", "train+val"):
        raise ValueError("pipeline.learned_fit_split must be 'val', 'train' or 'train+val'")
    curv = base.get("curvature") or {}
    sc = base.get("scoring") or {}
    comp = sc.get("components") or {}
    ev = base.get("eval") or {}
    split = ev.get("split") or {}
    spec = VARIANTS[variant]

    def pick(value, default):
        return default if value is None else value

    return {
        "variant": variant,
        "seed": int(seed),
        "variant_spec": spec.as_dict(),
        "pipeline": pipe,
        "split": {
            "cutoff_date": ev.get("cutoff_date"),
            "val_frac": float(pick(split.get("val_frac"), 0.10)),
            "min_test_pos": int(pick(split.get("min_test_pos"), 100)),
            "min_test_share": float(pick(split.get("min_test_share"), 0.15)),
        },
        "curvature": {
            "screen_percentile_low": float(pick(curv.get("screen_percentile_low"), pipe["screen_percentile_low"])),
            "screen_percentile_high": float(pick(curv.get("screen_percentile_high"),
                                                 pipe["screen_percentile_high"])),
            "screen_inclusive_ties": bool(pick(curv.get("screen_inclusive_ties"), False)),
            "screen_count_duplicates": bool(pick(curv.get("screen_count_duplicates"), False)),
            "orc_radius": int(pick(curv.get("orc_radius"), pipe["orc_radius"])),
            "orc_subgraph_edge_cap": int(pick(curv.get("orc_subgraph_edge_cap"), 5000)),
            "orc_prune": bool(pick(curv.get("orc_prune"), True)),
            "orc_max_events": pipe["orc_max_events"],
            "orc": dict(curv.get("orc") or {"alpha": 0.5, "method": "OTDSinkhornMix", "nbr_topk": 3000,
                                             "proc": 1, "shortest_path": "pairwise"}),
        },
        "scoring": {
            "cold_start_k": int(pick(sc.get("cold_start_k"), 3)),
            "weights": {c: float((sc.get("weights") or {}).get(c, 0.25)) for c in COMPONENTS},
            "zstats": {"robust": False, "std_floor": 1e-6, "exclude_cold_start": True, **(sc.get("zstats") or {})},
            "calibration": {"step": 0.1, "metric": "ap", **(sc.get("calibration") or {})},
            "learned_combiner": {"C": 1.0, "class_weight": "balanced", "random_state": 0, "max_iter": 1000,
                                 **(sc.get("learned_combiner") or {})},
            "components": {
                "embed_mode": pick(comp.get("embed_mode"), "auto"),
                "window_days": int(pick(comp.get("window_days"), 90)),
                "min_cell": int(pick(comp.get("min_cell"), 30)),
                "curvdyn_pca_dim": int(pick(comp.get("curvdyn_pca_dim"), 32)),
                "curvdyn_ridge_alpha": float(pick(comp.get("curvdyn_ridge_alpha"), 1.0)),
                "curvdyn_max_fit_rows": int(pick(comp.get("curvdyn_max_fit_rows"), 500_000)),
                "p_link_eps": float(pick(comp.get("p_link_eps"), 1e-7)),
            },
        },
        "eval": {"top_k": [int(k) for k in pick(ev.get("top_k"), [100, 1000])]},
    }


def run_config_hash(rcfg: dict) -> str:
    """config_hash of a resolved run config, over everything except variant and seed (they are separate
    stamp columns), so all cells of one ablation table share one hash when they share one configuration.
    pipeline.with_inputs is left out too: it only adds trace columns to the components frame."""
    hashed = {k: v for k, v in rcfg.items() if k not in ("variant", "seed", "variant_spec")}
    hashed["pipeline"] = {k: v for k, v in hashed["pipeline"].items() if k != "with_inputs"}  # trace only
    return _config_hash(hashed)


def data_hash(events, event_labels=None) -> str:
    """First 12 hex chars of sha256 over the canonical events (and labels): which data a run saw."""
    ev = _canonical_events(events)
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(ev[EVENT_COLUMNS], index=False).to_numpy().tobytes())
    if event_labels is not None:
        lab = event_labels[["event_id", "label"]].sort_values("event_id", kind="mergesort")
        h.update(pd.util.hash_pandas_object(lab.astype("int64"), index=False).to_numpy().tobytes())
    return h.hexdigest()[:12]


# ----------------------------------------------------------------- inputs


def _canonical_events(events) -> pd.DataFrame:
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events is missing column(s) {missing}; need {EVENT_COLUMNS}")
    ev = pd.DataFrame({
        "event_id": events["event_id"].to_numpy(dtype=np.int64),
        "src_id": events["src_id"].astype(str).to_numpy(dtype=object),
        "dst_id": events["dst_id"].astype(str).to_numpy(dtype=object),
        "t": events["t"].to_numpy(dtype=np.int64),
    })
    if ev["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    return ev.sort_values(["t", "event_id"], kind="mergesort").reset_index(drop=True)


def label_vector(event_ids, event_labels) -> np.ndarray:
    """0/1 per event id; ids missing from event_labels count as 0 (labels may list positives only)."""
    if event_labels is None:
        return np.zeros(len(event_ids), dtype=np.int64)
    if event_labels["event_id"].duplicated().any():
        raise ValueError("event_labels has duplicate event_id values")
    lab = pd.Series(event_labels["label"].to_numpy(), index=event_labels["event_id"].to_numpy(dtype=np.int64))
    y = lab.reindex(np.asarray(event_ids, dtype=np.int64)).fillna(0).to_numpy()
    if y.size and not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return y.astype(np.int64)


def cutoff_by_share(events, test_share) -> int:
    """Latest event time c such that events with t > c make up at least test_share of all events."""
    t = np.sort(events["t"].to_numpy(dtype=np.int64))
    cands = np.unique(t)[:-1]
    if cands.size == 0:
        raise ValueError("need at least two distinct event times to split")
    share = (t.size - np.searchsorted(t, cands, side="right")) / t.size
    ok = cands[share >= test_share]
    return int(ok.max()) if ok.size else int(cands.min())


def resolve_cutoff(events, event_labels, rcfg, cutoff=None) -> tuple[int, str]:
    """Cutoff (days since epoch) and the rule that produced it: argument, config date, choose_cutoff on event
    labels (positive_unit "event"; real-data runs freeze a paper-unit cutoff in S4.1 and pass it in), or
    the test-share rule."""
    if cutoff is not None:
        return int(cutoff), "argument"
    date = rcfg["split"]["cutoff_date"]
    if date:
        return (_dt.date.fromisoformat(str(date)) - EPOCH).days, "config eval.cutoff_date"
    if event_labels is not None:
        from citecurv.eval.split import choose_cutoff

        try:
            c = choose_cutoff(events, event_labels, min_test_pos=rcfg["split"]["min_test_pos"],
                              min_test_share=rcfg["split"]["min_test_share"], positive_unit="event")
            return int(c), "choose_cutoff (event unit)"
        except ValueError:
            pass
    return cutoff_by_share(events, rcfg["pipeline"]["test_share"]), "test_share"


# ----------------------------------------------------------------- curvature stage


def _memo_key(row_src, row_dst, eid, t, radius):
    return (int(eid), str(row_src), str(row_dst), int(t), int(radius))


def curvature_stage(events, split, rcfg, *, seed=0, orc_fn=None, store=None, orc_memo=None):
    """Curvature cache for the run's variant (None for no_curvature) plus thresholds and stage statistics.

    events: canonical events frame; split: array of 'train'/'val'/'test' aligned with it. Returns
    (curvature frame | None, thresholds | None, stats dict).
    """
    from citecurv.curvature.orc import escalate
    from citecurv.curvature.screen import apply_screen, forman_screen, screen_thresholds
    from citecurv.curvature.timeslice import EventStore
    from citecurv.scoring.components import matched_random_escalation

    mode = rcfg["variant_spec"]["curvature"]
    if mode == "none":
        return None, None, {"mode": "none"}
    cc = rcfg["curvature"]
    t0 = time.perf_counter()
    curv = forman_screen(events, count_duplicates=cc["screen_count_duplicates"])
    if not np.array_equal(curv["event_id"].to_numpy(), events["event_id"].to_numpy()):
        raise RuntimeError("forman_screen returned rows out of canonical order")
    train = np.asarray(split) == "train"
    th = screen_thresholds(curv, train, cc["screen_percentile_low"], cc["screen_percentile_high"],
                           inclusive=cc["screen_inclusive_ties"])
    curv = apply_screen(curv, th)
    n_screen = int(curv["escalated"].sum())
    stats = {"mode": mode, "n_events": int(len(curv)), "n_screen_flagged": n_screen,
             "screen_rate": n_screen / max(len(curv), 1), "screen_s": time.perf_counter() - t0}
    if mode == "screen":
        curv["escalated"] = False
    elif mode == "random+orc":
        ids = matched_random_escalation(curv, seed=seed)
        curv["escalated"] = curv["event_id"].isin(ids).to_numpy()
    esc = np.flatnonzero(curv["escalated"].to_numpy())
    stats["n_selected"] = int(len(esc))
    cap = cc["orc_max_events"]
    if cap is not None and len(esc) > int(cap):
        keep = np.sort(np.random.default_rng(seed).choice(esc, size=int(cap), replace=False))
        mask = np.zeros(len(curv), dtype=bool)
        mask[keep] = True
        curv["escalated"] = mask
        stats["orc_subsampled"] = True
    else:
        stats["orc_subsampled"] = False
    if mode in ("screen+orc", "random+orc"):
        radius = cc["orc_radius"]
        t1 = time.perf_counter()
        esc = np.flatnonzero(curv["escalated"].to_numpy())
        keys = [_memo_key(s, d, e, t, radius) for s, d, e, t in zip(
            curv["src_id"].to_numpy()[esc], curv["dst_id"].to_numpy()[esc],
            curv["event_id"].to_numpy()[esc], curv["t"].to_numpy()[esc])]
        n_hit = 0
        if orc_memo:
            o_col, r_col, s_col = (curv.columns.get_loc(c) for c in ("orc", "radius", "computed_at_stage"))
            for i, key in zip(esc, keys):
                if key in orc_memo:  # same output as a fresh computation: value, radius, stage
                    curv.iat[i, o_col] = orc_memo[key]
                    curv.iat[i, r_col] = float(radius)
                    curv.iat[i, s_col] = "escalate"
                    n_hit += 1
        store = store if store is not None else EventStore(events)
        curv, est = escalate(events, curv, store, radius, cc["orc_subgraph_edge_cap"], prune=cc["orc_prune"],
                             orc_fn=orc_fn, stage="escalate", **cc["orc"])
        if orc_memo is not None:
            vals = curv["orc"].to_numpy(dtype=float)[esc]
            for key, v in zip(keys, vals):
                if np.isfinite(v):
                    orc_memo[key] = float(v)
        stats.update({"n_escalated": int(len(esc)), "n_orc_memo_hits": n_hit,
                      "n_orc_computed": int(est["n_computed"]),
                      "orc_s": time.perf_counter() - t1,
                      "orc_s_per_computed": est["total_s_per_event"],
                      "mean_subgraph_edges": est["mean_subgraph_edges"],
                      "n_capped": est["n_capped"], "n_capped_lossy": est["n_capped_lossy"]})
    else:
        stats.update({"n_escalated": 0, "n_orc_memo_hits": 0, "n_orc_computed": 0})
    return curv, th, stats


# ----------------------------------------------------------------- combiner stage


def _sub_zstats(zs: ZStats, comps) -> ZStats:
    comps = tuple(comps)
    return ZStats(comps, {c: zs.center[c] for c in comps}, {c: zs.scale[c] for c in comps},
                  {c: zs.n_fit[c] for c in comps}, zs.robust, zs.std_floor)


def fixed_weights(cfg_weights: dict, enabled) -> dict:
    """Config weights restricted to the enabled components and renormalised on the simplex (uniform over
    the enabled components when their config weights sum to 0); disabled components get 0."""
    w = {c: (max(float(cfg_weights.get(c, 0.0)), 0.0) if c in enabled else 0.0) for c in COMPONENTS}
    s = sum(w.values())
    if not enabled:
        return w
    if s <= 0:
        return {c: (1.0 / len(enabled) if c in enabled else 0.0) for c in COMPONENTS}
    return {c: v / s for c, v in w.items()}


def _rows(split, names):
    return np.isin(np.asarray(split), list(names))


# ----------------------------------------------------------------- the run


def run_pipeline(events, works, *, authorships=None, embeddings=None, event_labels=None, cutoff=None,
                 variant="full", seed=0, cfg=None, tgn_scores=None, tgn_embeddings=None, orc_fn=None,
                 out_dir=None, orc_memo=None, store=None, tags=None) -> RunResult:
    """One run of one ablation cell; see the module docstring for the stages and their order.

    events: CONTRACTS.md events frame (event_id, src_id, dst_id, t; extra columns ignored). works: work_id +
    field (or concepts). authorships: accepted for interface symmetry with the baselines (no component reads
    it yet). embeddings: static node embeddings (ids, vectors), used by s_embed in static_cosine mode and by
    s_curvdyn when no TGN embeddings are given. event_labels: event_id, label (0/1; ids not listed count as
    0); needed for calibration and metrics, optional otherwise. cutoff: days since epoch (test = t > cutoff);
    None -> resolve_cutoff. variant: a key of eval.ablation.VARIANTS. cfg: loaded YAML dict (None = default).
    tgn_scores / tgn_embeddings: TGN outputs (CONTRACTS.md); ignored by no_temporal. orc_fn: ORC callable
    (G, edges, **knobs) -> {(u, v): kappa}; None = GraphRicciCurvature. out_dir: write the run artifacts
    there. orc_memo: dict reused across runs on the same stream (module docstring). store: a prebuilt
    EventStore of the same events. tags: free-form dict copied into metrics.json (e.g. level, data seed).

    Returns RunResult. Raises ValueError for an unknown variant or malformed inputs.
    """
    from citecurv.eval.metrics import evaluate, ranking_metrics
    from citecurv.eval.split import split_counts, split_events
    from citecurv.scoring import components as C

    t_run = time.perf_counter()
    timings = {}
    rcfg = resolve_config(cfg, variant=variant, seed=seed)
    spec = rcfg["variant_spec"]
    pipe = rcfg["pipeline"]
    ev = _canonical_events(events)
    y = label_vector(ev["event_id"].to_numpy(), event_labels)
    labels_df = pd.DataFrame({"event_id": ev["event_id"].to_numpy(), "label": y})

    # 1. split
    c_val, c_rule = resolve_cutoff(ev, labels_df if event_labels is not None else None, rcfg, cutoff)
    split = split_events(ev, c_val, val_frac=rcfg["split"]["val_frac"]).to_numpy(dtype=object)
    rcfg["split"].update({"cutoff_t": int(c_val), "cutoff_rule": c_rule,
                          "cutoff_date_resolved": (EPOCH + _dt.timedelta(days=int(c_val))).isoformat()})
    train = split == "train"
    if not train.any():
        raise ValueError("the split leaves no training rows")
    chash = run_config_hash(rcfg)
    dhash = data_hash(ev, labels_df if event_labels is not None else None)

    # 2. curvature
    t0 = time.perf_counter()
    curv, thresholds, curv_stats = curvature_stage(ev, split, rcfg, seed=seed, orc_fn=orc_fn, store=store,
                                                   orc_memo=orc_memo)
    timings["curvature_s"] = time.perf_counter() - t0

    # 3. components
    t0 = time.perf_counter()
    cc = rcfg["scoring"]["components"]
    use_tgn = spec["temporal_encoder"]
    comp = C.score_events(
        ev, works, curv, embeddings=embeddings,
        tgn_scores=tgn_scores if use_tgn else None, tgn_embeddings=tgn_embeddings if use_tgn else None,
        train_mask=train, mode=cc["embed_mode"] if use_tgn else "auto", variant=spec["component_variant"],
        k=rcfg["scoring"]["cold_start_k"], window_days=cc["window_days"], min_cell=cc["min_cell"],
        pca_dim=cc["curvdyn_pca_dim"], ridge_alpha=cc["curvdyn_ridge_alpha"],
        max_fit_rows=cc["curvdyn_max_fit_rows"], p_link_eps=cc["p_link_eps"], seed=seed,
        with_inputs=bool(pipe["with_inputs"]))
    if not np.array_equal(comp["event_id"].to_numpy(), ev["event_id"].to_numpy()):
        raise RuntimeError("score_events returned rows out of canonical order")
    disabled = list(comp.attrs.get("disabled_components", []))
    drop_temp = spec["drop_s_temp"] if spec["drop_s_temp"] is not None else bool(pipe["no_temporal_drop_s_temp"])
    if drop_temp and "s_temp" not in disabled:
        comp["s_temp"] = np.nan
        disabled.append("s_temp")
    embed_mode = comp.attrs.get("embed_mode")
    timings["components_s"] = time.perf_counter() - t0

    # 4. frozen z-statistics on train rows
    zc = rcfg["scoring"]["zstats"]
    zs = ZStats.fit(comp, COMPONENTS, split=split, robust=zc["robust"], std_floor=zc["std_floor"],
                    exclude_cold_start=zc["exclude_cold_start"])
    enabled = [c for c in COMPONENTS if c not in disabled and zs.n_fit[c] > 0]
    notes = list(comp.attrs.get("notes", []))
    for c in COMPONENTS:
        if c not in disabled and c not in enabled:
            notes.append(f"{c} has no finite training value; weight 0")
    if not enabled:
        raise ValueError("no component has a finite training value; nothing to combine")
    zs_en = _sub_zstats(zs, enabled)

    # 5. weights (or the learned head)
    t0 = time.perf_counter()
    cold = comp["cold_start"].to_numpy(dtype=bool)
    val = (split == "val") & ~cold
    have_labels = event_labels is not None
    n_val_pos = int(y[val].sum())
    calibration = None
    learned = None
    weights_source = "fixed_config"
    weights = fixed_weights(rcfg["scoring"]["weights"], enabled)
    if spec["combiner"] == "learned":
        fit_on = {"val": ("val",), "train": ("train",), "train+val": ("train", "val")}[pipe["learned_fit_split"]]
        rows = _rows(split, fit_on) & ~cold
        if have_labels and 0 < int(y[rows].sum()) < int(rows.sum()):
            lc_cfg = rcfg["scoring"]["learned_combiner"]
            learned = LearnedCombiner(zs_en, C=lc_cfg["C"], class_weight=lc_cfg["class_weight"],
                                      random_state=lc_cfg["random_state"], max_iter=lc_cfg["max_iter"])
            learned.fit(comp.loc[rows], y[rows])
            weights_source = f"learned_{pipe['learned_fit_split']}"
        else:
            notes.append(f"learned head not fitted: no positive (or no negative) {'+'.join(fit_on)} rows; "
                         "fixed config weights used")
    elif not pipe["calibrate_weights"]:
        notes.append("calibration switched off (pipeline.calibrate_weights = false): fixed config weights")
    elif have_labels and n_val_pos > 0:
        cal = rcfg["scoring"]["calibration"]
        w_en, best, table = calibrate_weights(comp.loc[val], y[val], zs_en, step=cal["step"],
                                              metric=cal["metric"])
        weights = {c: float(w_en.get(c, 0.0)) for c in COMPONENTS}
        weights_source = "calibrated_val"
        calibration = {"metric": cal["metric"], "best": best, "n_grid": int(len(table)),
                       "n_val_rows": int(val.sum()), "n_val_pos": n_val_pos}
    timings["weights_s"] = time.perf_counter() - t0

    # 6. combine
    scored = combine(comp, weights, zs)
    z_raw = zs.z_matrix(comp)
    z = np.where(np.isnan(z_raw), 0.0, z_raw)
    if learned is not None:
        weights = {c: float(learned.coef.get(c, 0.0)) for c in COMPONENTS}  # log-odds coefficients, not a simplex
        scored["total"] = learned.predict_score(comp)
        scored["n_components_used"] = (~np.isnan(z_raw[:, [COMPONENTS.index(c) for c in enabled]])).sum(axis=1)
    contrib_w = np.array([weights[c] for c in COMPONENTS])
    scored.loc[cold, "total"] = np.nan

    # 7. scores frame (contract columns, stamp, extras)
    stamp = {"model_variant": variant, "seed": int(seed), "config_hash": chash}
    extras = [f"z_{c}" for c in COMPONENTS] + ["n_components_used"]
    scores = scored[[c for c in SCORES_COLUMNS if c not in stamp] + extras].copy()
    for k, v in stamp.items():
        scores[k] = v
    scores["split"] = split
    scores = scores[SCORES_COLUMNS + extras + ["split"]]
    scores = scores.astype({"event_id": "int64", "t": "int64", "seed": "int64", "cold_start": bool,
                            "n_components_used": "int64"})

    # 8. metrics
    ks = tuple(rcfg["eval"]["top_k"])
    metrics = {
        "kind": "model", "method": variant, "variant": variant, "seed": int(seed), "config_hash": chash,
        "data_hash": dhash, "tags": dict(tags or {}),
        "cutoff_t": int(c_val), "cutoff_date": rcfg["split"]["cutoff_date_resolved"], "cutoff_rule": c_rule,
        "split_counts": split_counts(ev, split, labels_df if have_labels else None),
        "n_cold_start": int(cold.sum()),
        "embed_mode": embed_mode, "disabled_components": sorted(disabled), "enabled_components": enabled,
        "weights": weights, "weights_on_simplex": learned is None, "weights_source": weights_source,
        "learned": learned.to_dict() if learned is not None else None, "calibration": calibration,
        "curvature": curv_stats, "notes": notes,
    }
    if have_labels:
        test = split == "test"
        for part, mask in (("test", test), ("val", split == "val")):
            frame = scores.loc[mask, ["event_id", "total", "cold_start"]]
            metrics[part] = evaluate(frame, labels_df, score_col="total", ks=ks)
            metrics[f"{part}_n_pos_all"] = int(y[mask].sum())
            metrics[f"{part}_n_pos_cold_start"] = int(y[mask & cold].sum())
        keep = test & ~cold
        metrics["test_components"] = {c: ranking_metrics(z[keep, i], y[keep], ks=ks)
                                      for i, c in enumerate(COMPONENTS) if c in enabled}
        metrics["test_top100_contribution"] = _top_contribution(scores.loc[keep, "total"].to_numpy(),
                                                                z[keep] * contrib_w, y[keep], k=100)
    else:
        metrics["test"] = metrics["val"] = None
    timings["total_s"] = time.perf_counter() - t_run
    metrics["timings"] = timings

    result = RunResult(variant=variant, seed=int(seed), config=rcfg, config_hash=chash, data_hash=dhash,
                       cutoff=int(c_val), split=pd.Series(split, name="split"), curvature=curv,
                       thresholds=thresholds, components=comp, zstats=zs, weights=weights,
                       weights_source=weights_source, learned=learned, scores=scores, metrics=metrics,
                       stats={"curvature": curv_stats})
    if out_dir is not None:
        out = Path(out_dir)
        write_scores(scores, out / "scores.parquet", variant, seed, chash)
        write_run(out, metrics, {**rcfg, "config_hash": chash})
        write_json(out / "weights.json", {"weights": weights, "weights_on_simplex": learned is None,
                                          "weights_source": weights_source, "calibration": calibration,
                                          "learned": metrics["learned"], "enabled": enabled,
                                          "disabled": sorted(disabled)})
        write_json(out / "zstats.json", zs.to_dict())
        write_json(out / "thresholds.json", thresholds)
        result.out_dir = out
    return result


def _top_contribution(total, contrib, y, k=100) -> dict:
    """Mean w_c * z_c per component over the top-k rows by total (stable order), and their positive share."""
    if len(total) == 0:
        return {"k": 0}
    order = np.argsort(-total, kind="stable")[:k]
    mean = contrib[order].mean(axis=0)
    return {"k": int(len(order)), "n_pos": int(y[order].sum()), "mean_total": float(total[order].mean()),
            "mean_contribution": {c: float(v) for c, v in zip(COMPONENTS, mean)}}


def scores_sha256(path) -> str:
    """sha256 of a file's bytes (the NFR4 reproducibility hash compare)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    """sha256 over a frame's values and column names (independent of the parquet writer)."""
    h = hashlib.sha256(json.dumps(list(map(str, df.columns))).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def evaluate_baseline(result: RunResult, scores: pd.DataFrame, event_labels, ks=None) -> dict:
    """evaluate() of a baseline frame (event_id, score) on exactly the scored test events of a model run
    (test split, not cold-start), in canonical tie order, so every method in a table counts the same rows
    (plan S4.4: same test events, one metric code path). Raises when a scored test event has no score."""
    from citecurv.eval.metrics import evaluate

    base = result.scores.loc[(result.split == "test").to_numpy(), ["event_id", "cold_start"]]
    df = base.merge(scores[["event_id", "score"]], on="event_id", how="left")
    if df.loc[~df["cold_start"], "score"].isna().any():
        raise ValueError(f"{int(df.loc[~df['cold_start'], 'score'].isna().sum())} scored test events have no score")
    df["score"] = df["score"].fillna(0.0)  # cold-start rows are dropped by evaluate before ranking
    y = label_vector(df["event_id"].to_numpy(), event_labels)
    labels = pd.DataFrame({"event_id": df["event_id"].to_numpy(), "label": y})
    return evaluate(df, labels, score_col="score", ks=tuple(ks or result.config["eval"]["top_k"]))


def baseline_record(result: RunResult, kind: str, method: str, test: dict, tags=None) -> dict:
    """metrics.json content for a baseline or control evaluated with evaluate_baseline (eval.ablation.write_run)."""
    return {"kind": kind, "method": method, "seed": result.seed, "test": test, "tags": dict(tags or {}),
            "data_hash": result.data_hash, "config_hash": None,
            "test_n_pos_all": result.metrics.get("test_n_pos_all"),
            "test_n_pos_cold_start": result.metrics.get("test_n_pos_cold_start")}


# ----------------------------------------------------------------- reuse of a curvature cache


def orc_memo_from_cache(curvature, radius) -> dict:
    """ORC memo from a curvature cache written by the curvature stage (notebooks/30_curvature.ipynb), so a
    pipeline run reuses expensive ORC values instead of recomputing them. Only rows with a finite orc at the
    requested radius enter. Valid only for the same event stream (module docstring)."""
    c = curvature[np.isfinite(curvature["orc"].to_numpy(dtype=float))
                  & (curvature["radius"].to_numpy(dtype=float) == float(radius))]
    return {_memo_key(s, d, e, t, radius): float(v) for s, d, e, t, v in
            zip(c["src_id"], c["dst_id"], c["event_id"], c["t"], c["orc"])}


# ----------------------------------------------------------------- sanity gates (S3.4, S4.3)


def gate(passed: bool, value, rule: str) -> dict:
    """One sanity gate record: status PASS/FAIL, the measured value, the rule in words."""
    return {"status": "PASS" if passed else "FAIL", "value": value, "rule": rule}


def shuffle_test_timestamps(events, cutoff, seed=0) -> pd.DataFrame:
    """Copy of events whose test-period timestamps (t > cutoff) are permuted among the test events (seeded).
    Pre-cutoff rows are unchanged, so every object fitted on train/val rows stays the same (book 4.2.1)."""
    ev = events.copy()
    test = (ev["t"] > cutoff).to_numpy()
    rng = np.random.default_rng(seed)
    ev.loc[test, "t"] = rng.permutation(ev.loc[test, "t"].to_numpy())
    return ev


def gate_shuffle(full: RunResult, shuffled: RunResult) -> dict:
    """PASS when the run on shuffled test timestamps kept every fitted object (thresholds, ZStats, weights)
    and its test ranking moved toward chance: |AUROC - 0.5| smaller than in the original run. AP is reported
    but not gated, because a model ranking below chance gains AP when its ranking is destroyed."""
    frozen = (shuffled.thresholds == full.thresholds and shuffled.zstats.to_dict() == full.zstats.to_dict()
              and shuffled.weights == full.weights)
    a, s = full.metrics["test"], shuffled.metrics["test"]
    toward = abs(s["auroc"] - 0.5) < abs(a["auroc"] - 0.5)
    return gate(bool(frozen and toward),
                {"ap_full": a["ap"], "ap_shuffled": s["ap"], "auroc_full": a["auroc"],
                 "auroc_shuffled": s["auroc"], "fitted_objects_identical": frozen},
                "shuffled test timestamps: AUROC moves toward 0.5; thresholds, ZStats and weights unchanged")


def gate_reproducibility(path_a, path_b) -> dict:
    """PASS when two scores.parquet files are byte-identical (NFR4 hash compare)."""
    a, b = scores_sha256(path_a), scores_sha256(path_b)
    return gate(a == b, {"sha256_a": a, "sha256_b": b},
                "rerun with the same data, config and seed gives a byte-identical scores.parquet (NFR4)")


def gate_cold_start(result: RunResult, other_metrics=()) -> dict:
    """PASS when the cold-start rule is applied consistently: the flag equals a recount from the scores frame
    (#citations of dst with ts < t below k), totals are NaN exactly on cold-start rows, the test metrics count
    exactly the scored test rows, every other method's metrics (other_metrics: dicts with a "test" entry)
    count the same rows, and the z-statistics were fitted without cold-start rows."""
    s = result.scores
    k = result.config["scoring"]["cold_start_k"]
    dst, t = s["dst_id"].to_numpy(), s["t"].to_numpy()
    order = np.lexsort((t, dst))
    ds, ts = dst[order], t[order]
    block_start = np.searchsorted(ds, ds, "left")
    same_t_before = pd.DataFrame({"d": ds, "t": ts}).groupby(["d", "t"], sort=False).cumcount().to_numpy()
    n_prior = np.empty(len(ds), dtype=np.int64)
    n_prior[order] = np.arange(len(ds)) - block_start - same_t_before
    cold = s["cold_start"].to_numpy()
    split = result.split.to_numpy()
    checks = {
        "flag_matches_recount": bool(np.array_equal(cold, n_prior < k)),
        "total_nan_iff_cold": bool(np.array_equal(np.isnan(s["total"].to_numpy()), cold)),
        "test_n_matches": result.metrics.get("test") is None
        or result.metrics["test"]["n"] == int(((split == "test") & ~cold).sum()),
        "other_methods_same_n": all(m["test"]["n"] == result.metrics["test"]["n"] for m in other_metrics),
        "zstats_exclude_cold": all(
            result.zstats.n_fit[c] == int(((split == "train") & ~cold
                                           & np.isfinite(result.components[c].to_numpy())).sum())
            for c in COMPONENTS),
    }
    value = {**checks, "n_cold_start": int(cold.sum()), "share_cold_start": float(cold.mean()),
             "test_positives_all": result.metrics.get("test_n_pos_all"),
             "test_positives_cold_start": result.metrics.get("test_n_pos_cold_start")}
    return gate(all(checks.values()), value,
                f"cold_start == (#citations of dst with ts < t) < k={k}; cold-start rows excluded from "
                "z-statistics, totals and every method's metrics")


def _field_lookup(works, field_col="field") -> pd.Series:
    w = works.drop_duplicates("work_id")
    idx = w["work_id"].astype(str).to_numpy()
    if field_col in w.columns:
        return pd.Series(w[field_col].to_numpy(), index=idx)
    first = [c if isinstance(c, str) else (c[0] if len(c) else "unknown") for c in w["concepts"]]
    return pd.Series(first, index=idx)


def gate_fairness(result: RunResult, works, *, flag_share=0.01, max_share=0.5, part="test",
                  field_col="field") -> dict:
    """D4 (S3.4a): PASS when no single (field of the citing paper, calendar year of t) cell holds more than
    max_share of the top flag_share flags among scored `part` events ("test", "val", "train" or "all"), ranked
    by total with ties in canonical order. The field rule matches the components' reference cell
    (docs/score_components.md D1)."""
    s = result.scores
    keep = ~s["cold_start"].to_numpy()
    if part != "all":
        keep &= result.split.to_numpy() == part
    sub = s.loc[keep]
    if sub.empty:
        return gate(False, {"n_flags": 0}, "no scored events")
    n_flag = max(1, math.ceil(round(flag_share * len(sub), 9)))
    top = sub.iloc[np.argsort(-sub["total"].to_numpy(), kind="stable")[:n_flag]]
    field = _field_lookup(works, field_col).reindex(top["src_id"].astype(str)).fillna("unknown").to_numpy()
    year = top["t"].to_numpy().astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    counts = pd.Series([f"{f}-{y}" for f, y in zip(field, year)]).value_counts()
    share = float(counts.iloc[0] / n_flag)
    return gate(share <= max_share, {"max_cell_share": share, "max_cell": str(counts.index[0]),
                                     "n_flags": int(n_flag), "n_cells_flagged": int(len(counts)),
                                     "flag_event_ids": [int(e) for e in top["event_id"]]},
                f"max share of one (field, year) cell among the top {flag_share:.0%} {part} flags <= {max_share}")


def gate_top_breakdown(result: RunResult, k=20, part="test") -> dict:
    """S3.4d: PASS when each of the top-k scored events has a finite z for every enabled component, so its
    named breakdown is complete (no NaN or degenerate part)."""
    s = result.scores
    keep = ~s["cold_start"].to_numpy()
    if part != "all":
        keep &= result.split.to_numpy() == part
    sub = s.loc[keep]
    top = sub.iloc[np.argsort(-sub["total"].to_numpy(), kind="stable")[:k]]
    enabled = result.metrics["enabled_components"]
    complete = np.isfinite(top[[f"z_{c}" for c in enabled]].to_numpy()).all(axis=1)
    return gate(bool(complete.all()), {"k": int(len(top)), "n_complete": int(complete.sum()), "enabled": enabled},
                f"top-{k} {part} events have a finite z for every enabled component")

"""End-to-end scoring pipeline: split, curvature, components, combiner, scores, metrics (S3.4, S4.2, S4.3).

``run_pipeline(events, works, ...)`` performs one run of one ablation cell (``variant``) with one seed:

1. Canonical event order (t, event_id); time split (``eval.split``): test = t > cutoff, val = last
   ``val_frac`` of the pre-cutoff span, train = the rest.
2. Curvature (skipped by ``no_curvature``): Forman screen at event times (``curvature.screen``), two-sided
   percentile thresholds fitted on TRAIN rows and frozen, applied to every row; then directed ORC on the
   escalated rows (``curvature.orc.escalate``; ``orc_fn`` is injectable so tests run without
   GraphRicciCurvature). ``forman_only`` stops after the screen (no escalation). ``ollivier_only`` replaces
   the screen's escalation set by ``components.matched_random_escalation`` (same size, seeded, drawn inside
   each split with that split's screen count, so test-period events cannot change which train rows get
   ORC), so no Forman value chooses which events get ORC. Mode "closed" (the cells ``full``, ``no_temporal``,
   ``full_learned`` from 2026-11-26, option A; ``closed_orc_stage``) runs the screen for its thresholds and
   ``ref_known`` and computes the directed ORC of every event in closed form from the reference overlap
   (``orc_dir``), with no escalation and no ORC library call; the old ``full`` is the record cell ``plan_full``.
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
   excluded from fitting, calibration and metrics (CONTRACTS.md). Since 2026-11-16 (Maor's decision) they get
   a score of their own in the extra column ``total_cold`` (``cold_start_scores``: the components named in
   ``scoring.cold_start_components``, default s_geom, z-normalised on cold train rows), NaN on scored rows,
   with its own metrics (``test_cold``, ``val_cold``); the two rankings are never mixed.
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
import os
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
            "screen_unknown_without_refs": bool(pick(curv.get("screen_unknown_without_refs"), False)),
            "orc_radius": int(pick(curv.get("orc_radius"), pipe["orc_radius"])),
            "orc_subgraph_edge_cap": int(pick(curv.get("orc_subgraph_edge_cap"), 5000)),
            "orc_prune": bool(pick(curv.get("orc_prune"), True)),
            "orc_max_events": pipe["orc_max_events"],
            "orc": dict(curv.get("orc") or {"alpha": 0.5, "method": "OTDSinkhornMix", "nbr_topk": 3000,
                                             "proc": 1, "shortest_path": "pairwise"}),
        },
        "scoring": {
            "cold_start_k": int(pick(sc.get("cold_start_k"), 3)),
            "cold_start_components": _cold_start_components(sc.get("cold_start_components")),
            "weights": {c: float((sc.get("weights") or {}).get(c, 0.25)) for c in COMPONENTS},
            "zstats": {"robust": False, "std_floor": 1e-6, "exclude_cold_start": True, **(sc.get("zstats") or {})},
            "calibration": {"step": 0.1, "metric": "ap", **(sc.get("calibration") or {})},
            "learned_combiner": {"C": 1.0, "class_weight": "balanced", "random_state": 0, "max_iter": 1000,
                                 **(sc.get("learned_combiner") or {})},
            "components": {
                # label of the curvature formulation behind s_geom / s_curvdyn (option A, 2026-11-26): it changes
                # run_config_hash (resume never mixes formulations) and stays out of orc_settings_hash (ORC memos
                # stay valid). Bump the string whenever the closed mode or the orc_dir component changes meaning.
                "formulation": "orc_dir_closed_v1",
                "embed_mode": pick(comp.get("embed_mode"), "auto"),
                "window_days": int(pick(comp.get("window_days"), 90)),
                "min_cell": int(pick(comp.get("min_cell"), 30)),
                "curvdyn_pca_dim": int(pick(comp.get("curvdyn_pca_dim"), 32)),
                "curvdyn_ridge_alpha": float(pick(comp.get("curvdyn_ridge_alpha"), 1.0)),
                "curvdyn_max_fit_rows": int(pick(comp.get("curvdyn_max_fit_rows"), 500_000)),
                "p_link_eps": float(pick(comp.get("p_link_eps"), 1e-7)),
                "temporal_stat": str(pick(comp.get("temporal_stat"), "rate_ratio")),
                "temporal_prior_count": float(pick(comp.get("temporal_prior_count"), 1.0)),
                "temporal_prior_days": (None if comp.get("temporal_prior_days") is None
                                        else float(comp["temporal_prior_days"])),
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


SOURCE_COLUMNS = ("via_src", "via_dst")
NO_SOURCE_NOTE = ("event_labels has no via_src / via_dst columns: no results by label source or per paper "
                  "(ledger G10)")


def label_sources(event_ids, event_labels):
    """(via_src, via_dst) 0/1 vectors per event id, or None when event_labels lacks those columns (labels
    of synthetic injections, or files written before 2026-11-17). Ids missing from event_labels count as 0."""
    if event_labels is None or any(c not in event_labels.columns for c in SOURCE_COLUMNS):
        return None
    ids = np.asarray(event_ids, dtype=np.int64)
    by_id = event_labels.set_index(event_labels["event_id"].to_numpy(dtype=np.int64))
    return tuple(by_id[c].reindex(ids).fillna(0).to_numpy().astype(np.int64) for c in SOURCE_COLUMNS)


def _views(score, y, sources, ev, rows, ks):
    """eval.metrics.label_views on the rows of a mask (None when there are no source columns or no rows)."""
    from citecurv.eval.metrics import label_views

    if sources is None or not rows.any():
        return None
    return label_views(np.asarray(score, dtype=float)[rows], y[rows], sources[0][rows], sources[1][rows],
                       ev["src_id"].to_numpy()[rows], ev["dst_id"].to_numpy()[rows], ks=ks)


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


class OrcMemoMiss(ValueError):
    """A cell run with require_orc_memo=True found escalated rows without an ORC memo value (raised before any
    EventStore is built)."""


def screen_stage(events, split, rcfg, *, context=None, return_degrees=False):
    """The Forman screen of a run: (curvature frame with `escalated` from the train-fitted thresholds, thresholds,
    stats). events: canonical events frame (order asserted); split aligned with it; context rows as in
    curvature_stage. The first half of curvature_stage, shared with precompute_matrix_orc so cells and the
    precompute select from one code path. return_degrees: also the columns in_deg_src and out_deg_dst."""
    from citecurv.curvature import screen as screen_mod

    cc = rcfg["curvature"]
    t0 = time.perf_counter()
    n_context = 0 if context is None else int(len(context))
    curv = screen_mod.forman_screen(events, count_duplicates=cc["screen_count_duplicates"],
                                    unknown_without_refs=cc["screen_unknown_without_refs"], context=context,
                                    return_degrees=return_degrees)
    if not np.array_equal(curv["event_id"].to_numpy(), events["event_id"].to_numpy()):
        raise RuntimeError("forman_screen returned rows out of canonical order (pass canonical events)")
    n_unknown = int(curv["forman"].isna().sum())  # cited work with no reference at t (screen_unknown_without_refs)
    train = np.asarray(split) == "train"
    if train.any() and not np.isfinite(curv["forman"].to_numpy(dtype=float)[train]).any():
        raise ValueError(
            "no training row has a screen value: every cited work has no reference at its event's time. With "
            "curvature.screen_unknown_without_refs on, the reference lists of the cited works must be passed as "
            f"context rows (citecurv.data.cited.reference_events); got {n_context} context rows")
    th = screen_mod.screen_thresholds(curv, train, cc["screen_percentile_low"], cc["screen_percentile_high"],
                                      inclusive=cc["screen_inclusive_ties"])
    curv = screen_mod.apply_screen(curv, th)
    n_screen = int(curv["escalated"].sum())
    stats = {"n_events": int(len(curv)), "n_screen_flagged": n_screen,
             "screen_rate": n_screen / max(len(curv), 1), "n_screen_unknown": n_unknown,
             "n_context_rows": n_context, "screen_s": time.perf_counter() - t0}
    return curv, th, stats


def select_escalation(curv, split, mode, seed, cap, *, unknown_without_refs) -> pd.DataFrame:
    """Copy of a screen frame with `escalated` as cell `mode` uses it: "screen+orc" keeps the screen's flags,
    "screen" clears them, "random+orc" takes components.matched_random_escalation (per split, drawn only among
    rows whose cited work has a known reference list when unknown_without_refs, Maor 2026-11-25); then the
    optional ORC cap (a seeded subsample). Selection stats in .attrs["selection"]: n_selected, orc_subsampled,
    n_eligible."""
    from citecurv.scoring.components import matched_random_escalation

    out = curv.copy()
    eligible = (out["ref_known"].to_numpy(dtype=bool) if unknown_without_refs
                else np.ones(len(out), dtype=bool))
    if mode in ("screen", "closed"):  # "closed" computes no ORC either (closed-form directed ORC, option A)
        out["escalated"] = False
    elif mode == "random+orc":
        ids = matched_random_escalation(out, seed=seed, strata=split, eligible=eligible)  # per split
        out["escalated"] = out["event_id"].isin(ids).to_numpy()
    elif mode != "screen+orc":
        raise ValueError(f"unknown curvature mode {mode!r}")
    esc = np.flatnonzero(out["escalated"].to_numpy())
    sel = {"n_selected": int(len(esc)), "n_eligible": int(eligible.sum())}
    if cap is not None and len(esc) > int(cap):
        keep = np.sort(np.random.default_rng(seed).choice(esc, size=int(cap), replace=False))
        mask = np.zeros(len(out), dtype=bool)
        mask[keep] = True
        out["escalated"] = mask
        sel["orc_subsampled"] = True
    else:
        sel["orc_subsampled"] = False
    out.attrs["selection"] = sel
    return out


COUPLING_KEYS = ("event_id", "src_id", "dst_id", "t")


def _checked_coupling(coupling, curv, out_deg_distinct):
    """A passed reference_overlap frame, in the order of the canonical curvature frame, or ValueError when it does
    not belong to these events and this context (other ids, endpoints or times, e.g. the shuffled stream; another
    context, seen as a reference count that differs from the screen's distinct out-degree of the cited work)."""
    need = list(COUPLING_KEYS) + ["n_refs_dst", "n_shared"]
    missing = [c for c in need if c not in coupling.columns]
    if missing:
        raise ValueError(f"coupling frame lacks column(s) {missing}: pass curvature.overlap.reference_overlap(...) "
                         "of these events")
    if len(coupling) != len(curv) or not coupling["event_id"].is_unique:
        raise ValueError(f"coupling frame has {len(coupling)} rows (ids unique: {coupling['event_id'].is_unique}), "
                         f"the events {len(curv)}")
    c = coupling.sort_values(["t", "event_id"], kind="mergesort").reset_index(drop=True)
    for col in COUPLING_KEYS:
        a, b = c[col].to_numpy(), curv[col].to_numpy()
        same = (a.astype(np.int64) == b.astype(np.int64)) if col in ("event_id", "t") else (a.astype(str) == b.astype(str))
        if not same.all():
            raise ValueError(f"coupling frame does not match the events on column {col!r} "
                             f"({int((~same).sum())} rows differ): computed on another stream?")
    k = c["n_refs_dst"].to_numpy(dtype=float)
    if not np.array_equal(k, np.asarray(out_deg_distinct, dtype=float)):
        raise ValueError("coupling frame's n_refs_dst differs from the cited works' reference counts of this run "
                         f"on {int((k != np.asarray(out_deg_distinct, dtype=float)).sum())} rows: another context?")
    return c


def closed_orc_stage(events, split, rcfg, *, seed=0, context=None, coupling=None):
    """Curvature stage of mode "closed" (option A, plan 2026-11-26-orc-dir-closed-form): the Forman screen (for
    `forman`, `ref_known` and the thresholds, which this mode writes but does not use) plus ORC_DIR_COLUMNS:

    - n_refs_dst, n_shared: curvature.overlap.reference_overlap on the same graph state and context rows;
    - orc_dir = -(1 - alpha) * (1 - n_shared / n_refs_dst) (variants.orc_directed_uncited; Proposition 2 of
      docs/curvature_formulation_study.md), NaN where n_refs_dst == 0 (unknown reference list, G13) and on
      self-loop rows; alpha = curvature.orc.alpha;
    - citing_cited_at_t = in_deg_t(src) > 0: Proposition 2's premise fails and orc_dir is the closed form, not the
      library value (Maor 2026-11-26: closed form for every row, the flag, an exact-only view in the report).

    coupling: optional precomputed reference_overlap frame (checked by _checked_coupling); computed when None.
    No ORC is computed: `escalated` False, `orc` NaN. Leak argument: every count reads rows with ts <= t without
    the scored row (screen.py and overlap.py docstrings); thresholds are fitted on train rows."""
    from citecurv.curvature import overlap as overlap_mod
    from citecurv.curvature import screen as screen_mod
    from citecurv.curvature.screen import CURVATURE_COLUMNS, ORC_DIR_COLUMNS
    from citecurv.curvature.variants import orc_directed_uncited

    cc = rcfg["curvature"]
    curv, th, stats = screen_stage(events, split, rcfg, context=context, return_degrees=True)
    stats["mode"] = "closed"
    t1 = time.perf_counter()
    in_deg = curv["in_deg_src"].to_numpy(dtype=float)
    out_distinct = curv["out_deg_dst"].to_numpy(dtype=float)
    if cc["screen_count_duplicates"]:  # the overlap counts distinct references; compare with distinct degrees
        out_distinct = screen_mod.forman_screen(events, count_duplicates=False, unknown_without_refs=False,
                                                context=context, return_degrees=True)["out_deg_dst"].to_numpy(float)
    if coupling is None:
        coupling = overlap_mod.reference_overlap(events, context=context)
    c = _checked_coupling(coupling, curv, out_distinct)
    k = c["n_refs_dst"].to_numpy(dtype=float)
    s = c["n_shared"].to_numpy(dtype=float)
    loop = curv["src_id"].to_numpy().astype(str) == curv["dst_id"].to_numpy().astype(str)
    alpha = float((cc.get("orc") or {}).get("alpha", 0.5))
    orc_dir = np.where((k > 0) & ~loop, orc_directed_uncited(s, k, alpha), np.nan)
    curv = select_escalation(curv, split, "closed", seed, cc["orc_max_events"],
                             unknown_without_refs=cc["screen_unknown_without_refs"])
    stats.update(curv.attrs["selection"])
    curv["n_refs_dst"] = c["n_refs_dst"].to_numpy(dtype=np.int64)
    curv["n_shared"] = c["n_shared"].to_numpy(dtype=np.int64)
    curv["orc_dir"] = orc_dir
    curv["citing_cited_at_t"] = in_deg > 0
    finite = np.isfinite(orc_dir)
    n_flag = int((curv["citing_cited_at_t"].to_numpy() & finite).sum())
    stats.update({"n_escalated": 0, "n_orc_memo_hits": 0, "n_orc_computed": 0, "n_orc_dir_known": int(finite.sum()),
                  "n_citing_cited_at_t": n_flag,
                  "share_citing_cited_at_t": n_flag / int(finite.sum()) if finite.any() else None,
                  "orc_dir_s": time.perf_counter() - t1})
    return curv[list(CURVATURE_COLUMNS) + list(ORC_DIR_COLUMNS)], th, stats


def curvature_stage(events, split, rcfg, *, seed=0, orc_fn=None, store=None, orc_memo=None, context=None,
                    require_orc_memo=False, coupling=None):
    """Curvature cache for the run's variant (None for no_curvature) plus thresholds and stage statistics.

    events: canonical events frame; split: array of 'train'/'val'/'test' aligned with it. context: optional
    rows (event_id, src_id, dst_id, t; ids of their own) that take part in the graph and are not scored, e.g.
    the reference lists of the cited works (citecurv.data.cited.reference_events). They enter the screen's
    degrees and, when no store is passed, the EventStore behind the ORC subgraphs; a store passed in must
    already hold them. Thresholds are fitted on event rows only.

    ORC values come from orc_memo first (a key whose value is NaN is a hit: an attempted row, written as NaN).
    An EventStore is built, and escalate called, only for escalated non-self-loop rows without a memo key;
    with require_orc_memo=True such a miss raises OrcMemoMiss before anything is built. The frame returned
    holds exactly CURVATURE_COLUMNS, whether values were computed here or read from the memo. Returns
    (curvature frame | None, thresholds | None, stats dict).
    """
    from citecurv.curvature import timeslice as ts_mod
    from citecurv.curvature.orc import escalate
    from citecurv.curvature.screen import CURVATURE_COLUMNS

    mode = rcfg["variant_spec"]["curvature"]
    if mode == "none":
        return None, None, {"mode": "none"}
    if mode == "closed":  # closed-form directed ORC for every event; a passed coupling frame is used only here
        return closed_orc_stage(events, split, rcfg, seed=seed, context=context, coupling=coupling)
    cc = rcfg["curvature"]
    curv, th, stats = screen_stage(events, split, rcfg, context=context)
    stats["mode"] = mode
    n_context = stats["n_context_rows"]
    curv = select_escalation(curv, split, mode, seed, cc["orc_max_events"],
                             unknown_without_refs=cc["screen_unknown_without_refs"])
    stats.update(curv.attrs["selection"])
    if mode in ("screen+orc", "random+orc"):
        radius = cc["orc_radius"]
        t1 = time.perf_counter()
        esc = np.flatnonzero(curv["escalated"].to_numpy())
        keys = [_memo_key(s, d, e, t, radius) for s, d, e, t in zip(
            curv["src_id"].to_numpy()[esc], curv["dst_id"].to_numpy()[esc],
            curv["event_id"].to_numpy()[esc], curv["t"].to_numpy()[esc])]
        n_hit = 0
        is_loop = (curv["src_id"].to_numpy()[esc] == curv["dst_id"].to_numpy()[esc])
        hit = np.zeros(len(esc), dtype=bool)
        if orc_memo:
            o_col, r_col, s_col = (curv.columns.get_loc(c) for c in ("orc", "radius", "computed_at_stage"))
            for j, (i, key) in enumerate(zip(esc, keys)):
                if key in orc_memo:  # same output as a fresh computation: value (NaN for an attempted row), radius
                    curv.iat[i, o_col] = orc_memo[key]
                    curv.iat[i, r_col] = float(radius)
                    curv.iat[i, s_col] = "escalate"
                    hit[j] = True
                    n_hit += 1
        miss = esc[~hit & ~is_loop]
        stats.update({"n_escalated": int(len(esc)), "n_self_loops_escalated": int(is_loop.sum()),
                      "n_orc_memo_hits": n_hit})
        if len(miss) and require_orc_memo:
            first = [keys[j] for j in np.flatnonzero(~hit & ~is_loop)[:5]]
            raise OrcMemoMiss(f"{len(miss)} escalated row(s) have no ORC memo value (require_orc_memo=True); "
                              f"first keys {first}")
        if len(miss):
            if store is None:
                cols = ["src_id", "dst_id", "t"]
                store = ts_mod.EventStore(events if not n_context else pd.concat([events[cols], context[cols]],
                                                                                  ignore_index=True))
            curv, est = escalate(events, curv, store, radius, cc["orc_subgraph_edge_cap"], prune=cc["orc_prune"],
                                 orc_fn=orc_fn, stage="escalate", rows=miss, **cc["orc"])
            if orc_memo is not None:
                vals = curv["orc"].to_numpy(dtype=float)[esc]
                for key, v, was_hit in zip(keys, vals, hit):
                    if not was_hit and np.isfinite(v):
                        orc_memo[key] = float(v)
            stats.update({"n_orc_computed": int(est["n_computed"]), "n_errors": int(est["n_errors"]),
                          "orc_s_per_computed": est["total_s_per_event"],
                          "mean_subgraph_edges": est["mean_subgraph_edges"],
                          "n_capped": est["n_capped"], "n_capped_lossy": est["n_capped_lossy"]})
        else:
            stats.update({"n_orc_computed": 0, "n_errors": 0, "orc_s_per_computed": None, "mean_subgraph_edges": None,
                          "n_capped": None, "n_capped_lossy": None})
        stats["orc_s"] = time.perf_counter() - t1
    else:
        stats.update({"n_escalated": 0, "n_orc_memo_hits": 0, "n_orc_computed": 0})
    return curv[list(CURVATURE_COLUMNS)], th, stats


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


DEFAULT_COLD_START_COMPONENTS = ("s_geom",)


def _cold_start_components(value) -> list:
    """scoring.cold_start_components from a config: None -> the default (s_geom alone); a list of component
    names, kept in the given order; [] switches cold-start scoring off."""
    comps = list(DEFAULT_COLD_START_COMPONENTS if value is None else value)
    unknown = [c for c in comps if c not in COMPONENTS]
    if unknown or len(set(comps)) != len(comps):
        raise ValueError(f"scoring.cold_start_components must be distinct names from {list(COMPONENTS)}; got {comps}")
    return comps


def cold_start_scores(comp, split, cold, components, cfg_weights, disabled=(), *, robust=False, std_floor=1e-6):
    """A separate score for cold-start rows; returns (total_cold, info).

    total_cold: float array aligned with `comp`, NaN on every row that is not cold-start. On cold rows it is
    sum_c w_c * z_c over `components` (docs/score_components.md section 7): z_c uses the mean and sd of
    component c on the cold TRAIN rows (a ZStats of its own, since a component's distribution on papers
    without history differs from the scored rows), a NaN component contributes 0, and w is the config
    weights renormalised over the components in use (`fixed_weights`; no calibration on labels).
    A component is left out when it is disabled in this run or has no finite value on a cold train row;
    with none left, total_cold is all-NaN and info["note"] says why.
    info: components (in use), weights, zstats (ZStats.to_dict() or None), n_rows (cold rows), note.

    Leak argument. The statistics read component values of cold train rows only and no label; whether a
    train row is cold depends on events before it; the weights come from the config. So no validation or
    test value reaches a cold score other than the row's own component values.
    """
    cold = np.asarray(cold, dtype=bool)
    split = np.asarray(split)
    total = np.full(len(comp), np.nan)
    info = {"components": [], "weights": {}, "zstats": None, "n_rows": int(cold.sum()), "note": None}
    if not components:
        info["note"] = "cold-start scoring is off (scoring.cold_start_components is empty)"
        return total, info
    train_cold = cold & (split == "train")
    usable = [c for c in components
              if c not in disabled and np.isfinite(comp[c].to_numpy(dtype=float)[train_cold]).any()]
    if not usable:
        info["note"] = (f"cold-start rows are not scored: none of {list(components)} is enabled with a finite "
                        "value on a cold-start training row")
        return total, info
    sub = comp.loc[cold]
    zs = ZStats.fit(sub, usable, split=split[cold], robust=robust, std_floor=std_floor, exclude_cold_start=False)
    w_all = fixed_weights(cfg_weights, usable)
    weights = {c: float(w_all[c]) for c in usable}
    total[cold] = combine(sub, weights, zs)["total"].to_numpy()
    info.update(components=usable, weights=weights, zstats=zs.to_dict())
    return total, info


# ----------------------------------------------------------------- the run


def _resolve_run(events, event_labels, cutoff, variant, seed, cfg):
    """The resolved config, canonical events, labels, split cutoff and the two hashes of one run, exactly as
    run_pipeline computes them (shared with run_identity, so resume checks compare like with like)."""
    rcfg = resolve_config(cfg, variant=variant, seed=seed)
    ev = _canonical_events(events)
    y = label_vector(ev["event_id"].to_numpy(), event_labels)
    labels_df = pd.DataFrame({"event_id": ev["event_id"].to_numpy(), "label": y})
    c_val, c_rule = resolve_cutoff(ev, labels_df if event_labels is not None else None, rcfg, cutoff)
    rcfg["split"].update({"cutoff_t": int(c_val), "cutoff_rule": c_rule,
                          "cutoff_date_resolved": (EPOCH + _dt.timedelta(days=int(c_val))).isoformat()})
    chash = run_config_hash(rcfg)
    dhash = data_hash(ev, labels_df if event_labels is not None else None)
    return rcfg, ev, y, labels_df, int(c_val), c_rule, chash, dhash


def run_identity(data: dict, variant: str, seed: int, cfg=None, tags=None, context=None) -> dict:
    """What a finished run folder must match to be reused on resume: config_hash, data_hash (both as
    run_pipeline computes them), the commit tag and the context rows' hash (data_hash does not cover them).
    data: run_pipeline inputs (TGN outputs not needed)."""
    *_, chash, dhash = _resolve_run(data["events"], data.get("event_labels"), data.get("cutoff"), variant, seed, cfg)
    return {"config_hash": chash, "data_hash": dhash, "commit": (tags or {}).get("commit"),
            "context_hash": context_hash(context)}


RUN_FILES = ("config.json", "scores.parquet", "weights.json", "zstats.json", "thresholds.json", "escalated.npy")


def _file_record(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def run_pipeline(events, works, *, authorships=None, embeddings=None, event_labels=None, cutoff=None,
                 variant="full", seed=0, cfg=None, tgn_scores=None, tgn_embeddings=None, orc_fn=None,
                 out_dir=None, orc_memo=None, store=None, tags=None, context=None,
                 require_orc_memo=False, coupling=None) -> RunResult:
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
    context: rows that take part in the curvature stage's graph and are not scored (curvature_stage); no
    other stage reads them, and data_hash does not cover them (their count is in the curvature statistics).

    Returns RunResult. Raises ValueError for an unknown variant or malformed inputs.
    """
    from citecurv.eval.metrics import evaluate, ranking_metrics
    from citecurv.eval.split import split_counts, split_events
    from citecurv.scoring import components as C

    t_run = time.perf_counter()
    timings = {}
    rcfg, ev, y, labels_df, c_val, c_rule, chash, dhash = _resolve_run(events, event_labels, cutoff, variant,
                                                                       seed, cfg)
    spec = rcfg["variant_spec"]
    pipe = rcfg["pipeline"]
    sources = label_sources(ev["event_id"].to_numpy(), event_labels)

    # 1. split
    split = split_events(ev, c_val, val_frac=rcfg["split"]["val_frac"]).to_numpy(dtype=object)
    train = split == "train"
    if not train.any():
        raise ValueError("the split leaves no training rows")

    # 2. curvature
    t0 = time.perf_counter()
    curv, thresholds, curv_stats = curvature_stage(ev, split, rcfg, seed=seed, orc_fn=orc_fn, store=store,
                                                   orc_memo=orc_memo, context=context,
                                                   require_orc_memo=require_orc_memo, coupling=coupling)
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
        with_inputs=bool(pipe["with_inputs"]), temporal_stat=cc["temporal_stat"],
        temporal_prior_count=cc["temporal_prior_count"], temporal_prior_days=cc["temporal_prior_days"])
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

    # 6b. cold-start rows: their own score from the components that need no history of the cited work
    total_cold, cold_info = cold_start_scores(comp, split, cold, rcfg["scoring"]["cold_start_components"],
                                              rcfg["scoring"]["weights"], disabled, robust=zc["robust"],
                                              std_floor=zc["std_floor"])
    scored["total_cold"] = total_cold
    if cold_info["note"]:
        notes.append(cold_info["note"])

    # 7. scores frame (contract columns, stamp, extras)
    stamp = {"model_variant": variant, "seed": int(seed), "config_hash": chash}
    extras = [f"z_{c}" for c in COMPONENTS] + ["n_components_used", "total_cold"]
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
        "data_hash": dhash, "context_hash": context_hash(context), "tags": dict(tags or {}),
        "cutoff_t": int(c_val), "cutoff_date": rcfg["split"]["cutoff_date_resolved"], "cutoff_rule": c_rule,
        "split_counts": split_counts(ev, split, labels_df if have_labels else None),
        "n_cold_start": int(cold.sum()), "cold_start_scoring": cold_info,
        "embed_mode": embed_mode, "disabled_components": sorted(disabled), "enabled_components": enabled,
        "weights": weights, "weights_on_simplex": learned is None, "weights_source": weights_source,
        "learned": learned.to_dict() if learned is not None else None, "calibration": calibration,
        "curvature": curv_stats, "notes": notes,
    }
    if have_labels:
        test = split == "test"
        if sources is None:
            notes.append(NO_SOURCE_NOTE)
        total = scores["total"].to_numpy()
        for part, mask in (("test", test), ("val", split == "val")):
            frame = scores.loc[mask, ["event_id", "total", "cold_start"]]
            metrics[part] = evaluate(frame, labels_df, score_col="total", ks=ks)
            metrics[f"{part}_n_pos_all"] = int(y[mask].sum())
            metrics[f"{part}_n_pos_cold_start"] = int(y[mask & cold].sum())
            # the same list by label source and per paper (ledger G10), next to the total, never instead of it
            metrics[f"{part}_views"] = _views(total, y, sources, ev, mask & ~cold, ks)
            # cold-start rows of the part, ranked by their own score: a table of its own, never mixed with `part`
            m_cold = mask & cold
            has_cold = bool(cold_info["components"]) and bool(m_cold.any())
            metrics[f"{part}_cold"] = ranking_metrics(total_cold[m_cold], y[m_cold], ks=ks) if has_cold else None
            metrics[f"{part}_cold_views"] = _views(total_cold, y, sources, ev, m_cold, ks) if has_cold else None
        keep = test & ~cold
        metrics["test_components"] = {c: ranking_metrics(z[keep, i], y[keep], ks=ks)
                                      for i, c in enumerate(COMPONENTS) if c in enabled}
        metrics["test_top100_contribution"] = _top_contribution(scores.loc[keep, "total"].to_numpy(),
                                                                z[keep] * contrib_w, y[keep], k=100)
    else:
        for key in ("test", "val", "test_cold", "val_cold", "test_views", "val_views", "test_cold_views",
                    "val_cold_views"):
            metrics[key] = None
    timings["total_s"] = time.perf_counter() - t_run
    metrics["timings"] = timings

    result = RunResult(variant=variant, seed=int(seed), config=rcfg, config_hash=chash, data_hash=dhash,
                       cutoff=int(c_val), split=pd.Series(split, name="split"), curvature=curv,
                       thresholds=thresholds, components=comp, zstats=zs, weights=weights,
                       weights_source=weights_source, learned=learned, scores=scores, metrics=metrics,
                       stats={"curvature": curv_stats})
    if out_dir is not None:
        # every other run file first, metrics.json last with their sizes and hashes: a folder whose metrics.json
        # exists and matches its files is complete (run_matrix resume)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_scores(scores, out / "scores.parquet", variant, seed, chash)
        write_json(out / "config.json", {**rcfg, "config_hash": chash})
        write_json(out / "weights.json", {"weights": weights, "weights_on_simplex": learned is None,
                                          "weights_source": weights_source, "calibration": calibration,
                                          "learned": metrics["learned"], "enabled": enabled,
                                          "disabled": sorted(disabled)})
        write_json(out / "zstats.json", zs.to_dict())
        write_json(out / "thresholds.json", thresholds)
        esc_ids = (np.zeros(0, dtype=np.int64) if curv is None
                   else curv.loc[curv["escalated"].to_numpy(dtype=bool), "event_id"].to_numpy(dtype=np.int64))
        with open(out / "escalated.npy", "wb") as f:
            np.save(f, esc_ids)
        metrics["files"] = {name: _file_record(out / name) for name in RUN_FILES}
        write_run(out, metrics)
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


def baseline_views(result: RunResult, scores: pd.DataFrame, event_labels, ks=None):
    """eval.metrics.label_views of a baseline frame (event_id, score) on the same rows as evaluate_baseline:
    the scored test events of the model run, in canonical order (ledger G10). None when event_labels has no
    via_src / via_dst columns. Raises when a scored test event has no score."""
    keep = (result.split == "test").to_numpy() & ~result.scores["cold_start"].to_numpy(dtype=bool)
    df = result.scores.loc[keep, ["event_id", "src_id", "dst_id"]].merge(scores[["event_id", "score"]],
                                                                         on="event_id", how="left")
    if df["score"].isna().any():
        raise ValueError(f"{int(df['score'].isna().sum())} scored test events have no score")
    ids = df["event_id"].to_numpy()
    return _views(df["score"].to_numpy(), label_vector(ids, event_labels), label_sources(ids, event_labels), df,
                  np.ones(len(df), dtype=bool), tuple(ks or result.config["eval"]["top_k"]))


def baseline_record(result: RunResult, kind: str, method: str, test: dict, tags=None, views=None) -> dict:
    """metrics.json content for a baseline or control evaluated with evaluate_baseline (eval.ablation.write_run).
    views: the baseline_views result for the same frame (None when there are no source columns)."""
    return {"kind": kind, "method": method, "seed": result.seed, "test": test, "test_views": views,
            "tags": dict(tags or {}),
            "data_hash": result.data_hash, "config_hash": None,
            "test_n_pos_all": result.metrics.get("test_n_pos_all"),
            "test_n_pos_cold_start": result.metrics.get("test_n_pos_cold_start")}


# ----------------------------------------------------------------- reuse of a curvature cache


def orc_memo_from_cache(curvature, radius, include_attempted: bool = False) -> dict:
    """ORC memo from a curvature cache written by the curvature stage (notebooks/30_curvature.ipynb) or by
    precompute_matrix_orc, so a pipeline run reuses expensive ORC values instead of recomputing them. Rows with a
    finite orc at the requested radius enter; with include_attempted, also rows marked `orc_attempted` whose
    value is NaN (they enter as NaN, so a strict cell writes NaN instead of missing them; a frame without the
    column counts as none attempted). Valid only for the same event stream (module docstring)."""
    orc = curvature["orc"].to_numpy(dtype=float)
    at_r = curvature["radius"].to_numpy(dtype=float) == float(radius)
    keep = np.isfinite(orc) & at_r
    if include_attempted and "orc_attempted" in curvature.columns:
        keep |= np.isnan(orc) & curvature["orc_attempted"].fillna(False).to_numpy(dtype=bool) & at_r
    c = curvature[keep]
    return {_memo_key(s, d, e, t, radius): float(v) for s, d, e, t, v in
            zip(c["src_id"], c["dst_id"], c["event_id"], c["t"], c["orc"])}


# ----------------------------------------------------------------- ORC precompute for the matrix (S4.3)


def context_hash(context) -> str:
    """First 12 hex chars of sha256 over the canonical context rows; "none" without context rows."""
    if context is None or not len(context):
        return "none"
    c = pd.DataFrame({"event_id": context["event_id"].to_numpy(dtype=np.int64),
                      "src_id": context["src_id"].astype(str).to_numpy(dtype=object),
                      "dst_id": context["dst_id"].astype(str).to_numpy(dtype=object),
                      "t": context["t"].to_numpy(dtype=np.int64)})
    c = c.sort_values(["t", "event_id"], kind="mergesort")
    h = hashlib.sha256(pd.util.hash_pandas_object(c[EVENT_COLUMNS], index=False).to_numpy().tobytes())
    return h.hexdigest()[:12]


def orc_settings_hash(rcfg: dict) -> str:
    """Hash of the resolved curvature settings without orc.proc (worker processes change no value)."""
    cc = copy.deepcopy(rcfg["curvature"])
    cc["orc"] = {k: v for k, v in (cc.get("orc") or {}).items() if k != "proc"}
    return _config_hash(cc)


def build_context(works, refs, events):
    """Context rows of the cited works' reference lists (citecurv.data.cited.reference_events), the one place
    the notebook and the ORC job build them: the pulled works' own references are already event rows."""
    from citecurv.data.cited import reference_events

    return reference_events(refs, exclude_src=set(works["work_id"].astype(str)),
                            start_event_id=int(events["event_id"].max()) + 1)


def _write_heartbeat(path, payload: dict) -> None:
    """Write the heartbeat JSON atomically (local disk; read by the notebook's watchdog)."""
    if path is None:
        return
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


CKPT_ROW_COLUMNS = ("orc_subgraph_edges", "orc_capped", "orc_lossy", "orc_src_in_deg", "orc_attempted",
                    "orc_error")


def _ckpt_keys(frame, radius):
    return [_memo_key(s, d, e, t, radius) for s, d, e, t in
            zip(frame["src_id"], frame["dst_id"], frame["event_id"], frame["t"])]


def _sidecar_check(side: dict, want: dict, what: str, keys) -> None:
    for k in keys:
        if side.get(k) != want[k]:
            label = {"settings_hash": "settings", "data_hash": "data", "context_hash": "context"}.get(k, k)
            raise ValueError(f"{what}: stored {label} {side.get(k)!r} differs from {want[k]!r}")


def precompute_matrix_orc(events, cutoff, rcfg, seeds, *, out_path, context=None, known=None, known_max_t=None,
                          orc_fn=None, chunk=5000, batch=1, worker=None, order_seed=0, max_chunks=None,
                          heartbeat=None, freeze_gc=False, retry_errors=False, max_error_share=0.001) -> dict:
    """ORC values for every escalated row of every matrix cell, computed once, resumable (S4.3 plan, Task 3).

    events: events frame in any order (canonicalized here); cutoff: days since epoch, the split is built as
    run_pipeline builds it (val_frac of rcfg). Targets = the screen set (the cells full, no_temporal, full_learned
    of the plan formulation, i.e. the record cell plan_full since option A of 2026-11-26; the closed-mode cells
    need no ORC) united with the ollivier_only selection of every seed in `seeds` (one select_escalation code path), processed
    in a seeded random order (order_seed) so a chunk's timing is a fair running projection. An ORC cap
    (orc_max_events) is refused: its draw depends on the cell seed.

    out_path: parquet checkpoint (every row ever computed: curvature-cache columns plus the per-row ORC columns),
    written to a temporary file and moved into place after every chunk; sidecar <out_path>.json with the
    settings hash (orc_settings_hash), data_hash (events, no labels), context_hash, cutoff, radius, the union of
    the seeds and n_targets. A rerun refuses another settings, data, context, cutoff or radius, accepts a
    different seed list (fewer seeds compute nothing and drop no row) and takes ORC values by key from the file.
    known: a checkpoint path (its sidecar must match settings, context and radius; its data may differ: a
    shuffled stream) whose values fill target keys with t <= known_max_t (required with known), marked attempted.
    heartbeat: a local JSON path rewritten at each setup phase and before every ORC call ({phase, time, in_flight,
    clean}); a heartbeat left with clean false marks its in-flight rows attempted with orc_error
    "killed: timeout"; retry_errors recomputes rows with an error. The run stops (RuntimeError, before the chunk
    is written) when rows with an error exceed max_error_share of the targets. A dead ORC worker is restarted
    once. Returns a stats dict (see the plan's Interfaces block).
    """
    import gc

    from citecurv.curvature import timeslice as ts_mod
    from citecurv.curvature.orc import OrcWorker, OrcWorkerDied, escalate
    from citecurv.curvature.screen import CURVATURE_COLUMNS
    from citecurv.eval.split import split_counts, split_events

    t_start = time.perf_counter()
    cc = rcfg["curvature"]
    if cc.get("orc_max_events") is not None:
        raise ValueError("precompute_matrix_orc refuses an ORC cap (orc_max_events): its draw depends on the cell seed")
    if known is not None and known_max_t is None:
        raise ValueError("known values need known_max_t (only keys with t <= known_max_t may be reused)")
    radius = int(cc["orc_radius"])
    out_path = Path(out_path)
    side_path = out_path.with_suffix(".json")
    prev_hb = _read_json(heartbeat) if heartbeat is not None else None
    carried = (list(prev_hb.get("in_flight") or []) if prev_hb and not prev_hb.get("clean", True) else [])
    # rows in flight at a kill stay in the heartbeat until their marks are saved (a second kill must not lose them)
    hb = {"phase": "screen", "time": time.time(), "in_flight": carried, "clean": False}
    _write_heartbeat(heartbeat, hb)

    ev = _canonical_events(events)
    split = split_events(ev, int(cutoff), val_frac=rcfg["split"]["val_frac"]).to_numpy(dtype=object)
    want = {"settings_hash": orc_settings_hash(rcfg), "data_hash": data_hash(ev), "context_hash": context_hash(context),
            "cutoff": int(cutoff), "radius": radius}
    curv, _, sstats = screen_stage(ev, split, rcfg, context=context)
    uwr = cc["screen_unknown_without_refs"]
    target = select_escalation(curv, split, "screen+orc", 0, None, unknown_without_refs=uwr)["escalated"].to_numpy()
    for s in seeds:
        target = target | select_escalation(curv, split, "random+orc", int(s), None,
                                            unknown_without_refs=uwr)["escalated"].to_numpy()
    # the checkpoint frame holds the target rows only (a few percent of the events)
    frame = curv.loc[target].reset_index(drop=True)
    frame["escalated"] = True
    loop = (frame["src_id"] == frame["dst_id"]).to_numpy()
    for c, d in {"orc_subgraph_edges": np.nan, "orc_capped": False, "orc_lossy": False, "orc_src_in_deg": np.nan,
                 "orc_attempted": False, "orc_error": ""}.items():
        frame[c] = d
    frame["orc_error"] = frame["orc_error"].astype(object)
    keys = _ckpt_keys(frame, radius)
    pos_of = {k: i for i, k in enumerate(keys)}
    fill_cols = ["orc", "radius", "computed_at_stage", *CKPT_ROW_COLUMNS]

    def fill_from(src: pd.DataFrame, mask_fn=None):
        n = 0
        cols = [c for c in fill_cols if c in src.columns]
        for k, row in zip(_ckpt_keys(src, radius), src[cols].itertuples(index=False)):
            i = pos_of.get(k)
            if i is None or (mask_fn is not None and not mask_fn(k)):
                continue
            for c, v in zip(cols, row):
                frame.iat[i, frame.columns.get_loc(c)] = v
            frame.iat[i, frame.columns.get_loc("orc_attempted")] = True
            frame.iat[i, frame.columns.get_loc("radius")] = float(radius)
            n += 1
        return n

    # resume from the checkpoint (same settings, data, context, cutoff, radius)
    prior = None
    dirty = False
    killed_marked = False
    seed_union = sorted({int(s) for s in seeds})
    if out_path.exists():
        side = _read_json(side_path)
        if side is None:
            raise ValueError(f"checkpoint {out_path} has no readable sidecar {side_path}")
        _sidecar_check(side, want, f"checkpoint {out_path}", ("settings_hash", "data_hash", "context_hash", "cutoff",
                                                                "radius"))
        prior = pd.read_parquet(out_path)
        fill_from(prior)
        seed_union = sorted(set(seed_union) | {int(s) for s in side.get("seeds", [])})
    n_known = 0
    if known is not None:
        kpath = Path(known)
        kside = _read_json(kpath.with_suffix(".json"))
        if kside is None:
            raise ValueError(f"known checkpoint {kpath} has no readable sidecar")
        _sidecar_check(kside, want, f"known checkpoint {kpath}", ("settings_hash", "context_hash", "radius"))
        kframe = pd.read_parquet(kpath)
        kframe = kframe[kframe["orc_attempted"].fillna(False).astype(bool) if "orc_attempted" in kframe.columns
                        else np.isfinite(kframe["orc"].to_numpy(dtype=float))]
        att = frame["orc_attempted"].to_numpy(dtype=bool).copy()
        n_known = fill_from(kframe, mask_fn=lambda k: k[3] <= int(known_max_t) and not att[pos_of[k]])
        dirty = dirty or n_known > 0
    # rows in flight when an earlier run was killed
    if prev_hb and not prev_hb.get("clean", True) and prev_hb.get("in_flight"):
        ids = set(int(x) for x in prev_hb["in_flight"])
        hit = np.flatnonzero(frame["event_id"].isin(ids).to_numpy() & ~frame["orc_attempted"].to_numpy())
        dirty = dirty or len(hit) > 0
        killed_marked = len(hit) > 0
        for i in hit:
            frame.iat[i, frame.columns.get_loc("orc")] = np.nan
            frame.iat[i, frame.columns.get_loc("radius")] = float(radius)
            frame.iat[i, frame.columns.get_loc("computed_at_stage")] = "escalate"
            frame.iat[i, frame.columns.get_loc("orc_attempted")] = True
            frame.iat[i, frame.columns.get_loc("orc_error")] = "killed: timeout"
    if retry_errors:
        err = frame["orc_error"].astype(str).ne("").to_numpy()
        frame.loc[err, "orc_attempted"] = False
        frame.loc[err, "orc_error"] = ""
        frame.loc[err, "orc"] = np.nan

    tpos = np.flatnonzero(~loop)
    order = tpos[np.random.default_rng(order_seed).permutation(len(tpos))]
    n_targets = int(len(tpos))

    def pending():
        att = frame["orc_attempted"].to_numpy(dtype=bool)
        return order[~att[order]]

    def write_checkpoint():
        att = frame["orc_attempted"].to_numpy(dtype=bool)
        cols = list(CURVATURE_COLUMNS) + list(CKPT_ROW_COLUMNS)
        rows = frame.loc[att, cols]
        if prior is not None and len(prior):
            have = set(_ckpt_keys(rows, radius))
            extra = prior[[k not in have for k in _ckpt_keys(prior, radius)]]
            rows = pd.concat([rows, extra[[c for c in cols if c in extra.columns]]], ignore_index=True)
        tmp = out_path.with_suffix(".tmp")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
        side_tmp = side_path.with_suffix(".jtmp")
        side_tmp.write_text(json.dumps({**want, "seeds": seed_union, "n_targets": n_targets,
                                        "n_rows": int(len(rows))}, indent=1), encoding="utf-8")
        os.replace(side_tmp, side_path)

    if killed_marked:  # save the marks before the (long) store build
        write_checkpoint()
    hb["in_flight"] = []
    todo = pending()
    store = None
    n_chunks = n_done = 0
    restarted = False
    t_sub, t_orc, p95_edges, max_edges = [], [], [], []
    try:
        if len(todo):
            hb.update(phase="store", time=time.time())
            _write_heartbeat(heartbeat, hb)
            cols = ["src_id", "dst_id", "t"]
            rows_all = ev if context is None or not len(context) else pd.concat([ev[cols], context[cols]],
                                                                                 ignore_index=True)
            store = ts_mod.EventStore(rows_all)
            if freeze_gc:
                gc.freeze()

        def progress(ids):
            hb.update(phase="orc", time=time.time(), in_flight=[int(x) for x in ids])
            _write_heartbeat(heartbeat, hb)

        while len(todo) and (max_chunks is None or n_chunks < max_chunks):
            rows = todo[:chunk]
            before = int(frame["orc_attempted"].sum())
            try:
                frame_new, est = escalate(ev, frame, store, radius, cc["orc_subgraph_edge_cap"], prune=cc["orc_prune"],
                                          orc_fn=orc_fn, stage="escalate", rows=rows, batch=batch, worker=worker,
                                          progress=progress, **cc["orc"])
            except OrcWorkerDied as exc:
                if restarted or worker is None:
                    raise
                restarted = True
                frame = exc.partial_frame
                worker.close()
                worker = OrcWorker(worker.orc_fn_path, **worker.knobs)
                todo = pending()
                continue
            frame = frame_new
            n_err = int(frame["orc_error"].astype(str).ne("").sum())
            if n_err > max_error_share * max(n_targets, 1):
                raise RuntimeError(f"ORC precompute stopped: {n_err} rows with an error out of {n_targets} targets "
                                   f"(limit {max_error_share:.3%}); first: "
                                   f"{frame.loc[frame['orc_error'].astype(str).ne(''), 'orc_error'].iloc[0][:300]}")
            made = int(frame["orc_attempted"].sum()) - before
            if made == 0:
                raise RuntimeError(f"ORC precompute made no progress on a chunk of {len(rows)} rows")
            n_done += made
            n_chunks += 1
            t_sub.append(est["subgraph_s_per_event"])
            t_orc.append(est["orc_s_per_event"])
            p95_edges.append(est["p95_subgraph_edges"])
            max_edges.append(est["max_subgraph_edges"])
            write_checkpoint()
            todo = pending()
        if not out_path.exists() or (n_chunks == 0 and dirty):
            write_checkpoint()
    finally:
        if freeze_gc and store is not None:
            gc.unfreeze()
    att = frame["orc_attempted"].to_numpy(dtype=bool)
    orc = frame["orc"].to_numpy(dtype=float)
    errs = frame["orc_error"].astype(str).ne("").to_numpy()
    hb.update(phase="done", time=time.time(), in_flight=[], clean=True)
    _write_heartbeat(heartbeat, hb)

    def mean(x):
        x = [v for v in x if v is not None and np.isfinite(v)]
        return float(np.mean(x)) if x else None

    counts = split_counts(ev, split)
    return {"n_targets": n_targets, "n_known": int(n_known), "n_done": int(n_done),
            "n_pending": int(len(pending())), "n_attempted_nan": int((att & np.isnan(orc) & ~errs).sum()),
            "n_errors": int(errs.sum()), "n_chunks_run": int(n_chunks),
            "seconds": time.perf_counter() - t_start, "subgraph_s_per_event": mean(t_sub),
            "orc_s_per_event": mean(t_orc), "split_counts": counts, "n_screen_flagged": sstats["n_screen_flagged"],
            "n_screen_unknown": sstats["n_screen_unknown"],
            "screen_flagged_by_split": {part: int((curv["escalated"].to_numpy() & (split == part)).sum())
                                        for part in ("train", "val", "test")},
            "p95_subgraph_edges": max([v for v in p95_edges if np.isfinite(v)], default=None),
            "max_subgraph_edges": max(max_edges, default=None),
            "n_eligible": int(curv["ref_known"].sum()) if uwr else int(len(curv)), "seeds": seed_union}


def load_orc_memo(out_path, *, data_hash, context_hash, settings_hash, radius) -> dict:
    """The ORC memo of a precompute checkpoint, through its sidecar: refuses another data, context, settings or
    radius; attempted rows with a NaN value enter as NaN (orc_memo_from_cache include_attempted)."""
    out_path = Path(out_path)
    side = _read_json(out_path.with_suffix(".json"))
    if side is None:
        raise ValueError(f"checkpoint {out_path} has no readable sidecar")
    want = {"data_hash": data_hash, "context_hash": context_hash, "settings_hash": settings_hash, "radius": int(radius)}
    _sidecar_check(side, want, f"ORC memo {out_path}", ("data_hash", "context_hash", "settings_hash", "radius"))
    return orc_memo_from_cache(pd.read_parquet(out_path), radius, include_attempted=True)


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


def _cold_fitted(metrics) -> dict:
    """The fitted part of metrics["cold_start_scoring"]: components, weights, zstats (not the row count,
    which depends on test-period timestamps)."""
    info = metrics.get("cold_start_scoring") or {}
    return {k: info.get(k) for k in ("components", "weights", "zstats")}


def gate_shuffle(full: RunResult, shuffled: RunResult) -> dict:
    """PASS when the run on shuffled test timestamps kept every fitted object (thresholds, ZStats, weights)
    and its test ranking moved toward chance: |AUROC - 0.5| smaller than in the original run. AP is reported
    but not gated, because a model ranking below chance gains AP when its ranking is destroyed."""
    frozen = (shuffled.thresholds == full.thresholds and shuffled.zstats.to_dict() == full.zstats.to_dict()
              and shuffled.weights == full.weights
              and _cold_fitted(shuffled.metrics) == _cold_fitted(full.metrics))
    a, s = full.metrics["test"], shuffled.metrics["test"]
    toward = abs(s["auroc"] - 0.5) < abs(a["auroc"] - 0.5)
    return gate(bool(frozen and toward),
                {"ap_full": a["ap"], "ap_shuffled": s["ap"], "auroc_full": a["auroc"],
                 "auroc_shuffled": s["auroc"], "fitted_objects_identical": frozen},
                "shuffled test timestamps: AUROC moves toward 0.5; thresholds, ZStats, weights and the "
                "cold-start statistics unchanged")


def gate_shuffle_tgn(base: RunResult, shuffled: RunResult, merge_report: dict, *, drift_tol: float,
                     event_labels=None) -> dict:
    """S4.3 shuffle check with the TGN re-scored on the shuffled stream: PASS when gate_shuffle passes (fitted
    objects identical, AUROC closer to 0.5, each run on its own scored test rows) and the pre-cutoff drift of the
    shuffled TGN scoring (tgn.infer.merge_shuffled_tgn report) is <= drift_tol for p_link and neg_p_link. Both
    runs must come from one session (one host) so the fitted objects compare exactly. With event_labels, the
    value also holds readings on the test rows scored (not cold-start) in both runs, matched by event_id: total
    and per-component AUROC / AP on each side (reported, not gated)."""
    from citecurv.eval.metrics import ranking_metrics

    g = gate_shuffle(base, shuffled)
    drift = {k: float(merge_report.get(k, float("nan"))) for k in ("pre_max_abs_diff_p_link",
                                                                     "pre_max_abs_diff_neg_p_link")}
    drift_ok = all(np.isfinite(v) and v <= float(drift_tol) for v in drift.values())
    value = {**g["value"], **drift, "drift_tol": float(drift_tol), "drift_ok": bool(drift_ok),
             "gate_shuffle": g["status"]}
    if event_labels is not None:
        lab = event_labels.set_index(event_labels["event_id"].to_numpy(dtype=np.int64))["label"]

        def scored(res):
            s = res.scores
            keep = (res.split.to_numpy() == "test") & ~s["cold_start"].to_numpy(dtype=bool)
            return s.loc[keep].set_index(s.loc[keep, "event_id"].to_numpy(dtype=np.int64))

        a, b = scored(base), scored(shuffled)
        ids = a.index.intersection(b.index)
        y = lab.reindex(ids).fillna(0).to_numpy()
        comps = sorted(set(base.metrics.get("enabled_components", [])) & set(shuffled.metrics.get("enabled_components",
                                                                                                    [])))

        def readings(frame):
            out = {"total": ranking_metrics(frame.loc[ids, "total"].to_numpy(dtype=float), y)}
            for c in comps:
                z = frame.loc[ids, f"z_{c}"].to_numpy(dtype=float)
                out[c] = ranking_metrics(np.where(np.isnan(z), 0.0, z), y)
            return out

        value["matched"] = {"n_rows": int(len(ids)), "n_pos": int(y.sum()), "base": readings(a),
                            "shuffled": readings(b)}
    return gate(bool(g["status"] == "PASS" and drift_ok), value,
                "gate_shuffle passes and the shuffled TGN scoring reproduces pre-cutoff rows within drift_tol")


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
        "cold_total_only_on_cold": "total_cold" not in s.columns
        or bool(np.isnan(s["total_cold"].to_numpy(dtype=float)[~cold]).all()),
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
                "z-statistics, totals and every method's metrics; their own score (total_cold) is on no scored row")


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
    named breakdown is complete (no NaN or degenerate part). Under G13 a NaN s_geom / s_curvdyn on a row whose
    cited work has no known reference list (ref_known false in the curvature frame) is complete by rule: the
    curvature is unknown there by design. A result without a curvature frame (no_curvature) has no such rows."""
    s = result.scores
    keep = ~s["cold_start"].to_numpy()
    if part != "all":
        keep &= result.split.to_numpy() == part
    sub = s.loc[keep]
    top = sub.iloc[np.argsort(-sub["total"].to_numpy(), kind="stable")[:k]]
    enabled = result.metrics["enabled_components"]
    finite = np.isfinite(top[[f"z_{c}" for c in enabled]].to_numpy(dtype=float))
    curv = result.curvature
    if curv is not None and "ref_known" in curv.columns:
        unknown = ~top["event_id"].map(curv.set_index("event_id")["ref_known"]).fillna(True).to_numpy(dtype=bool)
        for j, c in enumerate(enabled):
            if c in ("s_geom", "s_curvdyn"):
                finite[:, j] |= unknown
    complete = finite.all(axis=1)
    return gate(bool(complete.all()), {"k": int(len(top)), "n_complete": int(complete.sum()), "enabled": enabled},
                f"top-{k} {part} events have a finite z for every enabled component")

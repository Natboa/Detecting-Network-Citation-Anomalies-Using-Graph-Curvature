"""SYNTHETIC dress rehearsal of the evaluation protocol (S3.4, S4.2-S4.6) on synthetic streams with planted cartels.

Every number this script produces comes from synthetic data (citecurv.data.synthetic + citecurv.eval.inject). None
of it is a measurement on real citations; it tests the pipeline and the protocol, not the method's real value.

Design (docs/rehearsal_report.md gives the reason for each choice):
- One data realisation per (level, seed): a synthetic stream (generate_citation_stream, seed = seed), cutoff at the
  latest event time leaving >= 20 % of events in test, val = last 10 % of the pre-cutoff span. Cartels with
  disjoint member sets are planted in three windows: train period (supervised baselines, learned head), val period
  (weight calibration), test period (evaluation; every test cartel event has t > cutoff). Levels: blatant,
  moderate, subtle (inject_cartel with eval.inject.OBVIOUSNESS) and clique (inject_clique). Positives = injected
  events. The seed therefore varies the data realisation (stream and cartels), not only the model.
- Model, group "matrix": every cell of eval.ablation.VARIANTS through eval.ablation.run_matrix /
  pipeline.run_pipeline, s_embed in static_cosine mode, real directed ORC (GraphRicciCurvature) on escalated
  events, ORC values shared across the cells of one realisation (exact: same stream). Extra cell
  full_learned_train: the learned head fitted on train-period labels instead of val labels.
- Group "equal_weights": the calibrated cells again with calibration switched off (fixed weights 0.25 each), a
  sensitivity setting that separates what the components carry from what the val calibration picks.
- Baselines on the same test events through the same metric code: tier-1 rules + rank average, tier-2 XGBoost
  (fitted on train-period labels), trivial controls (cited-paper age at t, reference overlap, random), and one
  PyGOD detector (DOMINANT) in the PyGOD venv for one (level, seed) cell, through a child process of this script.
- Sanity gates (pipeline.gate_*): timestamp shuffle (first seed of every level), reproducibility (fresh rerun ->
  identical sha256), cold-start consistency, D4 fairness, complete top-20 breakdowns; diagnostics: ORC on all
  scored test positives plus a negative sample, component inputs of positives vs negatives; one real TGN run
  (CPU) on a smaller stream, reported separately.

Usage (from the repo root, own venv with GraphRicciCurvature, torch, xgboost):
    PYTHONPATH=src nice -n 10 /home/user/venvs/w2a/bin/python scripts/rehearsal.py \
        --out-dir data/rehearsal --results results/rehearsal
    PYTHONPATH=src python -m citecurv.eval.ablation --tables data/rehearsal --out results/rehearsal
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SYNTHETIC_NOTE = ("SYNTHETIC DATA: synthetic citation streams with planted cartels (citecurv.data.synthetic, "
                  "citecurv.eval.inject); not a measurement on real citations")
LEVELS = ("blatant", "moderate", "subtle", "clique")
PERIODS = ("train", "val", "test")

DESIGN = {
    "n_works": 5000, "n_fields": 8, "years": [2005, 2025], "refs_mean": 25, "emb_dim": 64,
    "test_share": 0.20, "val_frac": 0.10,
    "n_cartels": {"train": 10, "val": 5, "test": 10},
    "recipient_share": 0.4,
    "clique": {"clique_size": 15, "drop_prob": 0.0},
    "baselines": {"window_days": 365, "min_venue_citers": 3, "max_ref_candidates": 20,
                  "xgb": {"num_boost_round": 300, "max_depth": 4, "eta": 0.1, "nthread": 1}},
    "pygod": {"detector": "DOMINANT", "epoch": 100, "hid_dim": 64, "level": "moderate", "seed": 0},
    "pipeline": {"screen_percentile_low": 0.01, "screen_percentile_high": 0.99, "orc_radius": 1,
                 "orc_max_events": None, "with_inputs": False},
    "embed_mode": "static_cosine",
    "fairness_flag_share": 0.01, "fairness_max_cell_share": 0.5,
    "orc_diagnostic_negatives": 1000,
    "tgn": {"n_works": 2000, "n_fields": 4, "n_cartels": {"train": 4, "val": 2, "test": 4}, "epochs": 2,
            "learning_rate": 1e-3, "num_threads": 1},
}


def cpu_seconds() -> float:
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} cpu={cpu_seconds() / 60:.1f}m] {msg}", flush=True)


# ----------------------------------------------------------------- data


def make_realisation(level: str, seed: int, design: dict) -> dict:
    """Synthetic stream + cartels in the train, val and test windows (disjoint members). SYNTHETIC."""
    from citecurv.data.synthetic import generate_citation_stream, to_days
    from citecurv.eval.inject import OBVIOUSNESS, inject_cartel, inject_clique
    from citecurv.pipeline import cutoff_by_share

    s = generate_citation_stream(n_works=design["n_works"], n_fields=design["n_fields"],
                                 years=tuple(design["years"]), refs_mean=design["refs_mean"],
                                 emb_dim=design["emb_dim"], seed=seed)
    events, works, auth = s["events"], s["works"], s["authorships"]
    cutoff = cutoff_by_share(events, design["test_share"])
    t0 = int(events["t"].min())
    val_start = cutoff - design["val_frac"] * (cutoff - t0)
    pub = to_days(works["pub_date"].to_numpy())
    in_period = {"train": (pub >= t0) & (pub <= val_start), "val": (pub > val_start) & (pub <= cutoff),
                 "test": pub > cutoff}
    t_min = {"train": t0, "val": int(math.floor(val_start)) + 1, "test": cutoff + 1}
    used: set = set()
    truth, cartels, counts = [], [], {}
    for i, period in enumerate(PERIODS):
        w = works[in_period[period]]
        rs = 1000 * seed + 10 * LEVELS.index(level) + i
        n = design["n_cartels"][period]
        if level == "clique":
            res = inject_clique(events, w, n_cliques=n, clique_size=design["clique"]["clique_size"], seed=rs,
                                t_min=t_min[period], drop_prob=design["clique"]["drop_prob"], exclude_ids=used)
        else:
            res = inject_cartel(events, w, n_cartels=n, seed=rs, t_min=t_min[period], authorships=auth,
                                exclude_ids=used, recipient_share=design["recipient_share"], **OBVIOUSNESS[level])
        events = res.events
        ids = res.truth_event_ids
        tt = events.set_index("event_id").loc[ids, "t"].to_numpy()
        ok = {"train": (tt >= t0) & (tt <= val_start), "val": (tt > val_start) & (tt <= cutoff),
              "test": tt > cutoff}[period]
        if not ok.all():
            raise AssertionError(f"{period} cartel events fall outside their period")
        c = res.cartels.copy()
        c["cartel_id"] = c["cartel_id"] + 100 * i
        c["period"] = period
        c["level"] = level
        cartels.append(c)
        used |= set(c["work_id"])
        truth.append(ids)
        counts[period] = int(len(ids))
    truth_ids = np.sort(np.concatenate(truth))
    labels = pd.DataFrame({"event_id": events["event_id"].to_numpy(np.int64),
                           "label": events["event_id"].isin(truth_ids).to_numpy().astype(np.int64)})
    return {"events": events, "works": works, "authorships": auth, "embeddings": s["embeddings"],
            "event_labels": labels, "cutoff": int(cutoff), "cartels": pd.concat(cartels, ignore_index=True),
            "truth_ids": truth_ids, "val_start": float(val_start), "injected": counts, "level": level,
            "seed": seed}


def pipeline_inputs(data: dict) -> dict:
    return {k: data[k] for k in ("events", "works", "authorships", "embeddings", "event_labels", "cutoff")}


def pipeline_cfg(design: dict, embed_mode=None, **pipeline) -> dict:
    from citecurv.config import load_config

    cfg = load_config()
    cfg["scoring"]["components"]["embed_mode"] = embed_mode or design["embed_mode"]
    cfg["eval"]["split"]["val_frac"] = design["val_frac"]
    cfg["pipeline"] = {**design["pipeline"], **pipeline}
    return cfg


# ----------------------------------------------------------------- baselines


def eval_frame(full, scores: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """evaluate() on the model's scored test events (same set for every method), canonical tie order."""
    from citecurv.eval.metrics import evaluate

    base = full.scores.loc[(full.split == "test").to_numpy(), ["event_id", "cold_start"]]
    df = base.merge(scores[["event_id", "score"]], on="event_id", how="left")
    if df["score"].isna().any():
        raise ValueError(f"{int(df['score'].isna().sum())} test events have no baseline score")
    return evaluate(df, labels, score_col="score")


def write_method(root, data, kind, method, test, full, extra=None):
    from citecurv.eval.ablation import run_dir_name, write_run

    m = {"kind": kind, "method": method, "seed": int(data["seed"]), "test": test,
         "tags": {"level": data["level"], "synthetic": True, "group": "matrix"},
         "data_hash": full.data_hash, "config_hash": None,
         "test_n_pos_all": full.metrics["test_n_pos_all"],
         "test_n_pos_cold_start": full.metrics["test_n_pos_cold_start"], **(extra or {})}
    write_run(Path(root) / "matrix" / data["level"] / "runs" / run_dir_name(method, data["seed"]), m)
    return m


def run_baselines(data, full, root, design, args, with_pygod) -> dict:
    """Tier-1 rules, rank average, tier-2 XGBoost, trivial controls and (optionally) PyGOD; returns
    {"metrics": {method: metrics}, "seconds": {...}}. Every method scores the same test events (the full
    model's scored test events) and goes through eval.metrics.evaluate."""
    from citecurv.data.synthetic import to_days
    from citecurv.eval.baselines import tier1_stats as t1
    from citecurv.eval.baselines import tier2_xgb as t2

    b = design["baselines"]
    events, works, labels = data["events"], data["works"], data["event_labels"]
    split = pd.DataFrame({"event_id": full.scores["event_id"].to_numpy(), "split": full.split.to_numpy()})
    train_ids = set(split.loc[split["split"] == "train", "event_id"])
    test_ids = set(split.loc[split["split"] == "test", "event_id"])
    out, secs = {}, {}

    t = time.perf_counter()
    feats = t1.compute_features(events, works, data["authorships"], window_days=b["window_days"],
                                min_venue_citers=b["min_venue_citers"], max_ref_candidates=b["max_ref_candidates"])
    ref = feats[feats["event_id"].isin(train_ids)]
    test = feats[feats["event_id"].isin(test_ids)]
    secs["tier1"] = time.perf_counter() - t
    out["tier1_rank_average"] = ("baseline", t1.rank_average(test, reference=ref))
    for c in ("self_citation", "venue_gini", "burst_z"):
        out[f"tier1_{c}"] = ("baseline", t1.rule_scores(test, c, reference=ref))
    out["ref_overlap"] = ("control", t1.rule_scores(test, "ref_overlap", reference=ref))

    t = time.perf_counter()
    x = t2.build_features(events, works, data["authorships"], data["embeddings"], tier1=feats)
    xs, _ = t2.fit_predict(x, labels, split, seed=data["seed"], nthread=b["xgb"]["nthread"],
                           num_boost_round=b["xgb"]["num_boost_round"],
                           params={"max_depth": b["xgb"]["max_depth"], "eta": b["xgb"]["eta"]})
    secs["tier2_xgb"] = time.perf_counter() - t
    out["tier2_xgb"] = ("baseline", xs)

    pub = pd.Series(to_days(works["pub_date"].to_numpy()), index=works["work_id"].to_numpy())
    ev_test = events[events["event_id"].isin(test_ids)]
    age = ev_test["t"].to_numpy() - pub.reindex(ev_test["dst_id"].to_numpy()).to_numpy()
    out["dst_age"] = ("control", pd.DataFrame({"event_id": ev_test["event_id"].to_numpy(), "score": -age}))
    rng = np.random.default_rng(10_000 + data["seed"])
    out["random"] = ("control", pd.DataFrame({"event_id": ev_test["event_id"].to_numpy(),
                                              "score": rng.random(len(ev_test))}))

    if with_pygod:
        t = time.perf_counter()
        try:
            name = f"pygod_{design['pygod']['detector'].lower()}"
            out[name] = ("baseline", run_pygod(data, ev_test, design, args))
        except Exception as exc:  # recorded, not fatal: the PyGOD tier is optional in the rehearsal
            log(f"PyGOD failed: {type(exc).__name__}: {exc}")
            secs["pygod_error"] = f"{type(exc).__name__}: {exc}"
        secs["pygod"] = time.perf_counter() - t

    metrics = {name: write_method(root, data, kind, name, eval_frame(full, df, labels), full)
               for name, (kind, df) in out.items()}
    return {"metrics": metrics, "seconds": secs}


def run_pygod(data, ev_test, design, args) -> pd.DataFrame:
    """One PyGOD detector on the cutoff snapshot, in the PyGOD venv (child process of this script)."""
    work = Path(args.out_dir) / "_pygod_tmp"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        data["events"][["event_id", "src_id", "dst_id", "t"]].to_parquet(work / "events.parquet", index=False)
        ev_test[["event_id", "src_id", "dst_id", "t"]].to_parquet(work / "test_events.parquet", index=False)
        ids, vec = data["embeddings"]
        np.savez(work / "emb.npz", ids=np.asarray(ids).astype(str), vectors=vec)
        p = design["pygod"]
        spec = {"cutoff": data["cutoff"], "seed": data["seed"], "detector": p["detector"], "epoch": p["epoch"],
                "hid_dim": p["hid_dim"]}
        (work / "spec.json").write_text(json.dumps(spec))
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        proc = subprocess.run([args.pygod_python, str(Path(__file__).resolve()), "--pygod-child", str(work)],
                              env=env, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])
        return pd.read_parquet(work / "scores.parquet")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def pygod_child(work: str) -> int:
    import torch

    from citecurv.eval.baselines.pygod_tier import score_pygod, snapshot_graph

    torch.set_num_threads(2)
    work = Path(work)
    spec = json.loads((work / "spec.json").read_text())
    events = pd.read_parquet(work / "events.parquet")
    test = pd.read_parquet(work / "test_events.parquet")
    z = np.load(work / "emb.npz", allow_pickle=False)
    snap = snapshot_graph(events, spec["cutoff"], z["ids"], z["vectors"], undirected=True)
    scores = score_pygod(spec["detector"], snap, test, seed=spec["seed"], epoch=spec["epoch"],
                         hid_dim=spec["hid_dim"])
    scores.to_parquet(work / "scores.parquet", index=False)
    return 0


# ----------------------------------------------------------------- diagnostics


def orc_diagnostic(data, full, oll, design) -> dict:
    """Does directed ORC separate cartel events at all, and does the screen send them to ORC? ORC is computed
    on every scored test positive plus a seeded sample of scored test negatives. Labels choose the rows, so this
    is a diagnostic only, never a model input."""
    from citecurv.curvature.orc import escalate
    from citecurv.curvature.timeslice import EventStore
    from citecurv.eval.metrics import auroc

    s, cur = full.scores, full.curvature
    keep = (full.split == "test").to_numpy() & ~s["cold_start"].to_numpy()
    y = data["event_labels"].set_index("event_id")["label"].reindex(s["event_id"]).to_numpy()
    pos = np.flatnonzero(keep & (y == 1))
    neg_all = np.flatnonzero(keep & (y == 0))
    rng = np.random.default_rng(4242)
    neg = np.sort(rng.choice(neg_all, size=min(design["orc_diagnostic_negatives"], len(neg_all)), replace=False))
    rows = np.sort(np.concatenate([pos, neg]))
    c = cur.copy()
    c["escalated"] = False
    c.iloc[rows, c.columns.get_loc("escalated")] = True
    c["orc"] = np.nan
    cc = full.config["curvature"]
    t0 = time.perf_counter()
    c, st = escalate(data["events"], c, EventStore(data["events"]), cc["orc_radius"], cc["orc_subgraph_edge_cap"],
                     prune=cc["orc_prune"], **cc["orc"])
    orc = c["orc"].to_numpy()[rows]
    yy = y[rows]
    ok = np.isfinite(orc)
    forman = cur["forman"].to_numpy()
    return {
        "n_pos": int(len(pos)), "n_neg_sampled": int(len(neg)), "n_orc_computed": int(st["n_computed"]),
        "seconds": time.perf_counter() - t0,
        "auroc_orc_higher_is_anomalous": auroc(orc[ok], yy[ok]),
        "mean_orc_pos": float(np.nanmean(orc[yy == 1])), "mean_orc_neg": float(np.nanmean(orc[yy == 0])),
        "auroc_minus_forman_scored_test": auroc(-forman[keep], y[keep]),
        "share_pos_escalated_by_screen": float(cur["escalated"].to_numpy()[pos].mean()) if len(pos) else None,
        "share_neg_escalated_by_screen": float(cur["escalated"].to_numpy()[neg_all].mean()),
        "share_pos_escalated_random": float(oll.curvature["escalated"].to_numpy()[pos].mean()) if len(pos) else None,
        "screen_rate_all_events": float(cur["escalated"].mean()),
    }


def component_inputs_diagnostic(data, full) -> dict:
    """Medians of the component inputs (in_* trace columns) and of cited-paper age for scored test positives
    vs negatives, plus the share of test positives lost to cold start."""
    from citecurv.data.synthetic import to_days

    c = full.components
    keep = (full.split == "test").to_numpy() & ~c["cold_start"].to_numpy()
    y = data["event_labels"].set_index("event_id")["label"].reindex(c["event_id"]).to_numpy()
    pub = pd.Series(to_days(data["works"]["pub_date"].to_numpy()), index=data["works"]["work_id"].to_numpy())
    age = c["t"].to_numpy() - pub.reindex(c["dst_id"]).to_numpy()
    cols = {"dst_age_days": age, "temp_x": c["in_temp_x"], "n_recent": c["in_n_recent"],
            "n_before": c["in_n_before"], "exposure_days": c["in_exposure_days"], "n_prior_dst": c["in_n_prior_dst"],
            "cosine": c["in_cosine"], "forman": c["in_forman"], "s_embed": c["s_embed"], "s_geom": c["s_geom"],
            "s_temp": c["s_temp"], "s_curvdyn": c["s_curvdyn"]}
    df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()})[keep]
    yk = y[keep]
    test = (full.split == "test").to_numpy()
    return {"median_pos": {k: float(np.nanmedian(df.loc[yk == 1, k])) for k in df},
            "median_neg": {k: float(np.nanmedian(df.loc[yk == 0, k])) for k in df},
            "test_positives_all": int(y[test].sum()),
            "test_positives_cold_start": int(y[test & c["cold_start"].to_numpy()].sum())}


# ----------------------------------------------------------------- explorer export


def export_explorer(data, full, out_dir) -> Path:
    """Inputs of one run for citecurv.viz.build_explorer, plus manifest.json (file -> contract name)."""
    from citecurv.eval.ablation import write_json

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {}

    def put(name, obj, contract, arg):
        p = out / name
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(p, index=False)
        else:
            write_json(p, obj)
        files[name] = {"contract": contract, "build_explorer_arg": arg}

    put("scores.parquet", full.scores, "scores.parquet (CONTRACTS.md 'Scores file') + z_*, n_components_used, "
        "split", "scores")
    put("components.parquet", full.components, "components (CONTRACTS.md 'Components') + in_* trace columns",
        "components")
    put("curvature.parquet", full.curvature, "curvature cache (CONTRACTS.md 'Curvature cache')", "curvature")
    put("events.parquet", data["events"], "events (CONTRACTS.md 'Events'), cartel rows included", "events")
    put("works.parquet", data["works"], "works (CONTRACTS.md 'Works') with field", "works")
    put("authorships.parquet", data["authorships"], "authorships (CONTRACTS.md 'Works')", "authorships")
    put("event_labels.parquet", data["event_labels"], "event_labels (CONTRACTS.md 'Labels', synthetic rule)",
        "labels")
    put("cartels.parquet", data["cartels"], "cartel members: eval.inject.CARTEL_COLUMNS + period + level",
        "cartels")
    put("cartel_event_ids.json", [int(x) for x in data["truth_ids"]], "injected event ids (ground truth)",
        "cartel-events")
    put("metrics.json", full.metrics, "run metrics dict (pipeline.run_pipeline)", "metrics")
    put("weights.json", full.weights, "combiner weights {component: w} on the simplex", "weights")
    put("zstats.json", full.zstats.to_dict(), "ZStats.to_dict() (frozen train statistics)", "zstats")
    ids, vec = data["embeddings"]
    np.savez(out / "embeddings.npz", ids=np.asarray(ids).astype(str), vectors=vec)
    files["embeddings.npz"] = {"contract": "embeddings.npz (CONTRACTS.md 'Node embeddings'; synthetic, d = 64)",
                               "build_explorer_arg": None}
    d = str(out)
    manifest = {
        "synthetic": True, "note": SYNTHETIC_NOTE,
        "run": {"level": data["level"], "seed": data["seed"], "variant": full.variant, "group": "matrix",
                "config_hash": full.config_hash, "data_hash": full.data_hash, "cutoff_t": full.cutoff},
        "files": files, "tables": {},
        "build_command": (
            f"PYTHONPATH=src python scripts/build_explorer.py --scores {d}/scores.parquet --components "
            f"{d}/components.parquet --curvature {d}/curvature.parquet --events {d}/events.parquet --works "
            f"{d}/works.parquet --authorships {d}/authorships.parquet --labels {d}/event_labels.parquet --metrics "
            f"{d}/metrics.json --weights {d}/weights.json --zstats {d}/zstats.json --cartels {d}/cartels.parquet "
            f"--cartel-events {d}/cartel_event_ids.json --table 'Main comparison={d}/table_main.csv' "
            f"--table 'Ablation={d}/table_ablation.csv' --data-note 'synthetic rehearsal (moderate cartels, seed 0)'"),
    }
    write_json(out / "manifest.json", manifest)
    return out / "manifest.json"


def add_tables_to_export(export_dir: Path, tbls: dict):
    from citecurv.eval.ablation import write_json

    man = json.loads((export_dir / "manifest.json").read_text())
    for name, key in (("table_main.csv", "main"), ("table_ablation.csv", "ablation"),
                      ("table_curvature_delta.csv", "curvature_delta")):
        df = tbls[key]
        df[[c for c in df.columns if not str(c).startswith("_")]].to_csv(export_dir / name, index=False)
        man["tables"][name] = {"contract": f"eval.ablation.tables()['{key}'] (all groups and levels)",
                               "build_explorer_arg": "table"}
    write_json(export_dir / "manifest.json", man)


def build_explorer_check(export_dir: Path, out_html: Path) -> dict:
    """Build the explorer page from the exported inputs (compatibility check with citecurv.viz)."""
    from citecurv.viz import build_explorer

    d = export_dir
    t0 = time.perf_counter()
    path = build_explorer(
        out_html, pd.read_parquet(d / "scores.parquet"), components=pd.read_parquet(d / "components.parquet"),
        curvature=pd.read_parquet(d / "curvature.parquet"), events=pd.read_parquet(d / "events.parquet"),
        works=pd.read_parquet(d / "works.parquet"), authorships=pd.read_parquet(d / "authorships.parquet"),
        labels=pd.read_parquet(d / "event_labels.parquet"),
        metrics=json.loads((d / "metrics.json").read_text()),
        tables={"Main comparison": pd.read_csv(d / "table_main.csv"),
                "Ablation": pd.read_csv(d / "table_ablation.csv")},
        cartels={"members": pd.read_parquet(d / "cartels.parquet"),
                 "event_ids": json.loads((d / "cartel_event_ids.json").read_text())},
        weights=json.loads((d / "weights.json").read_text()), zstats=json.loads((d / "zstats.json").read_text()),
        data_note="synthetic rehearsal (moderate cartels, seed 0)")
    return {"html": str(path), "mb": path.stat().st_size / 1e6, "seconds": time.perf_counter() - t0}


# ----------------------------------------------------------------- TGN extra run


def tgn_extra(design, root) -> dict:
    """One full-variant run with the real TGN (CPU, few epochs) on a smaller moderate realisation, paired with
    the static-embedding full run on the same stream. One seed: shows the path works, not a fair comparison."""
    from citecurv.eval.ablation import run_dir_name, write_run
    from citecurv.pipeline import run_pipeline
    from citecurv.tgn.convert import to_jodie
    from citecurv.tgn.infer import infer_tgn
    from citecurv.tgn.train import dyglib_split_ratios, train_tgn

    tg = design["tgn"]
    d = dict(design, n_works=tg["n_works"], n_fields=tg["n_fields"], n_cartels=tg["n_cartels"])
    data = make_realisation("moderate", 0, d)
    work = Path(root) / "_tgn_work"
    if work.exists():
        shutil.rmtree(work)
    name = "rehearsal"
    meta = to_jodie(data["events"], data["embeddings"], work / "data", name)
    val_ratio, test_ratio = dyglib_split_ratios(data["events"]["t"].to_numpy(), math.floor(data["val_start"]),
                                                data["cutoff"])
    t0 = time.perf_counter()
    tm = train_tgn(work / "data", name, epochs=tg["epochs"], seed=0, device="cpu",
                   node_feat_dim=meta["node_feat_dim"], out_dir=work / "run", direction_aware=True,
                   num_threads=tg["num_threads"], learning_rate=tg["learning_rate"], val_ratio=val_ratio,
                   test_ratio=test_ratio)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores, emb_path = infer_tgn(work / "run", work / "data", name, out_dir=work / "infer", seed=0,
                                 num_threads=tg["num_threads"])
    infer_s = time.perf_counter() - t0
    with np.load(emb_path) as z:
        tgn_emb = {k: z[k] for k in z.files}
    tags = {"level": "moderate", "synthetic": True, "group": "tgn_extra", "n_works": tg["n_works"]}
    out = {"design": {**tg, "n_events": int(len(data["events"])), "injected": data["injected"]},
           "train_s": train_s, "infer_s": infer_s,
           "dyglib": {k: tm.get(k) for k in ("epochs", "test", "best_epoch", "peak_rss_mb", "wall_clock_s")},
           "frac_p_link_gt_neg": float((scores["p_link"] > scores["neg_p_link"]).mean())}
    for label, kw, cfg in (("full_tgn", {"tgn_scores": scores, "tgn_embeddings": tgn_emb},
                            pipeline_cfg(design, embed_mode="auto")),
                           ("full_static", {}, pipeline_cfg(design))):
        res = run_pipeline(**pipeline_inputs(data), variant="full", seed=0, cfg=cfg, tags=tags, **kw)
        write_run(Path(root) / "tgn_extra" / "runs" / run_dir_name(label, 0), dict(res.metrics, method=label))
        out[label] = {"test": res.metrics["test"], "embed_mode": res.metrics["embed_mode"],
                      "weights": res.weights, "weights_source": res.weights_source,
                      "test_components": {c: {k: v[k] for k in ("auroc", "ap")}
                                          for c, v in res.metrics["test_components"].items()}}
    shutil.rmtree(work / "run" / "dyglib_work", ignore_errors=True)
    return out


# ----------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(REPO / "data" / "rehearsal"))
    ap.add_argument("--results", default=str(REPO / "results" / "rehearsal"))
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-works", type=int, default=DESIGN["n_works"])
    ap.add_argument("--orc-max-events", type=int, default=None)
    ap.add_argument("--skip-pygod", action="store_true")
    ap.add_argument("--pygod-python", default="/home/user/venvs/a5-pygod/bin/python")
    ap.add_argument("--skip-tgn", action="store_true")
    ap.add_argument("--skip-equal-weights", action="store_true")
    ap.add_argument("--sanity-level", default="moderate")
    ap.add_argument("--pygod-child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.pygod_child:
        return pygod_child(args.pygod_child)

    from citecurv.eval.ablation import (VARIANTS, load_records, run_dir_name, run_matrix, tables, write_bundle,
                                        write_json, write_run, write_tables)
    from citecurv.pipeline import (gate_cold_start, gate_fairness, gate_reproducibility, gate_shuffle,
                                   gate_top_breakdown, run_pipeline, shuffle_test_timestamps)

    design = copy.deepcopy(DESIGN)
    design["n_works"] = args.n_works
    design["pipeline"]["orc_max_events"] = args.orc_max_events
    root, results = Path(args.out_dir), Path(args.results)
    for p in (root, results):
        p.mkdir(parents=True, exist_ok=True)
    for sub in ("matrix", "equal_weights", "repro", "tgn_extra", "explorer_inputs", "explorer"):
        shutil.rmtree(root / sub, ignore_errors=True)
    t_wall, cpu0 = time.time(), cpu_seconds()
    summary = {"synthetic": True, "note": SYNTHETIC_NOTE, "design": design, "realisations": {}, "gates": {},
               "diagnostics": {}}
    variants = list(VARIANTS)
    calibrated = [v for v in variants if VARIANTS[v].combiner == "calibrated"]

    for level in args.levels:
        for seed in args.seeds:
            log(f"{level} seed {seed}: generating")
            data = make_realisation(level, seed, design)
            explorer_cell = level == args.sanity_level and seed == args.seeds[0]
            cfg = pipeline_cfg(design, with_inputs=explorer_cell)
            info = {"n_events": int(len(data["events"])), "cutoff_t": data["cutoff"], "injected": data["injected"]}
            kept, memos = {}, {}
            tags = {"level": level, "synthetic": True, "group": "matrix"}

            def keep(res):
                if res.variant in ("full", "ollivier_only"):
                    kept[res.variant] = res
                m = res.metrics
                log(f"  {res.variant}: AP {m['test']['ap']:.4f} AUROC {m['test']['auroc']:.4f} {res.weights_source} "
                    f"ORC computed {m['curvature'].get('n_orc_computed')}")

            t0 = cpu_seconds()
            run_matrix(variants, [seed], pipeline_inputs(data), root / "matrix" / level, cfg=cfg, tags=tags,
                       callback=keep, memos=memos)
            memo = memos[seed]
            cfg_lt = copy.deepcopy(cfg)
            cfg_lt["pipeline"]["learned_fit_split"] = "train"
            res = run_pipeline(**pipeline_inputs(data), variant="full_learned", seed=seed, cfg=cfg_lt, orc_memo=memo,
                               tags=tags)
            write_run(root / "matrix" / level / "runs" / run_dir_name("full_learned_train", seed),
                      dict(res.metrics, method="full_learned_train"), {**res.config, "config_hash": res.config_hash})
            log(f"  full_learned_train: AP {res.metrics['test']['ap']:.4f} AUROC {res.metrics['test']['auroc']:.4f}")
            info["model_cpu_s"] = cpu_seconds() - t0
            if not args.skip_equal_weights:
                t0 = cpu_seconds()
                cfg_eq = pipeline_cfg(design, calibrate_weights=False)
                run_matrix(calibrated, [seed], pipeline_inputs(data), root / "equal_weights" / level, cfg=cfg_eq,
                           tags={**tags, "group": "equal_weights"}, orc_memo=memo, share_orc=False)
                info["equal_weights_cpu_s"] = cpu_seconds() - t0
            full = kept["full"]
            t0 = cpu_seconds()
            with_pygod = (not args.skip_pygod and level == design["pygod"]["level"]
                          and seed == design["pygod"]["seed"])
            bl = run_baselines(data, full, root, design, args, with_pygod)
            info["baseline_cpu_s"] = cpu_seconds() - t0
            info["baseline_seconds"] = bl["seconds"]
            log("  baselines: " + ", ".join(f"{k} {v['test']['ap']:.4f}" for k, v in bl["metrics"].items()))

            gates = {"cold_start_consistency": gate_cold_start(full, list(bl["metrics"].values())),
                     "d4_fairness": gate_fairness(full, data["works"], flag_share=design["fairness_flag_share"],
                                                  max_share=design["fairness_max_cell_share"]),
                     "top20_breakdown": gate_top_breakdown(full, k=20)}
            if seed == args.seeds[0]:
                t0 = cpu_seconds()
                ev = shuffle_test_timestamps(data["events"], data["cutoff"], seed=777 + seed)
                pre = {k: v for k, v in memo.items() if k[3] <= data["cutoff"]}  # pre-cutoff state unchanged
                shuf = run_pipeline(**{**pipeline_inputs(data), "events": ev}, variant="full", seed=seed, cfg=cfg,
                                    orc_memo=pre, tags=tags)
                write_run(root / "matrix" / level / "runs" / run_dir_name("shuffle_full", seed),
                          dict(shuf.metrics, kind="sanity", method="shuffle_full"))
                gates["timestamp_shuffle"] = gate_shuffle(full, shuf)
                info["shuffle_cpu_s"] = cpu_seconds() - t0
                del shuf
            if explorer_cell:
                t0 = cpu_seconds()
                again = make_realisation(level, seed, design)
                rep = run_pipeline(**pipeline_inputs(again), variant="full", seed=seed, cfg=cfg,
                                   out_dir=root / "repro" / level / "runs" / run_dir_name("full", seed),
                                   tags={**tags, "group": "repro"})
                gates["reproducibility"] = gate_reproducibility(
                    root / "matrix" / level / "runs" / run_dir_name("full", seed) / "scores.parquet",
                    root / "repro" / level / "runs" / run_dir_name("full", seed) / "scores.parquet")
                gates["reproducibility"]["value"]["data_hash_equal"] = rep.data_hash == full.data_hash
                info["repro_cpu_s"] = cpu_seconds() - t0
                del again, rep
                t0 = cpu_seconds()
                summary["diagnostics"]["orc"] = orc_diagnostic(data, full, kept["ollivier_only"], design)
                summary["diagnostics"]["component_inputs"] = component_inputs_diagnostic(data, full)
                info["diagnostics_cpu_s"] = cpu_seconds() - t0
                log(f"  ORC diagnostic: {summary['diagnostics']['orc']}")
                summary["explorer_manifest"] = str(export_explorer(data, full, root / "explorer_inputs").resolve())
            write_run(root / "matrix" / level / "runs" / run_dir_name("gates", seed),
                      {"kind": "sanity", "method": "gates", "seed": seed, "tags": tags, "test": None, "gates": gates})
            summary["gates"][f"{level}_{seed}"] = {k: v["status"] for k, v in gates.items()}
            summary["realisations"].setdefault(level, {})[str(seed)] = info
            log(f"  gates: {summary['gates'][f'{level}_{seed}']}")
            del data, kept, memos, memo, full, bl

    if not args.skip_tgn:
        t0 = cpu_seconds()
        try:
            summary["tgn_extra"] = tgn_extra(design, root)
        except Exception as exc:  # optional extra; recorded, not fatal
            summary["tgn_extra"] = {"error": f"{type(exc).__name__}: {exc}"}
        summary["tgn_extra_cpu_s"] = cpu_seconds() - t0
        log(f"TGN extra: {summary['tgn_extra']}")

    records = load_records(root)
    write_bundle(records, results / "run_bundle.jsonl")
    tbls = tables(records)
    write_tables(tbls, results, title="Synthetic dress rehearsal: results tables",
                 banner="SYNTHETIC DATA: every number below comes from synthetic streams with planted cartels; "
                        "none is a measurement on real citations.")
    export = root / "explorer_inputs"
    if (export / "manifest.json").is_file():
        add_tables_to_export(export, tbls)
        try:
            summary["explorer_build"] = build_explorer_check(export, root / "explorer" / "rehearsal_moderate_0.html")
        except Exception as exc:  # compatibility check only
            summary["explorer_build"] = {"error": f"{type(exc).__name__}: {exc}"}
    summary["compute"] = {"cpu_minutes": (cpu_seconds() - cpu0) / 60, "wall_minutes": (time.time() - t_wall) / 60,
                          "peak_rss_mb_driver": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          "peak_rss_mb_children": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024}
    write_json(results / "rehearsal_summary.json", summary)
    log(f"done: {summary['compute']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

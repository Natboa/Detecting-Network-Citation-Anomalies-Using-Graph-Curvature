"""Variant x seed run matrix and results tables regenerated from run artifacts (tasks S4.3, S4.6).

VARIANTS: the ablation cells of book 4.2.1 / plan S4.3 plus the learned-combiner extra cell (L13). Each spec
fixes what the pipeline (``citecurv.pipeline.run_pipeline``) does for that cell:

==============  ==========================================================================================
cell            meaning
==============  ==========================================================================================
full            Forman screen + ORC escalation; all four components; calibrated fixed weights.
no_curvature    no curvature stage at all; s_geom and s_curvdyn off. Also serves as the "de-curvatured own
                model" baseline (plan S4.4).
forman_only     Forman screen only, no ORC escalation; s_geom = |z_F|, s_curvdyn target F.
ollivier_only   ORC on a seeded uniform random sample of the size of the screen's escalated set
                (components.matched_random_escalation); Forman values never read.
no_temporal     no TGN memory (s_embed static_cosine, static endpoint vectors for s_curvdyn) and, by default,
                s_temp off as well (pipeline.no_temporal_drop_s_temp; decision D11, PROVISIONAL).
full_learned    full components, LearnedCombiner (logistic head on the z-scores) instead of fixed weights.
==============  ==========================================================================================

Run artifacts: ``<out_dir>/runs/<variant>_<seed>/`` holds metrics.json, config.json (the resolved run
configuration with its config_hash), scores.parquet, weights.json, zstats.json, thresholds.json. Baselines and
sanity checks write the same metrics.json layout (``write_run``) with ``kind`` "baseline", "control" or
"sanity", so one reader regenerates every table.

``tables(source)`` reads every metrics.json under a folder (or a compact bundle written by ``write_bundle``)
and returns all results tables as DataFrames; ``python -m citecurv.eval.ablation --tables SRC --out DIR`` writes
them as CSV plus one markdown file (plan S4.6: no hand-edited numbers anywhere).

Leak argument: this module reads finished metrics and configurations only; nothing it computes feeds back into
any score.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VariantSpec:
    """What the pipeline does for one ablation cell."""

    name: str
    component_variant: str   # value for scoring.components.score_events(variant=...)
    curvature: str           # "screen+orc" | "screen" | "random+orc" | "none"
    temporal_encoder: bool   # True: TGN outputs are used when given; False: static embeddings only
    drop_s_temp: bool | None  # True/False, or None = follow pipeline.no_temporal_drop_s_temp
    combiner: str            # "calibrated" (fixed weights, calibrated on val) | "learned"
    description: str

    def as_dict(self) -> dict:
        return asdict(self)


VARIANTS = {
    "full": VariantSpec("full", "full", "screen+orc", True, False, "calibrated",
                        "Forman screen + ORC escalation, all four components, calibrated weights"),
    "no_curvature": VariantSpec("no_curvature", "no_curvature", "none", True, False, "calibrated",
                                "no curvature anywhere: s_geom and s_curvdyn off, no screen, no ORC"),
    "forman_only": VariantSpec("forman_only", "forman_only", "screen", True, False, "calibrated",
                               "Forman screen only, no ORC escalation"),
    "ollivier_only": VariantSpec("ollivier_only", "ollivier_only", "random+orc", True, False, "calibrated",
                                 "ORC on a random sample at the screen's escalation rate; Forman never read"),
    "no_temporal": VariantSpec("no_temporal", "no_temporal", "screen+orc", False, None, "calibrated",
                               "no TGN memory (static embeddings); s_temp off unless the D11 flag says otherwise"),
    "full_learned": VariantSpec("full_learned", "full", "screen+orc", True, False, "learned",
                                "full components, logistic head instead of calibrated fixed weights (L13)"),
}
ABLATION_VARIANTS = ("full", "no_curvature", "forman_only", "ollivier_only", "no_temporal")
EXTRA_CELLS = ("full_learned",)
KINDS = ("model", "baseline", "control", "sanity")


# ----------------------------------------------------------------- run artifacts


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, obj) -> Path:
    """JSON with sorted keys, NaN/inf as null, numpy values as Python values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_dir_name(method: str, seed: int) -> str:
    return f"{method}_{int(seed)}"


def write_run(run_dir, metrics: dict, config: dict | None = None) -> Path:
    """Write metrics.json (and config.json when given) into run_dir; returns run_dir.

    metrics must hold ``kind`` (one of KINDS), ``method`` and ``seed``; ``test`` is the evaluate() dict on the
    test events (or None), ``tags`` a dict (e.g. {"level": "moderate"}).
    """
    for key in ("kind", "method", "seed"):
        if key not in metrics:
            raise ValueError(f"metrics needs {key!r}")
    if metrics["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    run_dir = Path(run_dir)
    write_json(run_dir / "metrics.json", metrics)
    if config is not None:
        write_json(run_dir / "config.json", config)
    return run_dir


# ----------------------------------------------------------------- the matrix


def run_matrix(variants, seeds, data, out_dir, *, cfg=None, orc_fn=None, tags=None, share_orc=True,
               orc_memo=None, callback=None, memos=None, resume=False, **pipeline_kwargs) -> list[dict]:
    """Run every (variant, seed) cell and write ``out_dir/runs/<variant>_<seed>/``.

    data: mapping of run_pipeline inputs (events, works, embeddings, event_labels, cutoff, tgn_scores, ...)
    or a callable seed -> such a mapping (one data realisation per seed, as in the synthetic rehearsal).
    share_orc: one ORC memo per seed, shared by the variants of that seed (they see the same stream, so the
    cached ORC values are identical to recomputed ones). orc_memo: one memo used for every seed instead (the
    caller guarantees it belongs to the same stream, e.g. a rerun of one realisation in another setting).
    callback(result) receives each RunResult (to keep frames a caller needs). memos: optional dict that
    receives seed -> the ORC memo used for that seed (for follow-up runs on the same stream). resume: skip a
    cell whose metrics.json already exists (its stored metrics are returned instead; checkpoint-resume on
    Colab, plan 9.1). Returns the metrics dicts in run order.
    """
    from citecurv.curvature.timeslice import EventStore
    from citecurv.pipeline import run_pipeline

    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}")
    out = []
    for seed in seeds:
        todo = [v for v in variants
                if not (resume and (Path(out_dir) / "runs" / run_dir_name(v, seed) / "metrics.json").is_file())]
        for v in [v for v in variants if v not in todo]:
            out.append(json.loads((Path(out_dir) / "runs" / run_dir_name(v, seed) / "metrics.json").read_text()))
        if not todo:
            continue
        d = dict(data(seed) if callable(data) else data)
        memo = orc_memo if orc_memo is not None else ({} if share_orc else None)
        if memos is not None and memo is not None:
            memos[seed] = memo
        store = EventStore(d["events"]) if any(VARIANTS[v].curvature.endswith("orc") for v in todo) else None
        for v in todo:
            res = run_pipeline(**d, variant=v, seed=seed, cfg=cfg, orc_fn=orc_fn,
                               out_dir=Path(out_dir) / "runs" / run_dir_name(v, seed), orc_memo=memo, store=store,
                               tags=tags, **pipeline_kwargs)
            out.append(res.metrics)
            if callback is not None:
                callback(res)
    return out


# ----------------------------------------------------------------- reading runs


BUNDLE_KEYS = ("kind", "method", "seed", "tags", "config_hash", "data_hash", "test", "test_n_pos_all",
               "test_n_pos_cold_start", "weights_source", "weights", "test_components", "test_top100_contribution",
               "gates", "note")
EXTRA_GROUPS = ("tgn_extra", "repro")  # run groups reported outside the matrix tables


def load_records(source) -> list[dict]:
    """metrics dicts from a folder (every runs/<name>/metrics.json below it, sorted by path), a .jsonl bundle,
    or a list of dicts (returned as is)."""
    if isinstance(source, list):
        return source
    src = Path(source)
    if src.is_file():
        return [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not src.is_dir():
        raise FileNotFoundError(f"no run folder or bundle at {src}")
    paths = [p for p in sorted(src.rglob("metrics.json")) if p.parent.parent.name == "runs"]
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _group(r) -> str:
    return str((r.get("tags") or {}).get("group", "matrix"))


def write_bundle(records, path) -> Path:
    """Compact one-line-per-run bundle (only the keys the tables read), sorted for a stable diff."""
    rows = []
    for r in records:
        row = {k: r.get(k) for k in BUNDLE_KEYS if k in r}
        if row.get("test_components"):
            row["test_components"] = {c: {k: v.get(k) for k in ("auroc", "ap")}
                                      for c, v in row["test_components"].items()}
        if row.get("test_top100_contribution"):
            row["test_top100_contribution"] = {k: row["test_top100_contribution"].get(k)
                                               for k in ("k", "n_pos", "mean_contribution")}
        rows.append(row)
    rows.sort(key=lambda r: (_group(r), str((r.get("tags") or {}).get("level", "")), r["kind"], r["method"],
                             int(r["seed"])))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(_jsonable(r), sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


def collect_runs(source) -> pd.DataFrame:
    """One row per run: group, level, kind, method, seed, hashes, and the test metrics (recall@k added)."""
    rows = []
    for r in load_records(source):
        tags = r.get("tags") or {}
        test = r.get("test") or {}
        row = {"group": _group(r), "level": tags.get("level", ""), "kind": r["kind"],
               "method": r["method"], "seed": int(r["seed"]), "config_hash": r.get("config_hash"),
               "data_hash": r.get("data_hash"), "n_pos_all": r.get("test_n_pos_all"),
               "weights_source": r.get("weights_source")}
        for key, val in test.items():
            row[key] = np.nan if val is None else val
        n_pos = test.get("n_pos")
        for key in list(test):
            if key.startswith("found_at_"):
                k = key[len("found_at_"):]
                row[f"recall_at_{k}"] = (test[key] / n_pos) if n_pos else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["group", "level", "kind", "method", "seed"], kind="mergesort").reset_index(drop=True)


# ----------------------------------------------------------------- tables


def _fmt(mean, sd, n, digits=3):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "n/a"
    if n > 1 and sd is not None and not math.isnan(sd):
        return f"{mean:.{digits}f} ± {sd:.{digits}f}"
    return f"{mean:.{digits}f}"


def _ms(values):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan"), int(v.size)


LEVEL_ORDER = ("blatant", "moderate", "subtle", "clique")


def _level_rank(level):
    return (LEVEL_ORDER.index(level) if level in LEVEL_ORDER else len(LEVEL_ORDER), str(level))


def _key_rank(group, level):
    return (group != "matrix", str(group), _level_rank(level))


def _method_order(df):
    model = [v for v in VARIANTS if v in set(df["method"])]
    rest = sorted(set(df["method"]) - set(model))
    return model + rest


def _paired(df, method, ref, metric):
    a = df[df["method"] == method].set_index("seed")[metric]
    b = df[df["method"] == ref].set_index("seed")[metric]
    if not (a.index.is_unique and b.index.is_unique):
        raise ValueError(f"duplicate (method, seed) runs for {method!r} or {ref!r}")
    common = sorted(set(a.index) & set(b.index))
    return np.array([float(a[s]) - float(b[s]) for s in common]), common


METRIC_COLS = (("auroc", "AUROC"), ("ap", "AP"), ("p_at_100", "P@100"), ("recall_at_1000", "R@1000"))


def _summary_rows(df, methods, key):
    rows = []
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        row = {**key, "method": m, "kind": sub["kind"].iloc[0], "n_seeds": int(sub["seed"].nunique()),
               "n_pos": round(float(sub["n_pos"].mean()), 1) if "n_pos" in sub else np.nan,
               "n": round(float(sub["n"].mean()), 1) if "n" in sub else np.nan}
        for col, label in METRIC_COLS:
            if col in sub:
                mean, sd, n = _ms(sub[col].tolist())
                row[label] = _fmt(mean, sd, n)
                row[f"_{col}_mean"] = mean
        rows.append(row)
    return rows


def _compact(value, max_len=220):
    if isinstance(value, dict):
        value = {k: (f"<{len(v)} items>" if isinstance(v, list) and len(v) > 5 else v) for k, v in value.items()}
        text = json.dumps(_jsonable(value), sort_keys=True)
    else:
        text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def tables(source, *, control_method="dst_age") -> dict:
    """Every results table as {name: DataFrame}, regenerated from run artifacts only.

    source: folder of runs (runs/<name>/metrics.json files), a .jsonl bundle, or a list of metrics dicts.
    Rows are keyed by (group, level): group is the run setting (tags["group"], default "matrix"; e.g. a fixed-
    weights sensitivity setting) and level the data condition (tags["level"]). Tables: runs (long form), main
    (model vs baselines vs controls), ablation (variants with paired deltas vs full), curvature_delta (full
    minus no_curvature, book test iii), obviousness (recall@k of full per level, S4.5), learned_vs_fixed (L13),
    components (which component carries the signal in full), shuffle (timestamp-shuffle line), gates (sanity
    gates PASS/FAIL), extra (groups in EXTRA_GROUPS, e.g. the single TGN run). Paired deltas match runs by
    seed. Hidden helper columns starting with "_" hold unformatted means.
    """
    records = load_records(source)
    runs = collect_runs(records)
    out = {"runs": runs}
    if runs.empty:
        return out
    mat = runs[~runs["group"].isin(EXTRA_GROUPS) & (runs["kind"] != "sanity")]
    keys = sorted(set(zip(mat["group"], mat["level"])), key=lambda k: _key_rank(*k))

    main, abl, delta, obv, lvf, comp = [], [], [], [], [], []
    for group, level in keys:
        key = {"group": group, "level": level}
        d = mat[(mat["group"] == group) & (mat["level"] == level)]
        others = [m for m in _method_order(d) if m not in VARIANTS]
        base = _summary_rows(d, others, key)
        base.sort(key=lambda r: -np.nan_to_num(r.get("_ap_mean", np.nan), nan=-1.0))
        main += _summary_rows(d, [m for m in ("full", "no_curvature") if m in set(d["method"])], key) + base

        for m in [v for v in VARIANTS if v in set(d["method"])]:
            row = _summary_rows(d, [m], key)[0]
            for col, label in (("ap", "ΔAP vs full"), ("auroc", "ΔAUROC vs full"), ("p_at_100", "ΔP@100 vs full")):
                if m == "full" or col not in d or "full" not in set(d["method"]):
                    row[label] = "" if m == "full" else "n/a"
                    continue
                diffs, _ = _paired(d, m, "full", col)
                mean, sd, n = _ms(diffs.tolist())
                row[label] = _fmt(mean, sd, n)
            abl.append(row)

        if {"full", "no_curvature"} <= set(d["method"]):
            for col, label in (("ap", "AP"), ("auroc", "AUROC"), ("p_at_100", "P@100")):
                diffs, seeds = _paired(d, "full", "no_curvature", col)
                mean, sd, n = _ms(diffs.tolist())
                f_mean, f_sd, _ = _ms(d[d["method"] == "full"][col].tolist())
                c_mean, c_sd, _ = _ms(d[d["method"] == "no_curvature"][col].tolist())
                spread = np.nanmax([f_sd, c_sd]) if n > 1 else float("nan")
                same_sign = n > 1 and (np.all(diffs > 0) or np.all(diffs < 0))
                outside = bool(same_sign and abs(mean) > spread)
                delta.append({**key, "metric": label, "full": _fmt(f_mean, f_sd, n),
                              "no_curvature": _fmt(c_mean, c_sd, n), "delta (full - no_curv)": _fmt(mean, sd, n),
                              "per-seed deltas": ", ".join(f"s{s}: {x:+.4f}" for s, x in zip(seeds, diffs)),
                              "n_seeds": n,
                              "verdict": ("outside seed noise" if outside else "within seed noise") if n > 1
                              else "one seed: no noise estimate",
                              "_delta_mean": mean, "_delta_sd": sd})

        if "full" in set(d["method"]):
            f = d[d["method"] == "full"]
            row = {**key, "n_seeds": int(f["seed"].nunique())}
            for col, label in (("n_pos", "positives scored"), ("n_pos_all", "positives incl. cold start")):
                if col in f:
                    row[label] = round(float(pd.to_numeric(f[col], errors="coerce").mean()), 1)
            for col, label in (("recall_at_100", "recall@100"), ("recall_at_1000", "recall@1000"),
                               ("auroc", "AUROC"), ("ap", "AP")):
                if col in f:
                    mean, sd, n = _ms(f[col].tolist())
                    row[label] = _fmt(mean, sd, n)
            c = d[d["method"] == control_method]
            if not c.empty and "recall_at_1000" in c:
                mean, sd, n = _ms(c["recall_at_1000"].tolist())
                row[f"recall@1000 {control_method}"] = _fmt(mean, sd, n)
            obv.append(row)

            recs = [r for r in records if r["kind"] == "model" and r["method"] == "full" and _group(r) == group
                    and (r.get("tags") or {}).get("level", "") == level]
            for c in ("s_embed", "s_geom", "s_temp", "s_curvdyn"):
                tc = [((r.get("test_components") or {}).get(c) or {}) for r in recs]
                contrib = [(((r.get("test_top100_contribution") or {}).get("mean_contribution") or {}).get(c))
                           for r in recs]
                w = [(r.get("weights") or {}).get(c) for r in recs]
                crow = {**key, "component": c, "n_seeds": len(recs)}
                for name, vals in (("weight", w), ("AUROC alone", [x.get("auroc") for x in tc]),
                                   ("AP alone", [x.get("ap") for x in tc]),
                                   ("mean w*z on top-100", contrib)):
                    mean, sd, n = _ms([v for v in vals if v is not None])
                    crow[name] = _fmt(mean, sd, n)
                comp.append(crow)

        for m in [x for x in sorted(set(d["method"])) if x.startswith("full_learned")]:
            row = {**key, "cell": m}
            for col, label in (("ap", "AP"), ("auroc", "AUROC")):
                f_mean, f_sd, n_f = _ms(d[d["method"] == "full"][col].tolist())
                l_mean, l_sd, n_l = _ms(d[d["method"] == m][col].tolist())
                diffs, _ = _paired(d, m, "full", col)
                mean, sd, n = _ms(diffs.tolist())
                row[f"{label} fixed (full)"] = _fmt(f_mean, f_sd, n_f)
                row[f"{label} learned"] = _fmt(l_mean, l_sd, n_l)
                row[f"Δ{label} learned - fixed"] = _fmt(mean, sd, n)
            lvf.append(row)

    out["main"] = pd.DataFrame(main)
    out["ablation"] = pd.DataFrame(abl)
    out["curvature_delta"] = pd.DataFrame(delta)
    out["obviousness"] = pd.DataFrame(obv)
    out["learned_vs_fixed"] = pd.DataFrame(lvf)
    out["components"] = pd.DataFrame(comp)

    san = runs[runs["kind"] == "sanity"]
    shuf = []
    for group, level in sorted(set(zip(san["group"], san["level"])), key=lambda k: _key_rank(*k)):
        s = san[(san["group"] == group) & (san["level"] == level) & san["method"].str.startswith("shuffle")]
        f = mat[(mat["group"] == group) & (mat["level"] == level) & (mat["method"] == "full")]
        if s.empty:
            continue
        f = f[f["seed"].isin(s["seed"])]
        row = {"group": group, "level": level, "seeds": ",".join(str(x) for x in sorted(s["seed"].unique()))}
        for col, label in (("ap", "AP"), ("auroc", "AUROC"), ("p_at_100", "P@100")):
            fm, fs, fn = _ms(f[col].tolist())
            sm, ss, sn = _ms(s[col].tolist())
            row[f"{label} full"] = _fmt(fm, fs, fn)
            row[f"{label} shuffled"] = _fmt(sm, ss, sn)
        prev = (s["n_pos"] / s["n"]).mean()
        row["chance AP (prevalence)"] = f"{prev:.4f}"
        row["chance AUROC"] = "0.500"
        shuf.append(row)
    out["shuffle"] = pd.DataFrame(shuf)

    gates = []
    for r in records:
        for name, g in sorted((r.get("gates") or {}).items()):
            gates.append({"group": _group(r), "level": (r.get("tags") or {}).get("level", ""), "seed": r.get("seed"),
                          "gate": name, "status": g.get("status"), "value": _compact(g.get("value")),
                          "rule": g.get("rule", "")})
    cols = ["group", "level", "seed", "gate", "status", "value", "rule"]
    g = pd.DataFrame(gates, columns=cols)
    if not g.empty:
        g["_rank"] = [_key_rank(a, b) for a, b in zip(g["group"], g["level"])]
        g = g.sort_values(["gate", "_rank", "seed"], kind="mergesort").drop(columns="_rank")
    out["gates"] = g.reset_index(drop=True)

    extra = runs[runs["group"].isin(EXTRA_GROUPS)]
    rows = []
    for (group, level, method), sub in extra.groupby(["group", "level", "method"], sort=True):
        row = {"group": group, "level": level, "method": method, "kind": sub["kind"].iloc[0],
               "n_seeds": int(sub["seed"].nunique()), "n_pos": float(sub["n_pos"].mean()) if "n_pos" in sub else None}
        for col, label in METRIC_COLS:
            if col in sub:
                mean, sd, n = _ms(sub[col].tolist())
                row[label] = _fmt(mean, sd, n)
        rows.append(row)
    out["extra"] = pd.DataFrame(rows)
    return out


def _md_table(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if not str(c).startswith("_")]
    if df.empty or not cols:
        return "_(no rows)_\n"

    def cell(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        if isinstance(x, float):
            return f"{x:.4g}"
        return str(x).replace("|", "\\|")

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


TABLE_TITLES = {
    "main": "Main comparison: full model vs baselines and trivial controls (test events, mean ± sd over seeds)",
    "ablation": "Ablation: variants with paired deltas vs full (same data realisation per seed)",
    "curvature_delta": "Curvature-removal delta: full minus no_curvature (book test iii)",
    "obviousness": "Recall vs obviousness: full model per injection level (S4.5)",
    "learned_vs_fixed": "Learned combiner (L13) vs calibrated fixed weights",
    "components": "Which component carries the signal (full model: weight, standalone test metrics, top-100 w*z)",
    "shuffle": "Timestamp-shuffle sanity check",
    "gates": "Sanity gates",
    "extra": "Runs outside the matrix",
    "runs": "All runs (long form)",
}


def write_tables(tbls: dict, out_dir, *, title="Results tables", banner=None) -> list[Path]:
    """Write <name>.csv for every table (helper columns dropped) and tables.md with all of them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    parts = [f"# {title}\n"]
    if banner:
        parts.append(f"> {banner}\n")
    parts.append("Generated by `python -m citecurv.eval.ablation --tables` from run artifacts only; "
                 "do not edit by hand.\n")
    for name in [n for n in TABLE_TITLES if n in tbls] + sorted(set(tbls) - set(TABLE_TITLES)):
        df = tbls[name]
        cols = [c for c in df.columns if not str(c).startswith("_")]
        p = out / f"{name}.csv"
        df[cols].to_csv(p, index=False, lineterminator="\n")
        paths.append(p)
        if name == "runs":
            continue  # long form stays in CSV only
        parts.append(f"\n## {TABLE_TITLES.get(name, name)}\n\n{_md_table(df)}")
    md = out / "tables.md"
    md.write_text("".join(parts), encoding="utf-8")
    return [md, *paths]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate every results table from run artifacts (S4.6).")
    ap.add_argument("--tables", required=True, metavar="SRC", help="runs folder or .jsonl bundle")
    ap.add_argument("--out", required=True, help="output folder for tables.md and the CSV files")
    ap.add_argument("--bundle", default=None, help="also write the compact .jsonl bundle of all runs here")
    ap.add_argument("--title", default="Results tables")
    ap.add_argument("--banner", default=None, help="line shown under the title (e.g. a SYNTHETIC DATA notice)")
    a = ap.parse_args(argv)
    records = load_records(a.tables)
    if a.bundle:
        write_bundle(records, a.bundle)
    synthetic = any((r.get("tags") or {}).get("synthetic") for r in records)
    banner = a.banner or ("SYNTHETIC DATA: every number below comes from synthetic streams with planted cartels; "
                          "none is a measurement on real citations." if synthetic else None)
    paths = write_tables(tables(records), a.out, title=a.title, banner=banner)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

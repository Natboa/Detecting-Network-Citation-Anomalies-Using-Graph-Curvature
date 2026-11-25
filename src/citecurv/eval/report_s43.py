"""Derived records for the S4.3 tables: paper-level intervals, the ledger's label views, the G7 counting controls,
ORC coverage and the shuffle check's per-component readings (plan docs/superpowers/plans/2026-11-25-s43-ablation-
matrix.md, Task 7).

Every record is a metrics dict written with eval.ablation.write_run under <runs folder>/runs/<name>/metrics.json:
kind "report" (or "control" for the counting controls, which belong in the main table next to the baselines), an
integer seed (-1 for a record pooled over seeds), a group naming the table it feeds, and `inputs` = {path: sha256}
of every file it read, so eval.ablation.tables() marks a record whose inputs changed as stale.

Uncertainty. A few retracted papers carry most positive events, and same-day retraction batches tie papers
together, so rows are not independent draws. paired_paper_bootstrap resamples clusters (citing paper, cited paper,
retraction batch) with replacement, one resample per draw shared by every seed's pair, computes the metric per
seed on that seed's rows only (rows of different seeds are never ranked together: their totals use different
calibrated weights), averages the per-seed deltas, and reports the 2.5 and 97.5 percentiles per cluster key; the
headline interval is the widest. The weights calibrated on validation rows are held fixed, so the intervals leave
out the uncertainty of the weight choice itself.

`python -m citecurv.eval.report_s43 <spec.json>` regenerates every record from the run folders (run_report).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from citecurv.eval.ablation import run_dir_name, write_run
from citecurv.eval.metrics import ranking_metrics

SEED_VARIES = {
    "full": "TGN checkpoint, s_curvdyn subsample", "no_curvature": "TGN checkpoint",
    "forman_only": "TGN checkpoint, s_curvdyn subsample", "full_learned": "TGN checkpoint, s_curvdyn subsample",
    "ollivier_only": "TGN checkpoint, random ORC set, s_curvdyn subsample",
    "no_temporal": "s_curvdyn subsample only", "no_temporal_keep_temp": "s_curvdyn subsample only",
}
COMPARISONS = (("full", "no_curvature"), ("full", "forman_only"), ("full", "ollivier_only"),
               ("full", "no_temporal"), ("full", "full_learned"), ("full", "no_temporal_keep_temp"),
               ("no_temporal_keep_temp", "no_temporal"))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------- weighted ranking metrics


class _Ranked:
    """One ranked list prepared once (sorted by score, ties as blocks) so weighted AP / AUROC cost O(n) per draw."""

    def __init__(self, score, y):
        s = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(-s, kind="stable")
        self.order = order
        ss = s[order]
        self.y = y[order]
        change = np.r_[True, ss[1:] != ss[:-1]]
        self.block = np.cumsum(change) - 1
        self.n_blocks = int(self.block[-1] + 1) if len(ss) else 0

    def metrics(self, w=None) -> dict:
        w = np.ones(len(self.y)) if w is None else np.asarray(w, dtype=float)[self.order]
        wp = np.bincount(self.block, weights=w * self.y, minlength=self.n_blocks)
        wn = np.bincount(self.block, weights=w * (1.0 - self.y), minlength=self.n_blocks)
        Wp, Wn = wp.sum(), wn.sum()
        if Wp <= 0:
            return {"ap": float("nan"), "auroc": float("nan")}
        cum = np.cumsum(wp + wn)
        prec = np.divide(np.cumsum(wp), cum, out=np.zeros_like(cum), where=cum > 0)
        ap = float(np.sum(prec * wp) / Wp)
        auroc = float(np.sum(wp * ((Wn - np.cumsum(wn)) + 0.5 * wn)) / (Wp * Wn)) if Wn > 0 else float("nan")
        return {"ap": ap, "auroc": auroc}


def _scored_test(scores: pd.DataFrame, *, cold=False, col="total") -> pd.Series:
    s = scores
    keep = (s["split"].to_numpy() == "test") & (s["cold_start"].to_numpy(dtype=bool) == bool(cold))
    return pd.Series(s.loc[keep, col].to_numpy(dtype=float), index=s.loc[keep, "event_id"].to_numpy(dtype=np.int64))


def _labels(labels) -> pd.Series:
    return pd.Series(labels["label"].to_numpy(), index=labels["event_id"].to_numpy(dtype=np.int64))


# ----------------------------------------------------------------- clusters and the bootstrap


def cluster_keys(events, labels, batches=None) -> dict[str, pd.Series]:
    """Cluster label per event id: "src" (citing paper), "dst" (cited paper), "batch" (a labelled endpoint that
    belongs to a retraction batch maps to the batch id, every other event to its citing paper). batches: {work_id:
    batch id} for the papers of the batches (data.slice_query.mass_events joined to work ids)."""
    ids = events["event_id"].to_numpy(dtype=np.int64)
    src = events["src_id"].astype(str).to_numpy()
    dst = events["dst_id"].astype(str).to_numpy()
    batch = src.astype(object).copy()
    if batches:
        lab = labels.set_index(labels["event_id"].to_numpy(dtype=np.int64))
        vs = lab["via_src"].reindex(ids).fillna(0).to_numpy(dtype=bool) if "via_src" in lab else np.zeros(len(ids), bool)
        vd = lab["via_dst"].reindex(ids).fillna(0).to_numpy(dtype=bool) if "via_dst" in lab else np.zeros(len(ids), bool)
        bs = np.array([batches.get(x) for x in src], dtype=object)
        bd = np.array([batches.get(x) for x in dst], dtype=object)
        use_s = vs & (bs != None)  # noqa: E711  (elementwise on an object array)
        use_d = ~use_s & vd & (bd != None)  # noqa: E711
        batch[use_s] = ["batch:" + str(b) for b in bs[use_s]]
        batch[use_d] = ["batch:" + str(b) for b in bd[use_d]]
    return {"src": pd.Series(src, index=ids), "dst": pd.Series(dst, index=ids), "batch": pd.Series(batch, index=ids)}


def paired_paper_bootstrap(pairs, labels, keys: dict, *, metric="ap", n_draws=1000, seed=0, cold=False) -> dict:
    """Paired cluster bootstrap of metric(a) - metric(b). pairs: [(a_scores, b_scores), ...], one pair per seed
    (scores.parquet frames); cold: rank the cold-start rows by total_cold instead of the scored rows by total.
    Returns {delta, per_seed, per_key: {key: {lo, hi}}, lo, hi, interval_method, n_clusters, n_pos}."""
    y_of = _labels(labels)
    col = "total_cold" if cold else "total"
    prepared = []
    for a, b in pairs:
        sa, sb = _scored_test(a, cold=cold, col=col), _scored_test(b, cold=cold, col=col)
        ids = sa.index.intersection(sb.index)
        y = y_of.reindex(ids).fillna(0).to_numpy()
        prepared.append((ids, _Ranked(sa.loc[ids].to_numpy(), y), _Ranked(sb.loc[ids].to_numpy(), y), y))
    per_seed = [ra.metrics()[metric] - rb.metrics()[metric] for _, ra, rb, _ in prepared]
    out = {"delta": float(np.mean(per_seed)) if per_seed else float("nan"), "per_seed": per_seed, "per_key": {},
           "metric": metric, "n_draws": int(n_draws), "cold": bool(cold),
           "n_pos": int(prepared[0][3].sum()) if prepared else 0}
    all_ids = np.unique(np.concatenate([p[0].to_numpy() for p in prepared])) if prepared else np.zeros(0, np.int64)
    for name, ser in keys.items():
        codes_all, uniq = pd.factorize(ser.reindex(all_ids).astype(str))
        code_of = pd.Series(codes_all, index=all_ids)
        C = len(uniq)
        rng = np.random.default_rng([int(seed), sum(map(ord, name))])
        seed_codes = [code_of.reindex(p[0]).to_numpy() for p in prepared]
        draws = np.empty(n_draws)
        for k in range(n_draws):
            counts = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
            vals = []
            for (ids, ra, rb, _), codes in zip(prepared, seed_codes):
                w = counts[codes]
                vals.append(ra.metrics(w)[metric] - rb.metrics(w)[metric])
            vals = np.asarray(vals, dtype=float)
            draws[k] = float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else np.nan
        fin = draws[np.isfinite(draws)]
        lo, hi = np.percentile(fin, [2.5, 97.5]) if len(fin) else (np.nan, np.nan)
        out["per_key"][name] = {"lo": float(lo), "hi": float(hi), "n_clusters": int(C)}
    if out["per_key"]:
        widest = max(out["per_key"], key=lambda k: out["per_key"][k]["hi"] - out["per_key"][k]["lo"])
        out.update(lo=out["per_key"][widest]["lo"], hi=out["per_key"][widest]["hi"],
                   interval_method=f"cluster:{widest}", n_clusters=out["per_key"][widest]["n_clusters"])
    return out


# ----------------------------------------------------------------- masks (ledger G9, G12)


def mask_without_recordless(events, missing_ids) -> pd.Series:
    """G12: keep events whose cited work has an OpenAlex record (drop citations of an id without one)."""
    miss = {str(x) for x in missing_ids}
    keep = ~events["dst_id"].astype(str).isin(miss).to_numpy()
    return pd.Series(keep, index=events["event_id"].to_numpy(dtype=np.int64))


def mask_without_batches(events, labels, batch_papers: set) -> pd.Series:
    """G9: drop an event when any labelled endpoint is a paper of the given retraction batches (citing side when
    the label comes through the citing paper, cited side when through the cited paper)."""
    ids = events["event_id"].to_numpy(dtype=np.int64)
    lab = labels.set_index(labels["event_id"].to_numpy(dtype=np.int64))
    vs = lab["via_src"].reindex(ids).fillna(0).to_numpy(dtype=bool)
    vd = lab["via_dst"].reindex(ids).fillna(0).to_numpy(dtype=bool)
    papers = {str(x) for x in batch_papers}
    hit = (vs & events["src_id"].astype(str).isin(papers).to_numpy()) | \
          (vd & events["dst_id"].astype(str).isin(papers).to_numpy())
    return pd.Series(~hit, index=ids)


def mask_unseen_cited(events, split) -> tuple[pd.Series, float]:
    """G9: keep events whose cited work appears in no train-part event (as citing or cited work). Also returns the
    share of test events whose citing work appears in train-part events (a check: expected close to 0)."""
    split = np.asarray(split)
    tr = events[split == "train"]
    seen = set(tr["src_id"].astype(str)) | set(tr["dst_id"].astype(str))
    keep = ~events["dst_id"].astype(str).isin(seen).to_numpy()
    test = split == "test"
    src_seen = float(events.loc[test, "src_id"].astype(str).isin(seen).mean()) if test.any() else float("nan")
    return pd.Series(keep, index=events["event_id"].to_numpy(dtype=np.int64)), src_seen


# ----------------------------------------------------------------- records


def _run(run_dir) -> tuple[dict, pd.DataFrame]:
    run_dir = Path(run_dir)
    return json.loads((run_dir / "metrics.json").read_text()), pd.read_parquet(run_dir / "scores.parquet")


def _record(kind, method, seed, group, payload, inputs, level="real"):
    return {"kind": kind, "method": method, "seed": int(seed), "tags": {"group": group, "level": level},
            "test": payload.pop("test", None), **payload, "inputs": {str(p): sha256(p) for p in inputs}}


def interval_record(comparison, view, result: dict, seed: int, inputs, level="real") -> dict:
    a, b = comparison
    return _record("report", f"{a} - {b}", seed, "interval",
                   {"comparison": [a, b], "view": view, **result, "excludes_0": bool(result.get("lo", 0) > 0 or
                                                                                    result.get("hi", 0) < 0)},
                   inputs, level)


def masked_view_records(run_dir, labels, masks: dict, level="real") -> list[dict]:
    """The run's scored test rows re-evaluated inside each mask (ledger G9, G10, G12): one report record per mask."""
    m, scores = _run(run_dir)
    s = _scored_test(scores)
    y = _labels(labels).reindex(s.index).fillna(0).to_numpy()
    out = []
    for name, keep in masks.items():
        note = None
        if isinstance(keep, tuple):
            keep, note = keep
        k = keep.reindex(s.index).fillna(False).to_numpy(dtype=bool)
        test = ranking_metrics(s.to_numpy()[k], y[k])
        out.append(_record("report", m["method"], m["seed"], "masked_view",
                           {"mask": name, "test": test, "n_dropped": int((~k).sum()),
                            "n_pos_dropped": int(y[~k].sum()), "note": note},
                           [Path(run_dir) / "scores.parquet"], level))
    return out


def _directed(score: pd.Series, val_score: pd.Series, val_y) -> tuple[pd.Series, int]:
    ok = np.isfinite(val_score.to_numpy(dtype=float))
    a = ranking_metrics(val_score.to_numpy(dtype=float)[ok], np.asarray(val_y)[ok])["auroc"] if ok.any() else 0.5
    d = 1 if not np.isfinite(a) or a >= 0.5 else -1
    return score * d, d


def control_records(result_dir, controls: dict, labels, known=None, level="real") -> list[dict]:
    """Kind "control" records for the G7 counting controls next to the `full` run in result_dir. controls:
    {name: frame event_id, score} (higher = more suspicious before the sign rule). The sign is fixed by the
    validation rows (AUROC >= 0.5 keeps it). Readings: scored test rows with NaN ranked last (min finite - 1); the
    same rows restricted to known reference lists (known: Series event_id -> bool) next to `full` there; the
    cold-start test rows next to `full`'s total_cold."""
    m, scores = _run(result_dir)
    y_of = _labels(labels)
    s_test = _scored_test(scores)
    s_cold = _scored_test(scores, cold=True, col="total_cold")
    val = scores[(scores["split"].to_numpy() == "val") & ~scores["cold_start"].to_numpy(dtype=bool)]
    val_ids = val["event_id"].to_numpy(dtype=np.int64)
    out = []
    for name, frame in controls.items():
        sc = pd.Series(frame["score"].to_numpy(dtype=float), index=frame["event_id"].to_numpy(dtype=np.int64))
        directed, sign = _directed(sc, sc.reindex(val_ids), y_of.reindex(val_ids).fillna(0).to_numpy())

        def reading(ids, other=None):
            v = directed.reindex(ids).to_numpy(dtype=float)
            fin = np.isfinite(v)
            fill = (np.nanmin(v) - 1.0) if fin.any() else 0.0
            v = np.where(fin, v, fill)
            yy = y_of.reindex(ids).fillna(0).to_numpy()
            res = {"control": ranking_metrics(v, yy), "n_nan": int((~fin).sum())}
            if other is not None:
                res["full"] = ranking_metrics(other.reindex(ids).to_numpy(dtype=float), yy)
            return res

        main = reading(s_test.index)
        rec = {"test": main["control"], "direction": int(sign), "n_nan_test": main["n_nan"],
               "nan_rule": "NaN scores ranked last (min finite score minus 1)"}
        if known is not None:
            k_ids = s_test.index[known.reindex(s_test.index).fillna(False).to_numpy(dtype=bool)]
            rec["on_known"] = reading(k_ids, s_test)
        if len(s_cold):
            rec["on_cold"] = reading(s_cold.index, s_cold)
        out.append(_record("control", name, m["seed"], (m.get("tags") or {}).get("group", "matrix"), rec,
                           [Path(result_dir) / "scores.parquet"], level))
    return out


def orc_coverage(run_dir, orc_rows: pd.DataFrame, labels) -> dict:
    """How much ORC a cell actually used: escalated test rows, test positives with an ORC value (total, citing
    side, cited side), lossy share, share whose citing work was already cited at t (radius 1 then approximate),
    errors. orc_rows: the precompute checkpoint (event_id, orc, orc_lossy, orc_src_in_deg, orc_error)."""
    run_dir = Path(run_dir)
    m, scores = _run(run_dir)
    esc = np.load(run_dir / "escalated.npy")
    test_ids = set(scores.loc[scores["split"].to_numpy() == "test", "event_id"].to_numpy(dtype=np.int64))
    esc_test = np.array([e for e in esc if int(e) in test_ids], dtype=np.int64)
    rows = orc_rows.set_index(orc_rows["event_id"].to_numpy(dtype=np.int64)).reindex(esc_test)
    has = np.isfinite(rows["orc"].to_numpy(dtype=float))
    lab = labels.set_index(labels["event_id"].to_numpy(dtype=np.int64)).reindex(esc_test)
    y = lab["label"].fillna(0).to_numpy(dtype=bool)

    def side(col):
        return int((y & has & lab[col].fillna(0).to_numpy(dtype=bool)).sum()) if col in lab else None

    def share(col):
        return float(rows[col].fillna(False).astype(bool).mean()) if col in rows and len(rows) else None

    return {"method": m["method"], "seed": int(m["seed"]), "n_escalated_test": int(len(esc_test)),
            "n_pos_with_orc": int((y & has).sum()), "n_pos_with_orc_via_src": side("via_src"),
            "n_pos_with_orc_via_dst": side("via_dst"), "lossy_share": share("orc_lossy"),
            "src_cited_share": (float((rows["orc_src_in_deg"].fillna(0) > 0).mean()) if len(rows) else None),
            "n_errors": int(rows["orc_error"].fillna("").astype(str).ne("").sum()) if "orc_error" in rows else 0}


def orc_coverage_record(run_dir, orc_path, labels, level="real") -> dict:
    cov = orc_coverage(run_dir, pd.read_parquet(orc_path), labels)
    method, seed = cov.pop("method"), cov.pop("seed")
    return _record("report", method, seed, "orc_coverage", cov, [Path(run_dir) / "scores.parquet", orc_path], level)


def shuffle_components_record(gate_value: dict, interval: dict | None, seed: int, inputs, level="real") -> dict:
    """Per-component AUROC / AP of the shuffle check on the rows scored in both runs (gate_shuffle_tgn value), plus
    the paper-clustered interval of the AUROC change (base minus shuffled)."""
    matched = gate_value.get("matched") or {}
    rows = {c: {"base": matched.get("base", {}).get(c), "shuffled": matched.get("shuffled", {}).get(c)}
            for c in sorted(set(matched.get("base", {})) | set(matched.get("shuffled", {})))}
    return _record("report", "shuffle_full", seed, "shuffle_components",
                   {"components": rows, "n_rows": matched.get("n_rows"), "auroc_interval": interval}, inputs, level)


# ----------------------------------------------------------------- the whole regeneration


def run_report(spec: dict) -> list[dict]:
    """Regenerate the derived records from the run folders. spec: runs (folder holding runs/), labels (parquet with
    via_src / via_dst), seeds, comparisons (optional, default COMPARISONS where both cells exist), n_draws,
    metrics (default ["ap", "auroc"]), batches (optional JSON {work_id: batch id}), masks (optional {name: parquet
    with event_id, keep}), controls (optional {name: parquet with event_id, score}), known (optional parquet
    event_id, ref_known), orc (optional checkpoint parquet), events (parquet, needed for cluster keys). Writes each
    record under runs/runs/report_<n>/ and returns them."""
    root = Path(spec["runs"])
    labels = pd.read_parquet(spec["labels"])
    events = pd.read_parquet(spec["events"], columns=["event_id", "src_id", "dst_id", "t"])
    batches = json.loads(Path(spec["batches"]).read_text()) if spec.get("batches") else None
    keys = cluster_keys(events, labels, batches)
    seeds = [int(s) for s in spec["seeds"]]
    level = spec.get("level", "real")
    records = []

    def cell(name, s):
        return root / "runs" / run_dir_name(name, s)

    for a, b in [tuple(c) for c in spec.get("comparisons") or COMPARISONS]:
        have = [s for s in seeds if (cell(a, s) / "scores.parquet").exists() and (cell(b, s) / "scores.parquet").exists()]
        if not have:
            continue
        pairs = [(pd.read_parquet(cell(a, s) / "scores.parquet"), pd.read_parquet(cell(b, s) / "scores.parquet"))
                 for s in have]
        inputs = [cell(x, s) / "scores.parquet" for s in have for x in (a, b)]
        for metric in spec.get("metrics", ["ap", "auroc"]):
            for cold in (False, True):
                view = ("cold" if cold else "scored") + ":" + metric
                res = paired_paper_bootstrap(pairs, labels, keys, metric=metric, n_draws=int(spec.get("n_draws", 1000)),
                                             cold=cold)
                records.append(interval_record((a, b), view, res, -1, inputs, level))
                for s, pair in zip(have, pairs):
                    one = paired_paper_bootstrap([pair], labels, keys, metric=metric,
                                                 n_draws=int(spec.get("n_draws", 1000)), cold=cold)
                    records.append(interval_record((a, b), view, one, s, [cell(a, s) / "scores.parquet",
                                                                          cell(b, s) / "scores.parquet"], level))
    masks = {name: pd.read_parquet(p).set_index("event_id")["keep"] for name, p in (spec.get("masks") or {}).items()}
    known = (pd.read_parquet(spec["known"]).set_index("event_id")["ref_known"] if spec.get("known") else None)
    for d in sorted((root / "runs").iterdir()):
        mp = d / "metrics.json"
        if not mp.exists() or not (d / "scores.parquet").exists():
            continue
        m = json.loads(mp.read_text())
        if m.get("kind") != "model":
            continue
        if masks:
            records += masked_view_records(d, labels, masks, level)
        if spec.get("orc") and (d / "escalated.npy").exists():
            records.append(orc_coverage_record(d, spec["orc"], labels, level))
        if spec.get("controls") and m.get("method") == "full":
            ctl = {name: pd.read_parquet(p) for name, p in spec["controls"].items()}
            records += control_records(d, ctl, labels, known, level)
    for i, r in enumerate(records):
        write_run(root / "runs" / f"report_{r['tags']['group']}_{i:04d}", r)
    return records


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m citecurv.eval.report_s43 <spec.json>", file=sys.stderr)
        return 2
    recs = run_report(json.loads(Path(argv[0]).read_text(encoding="utf-8")))
    print(f"{len(recs)} records written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

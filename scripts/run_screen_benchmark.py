"""Forman screen benchmark (S2.2) and escalation-budget demo (S2.3 tooling), SYNTHETIC data only.

Everything here runs on a small local citation-like generator (not the project's
official synthetic generator, not real data). Numbers describe this machine and this
generator; they are not calibration results. The real S2.3 calibration reruns
escalation_budget() on the pilot slice.

Usage (from the repo root):
  PYTHONPATH=src python scripts/run_screen_benchmark.py --out results/screen_benchmark.json
The budget part needs GraphRicciCurvature (curv venv); without it, it is skipped.
"""

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from citecurv.curvature.screen import forman_screen, screen_report, spot_check  # noqa: E402
from citecurv.curvature.timeslice import EventStore  # noqa: E402

T0_DAY = 18262  # 2020-01-01 in days since 1970-01-01


def synthetic_citation_stream(n_papers, mean_refs=15.0, papers_per_day=36, p_uniform=0.3,
                              p_same_day=0.01, seed=0) -> pd.DataFrame:
    """Citation-like event stream (local stand-in generator, SYNTHETIC).

    Paper i is published on day T0_DAY + i // papers_per_day and cites
    k ~ 1 + Poisson(mean_refs - 1) distinct papers at its publication time (all its
    events share t). Each reference is drawn uniformly among papers published on
    earlier days with probability p_uniform, otherwise preferentially (uniformly
    from all earlier cited endpoints, i.e. proportional to in-degree). With
    probability p_same_day a reference goes to an earlier paper of the same day,
    so some citing papers already have in-citations at their own t (same-t policy).
    event_id is a random permutation, so it is not a position.
    """
    rng = np.random.default_rng(seed)
    day = np.arange(n_papers) // papers_per_day
    first_of_day = np.searchsorted(day, day)  # index of the first paper of each paper's day
    k = 1 + rng.poisson(mean_refs - 1.0, size=n_papers)
    cap = int(k.sum())
    src = np.empty(cap, dtype=np.int64)
    dst = np.empty(cap, dtype=np.int64)
    n_ev = 0
    pool = np.empty(cap, dtype=np.int64)  # cited endpoints, in order of appearance
    n_pool = 0
    n_pool_valid = 0  # pool[:n_pool_valid] was cited by papers of earlier days
    cur_day = -1
    for i in range(1, n_papers):
        if day[i] != cur_day:
            cur_day = day[i]
            n_pool_valid = n_pool
        n_old = int(first_of_day[i])  # papers published on earlier days
        n_same = i - n_old           # earlier papers of the same day
        ki = int(k[i])
        r1, r2 = rng.random(ki), rng.random(ki)
        same_m = (r1 < p_same_day) if n_same else np.zeros(ki, bool)
        if n_old == 0:
            if n_same == 0:
                continue
            same_m[:] = True
        unif_m = ~same_m & ((r2 < p_uniform) | (n_pool_valid == 0))
        pref_m = ~same_m & ~unif_m
        refs = np.empty(ki, dtype=np.int64)
        refs[same_m] = rng.integers(n_old, i, size=int(same_m.sum())) if n_same else 0
        refs[unif_m] = rng.integers(0, n_old, size=int(unif_m.sum()))
        refs[pref_m] = pool[rng.integers(0, n_pool_valid, size=int(pref_m.sum()))] if n_pool_valid else 0
        refs = np.unique(refs)
        m = refs.size
        src[n_ev:n_ev + m] = i
        dst[n_ev:n_ev + m] = refs
        n_ev += m
        pool[n_pool:n_pool + m] = refs
        n_pool += m
    src, dst = src[:n_ev], dst[:n_ev]
    names = np.array([f"W{j}" for j in range(n_papers)], dtype=object)
    return pd.DataFrame({
        "event_id": rng.permutation(n_ev).astype(np.int64) + 1_000_000,
        "src_id": names[src],
        "dst_id": names[dst],
        "t": (T0_DAY + day[src]).astype(np.int64),
    })


def max_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench_screen(n_events, seed=0, repeats=3):
    t0 = time.perf_counter()
    mean_refs = 15.0
    ev = synthetic_citation_stream(int(n_events / (mean_refs - 1.0)) + 100, mean_refs=mean_refs, seed=seed)
    ev = ev.sort_values(["t", "event_id"]).head(n_events).reset_index(drop=True)  # stream prefix
    gen_s = time.perf_counter() - t0
    out = {"n_events": len(ev), "n_nodes": int(pd.unique(pd.concat([ev.src_id, ev.dst_id])).size),
           "n_distinct_t": int(ev["t"].nunique()), "max_events_per_t": int(ev["t"].value_counts().max()),
           "generate_s": round(gen_s, 3)}
    times = {}
    for mode, kw in (("distinct", {}), ("count_duplicates", {"count_duplicates": True})):
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            curv = forman_screen(ev, **kw)
            ts.append(time.perf_counter() - t0)
        times[mode] = {"best_s": round(min(ts), 3), "mean_s": round(float(np.mean(ts)), 3),
                       "events_per_s": round(len(ev) / min(ts))}
    out["forman_screen"] = times
    curv = forman_screen(ev)
    out["coverage"] = float(curv["forman"].notna().mean()) * (len(curv) == len(ev))

    # full-scale cross-check of the row-count semantics against EventStore.degrees_at
    t0 = time.perf_counter()
    store = EventStore(ev)
    multi = forman_screen(ev, count_duplicates=True)
    mism = 0
    for s, d, t, f in zip(multi["src_id"], multi["dst_id"], multi["t"], multi["forman"]):
        in_deg, out_deg = store.degrees_at(t)
        mism += f != 2.0 - in_deg[s] - out_deg[d]
    out["eventstore_crosscheck"] = {"rows_checked": len(multi), "mismatches": int(mism),
                                    "seconds_incl_store_build": round(time.perf_counter() - t0, 2)}
    out["distinct_equals_count_duplicates"] = bool((multi["forman"].to_numpy() == curv["forman"].to_numpy()).all())
    chk = spot_check(ev, curv, n=20, seed=seed)
    out["spot_check_20"] = {"all_match": bool(chk["match"].all()), "n": len(chk)}

    t0 = time.perf_counter()
    rep = screen_report(curv)
    out["screen_report_s"] = round(time.perf_counter() - t0, 3)
    out["forman_distribution_all"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in rep.loc["all"].to_dict().items()}
    out["share_at_max_F2"] = round(float((curv["forman"] == 2.0).mean()), 5)
    out["max_rss_mb"] = round(max_rss_mb(), 1)
    return out, ev, curv


def budget_demo(n_papers, seed, sample, radii, pcts, edge_cap, variants, orc_knobs):
    """variants: iterable of (prune, shortest_path) pairs; one table block each."""
    from citecurv.curvature.orc import escalation_budget

    ev = synthetic_citation_stream(n_papers, seed=seed)
    curv = forman_screen(ev)
    store = EventStore(ev)
    # Reference rows = the first 80 % of the time span (stand-in for the training split).
    cut = curv["t"].min() + 0.8 * (curv["t"].max() - curv["t"].min())
    ref = (curv["t"] <= cut).to_numpy()
    tables = []
    for prune, shortest_path in variants:
        t0 = time.perf_counter()
        tab = escalation_budget(curv, store, radii=radii, pcts=pcts, sample=sample, ref_mask=ref,
                                edge_cap=edge_cap, prune=prune, seed=seed, shortest_path=shortest_path,
                                **orc_knobs)
        tab.insert(2, "prune", prune)
        tab.insert(3, "shortest_path", shortest_path)
        tables.append(tab)
        print(f"budget prune={prune} shortest_path={shortest_path}: {time.perf_counter() - t0:.1f}s", flush=True)
    tab = pd.concat(tables, ignore_index=True)
    meta = {"n_events": len(ev), "n_papers": n_papers, "seed": seed, "reference_rows": int(ref.sum()),
            "reference_rule": "t <= min_t + 0.8 * (max_t - min_t)", "sample_per_cell": sample,
            "edge_cap": edge_cap, "orc_knobs": orc_knobs, "max_rss_mb": round(max_rss_mb(), 1)}
    return meta, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/screen_benchmark.json")
    ap.add_argument("--n-events", type=int, default=1_000_000)
    ap.add_argument("--budget-papers", type=int, default=8000)
    ap.add_argument("--budget-sample", type=int, default=200)
    ap.add_argument("--skip-budget", action="store_true")
    args = ap.parse_args()

    result = {
        "label": "SYNTHETIC - local citation-like generator in scripts/run_screen_benchmark.py; "
                 "not real data, not the official synthetic generator",
        "machine": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "pandas": pd.__version__},
    }
    bench, _, _ = bench_screen(args.n_events)
    result["screen_benchmark"] = bench
    print(json.dumps(bench, indent=1), flush=True)

    if not args.skip_budget:
        try:
            import GraphRicciCurvature

            result["machine"]["GraphRicciCurvature"] = GraphRicciCurvature.__version__
            knobs = {"alpha": 0.5, "method": "OTDSinkhornMix", "nbr_topk": 3000, "proc": 1}
            meta, tab = budget_demo(args.budget_papers, seed=0, sample=args.budget_sample, radii=(1, 2),
                                    pcts=(0.01, 0.02, 0.05), edge_cap=5000,
                                    variants=[(False, "all_pairs"), (True, "all_pairs"),
                                              (False, "pairwise"), (True, "pairwise")],
                                    orc_knobs=knobs)
            result["escalation_budget_demo"] = {
                "label": "SYNTHETIC demo of the S2.3 tooling; real calibration needs the pilot slice",
                **meta,
                "table": json.loads(tab.to_json(orient="records", double_precision=6)),
            }
            print(tab.to_string(), flush=True)
        except ImportError:
            result["escalation_budget_demo"] = "skipped: GraphRicciCurvature not installed in this env"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

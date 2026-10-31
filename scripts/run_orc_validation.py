"""Directed Ollivier-Ricci validation for one GraphRicciCurvature install (S0.5 Block A, S1.5).

Checks, all on unweighted DiGraphs:
  1. toy graphs of tests/test_forman_directed.py CASES: ORC equals the hand-computed values
     in tests/test_orc.py (alpha 0.5), for OTD / OTDSinkhornMix / Sinkhorn / ATD;
  2. direction sensitivity: flipping the tested edge changes its curvature where the
     neighbourhood is asymmetric; reversing every edge leaves all values unchanged;
  3. OTD vs OTDSinkhornMix on the toys and on a 200-node random DAG (plan threshold 0.05),
     plus the Mix branch forced onto Sinkhorn by lowering its 2000-neighbour threshold;
  4. multiprocessing (proc > 1, both shortest-path modes, repeated pool creation) finishes
     inside a timeout and reproduces proc = 1 values;
  5. escalation subgraphs: ORC on EventStore.subgraph_at at radius 1 / 2 (+ prune) versus ORC
     on the whole time-t graph, on a small SYNTHETIC citation stream;
  6. library behaviour that changed between versions (weighted denominator, pairwise cache).
All values and timings go to the JSON given by --out.

Usage, from the repo root, inside a curv venv:
  PYTHONPATH=src python scripts/run_orc_validation.py --out results/orc_validation_0532.json
  python scripts/run_orc_validation.py --compare results/orc_validation_0532.json results/orc_validation_061.json
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings("ignore", message="Input histogram consists of integer")
warnings.filterwarnings("ignore", message=".*use of fork\\(\\) may lead to deadlocks.*")

METHODS = ("OTD", "OTDSinkhornMix", "Sinkhorn", "ATD")
TOL = 1e-9
MIX_THRESHOLD = 0.05  # IMPLEMENTATION_PLAN.md S1.5


def key(e):
    return f"{e[0]}->{e[1]}"


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def try_timed(fn, *a, **kw):
    """Like timed, but returns (None, seconds, "ExcType: message") on failure."""
    t0 = time.perf_counter()
    try:
        return fn(*a, **kw), time.perf_counter() - t0, None
    except Exception as exc:  # recorded in the JSON, e.g. ATD on nodes without neighbours
        return None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


def library_orc(G, edges, **kw):
    """Direct library call (for undirected graphs, which orc_directed rejects)."""
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    kw.setdefault("proc", 1)
    kw.setdefault("verbose", "ERROR")
    res = OllivierRicci(G, **kw).compute_ricci_curvature_edges(list(edges))
    return {tuple(e): float(v) for e, v in res.items()}


def env_info():
    import GraphRicciCurvature
    import networkit
    import ot
    import pandas
    import scipy

    return {
        "GraphRicciCurvature": GraphRicciCurvature.__version__, "networkit": networkit.__version__,
        "POT": ot.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
        "networkx": nx.__version__, "pandas": pandas.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "mp_default_start_method": mp.get_start_method(allow_none=True) or mp.get_context().get_start_method(),
    }


def toy_section():
    from citecurv.curvature.orc import orc_directed
    from test_forman_directed import CASES
    from test_orc import ORC_EXPECTED, ORC_FLIPPED_EXPECTED

    rows, t_total = [], 0.0
    for (edges, e, forman), exp, flip_exp in zip(CASES, ORC_EXPECTED, ORC_FLIPPED_EXPECTED):
        G = nx.DiGraph(edges)
        vals, secs, errors = {}, {}, {}
        for m in METHODS:
            r, dt, err = try_timed(orc_directed, G, [e], method=m)
            vals[m], secs[m] = (r[e] if r else None), dt
            if err:
                errors[m] = err
            t_total += dt
        all_otd = orc_directed(G, list(G.edges), method="OTD")
        Gf = G.copy()
        Gf.remove_edge(*e)
        Gf.add_edge(e[1], e[0])
        flipped = orc_directed(Gf, [(e[1], e[0])], method="OTD")[(e[1], e[0])]
        undirected = library_orc(G.to_undirected(), [e], method="OTD")[e]
        rows.append({
            "edges": [key(x) for x in edges], "tested_edge": key(e), "forman": forman,
            "orc_expected": exp, "orc": vals, "errors": errors, "seconds": secs,
            "orc_all_edges_otd": {key(k): v for k, v in all_otd.items()},
            "flipped_expected": flip_exp, "flipped_orc": flipped, "flip_changes_value": abs(flipped - vals["OTD"]) > TOL,
            "undirected_orc": undirected,
            "sign_orc_vs_sign_forman": [int(np.sign(round(vals["OTD"], 9))), int(np.sign(forman))],
        })
    checks = {
        "toy_hand_values": all(abs(r["orc"]["OTD"] - r["orc_expected"]) < TOL for r in rows),
        "toy_flip_values": all(abs(r["flipped_orc"] - r["flipped_expected"]) < TOL for r in rows),
        "toy_flip_changes": sum(r["flip_changes_value"] for r in rows),
        "toy_otd_vs_mix_max_abs_dev": max(abs(r["orc"]["OTD"] - r["orc"]["OTDSinkhornMix"]) for r in rows),
        "toy_otd_vs_sinkhorn_max_abs_dev": max(abs(r["orc"]["OTD"] - r["orc"]["Sinkhorn"]) for r in rows),
        "toy_otd_vs_atd_max_abs_dev": max((abs(r["orc"]["OTD"] - r["orc"]["ATD"]) for r in rows
                                           if r["orc"]["ATD"] is not None), default=None),
        "toy_atd_failures": sum("ATD" in r["errors"] for r in rows),
    }
    return {"cases": rows, "checks": checks, "seconds_total": t_total}


def upstream_section():
    from citecurv.curvature.orc import orc_directed

    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (1, 3), (3, 1)])
    exp = {(0, 1): -0.5, (1, 2): 0.0, (1, 3): 0.5, (2, 3): 0.5, (3, 1): 0.75}  # upstream test-suite
    got = orc_directed(G, list(G.edges), method="OTD")
    return {"values": {key(k): v for k, v in got.items()}, "expected": {key(k): v for k, v in exp.items()},
            "max_abs_dev": max(abs(got[k] - exp[k]) for k in exp)}


def random_dag(n=200, p=0.04, seed=0):
    U = nx.gnp_random_graph(n, p, seed=seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from((max(a, b), min(a, b)) for a, b in U.edges)  # newer (higher id) cites older
    assert nx.is_directed_acyclic_graph(G)
    return G


def dag_section():
    import GraphRicciCurvature.OllivierRicci as gr_or
    from citecurv.curvature.orc import orc_directed

    G = random_dag()
    E = sorted(G.edges)
    out = {"n_nodes": G.number_of_nodes(), "n_edges": len(E), "seconds": {}}
    vals, out["errors"] = {}, {}
    for m in METHODS:
        r, out["seconds"][m], err = try_timed(orc_directed, G, E, method=m)
        if err:
            out["errors"][m] = err
        else:
            vals[m] = r
    old = gr_or._OTDSinkhorn_threshold
    gr_or._OTDSinkhorn_threshold = 3  # force the Sinkhorn branch whenever both supports exceed 3 nodes
    try:
        vals["Mix_forced"], out["seconds"]["Mix_forced"] = timed(orc_directed, G, E, method="OTDSinkhornMix")
    finally:
        gr_or._OTDSinkhorn_threshold = old
    vals["OTD_pairwise"], out["seconds"]["OTD_pairwise"] = timed(orc_directed, G, E, method="OTD",
                                                                 shortest_path="pairwise")
    otd = vals["OTD"]

    def dev(other):
        d = np.array([abs(other[e] - otd[e]) for e in E])
        return {"max_abs_dev": float(d.max()), "mean_abs_dev": float(d.mean()), "n_dev_gt_1e-9": int((d > TOL).sum())}

    out["vs_OTD"] = {m: dev(vals[m]) for m in vals if m != "OTD"}
    n_forced = sum(1 for u, v in E if G.in_degree(u) + 1 > 3 and G.out_degree(v) + 1 > 3)
    out["vs_OTD"]["Mix_forced"]["n_edges_on_sinkhorn_branch"] = n_forced

    R = G.reverse(copy=True)
    rev, out["seconds"]["reversed_all"] = timed(orc_directed, R, [(v, u) for u, v in E], method="OTD")
    out["full_reversal_max_abs_dev"] = max(abs(rev[(v, u)] - otd[(u, v)]) for u, v in E)

    flips, t0 = [], time.perf_counter()
    for u, v in E[:: max(1, len(E) // 60)]:
        Gf = G.copy()
        Gf.remove_edge(u, v)
        Gf.add_edge(v, u)
        kf = orc_directed(Gf, [(v, u)], method="OTD")[(v, u)]
        flips.append({"edge": key((u, v)), "orc": otd[(u, v)], "flipped_orc": kf,
                      "in_u": G.in_degree(u), "out_v": G.out_degree(v), "changed": abs(kf - otd[(u, v)]) > TOL})
    out["seconds"]["single_flips"] = time.perf_counter() - t0
    out["single_edge_flips"] = {"n": len(flips), "n_changed": sum(f["changed"] for f in flips), "rows": flips}

    und, out["seconds"]["undirected"] = timed(library_orc, G.to_undirected(), E, method="OTD")
    d_und = np.array([abs(und[e] - otd[e]) for e in E])
    out["directed_vs_undirected"] = {"max_abs_diff": float(d_und.max()), "n_differ": int((d_und > TOL).sum())}
    out["orc_otd"] = {key(e): otd[e] for e in E}
    out["orc_otd_summary"] = {"min": float(min(otd.values())), "max": float(max(otd.values())),
                              "mean": float(np.mean(list(otd.values())))}
    return out, G, otd


def mp_worker(proc, shortest_path, method, n_repeat):
    """Runs in a subprocess: DAG ORC with the given knobs, then n_repeat small calls."""
    from citecurv.curvature.orc import orc_directed

    G = random_dag()
    E = sorted(G.edges)
    vals, dt = timed(orc_directed, G, E, method=method, proc=proc, shortest_path=shortest_path)
    t0 = time.perf_counter()
    small = nx.DiGraph([("x1", "u"), ("x2", "u"), ("u", "v"), ("v", "r1")])
    for _ in range(n_repeat):
        orc_directed(small, [("u", "v")], method=method, proc=proc, shortest_path=shortest_path)
    rep = time.perf_counter() - t0
    print(json.dumps({"values": {key(e): vals[e] for e in E}, "seconds_dag": dt,
                      "seconds_per_small_call": rep / max(n_repeat, 1)}))


def mp_section(otd_ref, timeout_s=300):
    configs = [(1, "all_pairs", "OTD", 30), (2, "all_pairs", "OTD", 30), (4, "all_pairs", "OTD", 30),
               (2, "pairwise", "OTD", 30), (4, "all_pairs", "OTDSinkhornMix", 30)]
    rows = []
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    for proc, sp, method, n_rep in configs:
        t0 = time.perf_counter()
        row = {"proc": proc, "shortest_path": sp, "method": method, "repeated_small_calls": n_rep}
        try:
            res = subprocess.run([sys.executable, __file__, "--mp-worker", str(proc), sp, method, str(n_rep)],
                                 capture_output=True, text=True, timeout=timeout_s, env=env)
            row["wall_s"] = time.perf_counter() - t0
            if res.returncode != 0:
                row.update(status="error", stderr_tail=res.stderr[-800:])
            else:
                data = json.loads(res.stdout.strip().splitlines()[-1])
                dev = max(abs(data["values"][k] - otd_ref[k]) for k in otd_ref)
                row.update(status="ok", seconds_dag=data["seconds_dag"],
                           seconds_per_small_call=data["seconds_per_small_call"], max_abs_dev_vs_proc1=dev)
        except subprocess.TimeoutExpired:
            row.update(status="timeout", wall_s=time.perf_counter() - t0)
        rows.append(row)
        print("mp", row.get("proc"), sp, method, row["status"], round(row["wall_s"], 2), flush=True)
    return {"timeout_s": timeout_s, "runs": rows,
            "all_finished": all(r["status"] == "ok" for r in rows),
            "max_abs_dev_vs_proc1": max(r.get("max_abs_dev_vs_proc1", np.inf) for r in rows)}


def exactness_section(n_papers=600, n_sample=40, seed=0):
    import pandas as pd

    from citecurv.curvature.orc import escalate, orc_directed
    from citecurv.curvature.screen import forman_screen
    from citecurv.curvature.timeslice import EventStore
    from run_screen_benchmark import synthetic_citation_stream

    ev = synthetic_citation_stream(n_papers, mean_refs=10.0, papers_per_day=10, p_same_day=0.05, seed=seed)
    curv = forman_screen(ev)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(curv), size=n_sample, replace=False))
    curv["escalated"] = False
    curv.iloc[pick, curv.columns.get_loc("escalated")] = True
    store = EventStore(ev)
    src, dst, ts = ev["src_id"].to_numpy(), ev["dst_id"].to_numpy(), ev["t"].to_numpy()

    full, t_full = {}, 0.0
    for r in curv[curv["escalated"]].itertuples(index=False):
        m = (ts <= r.t) & ~((src == r.src_id) & (dst == r.dst_id))
        G = nx.DiGraph()
        G.add_edges_from(zip(src[m], dst[m]))
        G.add_edge(r.src_id, r.dst_id)
        val, dt = timed(orc_directed, G, [(r.src_id, r.dst_id)], method="OTD")
        full[int(r.event_id)] = val[(r.src_id, r.dst_id)]
        t_full += dt
    out = {"label": "SYNTHETIC stream (scripts/run_screen_benchmark.py generator)",
           "n_events": len(ev), "n_sampled": int(n_sample),
           "n_src_with_in_citations_in_sample": 0,
           "full_graph_orc_s_per_event": t_full / n_sample, "variants": {}}
    for radius, prune in ((1, False), (1, True), (2, False), (2, True)):
        res, st = escalate(ev, curv, store, radius=radius, edge_cap=5000, prune=prune, method="OTD")
        got = dict(zip(res.loc[res["escalated"], "event_id"].astype(int), res.loc[res["escalated"], "orc"]))
        d = np.array([abs(got[k] - full[k]) for k in full])
        out["variants"][f"r{radius}{'_prune' if prune else ''}"] = {
            "max_abs_dev_vs_full": float(d.max()), "n_differ": int((d > TOL).sum()),
            "mean_subgraph_edges": st["mean_subgraph_edges"], "max_subgraph_edges": st["max_subgraph_edges"],
            "mean_kept_edges": st["mean_kept_edges"], "n_capped": st["n_capped"],
            "n_capped_lossy": st["n_capped_lossy"], "orc_s_per_event": st["orc_s_per_event"],
            "subgraph_s_per_event": st["subgraph_s_per_event"],
            "values": {str(k): got[k] for k in full},
        }
    in_src = {}
    for r in curv[curv["escalated"]].itertuples(index=False):
        in_src[r.event_id] = int(((dst == r.src_id) & (ts <= r.t)).sum())
    out["n_src_with_in_citations_in_sample"] = int(sum(v > 0 for v in in_src.values()))
    out["full_graph_values"] = {str(k): v for k, v in full.items()}
    out["value_distribution_full"] = pd.Series(list(full.values())).describe().round(6).to_dict()
    return out


def behaviour_section():
    """Version-dependent behaviour, measured (see docs/orc_version_comparison.md)."""
    out = {}
    tri = nx.Graph()
    tri.add_weighted_edges_from([(0, 1, 10.0), (0, 2, 1.0), (2, 1, 1.0)])
    out["weighted_triangle_undirected_orc_0_1"] = library_orc(tri, [(0, 1)], method="OTD")[(0, 1)]
    dtri = nx.DiGraph()
    dtri.add_weighted_edges_from([(0, 1, 10.0), (0, 2, 1.0), (2, 1, 1.0), (1, 3, 1.0)])
    out["weighted_directed_orc_0_1"] = library_orc(dtri, [(0, 1)], method="OTD")[(0, 1)]
    for sp in ("all_pairs", "pairwise"):
        out[f"weighted_directed_orc_0_1_{sp}"] = library_orc(dtri, [(0, 1)], method="OTD", shortest_path=sp)[(0, 1)]

    # pairwise-mode LRU cache across two graphs in one process (0.6.1 test-suite scenario)
    second = nx.Graph()
    second.add_weighted_edges_from([(0, 1, 10.0), (0, 2, 1.0), (2, 1, 1.0), (1, 3, 1.0)])
    expected = library_orc(second, [(0, 1)], method="OTD", shortest_path="all_pairs")[(0, 1)]
    library_orc(nx.path_graph(4), [(0, 1)], method="OTD", shortest_path="pairwise")
    actual = library_orc(second, [(0, 1)], method="OTD", shortest_path="pairwise")[(0, 1)]
    out["pairwise_cache_two_graphs"] = {"expected_all_pairs": expected, "pairwise_after_other_graph": actual,
                                        "leaks": abs(actual - expected) > TOL}

    from citecurv.curvature.orc import orc_directed

    small = nx.DiGraph([("u", "v")])
    orc_directed(small, [("u", "v")])
    t0 = time.perf_counter()
    for _ in range(30):
        orc_directed(small, [("u", "v")])
    out["seconds_per_call_tiny_graph_proc1"] = (time.perf_counter() - t0) / 30
    return out


def run(out_path):
    t0 = time.perf_counter()
    res = {"label": "Directed ORC validation (S0.5 Block A / S1.5); toys and DAG are fixed test graphs, "
                    "the escalation-subgraph section uses a SYNTHETIC stream",
           "env": env_info()}
    print("env", res["env"], flush=True)
    res["toy"] = toy_section()
    print("toy", res["toy"]["checks"], flush=True)
    res["upstream_directed_example"] = upstream_section()
    dag, _, otd = dag_section()
    res["dag200"] = dag
    print("dag", dag["vs_OTD"], dag["full_reversal_max_abs_dev"], flush=True)
    res["multiprocessing"] = mp_section({key(e): v for e, v in otd.items()})
    res["escalation_subgraphs"] = exactness_section()
    print("exactness", {k: v["max_abs_dev_vs_full"] for k, v in res["escalation_subgraphs"]["variants"].items()},
          flush=True)
    res["behaviour"] = behaviour_section()
    print("behaviour", res["behaviour"], flush=True)
    toy_c = res["toy"]["checks"]
    esc = res["escalation_subgraphs"]["variants"]
    res["checks"] = {
        "toy_hand_values": toy_c["toy_hand_values"],
        "direction_sensitive": toy_c["toy_flip_values"] and toy_c["toy_flip_changes"] == 5
        and dag["single_edge_flips"]["n_changed"] > 0,
        "full_reversal_invariant": dag["full_reversal_max_abs_dev"] < TOL,
        "upstream_directed_example": res["upstream_directed_example"]["max_abs_dev"] < 1e-6,
        "otd_vs_mix_toy_lt_0.05": toy_c["toy_otd_vs_mix_max_abs_dev"] < MIX_THRESHOLD,
        "otd_vs_mix_dag_lt_0.05": dag["vs_OTD"]["OTDSinkhornMix"]["max_abs_dev"] < MIX_THRESHOLD,
        "multiprocessing_no_hang": res["multiprocessing"]["all_finished"]
        and res["multiprocessing"]["max_abs_dev_vs_proc1"] < TOL,
        "radius2_subgraph_exact": esc["r2"]["max_abs_dev_vs_full"] < TOL and esc["r2_prune"]["max_abs_dev_vs_full"] < TOL,
    }
    res["checks"]["block_A_pass"] = all(res["checks"].values())
    res["seconds_total"] = time.perf_counter() - t0
    print("checks", res["checks"], flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")
    print("wrote", out_path, f"({res['seconds_total']:.1f}s)")


def flatten(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


def compare(path_a, path_b):
    A, B = (json.loads(Path(p).read_text()) for p in (path_a, path_b))
    fa, fb = dict(flatten(A)), dict(flatten(B))
    def is_timing(k):
        last = k.rsplit("/", 1)[-1]
        return "second" in k or "wall" in last or last.endswith("_s") or "_s_per_" in last

    value_keys = [k for k in fa if k in fb and isinstance(fa[k], float) and not is_timing(k)
                  and not k.startswith("env")]
    diffs = sorted(((abs(fa[k] - fb[k]), k) for k in value_keys), reverse=True)
    print(f"A = {A['env']['GraphRicciCurvature']}  B = {B['env']['GraphRicciCurvature']}")
    print(f"value entries compared: {len(value_keys)}; max abs diff: {diffs[0][0]:.3g}")
    for d, k in diffs[:12]:
        if d > 0:
            print(f"  {d:.6g}  {k}: {fa[k]} vs {fb[k]}")
    print("checks A:", A["checks"])
    print("checks B:", B["checks"])
    for name in ("toy", "dag200"):
        print(name, "seconds A/B:", A[name].get("seconds_total", A[name].get("seconds")),
              B[name].get("seconds_total", B[name].get("seconds")))
    print("mp A:", [(r["proc"], r["shortest_path"], r["status"], round(r.get("seconds_dag", -1), 3),
                     round(r.get("seconds_per_small_call", -1), 4)) for r in A["multiprocessing"]["runs"]])
    print("mp B:", [(r["proc"], r["shortest_path"], r["status"], round(r.get("seconds_dag", -1), 3),
                     round(r.get("seconds_per_small_call", -1), 4)) for r in B["multiprocessing"]["runs"]])
    for name in ("r1", "r2", "r2_prune"):
        a, b = A["escalation_subgraphs"]["variants"][name], B["escalation_subgraphs"]["variants"][name]
        print(name, "orc s/event A/B:", round(a["orc_s_per_event"], 4), round(b["orc_s_per_event"], 4))
    print("behaviour A:", A["behaviour"])
    print("behaviour B:", B["behaviour"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--mp-worker", nargs=4)
    args = ap.parse_args()
    if args.mp_worker:
        proc, sp, method, n_rep = args.mp_worker
        mp_worker(int(proc), sp, method, int(n_rep))
    elif args.compare:
        compare(*args.compare)
    elif args.out:
        run(args.out)
    else:
        ap.error("give --out or --compare")


if __name__ == "__main__":
    main()

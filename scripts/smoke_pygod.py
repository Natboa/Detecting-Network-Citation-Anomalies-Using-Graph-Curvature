"""S0.5 Block D gate (PyGOD + clique injector) and an S4.4 PyGOD-tier smoke run on a synthetic stream.

Checks, each PASS/FAIL with its error text:
  import_detectors          from pygod.detector import DOMINANT, CoLA, GADNR
  neighbor_sampler_backend  pyg-lib or torch-sparse importable (PyGOD's fit always goes through PyG's
                            NeighborLoader, which needs one of them, even in full-batch mode)
  gen_structural_outlier    pygod.generator.gen_structural_outlier(directed=True) on a small directed graph
                            returns the modified graph and an outlier mask with m*n ones, and adds exactly
                            n*C(m,2) one-directional edges inside the cliques
  detector_<name>           citecurv.eval.baselines.pygod_tier.score_pygod on a cutoff snapshot of a synthetic
                            stream: one finite score per test event, identical across two runs with one seed

All data here is synthetic (Price-model stream, random node features). Nothing in the output is a measurement on
real citations. Usage (from the repo root):

    PYTHONPATH=src /home/user/venvs/a5-pygod/bin/python scripts/smoke_pygod.py --out results/smoke_pygod.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def version_of(module: str):
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - record any import failure as data
        return f"not importable ({type(exc).__name__})"
    return getattr(mod, "__version__", "unknown")


def synthetic_stream(n_papers: int = 400, refs: int = 8, seed: int = 0) -> pd.DataFrame:
    """Price-model citation stream (synthetic): half of each paper's references copy an earlier event's dst."""
    rng = np.random.default_rng(seed)
    pub = np.sort(rng.integers(0, 3650, size=n_papers))
    src, dst = [], []
    for i in range(10, n_papers):
        for _ in range(refs):
            if rng.random() < 0.5 or not dst:
                d = int(rng.integers(0, i))
            else:
                d = dst[int(rng.integers(0, len(dst)))]
            src.append(i)
            dst.append(d)
    ev = pd.DataFrame({"src": src, "dst": dst}).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame({
        "event_id": np.arange(len(ev), dtype=np.int64) * 10 + 1,
        "src_id": [f"W{s}" for s in ev["src"]],
        "dst_id": [f"W{d}" for d in ev["dst"]],
        "t": pub[ev["src"].to_numpy()].astype(np.int64),
    })


def run_check(name: str, fn, report: dict) -> None:
    start = time.perf_counter()
    try:
        detail = fn() or {}
        status = "PASS" if detail.pop("_pass", True) else "FAIL"
    except Exception as exc:  # noqa: BLE001 - a smoke gate records failures instead of stopping
        detail = {"error": f"{type(exc).__name__}: {exc}", "traceback_tail": traceback.format_exc().splitlines()[-4:]}
        status = "FAIL"
    detail["status"] = status
    detail["seconds"] = round(time.perf_counter() - start, 3)
    report["checks"][name] = detail
    print(f"{status}  {name}  ({detail['seconds']} s)" + (f"  {detail.get('error', '')}" if status == "FAIL" else ""))


def check_imports():
    from pygod.detector import CoLA, DOMINANT, GADNR  # noqa: F401

    return {"imported": ["DOMINANT", "CoLA", "GADNR"]}


def check_sampler_backend():
    found = {}
    for mod in ("pyg_lib", "torch_sparse"):
        try:
            importlib.import_module(mod)
            found[mod] = version_of(mod)
        except Exception as exc:  # noqa: BLE001
            found[mod] = f"missing ({type(exc).__name__})"
    ok = any(not str(v).startswith("missing") for v in found.values())
    return {"_pass": ok, "backends": found}


def check_structural_outlier():
    import torch
    from pygod.generator import gen_structural_outlier
    from torch_geometric.data import Data

    torch.manual_seed(1)
    n_nodes, m, n = 60, 5, 2
    edge_index = torch.randint(0, n_nodes, (2, 150))
    data = Data(x=torch.randn(n_nodes, 8), edge_index=edge_index.clone())
    before = edge_index.shape[1]
    # seed=7, not 0: pygod 1.1.0 seeds only `if seed:`, so seed=0 silently leaves the RNG unseeded
    out, mask = gen_structural_outlier(data, m=m, n=n, directed=True, seed=7)
    added = out.edge_index[:, before:]
    outliers = set(torch.nonzero(mask).flatten().tolist())
    inside = all(int(a) in outliers and int(b) in outliers for a, b in added.t().tolist())
    pairs = {tuple(sorted(p)) for p in added.t().tolist()}
    ok = (
        isinstance(out, Data)
        and int(mask.sum()) == m * n
        and added.shape[1] == n * m * (m - 1) // 2
        and len(pairs) == added.shape[1]
        and inside
    )
    return {
        "_pass": ok,
        "n_outlier_nodes": int(mask.sum()),
        "edges_before": before,
        "edges_added": int(added.shape[1]),
        "expected_added": n * m * (m - 1) // 2,
        "added_edges_inside_cliques": inside,
        "one_direction_per_pair": len(pairs) == added.shape[1],
        "note": "pygod 1.1.0 seeds the generator only when seed is truthy; seed=0 is not reproducible",
    }


def make_detector_check(name: str, snapshot, test_events, epoch: int):
    from citecurv.eval.baselines.pygod_tier import score_pygod

    def check():
        t0 = time.perf_counter()
        a = score_pygod(name, snapshot, test_events, seed=0, epoch=epoch)
        first = time.perf_counter() - t0
        b = score_pygod(name, snapshot, test_events, seed=0, epoch=epoch)
        same_ids = a["event_id"].tolist() == test_events["event_id"].tolist()
        finite = bool(np.isfinite(a["score"]).all())
        max_diff = float(np.max(np.abs(a["score"].to_numpy() - b["score"].to_numpy())))
        return {
            "_pass": same_ids and finite and max_diff <= 1e-6,
            "n_scored": int(len(a)),
            "epoch": epoch,
            "fit_and_score_seconds": round(first, 3),
            "finite": finite,
            "rerun_max_abs_diff": max_diff,
            "score_min": float(a["score"].min()),
            "score_max": float(a["score"].max()),
        }

    return check


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/smoke_pygod.json")
    ap.add_argument("--epoch", type=int, default=10)
    args = ap.parse_args()

    report = {
        "what": "S0.5 Block D gate + S4.4 PyGOD-tier smoke (synthetic data only)",
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "torch": version_of("torch"),
            "torch_geometric": version_of("torch_geometric"),
            "pygod": version_of("pygod"),
            "numpy": version_of("numpy"),
            "pyg_lib": version_of("pyg_lib"),
            "torch_sparse": version_of("torch_sparse"),
            "torch_scatter": version_of("torch_scatter"),
            "cuda_available": None,
        },
        "checks": {},
    }
    try:
        import torch

        torch.set_num_threads(1)
        report["environment"]["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass

    run_check("import_detectors", check_imports, report)
    run_check("neighbor_sampler_backend", check_sampler_backend, report)
    run_check("gen_structural_outlier", check_structural_outlier, report)

    events = synthetic_stream()
    cutoff = int(np.quantile(events["t"], 0.8))
    test_events = events[events["t"] > cutoff]
    node_ids = sorted(set(events["src_id"]) | set(events["dst_id"]), key=lambda s: int(s[1:]))
    feats = np.random.default_rng(0).normal(size=(len(node_ids), 32)).astype(np.float32)
    snap_info = {"n_events": int(len(events)), "cutoff": cutoff, "n_test_events": int(len(test_events)),
                 "n_nodes": len(node_ids), "feature_dim": 32, "features": "random normal (synthetic stand-in)"}
    try:
        from citecurv.eval.baselines.pygod_tier import snapshot_graph

        snapshot = snapshot_graph(events, cutoff, node_ids, feats)
        snap_info.update(snapshot.stats)
    except Exception as exc:  # noqa: BLE001
        snapshot = None
        snap_info["error"] = f"{type(exc).__name__}: {exc}"
    report["synthetic_snapshot"] = snap_info
    try:
        from citecurv.eval.baselines.pygod_tier import PYGOD_ADJUSTMENTS

        report["score_pygod_adjustments"] = list(PYGOD_ADJUSTMENTS)
    except Exception as exc:  # noqa: BLE001
        report["score_pygod_adjustments"] = f"unavailable: {exc}"

    for name in ("DOMINANT", "CoLA", "GADNR"):
        if snapshot is None:
            report["checks"][f"detector_{name}"] = {"status": "FAIL", "error": "snapshot not built"}
            continue
        run_check(f"detector_{name}", make_detector_check(name, snapshot, test_events, args.epoch), report)

    report["overall"] = "PASS" if all(c["status"] == "PASS" for c in report["checks"].values()) else "FAIL"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"overall {report['overall']} -> {out}")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

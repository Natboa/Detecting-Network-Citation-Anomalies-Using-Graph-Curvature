"""Block B compatibility gate (S0.5) on a SYNTHETIC citation stream, run locally on CPU.

DyGLib's Wikipedia data cannot be downloaded from this environment (zenodo.org is refused by the
egress proxy), so the gate runs on a synthetic stream instead:

1. generate a seeded synthetic citation stream (papers with 64-d topic embeddings; each new paper
   cites earlier papers with preferential attachment and topic homophily);
2. convert it with ``citecurv.tgn.convert.to_jodie``;
3. run 1: stock TGN through DyGLib's own ``train_link_prediction.py`` (patch P5 and the leak hook
   off), ``--epochs`` epochs;
4. run 2: the citecurv configuration (direction-aware messages, patch P5), same seed and epochs,
   with the memory leak hook on for every epoch;
5. ``citecurv.tgn.infer`` on run 2's checkpoint, leak hook on;
6. write ``results/smoke_tgn.json``. Every number in it is a measurement on synthetic data.

Usage (torch venv, from the repo root):
    PYTHONPATH=src nice -n 10 /home/user/venvs/torch/bin/python scripts/smoke_tgn.py --work-dir <scratch>
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def synthetic_citation_stream(
    n_papers: int = 3000,
    refs_mean: float = 5.0,
    dim: int = 64,
    n_topics: int = 8,
    homophily: float = 4.0,
    days: int = 5 * 365,
    t0: int = 16801,  # 2016-01-01 in days since 1970-01-01
    seed: int = 0,
):
    """Return ``(events, (ids, vectors))`` in the contract formats (SYNTHETIC).

    Papers get a topic and a unit-norm embedding (topic centroid + noise) and a publication day.
    Paper j cites ``1 + Poisson(refs_mean - 1)`` papers published on an earlier day, drawn without
    replacement with probability proportional to ``(in_degree + 1) * exp(homophily * cos)``, so
    the link-prediction task carries a learnable signal. ``event_id`` values are shuffled and
    non-positional on purpose.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(n_topics, dim))
    topics = rng.integers(0, n_topics, size=n_papers)
    vec = centroids[topics] + 0.7 * rng.normal(size=(n_papers, dim))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    pub = np.sort(rng.integers(0, days, size=n_papers)) + t0
    ids = np.array([f"W{j:05d}" for j in range(n_papers)])

    in_deg = np.zeros(n_papers)
    src, dst, ts = [], [], []
    for j in range(n_papers):
        n_prior = int(np.searchsorted(pub, pub[j], side="left"))  # strictly earlier days only
        if n_prior == 0:
            continue
        k = min(n_prior, 1 + rng.poisson(refs_mean - 1))
        w = (in_deg[:n_prior] + 1.0) * np.exp(homophily * (vec[:n_prior] @ vec[j]))
        cited = rng.choice(n_prior, size=k, replace=False, p=w / w.sum())
        for c in cited:
            src.append(ids[j])
            dst.append(ids[c])
            ts.append(pub[j])
        in_deg[cited] += 1
    n = len(src)
    events = pd.DataFrame(
        {
            "event_id": (rng.permutation(n) * 3 + 1000).astype(np.int64),
            "src_id": src,
            "dst_id": dst,
            "t": np.asarray(ts, dtype=np.int64),
        }
    )
    return events, (ids, vec.astype(np.float32))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", required=True, help="scratch directory for data, runs, checkpoints")
    ap.add_argument("--out", default=str(REPO / "results" / "smoke_tgn.json"))
    ap.add_argument("--n-papers", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--num-threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch

    from citecurv.tgn.convert import to_jodie
    from citecurv.tgn.infer import infer_tgn
    from citecurv.tgn.train import train_tgn

    work = Path(args.work_dir)
    data_dir = work / "data"
    name = "synthcite"
    t_start = time.time()
    events, emb = synthetic_citation_stream(n_papers=args.n_papers, seed=args.seed)
    meta = to_jodie(events, emb, data_dir, name)
    gen_s = time.time() - t_start

    common = dict(
        data_dir=data_dir, name=name, seed=args.seed, device="cpu", node_feat_dim=meta["node_feat_dim"],
        num_threads=args.num_threads, learning_rate=args.learning_rate,
    )
    stock = train_tgn(out_dir=work / "run_stock", epochs=args.epochs, direction_aware=False, leak_check=False, **common)
    cite = train_tgn(out_dir=work / "run_citecurv", epochs=args.epochs, direction_aware=True, leak_check=True, **common)

    t0 = time.time()
    scores, emb_path = infer_tgn(work / "run_citecurv", data_dir, name, out_dir=work / "infer", seed=args.seed,
                                 leak_check=True, num_threads=args.num_threads)
    infer_s = time.time() - t0

    def run_summary(m):
        keep = ("epochs", "test", "new_node_test", "wall_clock_s", "peak_rss_mb", "leak_check", "returncode")
        return {k: m.get(k) for k in keep}

    pos, neg = scores["p_link"].to_numpy(), scores["neg_p_link"].to_numpy()
    report = {
        "synthetic": True,
        "note": "SYNTHETIC citation stream generated by scripts/smoke_tgn.py; not real data. "
                "Wikipedia (DyGLib DG_data) was not downloadable here: zenodo.org refused by the egress proxy.",
        "gate": "S0.5 Block B (DyGLib stock TGN under torch 2.x) run locally on CPU",
        "pass": bool(stock["returncode"] == 0 and all(np.isfinite(e["train_loss"]) for e in stock["epochs"])),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": "cpu",
            "num_threads": args.num_threads,
            "dyglib_commit": "3aacc36b94b8d2d8293d70a74fdf6d39089b4163",
        },
        "patches_needed": {
            "P1": "utils/DataLoader.py feature dims derived from data (needed for any non-172 dim)",
            "P3": "utils/DataLoader.py random.sample(set) -> random.sample(sorted(set)) (Python >= 3.11 TypeError)",
            "P4": "utils/EarlyStopping.py torch.load(weights_only=False) for the memory-bank pickle (torch >= 2.6)",
            "P6": "utils/load_configs.py --dataset_name choices dropped (custom dataset names)",
            "P7": "seed_offset (wrapper determinism; not a torch/CPU need)",
            "P5": "direction-aware messages, opt-in; off in the stock run",
        },
        "dataset": {k: meta[k] for k in ("n_events", "n_nodes", "node_feat_dim", "edge_feat_dim", "t_min", "t_max")},
        "generation_s": round(gen_s, 2),
        "run_stock": {"args": stock["resolved_args_subset"], **run_summary(stock)},
        "run_citecurv_leak_check": {"args": cite["resolved_args_subset"], **run_summary(cite)},
        "infer": {
            "n_scored": int(len(scores)),
            "wall_clock_s": round(infer_s, 2),
            "mean_p_link": float(pos.mean()),
            "mean_neg_p_link": float(neg.mean()),
            "frac_p_link_gt_neg": float((pos > neg).mean()),
            "embeddings_file": emb_path.name,
            "leak_check": json.loads((work / "infer" / "tgn_infer_meta.json").read_text())["leak_check"],
            "peak_rss_mb_driver_process": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

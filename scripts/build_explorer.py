"""Build the results explorer page (one offline HTML file).

Demo page from synthetic data (written under data/, which is gitignored):
    PYTHONPATH=src python scripts/build_explorer.py --demo

From pipeline outputs (parquet/JSON/CSV paths; only --scores is required):
    PYTHONPATH=src python scripts/build_explorer.py --scores data/runs/full_0/scores.parquet \
        --events data/edges_events.parquet --works data/works.parquet --labels data/event_labels.parquet \
        --curvature data/curvature_cache.parquet --metrics runs/full_0/metrics.json \
        --weights runs/full_0/weights.json --zstats runs/full_0/zstats.json \
        --table "Main comparison=results/main_table.csv" --data-note "" --out data/explorer/full_0.html

Cartel members for the synthetic injection track: --cartels members.parquet (cartel_id, work_id, role, level)
and optionally --cartel-events truth_ids.json (list of injected event ids).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from citecurv.viz import build_explorer, make_demo_data  # noqa: E402


def _read(path: str | None):
    if not path:
        return None
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise SystemExit(f"unsupported file type: {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true", help="build from synthetic demo data")
    ap.add_argument("--demo-papers", type=int, default=1500, help="papers in the synthetic demo stream")
    ap.add_argument("--out", default=None, help="output HTML (default data/explorer/explorer_demo.html or explorer.html)")
    for name in ("scores", "components", "curvature", "events", "works", "authorships", "labels", "metrics",
                 "weights", "zstats", "cartels", "cartel-events"):
        ap.add_argument(f"--{name}", default=None, help=f"path to the {name} artifact (.parquet/.csv/.json)")
    ap.add_argument("--table", action="append", default=[], metavar="NAME=PATH", help="comparison/ablation table")
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--radius", type=int, default=1, choices=(1, 2))
    ap.add_argument("--max-nodes", type=int, default=40)
    ap.add_argument("--flag-share", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Citation anomaly explorer")
    ap.add_argument("--data-note", default=None,
                    help='banner text; any note containing "synthetic" shows the synthetic-data banner')
    a = ap.parse_args(argv)

    common = dict(top_n=a.top_n, radius=a.radius, max_nodes=a.max_nodes, flag_share=a.flag_share, seed=a.seed,
                  title=a.title)
    if a.demo or not a.scores:
        if not a.demo:
            print("no --scores given: building the synthetic demo page", file=sys.stderr)
        data = make_demo_data(seed=a.seed, n_papers=a.demo_papers)
        out = Path(a.out or REPO / "data" / "explorer" / "explorer_demo.html")
        note = a.data_note if a.data_note is not None else "synthetic demo data"
        path = build_explorer(out, data_note=note, **data, **common)
    else:
        tables = {}
        for spec in a.table:
            name, _, path = spec.partition("=")
            tables[name] = _read(path)
        cartels = _read(a.cartels)
        if cartels is not None and a.cartel_events:
            cartels = {"members": cartels, "event_ids": _read(a.cartel_events)}
        out = Path(a.out or REPO / "data" / "explorer" / "explorer.html")
        note = a.data_note if a.data_note is not None else ""
        path = build_explorer(
            out, _read(a.scores), components=_read(a.components), curvature=_read(a.curvature),
            events=_read(a.events), works=_read(a.works), authorships=_read(a.authorships), labels=_read(a.labels),
            metrics=_read(a.metrics), tables=tables or None, cartels=cartels, weights=_read(a.weights),
            zstats=_read(a.zstats), data_note=note, **common,
        )
    size = path.stat().st_size / 1e6
    print(f"wrote {path} ({size:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

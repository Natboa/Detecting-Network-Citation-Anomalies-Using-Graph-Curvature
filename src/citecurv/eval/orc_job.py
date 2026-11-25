"""Child-process job for the S4.3 ORC precompute: `python -m citecurv.eval.orc_job <job.json>`.

The notebook starts this module in its own process group (subprocess.Popen(..., start_new_session=True)) and
watches its heartbeat file; on a stall it kills the whole group (os.killpg), which also ends the ORC worker and
GraphRicciCurvature's pool. The kernel itself holds nothing heavy, so an out-of-memory kill of the job cannot take
the kernel's watchdog with it, and the job's memory goes back to the system when it exits.

job.json keys: events, works, cited_joint (paths; cited_joint read with work_id, pub_date, ref_ids), shuffle (null
or {"cutoff", "seed"}: the job applies pipeline.shuffle_test_timestamps itself, no shuffled file is written),
config (YAML path), cutoff, seeds, out_path, known (null or a checkpoint path), known_max_t, batch, chunk, worker
(bool: run the ORC function in a spawned OrcWorker), order_seed, max_chunks, heartbeat (local path), result (path
of the result JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def run_job(job: dict, *, orc_fn=None, freeze_gc: bool = False) -> dict:
    """Load the inputs named in `job`, run pipeline.precompute_matrix_orc, write the result JSON; returns it."""
    from citecurv import pipeline as P
    from citecurv.config import load_config

    hb = job.get("heartbeat")
    P._write_heartbeat(hb, {"phase": "load", "in_flight": [], "clean": False})
    events = pd.read_parquet(job["events"])
    works = pd.read_parquet(job["works"], columns=["work_id"])
    refs = pd.read_parquet(job["cited_joint"], columns=["work_id", "pub_date", "ref_ids"])
    context, ctx_stats = P.build_context(works, refs, events)
    del refs
    if job.get("shuffle"):
        sh = job["shuffle"]
        events = P.shuffle_test_timestamps(events, int(sh["cutoff"]), seed=int(sh["seed"]))
    rcfg = P.resolve_config(load_config(job["config"]), variant="full")
    worker = None
    if job.get("worker"):
        from citecurv.curvature.orc import OrcWorker

        worker = OrcWorker("citecurv.curvature.orc:orc_directed", **rcfg["curvature"]["orc"])
    try:
        res = P.precompute_matrix_orc(
            events, int(job["cutoff"]), rcfg, list(job.get("seeds") or []), out_path=job["out_path"],
            context=context, known=job.get("known"), known_max_t=job.get("known_max_t"), orc_fn=orc_fn,
            chunk=int(job.get("chunk") or 5000), batch=int(job.get("batch") or 1), worker=worker,
            order_seed=int(job.get("order_seed") or 0), max_chunks=job.get("max_chunks"), heartbeat=hb,
            freeze_gc=freeze_gc)
    finally:
        if worker is not None:
            worker.close()
    res = {**res, "context_stats": ctx_stats, "context_hash": P.context_hash(context)}
    if job.get("result"):
        Path(job["result"]).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return res


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m citecurv.eval.orc_job <job.json>", file=sys.stderr)
        return 2
    run_job(json.loads(Path(argv[0]).read_text(encoding="utf-8")), freeze_gc=True)
    return 0


if __name__ == "__main__":  # a spawn child re-imports this module as __mp_main__
    sys.exit(main())

"""Thin wrapper around the vendored DyGLib link-prediction entry point (task S1.7).

``train_tgn(data_dir, name, epochs, seed, device, node_feat_dim, out_dir, **dyglib_args)`` trains
TGN with ``third_party/DyGLib/train_link_prediction.py`` on a dataset written by
``citecurv.tgn.convert.to_jodie`` and leaves in ``out_dir``: ``checkpoint/`` (DyGLib's best-on-
validation state dict and memory-message pickle), ``metrics.json`` (per-epoch loss and AP/AUC,
final test metrics, wall clock, peak RSS, leak-check counters) and ``resolved_args.json`` (every
argument DyGLib ran with, plus the wrapper's own).

Subprocess, not import. The entry point is a script whose logic sits under ``if __name__ ==
"__main__"``, resolves data and output paths relative to the working directory, imports top-level
packages named ``models`` and ``utils``, and ends with ``sys.exit()``. Running it in a child
process (``python -m citecurv.tgn.train --child cfg.json``) keeps those names out of the caller's
``sys.modules``, gives it a private working directory, lets a crash or an out-of-memory kill end
only the child, and makes the peak RSS a per-run number. Inside the child the script runs
unchanged through ``runpy``; the child only installs observers first (argument capture, a log
handler and, on request, the memory leak hook) so nothing in the vendored code is bypassed.

Determinism (NFR4). The seed reaches DyGLib as ``--seed_offset`` (patch P7) with ``--num_runs 1``;
DyGLib's ``set_random_seed`` seeds ``random``, numpy and torch and turns off cuDNN autotuning. The
child also calls ``torch.use_deterministic_algorithms(True, warn_only=True)`` and fixes the CPU
thread count when ``num_threads`` is given. DyGLib fixes its own evaluation seeds (neighbour
samplers 0/1, evaluation negatives 0-3) independently of the run seed.

Leak argument (NFR3). Two separate questions. (1) Does the memory see the future within a run?
``install_memory_leak_check`` asserts, before every batch's embeddings are computed, that every
pending memory message and every node's last memory update has ``ts`` no greater than the
earliest event time in that batch (run it with ``leak_check=True``). (2) Do the weights see
evaluation-period events? DyGLib splits by event-time quantiles (``val_ratio``/``test_ratio``)
and trains only on the first part; ``dyglib_split_ratios`` turns the project's train/val end
dates into ratios that put DyGLib's boundaries exactly on those dates, so the encoder is never
fitted on validation or test events. DyGLib additionally drops from training every edge that
touches a random 10 % of nodes seen after the training period (its inductive-evaluation set).
"""

from __future__ import annotations

import json
import logging
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from citecurv.config import REPO_ROOT

DYGLIB_ROOT_ENV = "CITECURV_DYGLIB_ROOT"
ENTRY_POINT = "train_link_prediction.py"

# DyGLib's own best TGN configuration (utils/load_configs.py, load_link_prediction_best_configs:
# num_neighbors=10, num_layers=1, recent sampling, dropout 0.1) plus its argparse defaults.
TGN_DEFAULTS = {
    "model_name": "TGN",
    "batch_size": 200,
    "num_neighbors": 10,
    "num_layers": 1,
    "num_heads": 2,
    "dropout": 0.1,
    "time_feat_dim": 100,
    "sample_neighbor_strategy": "recent",
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "weight_decay": 0.0,
    "patience": 20,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "negative_sample_strategy": "random",
}


# --------------------------------------------------------------------------- locating DyGLib

def resolve_dyglib_root(dyglib_root=None) -> Path:
    """Vendored DyGLib folder: argument, else $CITECURV_DYGLIB_ROOT, else <repo>/third_party/DyGLib."""
    root = Path(dyglib_root or os.environ.get(DYGLIB_ROOT_ENV) or REPO_ROOT / "third_party" / "DyGLib").resolve()
    if not (root / ENTRY_POINT).is_file():
        raise FileNotFoundError(
            f"DyGLib entry point not found under {root}; pass dyglib_root= or set ${DYGLIB_ROOT_ENV} "
            "to the third_party/DyGLib folder of a repository checkout"
        )
    return root


def _module_location(mod):
    f = getattr(mod, "__file__", None)
    if f:
        return Path(f).resolve()
    paths = list(getattr(mod, "__path__", []))
    return Path(paths[0]).resolve() if paths else None


def import_dyglib(dyglib_root=None) -> SimpleNamespace:
    """Import DyGLib's TGN pieces in this process (``models.*`` and ``utils.*`` become top-level names).

    Raises ImportError if another package already owns the names ``models`` or ``utils``.
    """
    root = resolve_dyglib_root(dyglib_root)
    for name in ("models", "utils"):
        mod = sys.modules.get(name)
        if mod is not None:
            loc = _module_location(mod)
            if loc is None or not loc.is_relative_to(root):
                raise ImportError(f"top-level '{name}' is already imported from {loc}; DyGLib needs that name")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib

    mm = importlib.import_module("models.MemoryModel")
    modules = importlib.import_module("models.modules")
    uu = importlib.import_module("utils.utils")
    dl = importlib.import_module("utils.DataLoader")
    for mod in (mm, uu):
        if not _module_location(mod).is_relative_to(root):
            raise ImportError(f"{mod.__name__} resolved to {_module_location(mod)}, not to {root}")
    return SimpleNamespace(
        root=root,
        MemoryModel=mm.MemoryModel,
        MergeLayer=modules.MergeLayer,
        get_neighbor_sampler=uu.get_neighbor_sampler,
        Data=dl.Data,
    )


# --------------------------------------------------------------------------- leak hook

class MemoryLeakError(AssertionError):
    """The memory module was about to use a message from after the current batch."""


class MemoryLeakCheck:
    """Counters of the leak hook; ``remove()`` restores the unpatched method."""

    def __init__(self, cls, original):
        self._cls, self._original = cls, original
        self.n_batches = 0          # calls of compute_src_dst_node_temporal_embeddings
        self.n_positive_batches = 0  # calls that also update memory
        self.n_pending_checked = 0  # pending messages inspected
        self.max_margin = None      # max over calls of (latest ts reaching memory - batch start time)
        self.violations = 0

    def remove(self):
        self._cls.compute_src_dst_node_temporal_embeddings = self._original

    def as_dict(self):
        return {
            "enabled": True,
            "n_batches": self.n_batches,
            "n_positive_batches": self.n_positive_batches,
            "n_pending_messages_checked": self.n_pending_checked,
            "max_margin": self.max_margin,
            "violations": self.violations,  # a violation also raises MemoryLeakError
        }


def install_memory_leak_check(memory_model_cls) -> MemoryLeakCheck:
    """Wrap ``MemoryModel.compute_src_dst_node_temporal_embeddings`` with a no-future assertion.

    Before each call (one call per batch and per positive/negative pass) the hook takes the batch
    start time ``t_b = min(node_interact_times)`` and asserts that (a) every pending raw message,
    which ``get_updated_memories`` folds into the memory used for this batch, has ``ts <= t_b``,
    and (b) every node's last memory update time is ``<= t_b``. After a positive call it asserts
    that the messages just stored are not later than the batch's last event. Together these say
    that the memory used to embed any event of the batch reflects only events with
    ``ts <= t_b <= t`` (no event later than the batch start). Raises ``MemoryLeakError``.
    """
    original = memory_model_cls.compute_src_dst_node_temporal_embeddings
    stats = MemoryLeakCheck(memory_model_cls, original)

    def checked(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids=None, edges_are_positive=True,
                num_neighbors=20):
        t_b = float(np.min(node_interact_times))
        latest = float(self.memory_bank.node_last_updated_times.detach().max().item())
        n_pending = 0
        for msgs in self.memory_bank.node_raw_messages.values():
            for _, ts in msgs:
                n_pending += 1
                latest = max(latest, float(ts))
        stats.n_batches += 1
        stats.n_pending_checked += n_pending
        margin = latest - t_b
        stats.max_margin = margin if stats.max_margin is None else max(stats.max_margin, margin)
        if latest > t_b:
            stats.violations += 1
            raise MemoryLeakError(f"memory would use ts={latest} for a batch starting at t={t_b}")
        out = original(self, src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                       node_interact_times=node_interact_times, edge_ids=edge_ids,
                       edges_are_positive=edges_are_positive, num_neighbors=num_neighbors)
        if edges_are_positive:
            stats.n_positive_batches += 1
            t_end = float(np.max(node_interact_times))
            stored = max((float(ts) for msgs in self.memory_bank.node_raw_messages.values() for _, ts in msgs),
                         default=-np.inf)
            if stored > t_end:
                stats.violations += 1
                raise MemoryLeakError(f"stored a message at ts={stored} after a batch ending at t={t_end}")
        return out

    memory_model_cls.compute_src_dst_node_temporal_embeddings = checked
    return stats


# --------------------------------------------------------------------------- split helper

def dyglib_split_ratios(ts, train_end_t, val_end_t) -> tuple[float, float]:
    """Return ``(val_ratio, test_ratio)`` so DyGLib trains on ``ts <= train_end_t`` and validates on
    ``train_end_t < ts <= val_end_t`` (its test part is then ``ts > val_end_t``).

    DyGLib sets ``val_time, test_time = np.quantile(ts, [1 - val_ratio - test_ratio, 1 - test_ratio])``
    and trains on ``ts <= val_time``. Each quantile is placed half an order statistic past the last
    event inside the period, so the cut falls strictly between two distinct timestamps; the result
    is checked against DyGLib's own formula before returning.
    """
    ts = np.sort(np.asarray(ts, dtype=np.float64))
    n = len(ts)
    k1 = int(np.searchsorted(ts, train_end_t, side="right"))
    k2 = int(np.searchsorted(ts, val_end_t, side="right"))
    if not (0 < k1 < k2 < n):
        raise ValueError(f"need events in all three periods; got train={k1}, val={k2 - k1}, test={n - k2}")
    q1, q2 = (k1 - 0.5) / (n - 1), (k2 - 0.5) / (n - 1)
    val_ratio, test_ratio = q2 - q1, 1.0 - q2
    val_time, test_time = np.quantile(ts, [1 - val_ratio - test_ratio, 1 - test_ratio])
    if int((ts <= val_time).sum()) != k1 or int((ts <= test_time).sum()) != k2:
        raise ValueError("could not place DyGLib's quantile cuts on the requested dates")
    return float(val_ratio), float(test_ratio)


# --------------------------------------------------------------------------- log parsing

_RE_EPOCH = re.compile(r"^Epoch: (\d+), learning rate: ([^,]+), train loss: ([-\d.e+naninf]+)$")
_RE_METRIC = re.compile(r"^(train|validate|new node validate|test|new node test) (\w+), ([-\d.e+naninf]+)$")
_RE_LOSS = re.compile(r"^(validate|new node validate|test|new node test) loss: ([-\d.e+naninf]+)$")
_PREFIX = {"train": "train", "validate": "val", "new node validate": "new_node_val", "test": "test",
           "new node test": "new_node_test"}
_SHORT = {"average_precision": "ap", "roc_auc": "auc"}


def parse_dyglib_messages(records) -> dict:
    """Turn DyGLib's root-logger messages ``[(created_time, message), ...]`` into metrics.

    Returns ``{"epochs": [...], "test": {...}, "new_node_test": {...}, "best_epoch": k}``. Each epoch
    has ``train_loss``, ``train_ap``, ``train_auc``, ``val_loss``, ``val_ap``, ``val_auc``,
    ``new_node_val_*`` and ``wall_clock_s`` (time since the previous epoch's summary line, or since
    the run started; it covers the training pass and both validation passes). DyGLib logs values
    rounded to 4 decimals. Final test metrics are the ones logged after "get final performance".
    """
    epochs, final, best_epoch = [], {"test": {}, "new_node_test": {}}, None
    last_t, in_final = None, False
    for created, msg in records:
        msg = msg.strip()
        if msg.startswith("********** Run") and "starts" in msg:
            last_t = created
            continue
        if msg.startswith("get final performance"):
            in_final = True
            continue
        if msg.startswith("save model"):
            best_epoch = len(epochs)
            continue
        m = _RE_EPOCH.match(msg)
        if m:
            epochs.append({
                "epoch": int(m.group(1)),
                "learning_rate": float(m.group(2)),
                "train_loss": float(m.group(3)),
                "wall_clock_s": None if last_t is None else round(created - last_t, 3),
            })
            last_t = created
            continue
        m = _RE_LOSS.match(msg)
        if m:
            part, val = _PREFIX[m.group(1)], float(m.group(2))
            if in_final and part in final:
                final[part]["loss"] = val
            elif epochs and part in ("val", "new_node_val"):
                epochs[-1][f"{part}_loss"] = val
            continue
        m = _RE_METRIC.match(msg)
        if m:
            part, name, val = _PREFIX[m.group(1)], _SHORT.get(m.group(2), m.group(2)), float(m.group(3))
            if in_final and part in final:
                final[part][name] = val
            elif epochs and part in ("train", "val", "new_node_val"):
                epochs[-1][f"{part}_{name}"] = val
    return {"epochs": epochs, "test": final["test"], "new_node_test": final["new_node_test"],
            "best_epoch": best_epoch}


# --------------------------------------------------------------------------- parent side

def _cli_args(params: dict) -> list[str]:
    out = []
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        elif v is not None:
            out += [f"--{k}", str(v)]
    return out


def _gpu_index(device: str) -> int:
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")


def _link_dataset(data_dir: Path, work: Path, name: str) -> None:
    target = work / "processed_data" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == data_dir.resolve():
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    try:
        target.symlink_to(data_dir.resolve(), target_is_directory=True)
    except OSError:  # file systems without symlinks (e.g. a mounted Drive): copy the three files
        target.mkdir()
        for suffix in (".csv", ".npy", "_node.npy"):
            shutil.copy2(data_dir / f"ml_{name}{suffix}", target / f"ml_{name}{suffix}")


def train_tgn(
    data_dir,
    name: str,
    epochs: int = 50,
    seed: int = 0,
    device: str = "cpu",
    node_feat_dim: int | None = None,
    out_dir=None,
    *,
    dyglib_root=None,
    direction_aware: bool = True,
    leak_check: bool = False,
    deterministic: bool = True,
    num_threads: int | None = None,
    python: str | None = None,
    timeout: float | None = None,
    **dyglib_args,
) -> dict:
    """Train DyGLib TGN on ``data_dir/ml_{name}.*`` and return the metrics dict (also in metrics.json).

    Args:
        data_dir: folder written by ``to_jodie`` (``ml_{name}.csv``, ``.npy``, ``_node.npy``, meta).
        name: dataset name.
        epochs: DyGLib ``--num_epochs`` (early stopping with ``patience`` still applies).
        seed: run seed (DyGLib ``--seed_offset``, patch P7).
        device: ``"cpu"``, ``"cuda"`` or ``"cuda:N"``; DyGLib falls back to CPU without CUDA, and
            the resolved device is recorded.
        node_feat_dim: expected node feature width; checked against ``ml_{name}_node.npy`` (since
            patch P1 DyGLib takes the width from the data, so this is a guard, not a setting).
        out_dir: run folder (default ``data_dir/../runs/TGN_{name}_seed{seed}``).
        dyglib_root: vendored DyGLib folder (see ``resolve_dyglib_root``).
        direction_aware: pass the meta file's ``direction_feat_index`` to DyGLib (patch P5); False
            runs upstream TGN message passing.
        leak_check: install ``install_memory_leak_check`` in the child.
        deterministic: ``torch.use_deterministic_algorithms(True, warn_only=True)`` in the child.
        num_threads: torch CPU threads in the child (also sets OMP/MKL thread variables).
        python: interpreter for the child (default: this one).
        timeout: seconds before the child is killed.
        **dyglib_args: any other ``train_link_prediction.py`` argument, e.g. ``learning_rate=1e-3``,
            ``batch_size=200``, ``val_ratio``/``test_ratio`` (see ``dyglib_split_ratios``).
    """
    data_dir = Path(data_dir).resolve()
    for suffix in (".csv", ".npy", "_node.npy"):
        if not (data_dir / f"ml_{name}{suffix}").is_file():
            raise FileNotFoundError(f"missing {data_dir / f'ml_{name}{suffix}'}")
    width = int(np.load(data_dir / f"ml_{name}_node.npy", mmap_mode="r").shape[1])
    if node_feat_dim is not None and int(node_feat_dim) != width:
        raise ValueError(f"node_feat_dim={node_feat_dim} but ml_{name}_node.npy has width {width}")
    meta_path = data_dir / f"ml_{name}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if direction_aware and "direction_feat_index" not in meta:
        raise ValueError(f"direction_aware=True needs {meta_path} with direction_feat_index (written by to_jodie)")

    root = resolve_dyglib_root(dyglib_root)
    out = Path(out_dir) if out_dir is not None else data_dir.parent / "runs" / f"TGN_{name}_seed{seed}"
    out = out.resolve()
    work = out / "dyglib_work"
    work.mkdir(parents=True, exist_ok=True)
    _link_dataset(data_dir, work, name)

    params = dict(TGN_DEFAULTS)
    params.update(dyglib_args)
    params.update(
        dataset_name=name,
        num_epochs=int(epochs),
        num_runs=1,
        seed_offset=int(seed),
        gpu=_gpu_index(device),
        test_interval_epochs=int(epochs) + 1,  # no mid-training test passes; one final test pass
        direction_feat_index=int(meta["direction_feat_index"]) if direction_aware else -1,
    )
    child_cfg = {
        "dyglib_root": str(root),
        "argv": _cli_args(params),
        "leak_check": bool(leak_check),
        "deterministic": bool(deterministic),
        "num_threads": num_threads,
        "report_path": str(out / "child_report.json"),
    }
    cfg_path = out / "child_config.json"
    cfg_path.write_text(json.dumps(child_cfg, indent=2), encoding="utf-8")

    env = dict(os.environ)
    src_dir = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join([src_dir, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else src_dir
    if num_threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[var] = str(num_threads)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by deterministic cuBLAS on GPU

    cmd = [python or sys.executable, "-m", "citecurv.tgn.train", "--child", str(cfg_path)]
    Path(child_cfg["report_path"]).unlink(missing_ok=True)  # never read a previous run's report
    t0 = time.time()
    with open(out / "dyglib_stdout.log", "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    wall = time.time() - t0

    report_path = Path(child_cfg["report_path"])
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    if proc.returncode != 0:
        tail = (out / "dyglib_stdout.log").read_text(errors="replace").splitlines()[-25:]
        raise RuntimeError(f"DyGLib training failed (exit {proc.returncode}); log tail:\n" + "\n".join(tail))

    parsed = parse_dyglib_messages(report.get("log_records", []))
    model_name = params["model_name"]
    ckpt_src = work / "saved_models" / model_name / name / f"{model_name}_seed{seed}"
    ckpt_dst = out / "checkpoint"
    if ckpt_dst.exists():
        shutil.rmtree(ckpt_dst)
    shutil.move(str(ckpt_src), str(ckpt_dst))

    resolved = report.get("resolved_args", {})
    resolved_all = {
        "dyglib": resolved,
        "wrapper": {
            "data_dir": str(data_dir), "name": name, "epochs": int(epochs), "seed": int(seed), "device": device,
            "node_feat_dim": width, "direction_aware": direction_aware, "leak_check": leak_check,
            "deterministic": deterministic, "num_threads": num_threads, "dyglib_root": str(root),
            "command": cmd, "dyglib_argv": child_cfg["argv"],
        },
    }
    (out / "resolved_args.json").write_text(json.dumps(resolved_all, indent=2), encoding="utf-8")

    keys = ("model_name", "batch_size", "num_neighbors", "num_layers", "num_heads", "dropout", "time_feat_dim",
            "sample_neighbor_strategy", "learning_rate", "num_epochs", "val_ratio", "test_ratio", "seed",
            "device", "direction_feat_index")
    metrics = {
        "name": name,
        "seed": int(seed),
        **parsed,
        "wall_clock_s": round(wall, 2),
        "child_wall_clock_s": report.get("wall_clock_s"),
        "peak_rss_mb": report.get("peak_rss_mb"),
        "cuda_peak_mb": report.get("cuda_peak_mb"),
        "torch_version": report.get("torch_version"),
        "device_resolved": resolved.get("device"),
        "leak_check": report.get("leak_check") or {"enabled": False},
        "returncode": proc.returncode,
        "checkpoint": {  # relative to the run folder, so a copied folder stays usable
            "model": f"checkpoint/{model_name}_seed{seed}.pkl",
            "memory_messages": f"checkpoint/{model_name}_seed{seed}_nonparametric_data.pkl",
        },
        "resolved_args_subset": {k: resolved.get(k) for k in keys},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# --------------------------------------------------------------------------- child side

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append((record.created, record.getMessage()))


def _child_main(cfg_path: str) -> int:
    import runpy

    cfg = json.loads(Path(cfg_path).read_text())
    t0 = time.time()
    report = {"returncode": None}
    root = Path(cfg["dyglib_root"])
    sys.path.insert(0, str(root))

    import torch

    report["torch_version"] = torch.__version__
    if cfg.get("num_threads"):
        torch.set_num_threads(int(cfg["num_threads"]))
    if cfg.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    import utils.load_configs as load_configs

    captured = []  # the argparse namespace; read at exit because DyGLib sets args.seed after parsing
    original_get_args = load_configs.get_link_prediction_args

    def get_args_and_record(*a, **k):
        args = original_get_args(*a, **k)
        captured.append(args)
        return args

    load_configs.get_link_prediction_args = get_args_and_record
    capture = _Capture()
    logging.getLogger().addHandler(capture)

    stats = None
    if cfg.get("leak_check"):
        from models.MemoryModel import MemoryModel

        stats = install_memory_leak_check(MemoryModel)

    script = root / ENTRY_POINT
    sys.argv = [str(script), *cfg["argv"]]
    rc = 1
    try:
        runpy.run_path(str(script), run_name="__main__")
        rc = 0
    except SystemExit as e:
        rc = 0 if e.code in (None, 0) else (e.code if isinstance(e.code, int) else 1)
    finally:
        report.update(
            returncode=rc,
            resolved_args={key: val for key, val in vars(captured[0]).items()
                           if isinstance(val, (str, int, float, bool, type(None)))} if captured else {},
            log_records=capture.records,
            leak_check=stats.as_dict() if stats is not None else {"enabled": False},
            peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
            cuda_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
            wall_clock_s=round(time.time() - t0, 2),
        )
        Path(cfg["report_path"]).write_text(json.dumps(report), encoding="utf-8")
    return rc


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        sys.exit(_child_main(sys.argv[2]))
    sys.exit("usage: python -m citecurv.tgn.train --child <config.json>  (internal; call train_tgn() instead)")

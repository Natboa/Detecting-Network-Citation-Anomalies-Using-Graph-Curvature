"""Streaming TGN scores and endpoint embeddings for every citation event (task S1.7).

``infer_tgn(run_dir, data_dir, name)`` loads a checkpoint written by ``citecurv.tgn.train.train_tgn``,
replays the converted event stream in canonical ``(t, event_id)`` order from an empty memory, and
writes the contract outputs (CONTRACTS.md, "TGN outputs"):

- ``tgn_event_scores`` (``tgn_event_scores.parquet``, or ``.csv`` when pyarrow is absent):
  ``event_id``, ``p_link`` (decoder probability of the observed citation), ``neg_p_link`` (the same
  for the citing paper and one seeded random destination, drawn uniformly from all papers that are
  ever cited in the input, as DyGLib's "random" negative strategy does);
- ``tgn_event_embeddings.npz``: ``event_id`` (M,), ``z_src`` (M, d), ``z_dst`` (M, d), the
  temporal embeddings of the two endpoints that produced ``p_link``.
These feed the ``s_embed`` and ``s_curvdyn`` score components.

What "before the batch's memory update" means. The loop mirrors DyGLib's evaluation loop
(``evaluate_models_utils.evaluate_model_link_prediction``): for each batch B it first embeds the
negative pairs (``edges_are_positive=False``, memory untouched), then the observed pairs
(``edges_are_positive=True``). Inside ``MemoryModel.compute_src_dst_node_temporal_embeddings`` the
embeddings are computed from ``get_updated_memories``, i.e. the stored memories plus the pending
messages of earlier batches folded in (one message per node, the latest, per DyGLib's
``MessageAggregator``); only afterwards are the memories of B's nodes overwritten and B's own
events turned into new pending messages. So for an event e in batch B, ``p_link``, ``z_src`` and
``z_dst`` depend on
  (1) memory that has absorbed every event of the batches before B and none of B;
  (2) static node features;
  (3) graph attention over temporal neighbours with ``ts < t_e`` strictly
      (``NeighborSampler.find_neighbors_before`` cuts with ``searchsorted(side="left")``), which may
      include earlier-day events of the same batch, but never e itself or anything at ``t_e`` or later.
Granularity is the batch: events of the same batch never see each other through memory. With
``align_to_timestamps=True`` (default) batch boundaries fall only between distinct days, so (1)
holds only events with ``ts < t_e``; same-day siblings (e.g. the other references of the same
citing paper) are not in memory when e is scored. A day with more than ``max_batch_size`` events is
split; events in its later chunks then see same-day siblings from earlier chunks in memory, which
the contract allows (``ts <= t``, own row excluded); the meta file counts such days.

Leak argument (NFR3). Points (1)-(3) above bound what the graph state contributes. The weights are
the other channel: they were fitted on DyGLib's training part, so scores of events inside that
part are in-sample for the weights. Choose ``val_ratio``/``test_ratio`` at training time with
``citecurv.tgn.train.dyglib_split_ratios`` so the training part ends at the project's training
cutoff; validation and test events are then out-of-sample. ``leak_check=True`` runs the memory
leak hook from ``citecurv.tgn.train`` over the whole replay. The memory is always reset first, so
the post-validation memory stored in the checkpoint is never used; memory parameters are not
loaded at all, which also lets the checkpoint score a dataset with a different node set (a model
trained on a pre-cutoff slice can score the full stream).

Determinism (NFR4): eval mode (no dropout), DyGLib's evaluation neighbour-sampler seed (1), a
seeded negative draw made once for the whole stream (independent of batching), CPU or
deterministic GPU kernels; same inputs and seed give the same outputs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from citecurv.tgn.convert import load_idmap

P_CLIP = 1e-12  # probabilities are clipped to [P_CLIP, 1 - P_CLIP] so log-scores stay finite


def batch_boundaries(ts, batch_size: int, align_to_timestamps: bool = True, max_batch_size: int | None = None):
    """Split a time-sorted stream into consecutive ``(start, end)`` slices.

    Without alignment: fixed ``batch_size`` slices (DyGLib's own batching). With alignment: whole
    timestamp groups are added to the open batch while it stays within ``batch_size``; a group that
    would overflow a non-empty batch starts the next one; a single group larger than
    ``max_batch_size`` (default ``10 * batch_size``) is cut into ``max_batch_size`` chunks.
    """
    ts = np.asarray(ts)
    n = len(ts)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not align_to_timestamps:
        return [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
    cap = int(max_batch_size or 10 * batch_size)
    if cap < batch_size:
        raise ValueError("max_batch_size must be >= batch_size")
    if np.any(np.diff(ts) < 0):
        raise ValueError("ts must be sorted")
    change = np.flatnonzero(np.diff(ts)) + 1
    starts = np.concatenate([[0], change]).astype(int)
    ends = np.concatenate([change, [n]]).astype(int)
    out, cur = [], 0
    for gs, ge in zip(starts, ends):
        if ge - cur > batch_size and gs > cur:
            out.append((cur, int(gs)))
            cur = int(gs)
        if ge - gs > cap:
            out.extend((s, min(s + cap, int(ge))) for s in range(int(gs), int(ge), cap))
            cur = int(ge)
    if cur < n:
        out.append((cur, n))
    return out


def paired_share(p_link, neg_p_link) -> dict:
    """How often each observed citation scores above its own negative, with ties reported apart.

    Exact ties are common: every cited work with no text vector and no earlier event gets the same
    embedding, so an observed and a negative destination that are both such works score the same
    (found on the pilot, 2026-11-12). ``strict`` counts a tie as a loss, ``ties_as_half`` counts it
    as half a win (the paired form of AUC), ``without_ties`` leaves ties out (None if all are ties).
    """
    p = np.asarray(p_link, dtype=np.float64)
    q = np.asarray(neg_p_link, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1 or len(p) == 0:
        raise ValueError("p_link and neg_p_link must be non-empty 1-d arrays of the same length")
    win, tie = p > q, p == q
    n_free = int((~tie).sum())
    return {
        "n": int(len(p)),
        "strict": float(win.mean()),
        "ties": float(tie.mean()),
        "ties_as_half": float((win + 0.5 * tie).mean()),
        "without_ties": float(win[~tie].sum() / n_free) if n_free else None,
    }


def _find_checkpoint(run_dir: Path) -> Path:
    metrics = run_dir / "metrics.json"
    if metrics.is_file():
        p = run_dir / json.loads(metrics.read_text())["checkpoint"]["model"]
        if p.is_file():
            return p
    cands = [p for p in sorted((run_dir / "checkpoint").glob("*.pkl")) if "nonparametric" not in p.name]
    if len(cands) != 1:
        raise FileNotFoundError(f"expected one model checkpoint in {run_dir / 'checkpoint'}, found {cands}")
    return cands[0]


def load_event_scores(out_dir) -> pd.DataFrame:
    """Read ``tgn_event_scores`` written by ``infer_tgn`` (parquet or csv)."""
    out = Path(out_dir)
    if (out / "tgn_event_scores.parquet").is_file():
        return pd.read_parquet(out / "tgn_event_scores.parquet")
    return pd.read_csv(out / "tgn_event_scores.csv")


def infer_tgn(
    run_dir,
    data_dir,
    name: str,
    out_dir=None,
    *,
    batch_size: int = 200,
    align_to_timestamps: bool = True,
    max_batch_size: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    dyglib_root=None,
    leak_check: bool = False,
    num_threads: int | None = None,
    save_embeddings: bool = True,
    embeddings_dtype: str = "float32",
):
    """Score every event of ``data_dir/ml_{name}.*`` with the TGN in ``run_dir``.

    Returns ``(scores, embeddings_path)``: the ``tgn_event_scores`` frame in canonical event order
    and the path of ``tgn_event_embeddings.npz`` (None when ``save_embeddings=False``). Files go to
    ``out_dir`` (default ``run_dir/infer``) together with ``tgn_infer_meta.json``.

    Size note: the npz holds 2 x M x d values; at M = 3e6 events and d = 768 that is about 18 GB in
    float32 (9 GB in float16, ``embeddings_dtype="float16"``), held in RAM before writing.
    """
    import torch
    import torch.nn as nn

    from citecurv.tgn.train import import_dyglib, install_memory_leak_check

    t_start = time.time()
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / "infer"
    out.mkdir(parents=True, exist_ok=True)
    if num_threads:
        torch.set_num_threads(int(num_threads))

    args = json.loads((run_dir / "resolved_args.json").read_text())["dyglib"]
    if args.get("model_name", "TGN") != "TGN":
        raise ValueError(f"infer_tgn supports TGN only, run has {args.get('model_name')}")
    ckpt = _find_checkpoint(run_dir)
    dg = import_dyglib(dyglib_root)

    df = pd.read_csv(data_dir / f"ml_{name}.csv")
    edge_feat = np.load(data_dir / f"ml_{name}.npy")
    node_feat = np.load(data_dir / f"ml_{name}_node.npy")
    # copies: pandas >= 3 may hand out read-only views, which torch.from_numpy warns about
    src = df.u.to_numpy(np.longlong, copy=True)
    dst = df.i.to_numpy(np.longlong, copy=True)
    ts = df.ts.to_numpy(np.float64, copy=True)
    eids = df.idx.to_numpy(np.longlong, copy=True)
    m = len(df)
    if not np.array_equal(eids, np.arange(1, m + 1)) or np.any(np.diff(ts) < 0):
        raise ValueError("ml_{name}.csv must be in canonical order with idx = 1..M (write it with to_jodie)")
    _, idmap = load_idmap(data_dir, name)
    event_id = idmap.event_id.to_numpy(np.int64)
    if len(event_id) != m:
        raise ValueError("id map and event file disagree on the number of events")

    data = dg.Data(src_node_ids=src, dst_node_ids=dst, node_interact_times=ts, edge_ids=eids, labels=np.zeros(m))
    sampler = dg.get_neighbor_sampler(data, sample_neighbor_strategy=args.get("sample_neighbor_strategy", "recent"),
                                      time_scaling_factor=args.get("time_scaling_factor", 1e-6), seed=1)
    edge_dst = np.zeros(edge_feat.shape[0], dtype=np.longlong)
    edge_dst[eids] = dst
    direction_feat_index = int(args.get("direction_feat_index", -1))

    torch.manual_seed(seed)
    backbone = dg.MemoryModel(node_raw_features=node_feat, edge_raw_features=edge_feat, neighbor_sampler=sampler,
                              time_feat_dim=int(args["time_feat_dim"]), model_name="TGN",
                              num_layers=int(args["num_layers"]), num_heads=int(args["num_heads"]),
                              dropout=float(args["dropout"]), device=device,
                              direction_feat_index=direction_feat_index, edge_dst_node_ids=edge_dst)
    d = node_feat.shape[1]
    model = nn.Sequential(backbone, dg.MergeLayer(input_dim1=d, input_dim2=d, hidden_dim=d, output_dim=1))
    state = torch.load(ckpt, map_location=device)
    state = {k: v for k, v in state.items() if "memory_bank." not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any("memory_bank." not in k for k in missing):
        raise RuntimeError(f"checkpoint does not match the model: missing={missing}, unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    backbone.memory_bank.__init_memory_bank__()
    backbone.set_neighbor_sampler(sampler)
    num_neighbors = int(args.get("num_neighbors", 20))

    rng = np.random.default_rng(seed)
    unique_dst = np.unique(dst)
    neg_dst = unique_dst[rng.integers(0, len(unique_dst), size=m)]

    bounds = batch_boundaries(ts, batch_size, align_to_timestamps, max_batch_size)
    cap = int(max_batch_size or 10 * batch_size)
    group_sizes = np.unique(ts, return_counts=True)[1]
    p_pos = np.empty(m, dtype=np.float64)
    p_neg = np.empty(m, dtype=np.float64)
    z_src = np.empty((m, d), dtype=embeddings_dtype) if save_embeddings else None
    z_dst = np.empty((m, d), dtype=embeddings_dtype) if save_embeddings else None

    hook = install_memory_leak_check(dg.MemoryModel) if leak_check else None
    try:
        with torch.no_grad():
            for s, e in bounds:
                b_src, b_dst, b_t, b_eid, b_neg = src[s:e], dst[s:e], ts[s:e], eids[s:e], neg_dst[s:e]
                neg_src_emb, neg_dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=b_src, dst_node_ids=b_neg, node_interact_times=b_t, edge_ids=None,
                    edges_are_positive=False, num_neighbors=num_neighbors)
                src_emb, dst_emb = backbone.compute_src_dst_node_temporal_embeddings(
                    src_node_ids=b_src, dst_node_ids=b_dst, node_interact_times=b_t, edge_ids=b_eid,
                    edges_are_positive=True, num_neighbors=num_neighbors)
                logit_pos = model[1](input_1=src_emb, input_2=dst_emb).squeeze(dim=-1)
                logit_neg = model[1](input_1=neg_src_emb, input_2=neg_dst_emb).squeeze(dim=-1)
                p_pos[s:e] = torch.sigmoid(logit_pos.double()).cpu().numpy()
                p_neg[s:e] = torch.sigmoid(logit_neg.double()).cpu().numpy()
                if save_embeddings:
                    z_src[s:e] = src_emb.cpu().numpy()
                    z_dst[s:e] = dst_emb.cpu().numpy()
    finally:
        if hook is not None:
            hook.remove()

    scores = pd.DataFrame({
        "event_id": event_id,
        "p_link": np.clip(p_pos, P_CLIP, 1 - P_CLIP),
        "neg_p_link": np.clip(p_neg, P_CLIP, 1 - P_CLIP),
    })
    try:
        scores.to_parquet(out / "tgn_event_scores.parquet", index=False)
        scores_file = "tgn_event_scores.parquet"
    except ImportError:
        scores.to_csv(out / "tgn_event_scores.csv", index=False)
        scores_file = "tgn_event_scores.csv"
    emb_path = None
    if save_embeddings:
        emb_path = out / "tgn_event_embeddings.npz"
        np.savez(emb_path, event_id=event_id, z_src=z_src, z_dst=z_dst)

    meta = {
        "run_dir": str(run_dir), "checkpoint": str(ckpt), "data_dir": str(data_dir), "name": name,
        "n_events": m, "n_batches": len(bounds), "batch_size": batch_size,
        "align_to_timestamps": align_to_timestamps, "max_batch_size": cap,
        "n_split_timestamp_groups": int((group_sizes > cap).sum()) if align_to_timestamps else None,
        "seed": seed, "device": device, "direction_feat_index": direction_feat_index,
        "num_neighbors": num_neighbors, "p_clip": P_CLIP, "scores_file": scores_file,
        "embeddings_file": emb_path.name if emb_path else None, "embeddings_dtype": embeddings_dtype,
        "leak_check": hook.as_dict() if hook is not None else {"enabled": False},
        "wall_clock_s": round(time.time() - t_start, 2),
    }
    (out / "tgn_infer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return scores, emb_path

"""Citation events to DyGLib's JODIE-format files (task S1.7).

``to_jodie(edges_df, node_emb, out_dir, name)`` writes, under ``out_dir``:

- ``ml_{name}.csv``: columns ``u, i, ts, label, idx`` (plus pandas' unnamed index column, as
  upstream ``preprocess_data.py`` writes it). One row per citation event, ordered by
  ``(t, event_id)``; ``u`` cites ``i`` at day ``ts``. Node and edge indices start at 1 because
  DyGLib reserves index 0 for padding (neighbour slots that are empty point at node 0 / edge 0).
  ``label`` is always 0: DyGLib's link-prediction task does not read it, and keeping weak labels
  out of the training files removes one way for them to leak into the encoder.
- ``ml_{name}.npy``: edge features, shape ``(M + 1, 1)`` float32. Row 0 is zero padding; column
  ``DIRECTION_FEAT_INDEX`` holds the citation-direction indicator, +1 on every event ("u cites i").
- ``ml_{name}_node.npy``: node features, shape ``(N + 1, d)`` float32. Row 0 is zero padding and
  row k holds the embedding of the node with index k (SPECTER2 on real data). A node without an
  embedding gets a zero row (``on_missing="zero"``, counted in the meta file) or raises
  (``on_missing="raise"``).
- ``ml_{name}_idmap.npz``: ``work_id[k - 1]`` is the work behind node index k and
  ``event_id[idx - 1]`` the event behind edge index idx (``load_idmap`` returns both as frames),
  so TGN outputs join back to ``event_id``.
- ``ml_{name}_meta.json``: counts, dimensions, the direction column, file names.

Why a constant direction column is not enough on stock DyGLib. DyGLib's TGN
(``models/MemoryModel.py``) turns each event (u, i, t, x_e) into two raw messages, one per
endpoint, both laid out from the receiver's side: ``m_u = [s_u, s_i, phi(t - t_u), x_e]`` and
``m_i = [s_i, s_u, phi(t - t_i), x_e]``. Both carry the same edge-feature row ``x_e``. The
graph-attention embedding reads neighbours from an undirected adjacency list
(``utils/utils.py:get_neighbor_sampler`` files every event under both endpoints with the same edge
id), so attention keys also see the same ``x_e`` from either side. Upstream zero-padding of edge
features to 172 columns changes nothing here. A column that is +1 on every row is therefore
identical in the citing and in the cited endpoint's message: it acts as a constant bias, and the
memory updater cannot tell whether the receiving paper cited or was cited. Bidirectional
messaging erases direction unless the indicator is read relative to the receiver.

Vendored patch P5 (``third_party/DyGLib/PATCHES.md``) does that when DyGLib runs with
``--direction_feat_index 0`` (``citecurv.tgn.train`` passes it from the meta file): the cited
endpoint's message, and every attention key in which the centre node is the cited endpoint of the
neighbouring event, see the column negated. The citing paper sees +1 and the cited paper -1. With
the flag off (upstream default) the column is inert. The decoder (``MergeLayer`` on
``[z_src, z_dst]``) is order-sensitive on its own and needs no patch.

Leak argument (NFR3). Conversion re-indexes; it computes no statistic across events, so no row
carries information from another row. Node features are per-paper text embeddings, fixed at
publication; DyGLib reads a node's features only when that node takes part in an event or appears
as a temporal neighbour with ts < t (``NeighborSampler.find_neighbors_before`` is strict), both of
which happen at or after the paper exists. The label column is constant, so no weak label reaches
training. Index assignment (first appearance in time order) depends only on the order of events,
not on their outcome.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

DIRECTION_FEAT_INDEX = 0
EDGE_FEATURE_NAMES = ("direction",)
REQUIRED_COLUMNS = ("event_id", "src_id", "dst_id", "t")


def _split_embeddings(node_emb):
    """Return (ids, vectors) from a tuple/list or an npz-style mapping with 'ids' and 'vectors'."""
    if isinstance(node_emb, Mapping) or hasattr(node_emb, "files"):
        ids, vectors = node_emb["ids"], node_emb["vectors"]
    else:
        ids, vectors = node_emb
    ids = np.asarray(ids).astype(str)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(ids) != len(vectors):
        raise ValueError(f"embeddings must be ids (N,) and vectors (N, d); got {ids.shape} and {vectors.shape}")
    return ids, vectors


def to_jodie(
    edges_df: pd.DataFrame,
    node_emb,
    out_dir,
    name: str,
    *,
    node_feat_dim: int | None = None,
    on_missing: str = "zero",
) -> dict:
    """Write the DyGLib dataset ``name`` for the citation events in ``edges_df``.

    Args:
        edges_df: contract ``events`` frame (``event_id``, ``src_id``, ``dst_id``, ``t``); any row
            order. ``event_id`` must be unique.
        node_emb: ``(ids, vectors)``, a mapping/npz with ``ids`` and ``vectors``, or ``None`` for
            all-zero node features (upstream DyGLib's behaviour; needs ``node_feat_dim``).
        out_dir: directory to write into (created if missing).
        name: dataset name; DyGLib's ``--dataset_name``.
        node_feat_dim: required with ``node_emb=None``; otherwise, if given, must equal the
            embedding width (a guard against feeding the wrong embedding file).
        on_missing: ``"zero"`` (zero row, counted in the meta) or ``"raise"`` (KeyError) for a
            work that appears in an event but has no embedding.

    Returns:
        The meta dict, also written to ``ml_{name}_meta.json``.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in edges_df.columns]
    if missing_cols:
        raise ValueError(f"edges_df lacks columns {missing_cols}")
    if edges_df["event_id"].duplicated().any():
        raise ValueError("edges_df has duplicate event_id values")
    if on_missing not in ("zero", "raise"):
        raise ValueError("on_missing must be 'zero' or 'raise'")

    ev = edges_df.loc[:, list(REQUIRED_COLUMNS)].copy()
    ev["event_id"] = ev["event_id"].astype(np.int64)
    ev["t"] = ev["t"].astype(np.int64)
    ev["src_id"] = ev["src_id"].astype(str)
    ev["dst_id"] = ev["dst_id"].astype(str)
    ev = ev.sort_values(["t", "event_id"], kind="mergesort").reset_index(drop=True)

    # node index = first appearance in canonical order, src before dst within a row
    interleaved = np.column_stack([ev["src_id"].to_numpy(), ev["dst_id"].to_numpy()]).ravel()
    work_ids = pd.unique(interleaved)
    node_index = {w: k + 1 for k, w in enumerate(work_ids)}

    n_events, n_nodes = len(ev), len(work_ids)
    jodie = pd.DataFrame(
        {
            "u": ev["src_id"].map(node_index).to_numpy(np.int64),
            "i": ev["dst_id"].map(node_index).to_numpy(np.int64),
            "ts": ev["t"].to_numpy(np.int64),
            "label": np.zeros(n_events, dtype=np.int64),
            "idx": np.arange(1, n_events + 1, dtype=np.int64),
        }
    )

    edge_feat = np.zeros((n_events + 1, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    edge_feat[1:, DIRECTION_FEAT_INDEX] = 1.0

    n_missing = 0
    if node_emb is None:
        if node_feat_dim is None:
            raise ValueError("node_feat_dim is required when node_emb is None")
        node_feat = np.zeros((n_nodes + 1, int(node_feat_dim)), dtype=np.float32)
    else:
        ids, vectors = _split_embeddings(node_emb)
        if node_feat_dim is not None and int(node_feat_dim) != vectors.shape[1]:
            raise ValueError(f"node_feat_dim={node_feat_dim} but embeddings have width {vectors.shape[1]}")
        row_of = pd.Series(np.arange(len(ids)), index=pd.Index(ids))
        if row_of.index.has_duplicates:
            raise ValueError("embedding ids are not unique")
        rows = row_of.reindex(work_ids)
        absent = rows.isna().to_numpy()
        if absent.any() and on_missing == "raise":
            raise KeyError(f"no embedding for {int(absent.sum())} works, e.g. {list(work_ids[absent][:5])}")
        n_missing = int(absent.sum())
        node_feat = np.zeros((n_nodes + 1, vectors.shape[1]), dtype=np.float32)
        present = np.flatnonzero(~absent)
        node_feat[present + 1] = vectors[rows.to_numpy()[present].astype(np.int64)]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "csv": f"ml_{name}.csv",
        "edge_feat": f"ml_{name}.npy",
        "node_feat": f"ml_{name}_node.npy",
        "idmap": f"ml_{name}_idmap.npz",
    }
    jodie.to_csv(out / files["csv"])
    np.save(out / files["edge_feat"], edge_feat)
    np.save(out / files["node_feat"], node_feat)
    np.savez(
        out / files["idmap"],
        work_id=np.asarray(work_ids, dtype=str),
        event_id=ev["event_id"].to_numpy(np.int64),
    )

    meta = {
        "name": name,
        "n_events": n_events,
        "n_nodes": n_nodes,
        "node_feat_dim": int(node_feat.shape[1]),
        "edge_feat_dim": int(edge_feat.shape[1]),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "direction_feat_index": DIRECTION_FEAT_INDEX,
        "n_missing_embeddings": n_missing,
        "t_min": int(ev["t"].min()) if n_events else None,
        "t_max": int(ev["t"].max()) if n_events else None,
        "files": files,
    }
    (out / f"ml_{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_idmap(out_dir, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(nodes, events)``: ``nodes`` has ``node_idx, work_id``; ``events`` has ``idx, event_id``."""
    with np.load(Path(out_dir) / f"ml_{name}_idmap.npz", allow_pickle=False) as z:
        work_id, event_id = z["work_id"], z["event_id"]
    nodes = pd.DataFrame({"node_idx": np.arange(1, len(work_id) + 1, dtype=np.int64), "work_id": work_id.astype(str)})
    events = pd.DataFrame({"idx": np.arange(1, len(event_id) + 1, dtype=np.int64), "event_id": event_id})
    return nodes, events

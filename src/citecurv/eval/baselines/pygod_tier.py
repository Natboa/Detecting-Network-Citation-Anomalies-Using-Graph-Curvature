"""PyGOD GNN baseline tier: DOMINANT, CoLA, GADNR on the cutoff snapshot (task S4.4, decision L17).

Fair-comparison protocol (book 3.3, plan S4.4): a static detector sees one graph, the snapshot of all events with
t <= cutoff, and scores the same test events as every other method. PyGOD detectors score nodes, so an event
gets ``edge_score = max(score(src), score(dst))``: an edge is as suspicious as its more suspicious endpoint.

Snapshot construction (``snapshot_graph``):
- nodes: exactly ``node_ids`` in the given order (pass every paper that appears in the snapshot or as an endpoint
  of an event to score; test-period papers then enter as isolated nodes carrying only their features);
- edges: the distinct (src, dst) pairs of events with t <= cutoff whose endpoints are both in ``node_ids``
  (events with an unknown endpoint are counted in ``n_events_dropped``); self-loops are removed;
- ``undirected=True`` (default) adds the reverse of every pair. PyGOD's GCN backbones pass messages from
  edge_index[0] to edge_index[1], so a directed citation graph would leave every citing paper without messages
  from its references. Pass False to keep citation direction. PROVISIONAL default, open for review.

Leak argument: the structure contains only events with t <= cutoff, which precedes every test event; labels are
never read (the detectors are unsupervised). Node features are text embeddings, fixed at each paper's
publication. The detectors are fitted transductively on all snapshot nodes, so a test paper's feature vector can
influence the fitted weights and, through them, other nodes' scores; this is the standard static setting the
protocol prescribes, and it uses no structure or label from after the cutoff.

Environment (verified 2026-10-31 on CPU, results/smoke_pygod.json): Python 3.12, torch 2.15.0 (PyPI),
torch_geometric 2.6.1, pygod 1.1.0, plus torch-scatter 2.1.2 and torch-sparse 0.6.18 compiled from their PyPI
source distributions (PyPI has no wheels for them). ``pip install pygod==1.1.0 torch_geometric==2.6.*`` alone is
not enough: DOMINANT and CoLA then fail with "'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'". On
Colab, install the prebuilt pyg-lib or torch-sparse wheel from data.pyg.org that matches the runtime's torch and
CUDA (that index was not reachable where this was verified). torch, torch_geometric and pygod are imported
lazily, so this module imports in the light environments; only ``snapshot_graph`` and ``score_pygod`` need them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DETECTORS = {"dominant": "DOMINANT", "cola": "CoLA", "gadnr": "GADNR"}

# What score_pygod changes relative to calling pygod 1.1.0 directly (each one documented at its function).
PYGOD_ADJUSTMENTS = (
    "DOMINANT: dense adjacency sized to all N nodes (upstream to_dense_adj without max_num_nodes raises "
    "IndexError when the highest-indexed nodes are isolated)",
    "GADNR: backbone wrapper drops the stray tot_nodes keyword rejected by PyG >= 2.5 (removed upstream in pygod "
    "commit 9d97307, 2024-02-21, not released on PyPI)",
    "GADNR: isolated nodes get a self-loop (full-batch preprocessing keeps only nodes present in edge_index)",
    "GADNR: O(E) re-implementation of GADNRBase.process_graph with identical outputs (upstream is O(E*N))",
    "GADNR: the multiprocessing.Pool(4) opened by GADNRBase.__init__ is terminated after scoring",
    "all: detectors run on a copy of the snapshot (GADNR rewrites x and edge_index of its input in place)",
)


@dataclass
class Snapshot:
    data: Any  # torch_geometric.data.Data with x (N, d) float32 and edge_index (2, M) int64
    node_ids: np.ndarray
    node_index: dict
    cutoff: int
    stats: dict = field(default_factory=dict)


def snapshot_edges(events: pd.DataFrame, cutoff: int, node_ids, *, undirected: bool = True):
    """Edge index of the static snapshot: returns (edge_index (2, M) int64, node_index dict, stats dict).

    Pure numpy/pandas, so it runs without torch. See the module docstring for the rules.
    """
    node_ids = np.asarray(list(node_ids), dtype=object)
    node_index = {k: i for i, k in enumerate(node_ids.tolist())}
    if len(node_index) != len(node_ids):
        raise ValueError("node_ids must be unique")
    past = events[events["t"].to_numpy() <= cutoff]
    s = past["src_id"].map(node_index)
    d = past["dst_id"].map(node_index)
    known = s.notna().to_numpy() & d.notna().to_numpy()
    s = s.to_numpy()[known].astype(np.int64)
    d = d.to_numpy()[known].astype(np.int64)
    keep = s != d
    pairs = np.stack([s[keep], d[keep]])
    if undirected:
        pairs = np.concatenate([pairs, pairs[::-1]], axis=1)
    if pairs.shape[1]:
        pairs = np.unique(pairs, axis=1)
    stats = {
        "n_events_used": int(known.sum()),
        "n_events_dropped": int((~known).sum()),
        "n_self_loops_removed": int((~keep).sum()),
        "n_edges": int(pairs.shape[1]),
        "undirected": bool(undirected),
    }
    return pairs.astype(np.int64), node_index, stats


def snapshot_graph(events: pd.DataFrame, cutoff: int, node_ids, node_features, *, undirected: bool = True) -> Snapshot:
    """Static snapshot at ``cutoff`` as a PyG ``Data`` (x = node_features rows aligned with node_ids)."""
    import torch
    from torch_geometric.data import Data

    x = np.asarray(node_features, dtype=np.float32)
    ids = np.asarray(list(node_ids), dtype=object)
    if x.ndim != 2 or x.shape[0] != len(ids):
        raise ValueError(f"node_features must be (len(node_ids), d); got {x.shape} for {len(ids)} ids")
    edges, node_index, stats = snapshot_edges(events, cutoff, ids, undirected=undirected)
    data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edges))
    data.num_nodes = len(ids)
    return Snapshot(data=data, node_ids=ids, node_index=node_index, cutoff=int(cutoff), stats=stats)


def edge_scores_from_nodes(node_scores, node_index: dict, events: pd.DataFrame) -> pd.DataFrame:
    """Map node scores to events by max(score(src), score(dst)); DataFrame(event_id, score) in events order."""
    node_scores = np.asarray(node_scores, dtype=np.float64)
    s = events["src_id"].map(node_index)
    d = events["dst_id"].map(node_index)
    unknown = s.isna() | d.isna()
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} events have an endpoint that is not in the snapshot node set; include every "
            "endpoint of the events to score in node_ids (as an isolated node if it has no edge by the cutoff)"
        )
    score = np.maximum(node_scores[s.to_numpy(np.int64)], node_scores[d.to_numpy(np.int64)])
    return pd.DataFrame({"event_id": events["event_id"].to_numpy(), "score": score})


def gadnr_compatible_backbone(backbone=None):
    """Wrap a PyG backbone so it ignores the stray ``tot_nodes`` keyword that pygod 1.1.0's GADNR passes on.

    pygod 1.1.0 (PyPI, 2024-02-04) calls ``GADNRBase(..., tot_nodes=...)``; GADNRBase forwards unknown keywords
    to the backbone constructor, and PyG >= 2.5 ``MessagePassing.__init__`` no longer accepts ``**kwargs``, so
    GADNR fails with "unexpected keyword argument 'tot_nodes'" on the planned PyG 2.5-2.6 pin (PyG 2.4.0 still
    swallowed it). Upstream removed the keyword in pygod commit 9d97307 (2024-02-21), which is not on PyPI.
    The wrapper drops only that keyword; the model is otherwise PyGOD's own.
    """
    if backbone is None:
        from torch_geometric.nn import GCN as backbone

    def build(*args, tot_nodes=None, **kwargs):
        return backbone(*args, **kwargs)

    build.__name__ = getattr(backbone, "__name__", "backbone")
    return build


def gadnr_process_graph(data, input_id=None):
    """Same outputs as pygod 1.1.0 ``GADNRBase.process_graph``, with set membership instead of list membership.

    Upstream (pygod/nn/gadnr.py, BSD-2-Clause) tests ``in_node.item() in input_id`` against a Python list inside
    the loop over all edges, which costs O(E * N) and does not finish at the slice scale (~3 * 10^5 nodes).
    This re-implementation follows the same steps (row-normalise x, make undirected, add self-loops, collect
    each node's neighbour list in edge order, degree list in input_id order) in O(E).
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.utils import add_self_loops, to_undirected

    data.x = F.normalize(data.x, p=1, dim=1)
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index)
    out_nodes = data.edge_index[0, :].tolist()
    in_nodes = data.edge_index[1, :].tolist()
    id_mapping = {}
    if input_id is None:
        input_id = torch.unique(data.edge_index).tolist()
    else:
        n_id = data.n_id.tolist()
        for edge_id, node_id in enumerate(n_id):
            id_mapping[node_id] = edge_id
        in_nodes = [n_id[i] for i in in_nodes]
        out_nodes = [n_id[i] for i in out_nodes]
    wanted = set(input_id)
    neighbor_dict: dict[int, list[int]] = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node in wanted:
            neighbor_dict.setdefault(in_node, []).append(out_node)
    neighbor_num_list = torch.tensor([len(neighbor_dict[i]) for i in input_id if i in neighbor_dict])
    return data, neighbor_dict, neighbor_num_list, id_mapping


def dominant_process_graph(data):
    """pygod 1.1.0 ``DOMINANTBase.process_graph`` with the adjacency sized to all N nodes.

    Upstream calls ``to_dense_adj(edge_index)`` without ``max_num_nodes``, so the matrix has max-index + 1 rows;
    when the highest-indexed nodes are isolated (test-period papers in a cutoff snapshot) the fit fails with an
    IndexError. Same dense target otherwise.
    """
    from torch_geometric.utils import to_dense_adj

    data.s = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]


def add_isolated_self_loops(data):
    """Give every node without an edge a self-loop (in place), so GADNR's full-batch mode keeps it.

    GADNR's preprocessing takes the node set from ``edge_index`` and its full-batch decoder assumes that set is
    all N nodes; an isolated node (every test-period paper is isolated in the cutoff snapshot) breaks the shapes.
    With the loop, an isolated node's neighbourhood is the node itself.
    """
    import torch

    present = torch.zeros(data.num_nodes, dtype=torch.bool)
    present[data.edge_index.flatten()] = True
    iso = torch.nonzero(~present).flatten()
    if len(iso):
        data.edge_index = torch.cat([data.edge_index, torch.stack([iso, iso])], dim=1)
    return data


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def score_pygod(detector_name: str, snapshot: Snapshot, events_to_score: pd.DataFrame, seed: int, **kw) -> pd.DataFrame:
    """Fit one PyGOD detector on the snapshot and score events by the max rule; DataFrame(event_id, score).

    detector_name: 'DOMINANT', 'CoLA' or 'GADNR' (case-insensitive). ``kw`` goes to the detector constructor
    (epoch, hid_dim, num_layers, lr, batch_size, num_neigh, gpu, ...); defaults are PyGOD's own except
    verbose=0. Node scores come from ``decision_function`` on the snapshot after fitting (eval mode). Python,
    NumPy and torch RNGs are reseeded before the fit and again before scoring, so the same snapshot and seed give
    the same scores on CPU. The detector works on a copy of ``snapshot.data`` (GADNR modifies its input).

    Environment (verified in results/smoke_pygod.json): DOMINANT and CoLA always sample through PyG's
    NeighborLoader, which needs ``pyg-lib`` or ``torch-sparse`` installed next to torch_geometric; GADNR in its
    default full-batch mode does not. Documented adjustments for pygod 1.1.0, each applied only for the call and
    restored afterwards: DOMINANT's dense adjacency is built with ``max_num_nodes = N`` (``dominant_process_graph``);
    GADNR gets ``gadnr_compatible_backbone`` (PyG >= 2.5 keyword clash), ``add_isolated_self_loops`` (isolated
    nodes) and ``gadnr_process_graph`` (O(E) preprocessing). CoLA runs unmodified.

    Scale (read from the pygod 1.1.0 source; CurvGAD's own limit is in curvgad_notes.md): DOMINANT
    materialises a dense N x N adjacency of the whole snapshot once per fit, whatever ``batch_size`` is, so N is
    bounded by memory (4 N^2 bytes; 3 * 10^5 nodes would need ~360 GB). GADNR full-batch keeps an (N, hid, hid)
    covariance tensor (hid = 64: 16 KB per node). CoLA and DOMINANT train in mini-batches with ``batch_size`` and
    ``num_neigh``. Running them on the full slice therefore needs a node-capped snapshot (e.g. the k-hop
    neighbourhood of the test-event endpoints); the cap belongs in configs (S4.4).
    """
    key = str(detector_name).lower()
    if key not in DETECTORS:
        raise ValueError(f"unknown detector {detector_name!r}; expected one of {sorted(DETECTORS.values())}")
    import pygod.detector as detectors

    cls = getattr(detectors, DETECTORS[key])
    params = {"verbose": 0, **kw}
    data = snapshot.data.clone()
    patches = []  # (class, attribute name, original attribute)
    if key == "dominant":
        from pygod.nn.dominant import DOMINANTBase

        patches.append((DOMINANTBase, "process_graph", dominant_process_graph))
    if key == "gadnr":
        from pygod.nn.gadnr import GADNRBase

        params["backbone"] = gadnr_compatible_backbone(params.get("backbone"))
        add_isolated_self_loops(data)
        patches.append((GADNRBase, "process_graph", gadnr_process_graph))
    originals = [(cls_, name, cls_.__dict__[name]) for cls_, name, _ in patches]
    det = None
    try:
        for cls_, name, fn in patches:
            setattr(cls_, name, staticmethod(fn))
        _seed_everything(seed)
        det = cls(**params)
        det.fit(data)
        _seed_everything(seed)
        node_scores = det.decision_function(data).detach().cpu().numpy()
    finally:
        for cls_, name, original in originals:
            setattr(cls_, name, original)
        pool = getattr(getattr(det, "model", None), "pool", None)
        if pool is not None:  # GADNRBase opens multiprocessing.Pool(4) in __init__ and never closes it
            pool.terminate()
    return edge_scores_from_nodes(node_scores, snapshot.node_index, events_to_score)

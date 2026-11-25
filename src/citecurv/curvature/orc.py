"""Directed Ollivier-Ricci curvature on time-t subgraphs (tasks S1.5, S2.3).

- orc_directed(G, edges, ...): thin wrapper around GraphRicciCurvature's
  OllivierRicci (Ni et al., 2015; Ollivier, 2009) for a directed nx.DiGraph,
  computing only the requested edges. GraphRicciCurvature is imported lazily,
  so this module imports in every stage env; the library itself runs on
  Linux/Colab (fork multiprocessing).
- escalate(...): fills the `orc` column of the curvature cache for rows the
  Forman screen escalated, on EventStore.subgraph_at(u, v, t, radius) plus the
  scored edge.
- escalation_budget(...): the S2.3 table (escalation rate, subgraph sizes, cap
  hits, measured seconds per flagged event) per (tail percentile, radius) cell.

Directed ORC in GraphRicciCurvature 0.5.3.2 / 0.6.1 (read from the source):
for an edge u -> v the source measure puts mass alpha on u and spreads 1 - alpha
over the predecessors (in-neighbours) of u, the target measure puts alpha on v
and spreads 1 - alpha over the successors (out-neighbours) of v; a node without
such neighbours keeps all its mass. kappa(u, v) = 1 - W1(m_u, m_v) / d(u, v)
with directed shortest-path costs. In 0.5.3.2 the denominator is the edge weight,
in 0.6.x the shortest-path distance; both are 1 on our unweighted graphs.

Which edges matter (unweighted case). Let S = {u} + pred(u), T = {v} + succ(v).
Only distances d(p, w) for p in S, w in T enter W1, and the path p -> u -> v -> w
bounds each of them by 3. A shorter path has length <= 2, so it uses only edges
leaving S or edges x -> w with x in succ(S) and w in T. All other edges cannot
change kappa(u, v). edge_tiers() ranks edges by this rule; it is also why a
radius-2 subgraph reproduces ORC on the whole time-t graph exactly while radius
1 can miss length-2 paths through second-hop nodes (checked against the real
library in tests/test_orc.py and scripts/run_orc_validation.py).

Consequence for citation events: a citing paper u is almost never cited by
time t = its own publication date, so m_u is a Dirac at u, d(u, w) = 1 when u
also cites w and 2 otherwise (via v), and
    kappa(u, v) = -(1 - alpha) * (1 - |succ(u) & succ(v)| / |succ(v)|)
(0 when v cites nothing): minus (1 - alpha) times the share of v's references
that u does not also cite. Radius 1 is then already exact.
"""

import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from citecurv.curvature.screen import apply_screen, screen_thresholds
from citecurv.curvature.timeslice import EventStore

INSTALL_HINT = (
    "orc_directed needs GraphRicciCurvature (curv stage env, see configs/pins.md), e.g. "
    "pip install GraphRicciCurvature==0.5.3.2 'scipy<=1.13.1' POT==0.9.6 networkit==11.2.1"
)

BUDGET_COLUMNS = [
    "pct", "radius", "lo", "hi", "n_events", "n_flagged", "escalation_rate", "rate_low", "rate_high",
    "n_sampled", "mean_subgraph_edges", "p95_subgraph_edges", "max_subgraph_edges", "cap_hits",
    "cap_hits_lossy", "cap_hit_rate", "subgraph_s_per_event", "orc_s_per_event", "total_s_per_event",
    "est_hours_all_flagged",
]


def orc_directed(G, edges, alpha=0.5, method="OTDSinkhornMix", nbr_topk=3000, proc=1, **extra) -> dict:
    """Directed Ollivier-Ricci curvature of the requested edges of DiGraph G.

    edges: iterable of (u, v) that are edges of G (no self-loops: ORC is not
    defined for them and the library drops self-loops). Knobs are passed to
    OllivierRicci unchanged: alpha (mass kept on the node), method ("OTD",
    "ATD", "Sinkhorn", "OTDSinkhornMix"), nbr_topk, proc (worker processes; the
    library always forks a pool, also for proc=1), and **extra (e.g.
    shortest_path="pairwise", exp_power, base). G is copied by the library;
    edges without a "weight" attribute get weight 1. The library still builds
    all-pairs shortest paths over all of G (shortest_path="all_pairs"), so cost
    depends on |G|, not only on len(edges).

    Returns {(u, v): float} for exactly the requested edges. Not thread-safe:
    the library keeps its state in module globals.
    """
    if not G.is_directed():
        raise ValueError("orc_directed expects a directed nx.DiGraph")
    edges = [tuple(e) for e in edges]
    for u, v in edges:
        if u == v:
            raise ValueError(f"self-loop {(u, v)}: Ollivier-Ricci curvature is not defined")
        if not G.has_edge(u, v):
            raise ValueError(f"edge {(u, v)} is not in G")
    if not edges:
        return {}  # the library would compute every edge for an empty list
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc

    extra.setdefault("verbose", "ERROR")
    with warnings.catch_warnings():
        # POT warns when a measure is the integer list [1] (a node with no
        # relevant neighbours keeps all its mass); the cost is unaffected.
        warnings.filterwarnings("ignore", message="Input histogram consists of integer")
        orc = OllivierRicci(G, alpha=alpha, method=method, nbr_topk=nbr_topk, proc=proc, **extra)
        result = orc.compute_ricci_curvature_edges(edges)
    return {e: float(result[e]) for e in edges}


def edge_tiers(G, u, v) -> dict:
    """Relevance tier of every edge of G for the directed ORC of (u, v).

    0: the scored edge (u, v).
    1: edges that define the two measures: in-edges of u, out-edges of v.
    2: other edges on a possible S -> T path of length <= 2: edges leaving
       S = {u} + pred(u), and edges x -> w with x in succ(S), w in T = {v} + succ(v).
    3: everything else, including self-loops (GraphRicciCurvature drops them).
    For unweighted graphs, removing tier-3 edges leaves kappa(u, v) unchanged
    (module docstring).
    """
    pred_u = set(G.predecessors(u)) - {u}
    succ_v = set(G.successors(v)) - {v}
    S = pred_u | {u}
    T = succ_v | {v}
    X = set()
    for s in S:
        X.update(G.successors(s))
    tiers = {}
    for a, b in G.edges:
        if a == b:
            tiers[(a, b)] = 3
        elif (a, b) == (u, v):
            tiers[(a, b)] = 0
        elif (b == u and a in pred_u) or (a == v and b in succ_v):
            tiers[(a, b)] = 1
        elif a in S or (a in X and b in T):
            tiers[(a, b)] = 2
        else:
            tiers[(a, b)] = 3
    return tiers


def cap_subgraph(G, u, v, edge_cap, prune=False):
    """Deterministic edge-cap truncation of an escalation subgraph.

    Edges are ranked by (tier from edge_tiers, most recent attribute "t"
    first, then str(source), str(target)) and the first edge_cap are kept;
    prune=True additionally drops every tier-3 edge. The result holds u, v and
    the endpoints of the kept edges, with edge attributes copied. Truncation is
    lossless for ORC (unweighted) as long as only tier-3 edges are dropped;
    info["lossy"] says whether a tier <= 2 edge was cut.

    Returns (H, info) with info = {"n_edges", "n_kept", "capped", "lossy"}.
    """
    tiers = edge_tiers(G, u, v)
    n_edges = G.number_of_edges()
    cand = [e for e, k in tiers.items() if k <= 2] if prune else list(tiers)
    capped = len(cand) > edge_cap
    if not capped and not prune:
        return G, {"n_edges": n_edges, "n_kept": n_edges, "capped": False, "lossy": False}
    cand.sort(key=lambda e: (tiers[e], -G.edges[e].get("t", 0), str(e[0]), str(e[1])))
    kept = cand[:edge_cap]
    lossy = any(tiers[e] <= 2 for e in cand[edge_cap:])
    H = nx.DiGraph()
    H.add_nodes_from([u, v])
    H.add_edges_from((a, b, G.edges[a, b]) for a, b in kept)
    return H, {"n_edges": n_edges, "n_kept": len(kept), "capped": capped, "lossy": lossy}


FIRST_ROWS = 5  # escalate raises when each of its first FIRST_ROWS computed rows failed (systematic failure)
ROW_COLUMNS = ("orc_subgraph_edges", "orc_capped", "orc_lossy", "orc_src_in_deg", "orc_attempted", "orc_error")


class OrcWorkerDied(RuntimeError):
    """The ORC worker process ended (EOF, broken pipe, killed). Not a row error: the rows of the batch in flight
    stay pending. ``partial_frame`` holds escalate's frame with the batches finished before the death."""

    def __init__(self, msg, partial_frame=None):
        super().__init__(msg)
        self.partial_frame = partial_frame


class OrcFnError(RuntimeError):
    """An exception raised by the ORC function inside the worker, sent back as text (a row error)."""


def _resolve(path: str):
    import importlib

    mod, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:function', got {path!r}")
    return getattr(importlib.import_module(mod), attr)


def _worker_main(conn, orc_fn_path, knobs):
    """Loop of the ORC worker process: build each graph in the node order given, call the ORC function, send the
    values or the error text back."""
    import traceback

    fn = _resolve(orc_fn_path)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "close":
            return
        _, nodes, edges, scored = msg
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from((a, b, {"t": t}) for a, b, t in edges)
        try:
            conn.send(("ok", fn(G, scored, **knobs)))
        except Exception:
            conn.send(("err", traceback.format_exc(limit=3)))


class OrcWorker:
    """A long-lived ORC process started with the "spawn" context, so GraphRicciCurvature forks its pool from a
    small process instead of one that holds a large EventStore. A plain non-daemonic Process with a pipe, never a
    Pool: the library opens its own fork pool inside it, which a daemonic process may not do.

    orc_fn_path: "module:function" importable in the child (default ORC: "citecurv.curvature.orc:orc_directed").
    knobs: the ORC knobs, fixed for the worker's life (escalate checks them against its own).
    """

    def __init__(self, orc_fn_path: str = "citecurv.curvature.orc:orc_directed", **knobs):
        import multiprocessing as mp

        self.orc_fn_path = orc_fn_path
        self.knobs = dict(knobs)
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_worker_main, args=(child, orc_fn_path, self.knobs), daemon=False)
        self._proc.start()
        child.close()

    def compute(self, nodes: list, edges: list, scored: list) -> dict:
        """nodes: in the order the graph must have them; edges: (a, b, t); scored: edges to return."""
        try:
            self._conn.send(("compute", nodes, edges, scored))
            status, payload = self._conn.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise OrcWorkerDied(f"ORC worker ended: {exc!r} (exit code {self._proc.exitcode})") from exc
        if status == "err":
            raise OrcFnError(payload)
        return payload

    def alive(self) -> bool:
        return self._proc.is_alive()

    def close(self, timeout: float = 10.0):
        try:
            if self._proc.is_alive():
                self._conn.send(("close",))
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout)
        self._conn.close()


def _orc_knobs_core(knobs) -> dict:
    return {k: v for k, v in dict(knobs).items() if k != "proc"}


def escalate(events, curv, store=None, radius=1, edge_cap=5000, *, prune=False, orc_fn=None,
             max_events=None, stage="escalate", rows=None, batch=1, worker=None, progress=None, **orc_knobs):
    """Fill `orc` for escalated rows of the curvature cache.

    events: the events frame (used only to build an EventStore when store is
    None). curv: curvature cache frame (CONTRACTS.md) with `escalated` set by
    apply_screen. rows: explicit row positions to compute, in the order given
    (self-loops among them are skipped and counted); default = each escalated
    row that is not a self-loop, has orc NaN and is not yet attempted
    (`orc_attempted`, when the column exists), in the frame's order, so a
    second call resumes after a checkpoint. At most max_events of those rows:

      1. G = store.subgraph_at(u, v, t, radius): edges with ts <= t, the scored
         instance (u, v, t) excluded, same-t rows of other events included.
      2. Add the scored edge u -> v itself (if no earlier instance exists).
      3. Enforce edge_cap with cap_subgraph (deterministic; prune drops
         tier-3 edges, which leaves the unweighted ORC value unchanged).
      4. orc = orc_fn(G, [(u, v)], **orc_knobs)[(u, v)] (default orc_directed).

    Why adding the scored edge is not leakage: the curvature of an edge is only
    defined for an edge of the graph (kappa = 1 - W1(m_u, m_v) / d(u, v));
    without it d(u, v) and the support distances can be infinite. The edge is
    the event being scored, so it carries nothing beyond the event's own
    endpoints and time, which every score component already uses. It does not
    enter either measure (m_u lives on u and its predecessors, m_v on v and its
    successors); it only fixes the transport geometry. Everything else comes
    from ts <= t (NFR3).

    Self-loop rows are skipped (orc stays NaN; ORC is undefined for them).
    Every attempted row (a value, NaN or an error) gets radius and
    computed_at_stage = stage and the per-row columns of ROW_COLUMNS:
    orc_subgraph_edges (before the cap, scored edge included), orc_capped,
    orc_lossy, orc_src_in_deg (in-degree of the citing work at t in its
    subgraph; 0 means radius 1 is exact), orc_attempted, orc_error ("" when
    fine).

    batch > 1: up to `batch` subgraphs, nodes relabelled (k, node) with each
    subgraph's node order kept, joined into one DiGraph; one ORC call scores
    all their edges (one pool fork instead of `batch`). The subgraphs share no
    node, so with shortest_path="pairwise" every value equals its per-event
    value; it is refused otherwise. An exception raised by the ORC function is
    retried row by row; a row that still raises is recorded as an error.
    worker: an OrcWorker that runs the ORC function in its own process (its
    knobs, without proc, must equal orc_knobs). A dead worker raises
    OrcWorkerDied (rows of the batch in flight stay pending; the frame with the
    batches done before is on the exception). progress(event_ids) is called
    before every ORC call with the ids in flight.

    Returns (updated copy of curv, stats dict): n_escalated, n_pending,
    n_computed, n_self_loops, n_skipped_self_loops, n_errors, rows (per-row
    columns of the rows computed here, with event_id), mean/p95/max_subgraph_edges,
    mean/max_kept_edges, n_capped, n_capped_lossy, subgraph_s_per_event,
    orc_s_per_event, total_s_per_event, seconds_total.
    """
    batch = int(batch)
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if worker is not None and _orc_knobs_core(worker.knobs) != _orc_knobs_core(orc_knobs):
        raise ValueError(f"worker knobs {worker.knobs} differ from the ORC knobs {orc_knobs}")
    if batch > 1 and orc_knobs.get("shortest_path") != "pairwise":
        raise ValueError("batch > 1 needs shortest_path='pairwise' (joined subgraphs must keep their own distances)")
    orc_fn = orc_fn or orc_directed
    out = curv.copy()
    out["orc"] = out["orc"].astype(float)
    out["radius"] = out["radius"].astype(float)
    defaults = {"orc_subgraph_edges": np.nan, "orc_capped": False, "orc_lossy": False, "orc_src_in_deg": np.nan,
                "orc_attempted": False, "orc_error": ""}
    for c, d in defaults.items():
        if c not in out.columns:
            out[c] = d
    out["orc_subgraph_edges"] = out["orc_subgraph_edges"].astype(float)
    out["orc_src_in_deg"] = out["orc_src_in_deg"].astype(float)
    for c in ("orc_capped", "orc_lossy", "orc_attempted"):
        out[c] = out[c].fillna(False).astype(bool)
    out["orc_error"] = out["orc_error"].fillna("").astype(object)
    esc = out["escalated"].to_numpy(dtype=bool)
    loop = (out["src_id"] == out["dst_id"]).to_numpy()
    n_skipped_loops = 0
    if rows is None:
        pending = np.flatnonzero(esc & ~loop & out["orc"].isna().to_numpy() & ~out["orc_attempted"].to_numpy())
    else:
        rows = np.asarray(rows, dtype=np.int64)
        n_skipped_loops = int(loop[rows].sum())
        pending = rows[~loop[rows]]
    n_pending = len(pending)
    if max_events is not None:
        pending = pending[:max_events]
    if len(pending) and store is None:
        store = EventStore(events)

    col = {c: out.columns.get_loc(c) for c in ("event_id", "src_id", "dst_id", "t", "orc", "radius",
                                               "computed_at_stage", *ROW_COLUMNS)}
    sizes, kept_sizes, t_sub, t_orc = [], [], [], []
    n_capped = n_lossy = n_errors = 0
    done = []
    checked_first = False
    t_start = time.perf_counter()

    def call(graphs):
        """One ORC call for a list of (H, u, v); returns values in order. Single graph: no relabelling."""
        if len(graphs) == 1:
            H, u, v = graphs[0]
            J, scored = H, [(u, v)]
        else:
            J = nx.DiGraph()
            scored = []
            for k, (H, u, v) in enumerate(graphs):
                J.add_nodes_from((k, n) for n in H.nodes)
                J.add_edges_from(((k, a), (k, b), d) for a, b, d in H.edges(data=True))
                scored.append(((k, u), (k, v)))
        if worker is not None:
            got = worker.compute(list(J.nodes), [(a, b, d.get("t")) for a, b, d in J.edges(data=True)], scored)
        else:
            got = orc_fn(J, scored, **orc_knobs)
        return [float(got[e]) for e in scored]

    def record(i, H_info, val, err):
        nonlocal n_errors
        info, in_deg = H_info
        out.iat[i, col["orc"]] = float(val)
        out.iat[i, col["radius"]] = float(radius)
        out.iat[i, col["computed_at_stage"]] = stage
        out.iat[i, col["orc_subgraph_edges"]] = float(info["n_edges"])
        out.iat[i, col["orc_capped"]] = bool(info["capped"])
        out.iat[i, col["orc_lossy"]] = bool(info["lossy"])
        out.iat[i, col["orc_src_in_deg"]] = float(in_deg)
        out.iat[i, col["orc_attempted"]] = True
        out.iat[i, col["orc_error"]] = err
        n_errors += bool(err)
        done.append(i)

    for lo in range(0, len(pending), batch):
        group = pending[lo:lo + batch]
        built = []
        t0 = time.perf_counter()
        for i in group:
            u, v, t = out.iat[i, col["src_id"]], out.iat[i, col["dst_id"]], int(out.iat[i, col["t"]])
            G = store.subgraph_at(u, v, t, radius)
            if not G.has_edge(u, v):
                G.add_edge(u, v, t=t)
            in_deg = len(set(G.predecessors(u)) - {u})
            H, info = cap_subgraph(G, u, v, edge_cap, prune=prune)
            built.append((i, H, u, v, info, in_deg))
            sizes.append(info["n_edges"])
            kept_sizes.append(info["n_kept"])
            n_capped += info["capped"]
            n_lossy += info["lossy"]
        t1 = time.perf_counter()
        ids = [int(out.iat[i, col["event_id"]]) for i, *_ in built]
        try:
            if progress is not None:
                progress(ids)
            vals = call([(H, u, v) for _, H, u, v, _, _ in built])
            for (i, _, _, _, info, in_deg), val in zip(built, vals):
                record(i, (info, in_deg), val, "")
        except OrcWorkerDied as exc:
            exc.partial_frame = out
            raise
        except (ImportError, TypeError):
            raise  # a missing library or a wrong knob is not a row error
        except Exception:
            for i, H, u, v, info, in_deg in built:
                try:
                    if progress is not None:
                        progress([int(out.iat[i, col["event_id"]])])
                    val, err = call([(H, u, v)])[0], ""
                except OrcWorkerDied as exc:
                    exc.partial_frame = out
                    raise
                except (ImportError, TypeError):
                    raise
                except Exception as exc:  # a row error: recorded, the run moves on
                    val, err = float("nan"), f"{type(exc).__name__}: {exc}"[:2000]
                record(i, (info, in_deg), val, err)
        t2 = time.perf_counter()
        if not checked_first and len(done) >= FIRST_ROWS:
            checked_first = True
            first = [out.iat[i, col["orc_error"]] for i in done[:FIRST_ROWS]]
            if all(first):
                raise RuntimeError(f"ORC failed on each of the first {FIRST_ROWS} rows (a systematic failure, not "
                                   f"row errors): {first[0][:500]}")
        t_sub.extend([(t1 - t0) / len(built)] * len(built))
        t_orc.extend([(t2 - t1) / len(built)] * len(built))

    def mean(x):
        return float(np.mean(x)) if x else float("nan")

    rows_frame = out.iloc[done][["event_id", *ROW_COLUMNS]].reset_index(drop=True)
    stats = {
        "n_escalated": int(esc.sum()),
        "n_pending": int(n_pending),
        "n_computed": len(done),
        "n_self_loops": int((esc & loop).sum()),
        "n_skipped_self_loops": n_skipped_loops,
        "n_errors": int(n_errors),
        "rows": rows_frame,
        "mean_subgraph_edges": mean(sizes),
        "p95_subgraph_edges": float(np.percentile(sizes, 95)) if sizes else float("nan"),
        "max_subgraph_edges": int(max(sizes)) if sizes else 0,
        "mean_kept_edges": mean(kept_sizes),
        "max_kept_edges": int(max(kept_sizes)) if kept_sizes else 0,
        "n_capped": int(n_capped),
        "n_capped_lossy": int(n_lossy),
        "subgraph_s_per_event": mean(t_sub),
        "orc_s_per_event": mean(t_orc),
        "total_s_per_event": mean([a + b for a, b in zip(t_sub, t_orc)]),
        "seconds_total": time.perf_counter() - t_start,
    }
    return out, stats


def escalation_budget(curv, store, radii=(1, 2), pcts=(0.01, 0.02, 0.05), sample=200, orc_fn=None, *,
                      ref_mask=None, edge_cap=5000, prune=False, inclusive=False, seed=0,
                      **orc_knobs) -> pd.DataFrame:
    """S2.3 grid: one row per (tail percentile p, radius).

    For each p, cut-offs are the p and 1 - p quantiles of `forman` over the
    reference rows (ref_mask; None = all rows, acceptable only for tooling
    demos, real calibration passes the training mask), applied to all rows
    (strict tails unless inclusive). escalation_rate = flagged share of all
    rows, split into rate_low / rate_high. Then the same seeded sample of at
    most `sample` flagged rows is escalated at every radius (orc_fn, edge_cap,
    prune and **orc_knobs as in escalate) to measure subgraph size, cap hits
    and seconds per flagged event. est_hours_all_flagged = n_flagged x
    total_s_per_event / 3600 (single process, measured on this machine).
    """
    if ref_mask is None:
        ref_mask = np.ones(len(curv), dtype=bool)
    rng = np.random.default_rng(seed)
    rows = []
    f = curv["forman"].to_numpy(dtype=float)
    for p in pcts:
        th = screen_thresholds(curv, ref_mask, p, 1.0 - p, inclusive=inclusive)
        lo, hi = th["global"]["lo"], th["global"]["hi"]
        fl = apply_screen(curv, th)
        mask = fl["escalated"].to_numpy()
        low = (f <= lo) if inclusive else (f < lo)
        n, n_flag = len(fl), int(mask.sum())
        idx = np.flatnonzero(mask)
        pick = np.sort(rng.choice(idx, size=min(sample, idx.size), replace=False)) if idx.size else idx
        for radius in radii:
            sub = fl.copy()
            sub["escalated"] = False
            sub.iloc[pick, sub.columns.get_loc("escalated")] = True
            _, st = escalate(None, sub, store, radius, edge_cap, prune=prune, orc_fn=orc_fn, **orc_knobs)
            k = st["n_computed"]
            rows.append({
                "pct": float(p), "radius": int(radius), "lo": lo, "hi": hi, "n_events": n,
                "n_flagged": n_flag, "escalation_rate": n_flag / n if n else float("nan"),
                "rate_low": float((mask & low).sum()) / n if n else float("nan"),
                "rate_high": float((mask & ~low).sum()) / n if n else float("nan"),
                "n_sampled": k, "mean_subgraph_edges": st["mean_subgraph_edges"],
                "p95_subgraph_edges": st["p95_subgraph_edges"], "max_subgraph_edges": st["max_subgraph_edges"],
                "cap_hits": st["n_capped"], "cap_hits_lossy": st["n_capped_lossy"],
                "cap_hit_rate": st["n_capped"] / k if k else float("nan"),
                "subgraph_s_per_event": st["subgraph_s_per_event"], "orc_s_per_event": st["orc_s_per_event"],
                "total_s_per_event": st["total_s_per_event"],
                "est_hours_all_flagged": n_flag * st["total_s_per_event"] / 3600.0 if k else float("nan"),
            })
    return pd.DataFrame(rows, columns=BUDGET_COLUMNS)

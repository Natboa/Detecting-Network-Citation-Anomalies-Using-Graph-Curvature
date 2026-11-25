"""Module-level fake ORC functions for tests that need them by import path ("orc_fakes:<name>"), e.g. inside a
spawned OrcWorker. Not a test module (no test_ prefix)."""

import math
import multiprocessing as mp
import os

import networkx as nx


def component_fake(G, edges, **knobs):
    """For each requested edge (u, v): edges of the weakly connected component of u, plus the largest edge time
    in that component / 1e6. Equal for a batched (joined, relabelled) and a per-event call exactly when the joined
    subgraphs stay apart; any merge changes the edge count or the largest time."""
    out = {}
    for u, v in edges:
        comp = nx.node_connected_component(G.to_undirected(as_view=True), u)
        sub = G.subgraph(comp)
        t_max = max((d.get("t", 0) for _, _, d in sub.edges(data=True)), default=0)
        out[(u, v)] = float(sub.number_of_edges()) + float(t_max) / 1e6
    return out


def nan_fake(G, edges, **knobs):
    return {tuple(e): math.nan for e in edges}


def _noop():
    return None


def child_starting_fake(G, edges, **knobs):
    """Starts and joins a spawn child before answering, as GraphRicciCurvature opens its own pool: a daemonic host
    process may not do this."""
    p = mp.get_context("spawn").Process(target=_noop)
    p.start()
    p.join()
    return component_fake(G, edges)


def suicide_fake(G, edges, **knobs):
    os._exit(3)


def raising_fake(G, edges, **knobs):
    raise RuntimeError("fake failure inside orc_fn")

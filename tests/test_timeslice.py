"""S2.1: EventStore — graph state as it existed at time t.

Spec under test (IMPLEMENTATION_PLAN.md S2.1):
- subgraph_at(u, v, t, radius): union of radius-hop neighborhoods of u and v
  (hops traverse edges in either direction, edges stay directed) over events
  with ts <= t, excluding the scored (u, v) instance at ts == t itself.
- degrees_at(t): in/out event counts with ts <= t; t must be non-decreasing.
"""

import random

import networkx as nx
import pytest

from citecurv.curvature.timeslice import EventStore

HAND_STREAM = [
    ("a", "b", 1), ("c", "b", 2), ("b", "d", 3), ("e", "a", 4), ("a", "d", 5),
    ("d", "f", 6), ("c", "a", 7), ("f", "b", 8), ("e", "d", 9), ("a", "f", 10),
]


def brute_subgraph(events, u, v, t, radius):
    kept = [
        (s, d, ts) for s, d, ts in events
        if ts <= t and not (s == u and d == v and ts == t)
    ]
    G_all = nx.DiGraph()
    G_all.add_nodes_from([u, v])
    for s, d, ts in kept:
        if not G_all.has_edge(s, d):
            G_all.add_edge(s, d)
    und = G_all.to_undirected(as_view=True)
    nodes = {u, v}
    frontier = {u, v}
    for _ in range(radius):
        nxt = {n for f in frontier if f in und for n in und.neighbors(f)} - nodes
        nodes |= nxt
        frontier = nxt
    return nodes, {(s, d) for s, d in G_all.edges if s in nodes and d in nodes}


def test_hand_stream_radius_1():
    store = EventStore(HAND_STREAM)
    G = store.subgraph_at("a", "d", 5, radius=1)
    assert set(G.nodes) == {"a", "b", "d", "e"}
    assert set(G.edges) == {("a", "b"), ("b", "d"), ("e", "a")}
    assert not G.has_edge("a", "d")  # the scored instance is excluded


def test_hand_stream_radius_2_pulls_second_hop():
    store = EventStore(HAND_STREAM)
    G = store.subgraph_at("a", "d", 5, radius=2)
    assert set(G.nodes) == {"a", "b", "c", "d", "e"}
    assert set(G.edges) == {("a", "b"), ("b", "d"), ("e", "a"), ("c", "b")}


def test_hand_stream_later_event():
    store = EventStore(HAND_STREAM)
    G = store.subgraph_at("e", "d", 9, radius=1)
    assert set(G.nodes) == {"a", "b", "d", "e", "f"}
    assert set(G.edges) == {
        ("a", "b"), ("a", "d"), ("b", "d"), ("e", "a"), ("d", "f"), ("f", "b"),
    }


def test_earlier_same_pair_instance_survives_exclusion():
    events = [("u", "v", 1), ("x", "u", 2), ("u", "v", 5)]
    G = EventStore(events).subgraph_at("u", "v", 5, radius=1)
    assert G.has_edge("u", "v")  # the ts=1 instance remains; only ts=5 is excluded


def test_brute_force_sweep_random_stream():
    rng = random.Random(42)
    nodes = list(range(30))
    events = []
    for _ in range(200):
        s, d = rng.sample(nodes, 2)
        events.append((s, d, rng.randint(1, 1000)))
    store = EventStore(events)
    for s, d, ts in events:
        for radius in (1, 2):
            G = store.subgraph_at(s, d, ts, radius=radius)
            exp_nodes, exp_edges = brute_subgraph(events, s, d, ts, radius)
            assert set(G.nodes) == exp_nodes, (s, d, ts, radius)
            assert set(G.edges) == exp_edges, (s, d, ts, radius)


def test_degrees_at_monotonic_advance():
    store = EventStore(HAND_STREAM)
    in1, out1 = store.degrees_at(1)
    assert (in1["b"], out1["a"]) == (1, 1)
    in5, out5 = store.degrees_at(5)
    assert (in5["b"], in5["d"], in5["a"]) == (2, 2, 1)
    assert (out5["a"], out5["c"], out5["b"], out5["e"]) == (2, 1, 1, 1)
    in10, out10 = store.degrees_at(10)
    assert (in10["b"], in10["d"], in10["a"], in10["f"]) == (3, 3, 2, 2)
    assert (out10["a"], out10["c"], out10["e"]) == (3, 2, 2)


def test_degrees_at_rejects_going_backwards():
    store = EventStore(HAND_STREAM)
    store.degrees_at(5)
    with pytest.raises(ValueError, match="monotonic"):
        store.degrees_at(1)

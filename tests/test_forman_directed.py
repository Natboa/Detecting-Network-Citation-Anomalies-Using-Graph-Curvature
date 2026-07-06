"""S1.4: directed Forman-Ricci — hand-computed cases + weighted-reduces property.

Unweighted closed form for edge e = u->v (Saucan et al., 2019):
F(e) = 2 - in_deg(u) - out_deg(v)
"""

import networkx as nx

from citecurv.curvature.forman_directed import forman_directed, forman_edge

CASES = [
    # (edges, tested_edge, expected F = 2 - in_deg(u) - out_deg(v))
    ([("u", "v")], ("u", "v"), 2.0),                                    # isolated edge
    ([("a", "b"), ("b", "c")], ("a", "b"), 1.0),                        # chain head
    ([("a", "b"), ("b", "c")], ("b", "c"), 1.0),                        # chain tail
    ([("x1", "u"), ("x2", "u"), ("x3", "u"), ("u", "v")], ("u", "v"), -1.0),  # 3 in-edges at source
    ([("a", "b"), ("b", "c"), ("c", "a")], ("a", "b"), 0.0),            # directed 3-cycle
    ([("u", "v"), ("v", "r1"), ("v", "r2"), ("v", "r3"), ("v", "r4"), ("v", "r5")],
     ("u", "v"), -3.0),                                                  # target cites 5
]


def test_unweighted_hand_cases():
    for edges, (u, v), expected in CASES:
        G = nx.DiGraph(edges)
        assert forman_edge(G, u, v) == expected, f"edge {(u, v)} in {edges}"


def test_full_graph_sweep_matches_per_edge():
    G = nx.DiGraph([e for edges, _, _ in CASES for e in edges])
    result = forman_directed(G)
    assert set(result) == set(G.edges)
    for (u, v), val in result.items():
        assert val == forman_edge(G, u, v)


def test_weighted_with_unit_weights_reduces_to_unweighted():
    for seed in (0, 1, 2):
        G = nx.gnp_random_graph(30, 0.1, seed=seed, directed=True)
        nx.set_edge_attributes(G, 1.0, "w")
        nx.set_node_attributes(G, 1.0, "node_w")
        for u, v in G.edges:
            weighted = forman_edge(G, u, v, weight="w", node_weight="node_w")
            assert abs(weighted - forman_edge(G, u, v)) < 1e-12, (u, v)

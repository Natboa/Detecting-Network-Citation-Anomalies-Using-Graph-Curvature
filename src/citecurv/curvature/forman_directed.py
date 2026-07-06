"""Directed Forman-Ricci curvature, closed form from Saucan et al. (2019) (task S1.4).

For a directed edge e = u -> v with edge weight w_e and node weights w_u, w_v:

    F(e) = w_e * [ (w_u/w_e - sum over in-edges e_l of u:  w_u / sqrt(w_e * w_el))
                 + (w_v/w_e - sum over out-edges e_l of v: w_v / sqrt(w_e * w_el)) ]

With all weights 1 this reduces to F(e) = 2 - in_deg(u) - out_deg(v).

Only in-edges at the source and out-edges at the target enter the sums; e
itself is an out-edge of u and an in-edge of v, so it never appears in either
sum and no self-exclusion is needed. Cost is O(in_deg(u) + out_deg(v)) per
edge. No package implements this form (verified 2026-07-02), hence the custom
module. Pure NetworkX/math — runs on any platform.
"""

from math import sqrt


def forman_edge(G, u, v, weight=None, node_weight=None) -> float:
    """Directed Forman-Ricci of edge (u, v). weight / node_weight name the
    edge/node attributes to use; both None means the unweighted fast path.
    Missing attributes default to 1.0."""
    if weight is None and node_weight is None:
        return 2.0 - G.in_degree(u) - G.out_degree(v)

    def edge_w(data):
        return data.get(weight, 1.0) if weight else 1.0

    def node_w(n):
        return G.nodes[n].get(node_weight, 1.0) if node_weight else 1.0

    w_e = edge_w(G[u][v])
    w_u, w_v = node_w(u), node_w(v)
    in_sum = sum(w_u / sqrt(w_e * edge_w(d)) for _, _, d in G.in_edges(u, data=True))
    out_sum = sum(w_v / sqrt(w_e * edge_w(d)) for _, _, d in G.out_edges(v, data=True))
    return w_e * ((w_u / w_e - in_sum) + (w_v / w_e - out_sum))


def forman_directed(G, weight=None, node_weight=None) -> dict[tuple, float]:
    """Curvature for every edge of the DiGraph, keyed by (u, v)."""
    return {
        (u, v): forman_edge(G, u, v, weight=weight, node_weight=node_weight)
        for u, v in G.edges
    }

# GraphRicciCurvature 0.5.3.2 vs 0.6.1 for directed Ollivier-Ricci curvature

Status: PROVISIONAL, decision for Maor (pin in `configs/pins.md`, plan decision L5). Written 2026-10-31.

## Question and short answer

The plan pins GraphRicciCurvature 0.5.3.2 (L5, "directed support confirmed in source; dormant upstream").
Since then 0.5.4, 0.6.0 and 0.6.1 were released. Does anything change for the directed ORC we compute?

- On the graphs we use (unweighted directed subgraphs), 0.5.3.2 and 0.6.1 return the same values: 1,291
  measured values agree to within 6.7e-16 (floating-point rounding only).
- 0.6.x changes the curvature denominator from the edge weight to the shortest-path distance. This only
  matters for weighted graphs, where 0.5.3.2's value is wrong whenever a cheaper alternative path exists.
- Directed handling, all defaults and the dependency metadata (including `scipy<=1.13.1`) are unchanged.
- Neither version installs on Python 3.13 with its declared dependencies. Both work on 3.13 when installed
  with `--no-deps`, because the package never imports scipy (measured, values identical to 3.12).

Recommendation: see the last section.

## Releases (PyPI upload times, UTC)

| Version | Uploaded | Note |
|---|---|---|
| 0.5.3.2 | 2024-06-26 | current pin |
| 0.5.4 | 2026-08-13 | |
| 0.6.0 | 2026-08-24 04:55 | |
| 0.6.1 | 2026-08-24 05:41 | |

Upstream is active again, so the "dormant upstream" part of the L5 rationale no longer holds.

## Source differences (wheels of 0.5.3.2 and 0.6.1, `OllivierRicci.py`)

| Change | Where | Effect on this project |
|---|---|---|
| Curvature denominator: `1 - W / w(u,v)` (edge weight) became `1 - W / d(u,v)` (graph shortest-path distance); the zero-weight guard now tests `d(u,v)` | `_compute_ricci_curvature_single_edge` | None on unweighted graphs (`d(u,v) = w(u,v) = 1` for an edge). Different on weighted graphs with a shorter alternative path (measured below). |
| `shortest_path="pairwise"` builds each cost row with one multi-target Dijkstra per source node (`_source_target_shortest_paths`) instead of one bidirectional Dijkstra per pair | `_distribute_densities` | Speed only; values identical (measured: 0 deviation on the 200-node DAG). Present in 0.6.1, not in the 0.6.0 wheel, although CHANGELOG.md lists it under 0.5.4. |
| The two pairwise LRU caches are cleared at the start of every computation | `_compute_ricci_curvature_edges` | No measured effect for us: upstream's new regression test (two graphs in one process) gives the correct value on 0.5.3.2 as well, because the edge computations run in forked pool workers and the parent's cache stays empty. |
| Packaging: `pyproject.toml`, version derived from git tags (`_version.py`), `__version__` import fallback | `__init__.py` | None. |
| FormanRicci: balanced Forman curvature (Topping et al., 2022) and non-integer node labels | `FormanRicci.py` | None: we use our own directed Forman module (L4). |

Unchanged between the two versions (read from both sources):

- Directed handling: for an edge u -> v the source measure puts mass `alpha` on u and spreads `1 - alpha` over
  the predecessors of u; the target measure does the same with v and the successors of v. A node without such
  neighbours keeps all its mass (a Dirac measure). Costs are directed shortest-path lengths.
- Defaults: `alpha=0.5`, `method="OTDSinkhornMix"`, `base=e`, `exp_power=2`, `proc=cpu_count()`,
  `chunksize=None`, `shortest_path="all_pairs"`, `cache_maxsize=1000000`, `nbr_topk=3000`; the Mix method
  switches to Sinkhorn (regularisation 0.1) only when both supports exceed 2,000 nodes.
- A `fork` multiprocessing pool is created for every call, also for `proc=1`; self-loops are removed; edges get
  weight 1.0 when no edge carries the weight attribute.
- Dependency metadata: `scipy<=1.13.1`, `pot>=0.8.0`, `networkit>=6.1`, `networkx>=2.0`, `numpy`, `cython`,
  `packaging`, `python-louvain`.
- `method="ATD"` divides by `|pred(u)| * |succ(v)|` and raises `ZeroDivisionError` when either set is empty
  (5 of the 6 toy graphs, and the DAG, in both versions). A citing paper has no in-citations at its own
  publication time in almost every case, so ATD is not usable for this project.

## Measured value differences on identical graphs

Script: `scripts/run_orc_validation.py`, one run per environment; JSON in `results/`. Environments:
0.5.3.2 with POT 0.9.6, networkit 11.2.1, scipy 1.13.1, numpy 2.2.6; 0.6.1 with what it resolves to
unpinned (POT 0.9.7.post1, networkit 11.2.2, scipy 1.13.1, numpy 2.2.6); both Python 3.12.3, Linux, 4 CPUs.

| Graphs (all unweighted unless stated) | Values compared | Max abs difference 0.5.3.2 vs 0.6.1 |
|---|---|---|
| 6 toy graphs of `tests/test_forman_directed.py` (OTD, Mix, Sinkhorn; all edges; flipped edge; undirected) | 70 | 0 |
| Upstream directed example `[(0,1),(1,2),(2,3),(1,3),(3,1)]` | 5 | 0 |
| 200-node random DAG, 849 edges (OTD values; deviations of Mix, Sinkhorn, forced Mix, pairwise; reversed graph) | 849 + 13 summaries | 6.7e-16 |
| 61 single-edge flips on the DAG (value before and after) | 122 | 4.4e-16 |
| 40 escalation subgraphs of a SYNTHETIC stream, radius 1 / 2, with and without pruning, and the full time-t graph | 220 | 2.2e-16 |
| Weighted undirected triangle (0-1: 10, 0-2: 1, 2-1: 1), edge (0,1) | 1 | 0.9 vs 0.5 (0.5 is upstream's new expected value) |
| Weighted directed graph (0->1: 10, 0->2: 1, 2->1: 1, 1->3: 1), edge (0,1) | 1 | 0.75 vs -0.25 |

Hand check of the directed weighted case: m_0 is a Dirac at 0, m_1 = 0.5 at 1 + 0.5 at 3, d(0,1) = 2 (via 2),
d(0,3) = 3, so W = 2.5. 0.5.3.2 divides by the edge weight 10 (kappa = 0.75); 0.6.1 divides by d(0,1) = 2
(kappa = -0.25). Only the second is the Ollivier (2009) definition on the weighted shortest-path metric.

Checks that pass in both versions (field `checks` in each JSON): hand-computed toy values (alpha 0.5:
0, -0.5, -0.5, -0.5, 0.5, -0.5); direction sensitivity (flipping the tested edge changes 5 of 6 toy values and
56 of 61 DAG edges, reversing every edge changes none); upstream directed example; OTD vs OTDSinkhornMix
max deviation 0.0 on toys and DAG (plan threshold 0.05: Mix equals OTD below 2,000 support nodes; with the
threshold forced down to 3, 141 DAG edges take the Sinkhorn branch and deviate by at most 0.0141; plain
Sinkhorn deviates by at most 0.0141 on the DAG and 4.2e-5 on the toys); multiprocessing with `proc` 1, 2, 4
and both shortest-path modes finishes well inside a 300 s timeout, 30 repeated calls each, values equal to
`proc=1`; radius-2 escalation subgraphs reproduce ORC on the whole time-t graph exactly.

Timings are the same within noise: about 21 ms per call on a tiny graph (dominated by forking the pool),
0.10-0.12 s for all 849 DAG edges with OTD.

Python 3.12 emits `DeprecationWarning: This process ... is multi-threaded, use of fork() may lead to
deadlocks` (networkit starts threads). No hang occurred in any run here (5 configurations x 31 calls, plus the
test-suite), but the risk is real in principle; `proc=1` keeps exposure lowest.

## Python 3.13 (Colab default since 2026-08-25)

- `scipy<=1.13.1` has no Python 3.13 wheel. With binary wheels only, `uv` reports the requirement set as
  unsatisfiable for 0.5.3.2 (and the metadata of 0.6.1 is identical). Building scipy 1.13.1 from source failed
  here at the meson step for lack of a Fortran compiler; whether it builds on Colab was not tested.
- GraphRicciCurvature never imports scipy (checked in both sources); POT needs `scipy>=1.6` but not an old one.
  Installing the declared dependencies with a current scipy and then
  `pip install --no-deps GraphRicciCurvature==<version>` works on Python 3.13.18:
  0.5.3.2 with POT 0.9.6, networkit 11.2.1, scipy 1.18.1, numpy 2.5.4; 0.6.1 with POT 0.9.7.post1,
  networkit 11.2.2, scipy 1.18.1, numpy 2.5.4. All validation checks pass, `tests/test_orc.py` passes, and the
  values match the Python 3.12 runs to within 4.4e-16 (`results/orc_validation_*_py313.json`).
- This was measured in a Linux container, not on Colab itself.

## Recommendation for `configs/pins.md` (not applied; Maor decides)

1. Library: move the curv pin to GraphRicciCurvature 0.6.1 and keep POT 0.9.6 and networkit 11.2.1 (this
   combination resolves; checked with `uv pip compile`). Reasons: identical values on everything the pipeline
   computes today, correct weighted ORC if a weighted variant is ever added (0.5.3.2 would silently be wrong
   there), a maintained upstream, and a faster `pairwise` mode. Keeping 0.5.3.2 is also safe for the current
   unweighted design; the cost of staying is only the weighted-graph bug.
2. Python: keep Python 3.12 as the primary curv runtime (Colab runtime version 26.07, which
   `notebooks/30_curvature.ipynb` checks for), with `scipy<=1.13.1` as now. Record the validated Python 3.13
   path as the fallback: `pip install POT==0.9.6 networkit==11.2.1 networkx python-louvain cython packaging`
   then `pip install --no-deps GraphRicciCurvature==<pin>`, scipy taken from the runtime.
3. Knobs: never `method="ATD"`; `OTDSinkhornMix` and `OTD` are equivalent at our support sizes;
   `shortest_path="pairwise"` avoids the all-pairs matrix (same values; in a SYNTHETIC probe the process peak
   RSS rose from 272 MB to 559 MB when all-pairs ORC ran on unpruned radius-2 subgraphs of up to 2,293 nodes).

Suggested row: `curv | GraphRicciCurvature 0.6.1 · POT 0.9.6 · networkit 11.2.1 · scipy ≤1.13.1 (py3.12;
on py3.13 install GraphRicciCurvature with --no-deps) · networkx`.

## Reproduce

```bash
uv venv -p python3.12 venv0532 && uv pip install -p venv0532/bin/python GraphRicciCurvature==0.5.3.2 \
    "scipy<=1.13.1" POT==0.9.6 networkit==11.2.1 networkx numpy pandas pytest pyyaml
PYTHONPATH=src venv0532/bin/python scripts/run_orc_validation.py --out results/orc_validation_0532.json
# same for GraphRicciCurvature==0.6.1 -> results/orc_validation_061.json, then:
python scripts/run_orc_validation.py --compare results/orc_validation_0532.json results/orc_validation_061.json
```

## References

Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., & Saucan, E. (2015). Ricci curvature of the Internet topology.
*2015 IEEE Conference on Computer Communications (INFOCOM)*, 2758-2766.

Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. *Journal of Functional Analysis,
256*(3), 810-864.

Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X., & Bronstein, M. M. (2022). Understanding
over-squashing and bottlenecks on graphs via curvature. *International Conference on Learning
Representations*.

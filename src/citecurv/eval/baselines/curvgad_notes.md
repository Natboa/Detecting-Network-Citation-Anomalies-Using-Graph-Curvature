# CurvGAD sealed-environment notes (task S4.4, decision L17)

CurvGAD (Grover, Gordon, & Faloutsos, 2025; ICML 2025, arXiv:2502.08605) is the curvature-based graph anomaly
detector in our baseline set. It runs in its own environment and is never imported by `citecurv`; we only feed it
our cutoff snapshot and read back one score per node. Its code is not vendored into this repository.

## 1. What was checked, and how

| Item | Status |
|------|--------|
| Source | `git clone https://github.com/karish-grover/curvgad`, HEAD `81a9d94a0bc023ebc1eb551e2b515819f46ba0d2` (2025-07-22, "Update README.md"). All file and line references below are to this commit. |
| License | MIT, copyright 2025 Karish Grover, Geoffrey J. Gordon, Christos Faloutsos. Using and citing it is fine; keep it out of our repo anyway (sealed artifact, plan S4.4). |
| Dependency stack | Installed into a throwaway venv on CPU (Linux, Python 3.12.3) and imported together: torch 2.2.1+cu121, dgl 2.1.0, torchdata 0.7.1, geoopt 0.5.1, torch_geometric 2.6.1, GraphRicciCurvature 0.5.3.2, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.9.1, catboost 1.2.10, networkx 3.7. A DGL graph was built and `dgl.data.utils` loaded. |
| CurvGAD code itself | **Not executed here** (running the cloned code was not permitted in this session). Everything about its behaviour below comes from reading the source. The first Colab run is the real test. |
| GPU path | Not testable here (no GPU; `data.dgl.ai` is unreachable from this container). Marked "verify on Colab" where it matters. |

## 2. Sealed environment recipe (Colab)

Rules: one fresh venv used only for CurvGAD; never `pip install -r requirements.txt` (see risk R1); nothing from this
env is installed into the citecurv envs, and vice versa.

DGL is the binding constraint. DGL ships a C++ library per exact torch version: the PyPI wheel `dgl==2.1.0`
(2024-03-05, the last DGL release with Linux wheels on PyPI) contains `libgraphbolt_pytorch_<v>.so` for torch 2.0.0,
2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0 and 2.2.1 only (verified from the wheel), and `import dgl` fails on any other
torch ("Cannot find DGL C++ graphbolt library ... libgraphbolt_pytorch_2.15.0.so" with torch 2.15). DGL 2.1.0 also
imports `torchdata.datapipes`, removed in recent torchdata, so torchdata must be pinned to 0.7.1, and it needs
`pyyaml` and `pydantic`, which its wheel does not declare.

CPU recipe (verified to install and import, section 1):

```bash
pip install uv                                    # the verified route; `python -m venv` also works where ensurepip exists
uv venv -p 3.12 /content/curvgad_env              # separate from Colab's preinstalled torch
uv pip install -p /content/curvgad_env/bin/python \
  "torch==2.2.1" "numpy==1.26.4" "dgl==2.1.0" "torchdata==0.7.1" pyyaml pydantic \
  "geoopt==0.5.1" "torch_geometric==2.6.1" "GraphRicciCurvature==0.5.3.2" "scipy<=1.13.1" \
  scikit-learn pandas networkx matplotlib seaborn psutil sympy catboost openpyxl
git clone https://github.com/karish-grover/curvgad /content/curvgad
git -C /content/curvgad checkout 81a9d94a0bc023ebc1eb551e2b515819f46ba0d2
```

GPU recipe (verify on Colab): the PyPI `torch==2.2.1` wheel is already the CUDA 12.1 build (`2.2.1+cu121`), but
the PyPI `dgl==2.1.0` wheel is a CPU build (its library carries the message "is not enabled. Please install the
cuda version of dgl", and no CUDA symbols). For a GPU run, replace the dgl pin with DGL's own CUDA wheel for torch
2.2 / cu121 from the DGL wheel index (`https://data.dgl.ai/wheels/...`, see the DGL install page; not reachable
from the container these notes were written in). Then check `dgl.graph(([0],[1])).to("cuda")` before anything else.
If no matching CUDA wheel exists any more, run on a CPU high-RAM runtime with the CPU recipe and the node cap from
section 4 (dense N x N work, see R3).

Record in the run log: `pip freeze`, the CurvGAD commit, GPU type, wall-clock per epoch, peak memory.

## 3. Input format the code expects

`utils.Dataset(name)` loads `datasets/<name>` with `dgl.data.utils.load_graphs` (paths are relative to the working
directory, which must be the repository root). The single DGL graph must carry:

| ndata key | dtype / shape | meaning |
|-----------|---------------|---------|
| `feature` | float32 (N, d) | node features (StandardScaler is applied inside the detector) |
| `label` | int64 (N,) | 0/1 node labels; **used for training** (see R5) |
| `train_mask`, `val_mask`, `test_mask` | bool (N,) | splits; or `train_masks`/`val_masks`/`test_masks` (N, T) for T trials |

Plus two precomputed files, also relative to the repository root:

- `curvature_data/<prefix>_curvature_matrix.npz`: scipy CSR (N, N) of edge curvatures, written by
  `preprocessing/compute_curvature.py --use_accelerated --save_matrix`. `<prefix>` is the dataset name cut at its
  first underscore (`curvgad_detector.py`, `load_curvature_data`), so use a name without underscores (`citecurv`).
- optional `ricci_flow_data/<name>_ricci_flow_adj_matrix.npz`: never produced by the released code (R4); without
  it the detector uses unit edge weights.

`benchmark.py` is not usable as a driver: its dataset list is hard-coded to `['chameleon']`, datasets are selected
by list index, and `device` is hard-coded to `'cuda'`. Use the driver in section 5 instead.

## 4. Feeding our cutoff snapshot

Same fair-comparison protocol as the PyGOD tier (`pygod_tier.py`): structure = distinct pairs of events with
t <= cutoff, made undirected (CurvGAD's curvature step converts to an undirected `nx.Graph` anyway), nodes =
snapshot papers plus the endpoints of the events to score (test-period papers enter as isolated nodes), features =
SPECTER2 vectors.

1. In a citecurv env, export the snapshot without torch:

   ```python
   from citecurv.eval.baselines.pygod_tier import snapshot_edges
   edges, node_index, stats = snapshot_edges(events, cutoff, node_ids, undirected=True)
   np.savez("curvgad_input.npz", edge_index=edges, node_ids=np.array(node_ids), x=features,
            label=node_label, train=train_mask, val=val_mask, test=test_mask)
   ```

2. **Node cap (required).** The released code builds dense N x N matrices every epoch (R3), so the full slice
   (~3 * 10^5 nodes) cannot run. Use a node-capped subgraph: a seeded sample of test events (all positive events up
   to a cap, plus random negatives), their endpoints, and their snapshot neighbours, up to `max_nodes`
   (`configs`: `baselines.curvgad.max_nodes`). Rough arithmetic, not a measurement: one dense float32 N x N matrix
   is 4 N^2 bytes (1.6 GB at N = 20,000) and several are alive at once, so N of order 10^4 is the realistic range
   on a 16 GB T4. Every method is then also scored on the same event subset for this comparison row.
   PROVISIONAL: the sampling rule is a decision for Maor (it changes base rates, so P@k on the subset is not
   comparable with P@k on the full test set).

3. **Labels and masks without test leakage (R5).** CurvGAD trains a classifier on `label[train_mask]`. Under L12 an
   event is positive when src or dst is retracted, so a test event citing a training-positive paper would inherit
   that paper's fitted score through the max rule. Use paper-level labels (paper retracted under the L12 family),
   and put in `train_mask` only snapshot nodes that are not an endpoint of any val or test event; `val_mask` =
   endpoints of val events not in train; `test_mask` = all remaining nodes. Every mask needs both classes (the
   repo's evaluation calls `roc_auc_score` on val and test; the detector divides by the number of train positives).

## 5. Driver and score export

A short driver kept in the Colab notebook (our code; it only calls CurvGAD's API). Run it with the sealed env's
python from the CurvGAD repository root.

```python
import os, sys, numpy as np, torch, dgl
from dgl.data.utils import save_graphs
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "preprocessing"))

z = np.load("/content/drive/.../curvgad_input.npz", allow_pickle=True)
g = dgl.graph((torch.from_numpy(z["edge_index"][0]), torch.from_numpy(z["edge_index"][1])),
              num_nodes=len(z["node_ids"]))
g.ndata["feature"] = torch.from_numpy(z["x"]).float()
g.ndata["label"] = torch.from_numpy(z["label"]).long()
for k in ("train", "val", "test"):
    g.ndata[f"{k}_mask"] = torch.from_numpy(z[k]).bool()
os.makedirs("datasets", exist_ok=True); os.makedirs("plots", exist_ok=True)   # train() saves a plot (R8)
save_graphs("datasets/citecurv", [g])

from utils import Dataset
from compute_curvature import compute_and_store_curvature
data = Dataset("citecurv"); data.split(False, 0)
compute_and_store_curvature(data, "citecurv", use_accelerated=True, save_matrix=True)

from models.curvgad_detector import CurvGADDetector
seed = 0                                                      # loop over configs seed_list [0..4]
import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
train_config = {"dataset_name": "citecurv", "device": "cuda" if torch.cuda.is_available() else "cpu",
                "epochs": 200, "patience": 100, "metric": "AUPRC", "inductive": False,
                "optimizer": "riemannian_adam", "use_autoencoder": True, "alpha": 1.0,
                "use_ricci_flow_weights": True, "seed": seed}
model_config = {"model": "CurvGAD", "lr": 0.01, "drop_rate": 0.0, "K": 7, "manifolds_config_str": "H32S32E32"}
det = CurvGADDetector(train_config, model_config, data)
det.train()

# one score per node: the same forward pass the repo uses for its test evaluation (curvgad_detector.py, train())
det.model_curv.eval(); det.model_adj_feat.eval()
with torch.no_grad():
    logits_curv = det.model_curv(det.source_graph); h_c = det.model_curv.get_embeddings()
    det.model_adj_feat(det.rf_graph); h_a = det.model_adj_feat.get_embeddings()
    _, node_curv = det.model_curv.decode_curvature(h_c, det.edge_indices)
    anomaly = det.compute_combined_anomaly_scores(node_curv, det.model_adj_feat.decode_adj(h_a, det.rf_edge_indices),
                                                  det.model_adj_feat.decode_features(h_a), det.num_nodes)
    logits = det.beta * logits_curv + (1 - det.beta) * anomaly.unsqueeze(-1)
    node_scores = logits.softmax(dim=1)[:, 1].cpu().numpy()
np.savez(f"/content/drive/.../curvgad_node_scores_seed{seed}.npz", node_ids=z["node_ids"], score=node_scores)
```

Hyperparameters are those of the README example (`K 7`, `H32S32E32`, `lr 0.01`, `--use_autoencoder`,
`--optimizer riemannian_adam`); they belong in `configs` (`baselines.curvgad`).

Back in a citecurv env, map node scores to events with the same rule as the PyGOD tier:

```python
from citecurv.eval.baselines.pygod_tier import edge_scores_from_nodes
z = np.load("curvgad_node_scores_seed0.npz", allow_pickle=True)
index = {k: i for i, k in enumerate(z["node_ids"].tolist())}
scores = edge_scores_from_nodes(z["score"], index, test_events_subset)   # DataFrame(event_id, score)
```

`edge_scores_from_nodes` raises if an event endpoint is missing from the node set, so a too-small node cap fails
loudly instead of silently dropping events.

## 6. Known risks (from the source at 81a9d94)

- **R1 requirements.txt is not installable.** Besides unpinned `>=` ranges it lists standard-library modules
  (`argparse`, `warnings`, `json`, `os`, `time`, `random`, `itertools`, `collections`). PyPI has no installable
  distribution for `warnings`, `json`, `os`, `random`, `itertools`, `collections` (no project) or `time` (project
  without files), so `pip install -r requirements.txt` stops with "No matching distribution". Use section 2.
- **R2 DGL pins torch.** See section 2; DGL also needs a CUDA build for GPU runs. DGL Linux wheels on PyPI end at
  2.1.0 (2024-03-05).
- **R3 Dense N x N memory and time.** `CurvGAD.forward` densifies the normalised Laplacian
  (`curvgad_gnn.py`, `torch.sparse_coo_tensor(...).to_dense()`) on every forward pass, and
  `compute_node_anomaly_from_adj` densifies the predicted and true adjacency (`curvgad_detector.py`). Each
  Chebyshev step multiplies by the dense Laplacian (O(N^2 d)); `train()` runs three forward passes per epoch
  (train, val, test). Node cap required (section 4).
- **R4 Ricci-flow weights are never written.** In `preprocessing/ollivier_ricci_flow.py` the final
  `save_adjacency_matrix(graph, output_file)` call is commented out, so `ricci_flow_data/` stays empty and
  `load_ricci_flow_weights` silently falls back to unit weights (it prints "Using Ricci flow matrix" only when the
  file exists). The released code therefore runs the curvature-invariant branch on the plain adjacency. Report this
  as "CurvGAD as released", not as the paper's full method.
- **R5 Semi-supervised, so label leakage is possible.** The loss includes cross-entropy on `label[train_mask]`
  (`loss_cls`), weighted by a learned `d`. Masks must follow section 4.3. It also means CurvGAD uses labels while
  the PyGOD tier does not; state this in the comparison table.
- **R6 Curvature term does not vary by node.** In `compute_combined_anomaly_scores`, `anomaly_scores_curv =
  torch.norm(predicted - true node curvatures, p='fro')` is a single scalar for the whole graph, so in the final
  node score it shifts every node equally; node-level differences come from the adjacency and feature terms and from
  the curvature-pipeline classifier logits.
- **R7 Saved curvature is an approximation.** `compute_curvature.py --save_matrix` saves only the accelerated
  degree/triangle bound (`accelerated_ricci_approximation`), never exact Ollivier-Ricci; without
  `--use_accelerated` it tries to save `None` and fails. It also needs `GraphRicciCurvature` importable even for the
  approximation (module-level import), and the README invocation `cd preprocessing && python compute_curvature.py`
  cannot import `utils` (it lives at the repository root); call it from the root with `PYTHONPATH=.` or import it
  as in section 5. The curvature is undirected (`convert_multidigraph_to_graph` builds an `nx.Graph`).
- **R8 Smaller traps.** The optimizer switch is inverted (`optimizer == 'riemannian_adam'` builds `torch.optim.Adam`,
  anything else builds `geoopt.optim.RiemannianAdam`); `train()` returns the test metrics of the last completed
  epoch, not of the best validation epoch; `train()` saves `plots/<name>_anomaly_scores_vs_curvature.png` and
  fails if `plots/` is missing; `benchmark.py` seeds with 3407 + 10 t, our runs use configs `seed_list`.

## 7. Open decisions (PROVISIONAL until Maor decides)

- Node-cap sampling rule and `max_nodes` (section 4.2), and whether the CurvGAD comparison is a separate table row
  on the shared event subset.
- Mask construction for the semi-supervised loss (section 4.3).
- If the GPU wheel route fails on Colab, whether CPU-only CurvGAD at a smaller cap is acceptable, or the fallback in
  the plan's risk register (report CurvGAD from its published numbers only, marked as such) applies.

## Reference

Grover, K., Gordon, G. J., & Faloutsos, C. (2025). CurvGAD: Leveraging curvature for enhanced graph anomaly
detection. *International Conference on Machine Learning (ICML 2025)*. arXiv:2502.08605. (The proceedings
volume is not verified here; the plan names PMLR v267.)

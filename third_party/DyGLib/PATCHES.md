# Vendored DyGLib: provenance and patches

## Provenance

- Upstream: https://github.com/yule-BUAA/DyGLib (Yu et al., 2023), MIT licence, `LICENSE` kept unchanged.
- Commit: `3aacc36b94b8d2d8293d70a74fdf6d39089b4163` ("update paper bibtex", 2023-09-28). This was
  the head of `master` when vendored (2026-10-31) and is the last upstream commit: the library
  has had no changes for three years, so fixes for current Python and torch live here.
- Copied: every tracked file of that commit except the bundled Myket dataset
  (`DG_data/myket/`, `processed_data/myket/`, about 64 MB) and the two README figures
  (`figures/*.jpg`, about 360 KB; the README image links are therefore broken). No `.git`.
- The unmodified copy is its own commit in this repository
  (`chore: vendor DyGLib at upstream 3aacc36 unmodified (S1.7)`), so `git diff <that commit> --
  third_party/DyGLib` shows exactly the changes listed below.
- Every changed line carries a trailing `# citecurv Pn` comment.

## Patch list

Line numbers refer to the patched files. "Needed by" says why the patch exists; P3 and P4 were
confirmed failing on the unpatched code (Python 3.12.3, torch 2.15.0, see "Verification").

| Id | File | Needed by | Effect when unused |
|----|------|-----------|--------------------|
| P1 | `utils/DataLoader.py` | any feature width other than 172 (SPECTER2 is 768) | none for 172-wide data |
| P2 | (reserved) | S1.8 text-initialised memory; not applied, see the last section | — |
| P3 | `utils/DataLoader.py` | Python >= 3.11 | — |
| P4 | `utils/EarlyStopping.py` | torch >= 2.6 | — |
| P5 | `models/MemoryModel.py`, `train_link_prediction.py`, `utils/load_configs.py` | keeping citation direction under bidirectional messaging | off by default (`--direction_feat_index -1`): upstream code path |
| P6 | `utils/load_configs.py` | datasets not named after the 14 benchmarks | — |
| P7 | `train_link_prediction.py`, `utils/load_configs.py` | the citecurv seed list (one run per seed) | `--seed_offset 0`: upstream seeds |
| P8 | `models/MemoryModel.py`, `train_link_prediction.py`, `utils/load_configs.py` | training time on graphs with many nodes (same results) | off by default (no `--lazy_memory`): upstream code path; citecurv turns it on |

### P1: feature widths derived from the data

`utils/DataLoader.py` line 81 (`get_link_prediction_data`) and line 192 (`get_node_classification_data`):

- before: `    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172`
- after: `    NODE_FEAT_DIM, EDGE_FEAT_DIM = node_raw_features.shape[1], edge_raw_features.shape[1]  # citecurv P1: derived from the data (upstream: = 172)`
- reason: upstream hardcodes 172 and asserts that the stored features are not wider, so 768-wide
  SPECTER2 node features fail the assert. The asserts and the padding code stay; with derived widths
  the padding is a no-op. Every model in the library sizes its layers from the two widths
  separately, so node and edge widths need not match. Side effect: benchmark datasets whose edge
  features are narrower than 172 (e.g. Myket, 2 columns) are no longer zero-padded, so their numbers
  are not bit-identical to upstream (layer fan-in, hence initialisation, changes). Wikipedia and
  Reddit are 172-wide and unaffected.

### P3: `random.sample` on a set

`utils/DataLoader.py` line 115:

- before: `    new_test_node_set = set(random.sample(test_node_set, int(0.1 * num_total_unique_node_ids)))`
- after: `    new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * num_total_unique_node_ids)))  # citecurv P3: Python >= 3.11 rejects sets`
- reason: since Python 3.11 `random.sample` raises `TypeError: Population must be a sequence` for
  sets. `sorted` makes the population order independent of set internals, so the draw stays fixed by
  `random.seed(2020)`. Python <= 3.10 converted the set with `tuple(...)`, so the sampled
  "new test nodes" differ from what upstream drew on old Python.

### P4: loading the memory-message pickle under torch >= 2.6

`utils/EarlyStopping.py` line 86:

- before: `            model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location)`
- after: `            model[0].memory_bank.node_raw_messages = torch.load(self.save_model_nonparametric_data_path, map_location=map_location, weights_only=False)  # citecurv P4: torch >= 2.6`
- reason: torch 2.6 changed the `torch.load` default to `weights_only=True`. The memory-message file
  is a `defaultdict` of `(Tensor, numpy.float64)` tuples, which that mode refuses
  (`UnpicklingError: Weights only load failed`), so every TGN/JODIE/DyRep run crashed when
  reloading its best checkpoint. The file is written by the same run, so trusting it is safe. The
  state-dict load on line 84 holds tensors only and works unchanged.

### P5: direction-aware messages and attention keys (opt-in)

Background (`src/citecurv/tgn/convert.py` docstring has the full argument): DyGLib builds one
message per endpoint, `[s_self, s_other, phi(dt), x_e]`, with the same edge-feature row `x_e` for
both, and its neighbour lists are undirected with the same edge id on both sides. A citation
direction column that is +1 on every row is therefore identical for the citing and the cited
paper, so the memory and the attention cannot tell which endpoint cited. P5 makes the cited
endpoint read the column negated (+1 citing, -1 cited), only when `--direction_feat_index >= 0`.

`utils/load_configs.py`, new line 52:

- before: (none)
- after: `    parser.add_argument('--direction_feat_index', type=int, default=-1, help='citecurv P5: edge-feature column of the citation-direction indicator, -1 = off')`
- reason: switch; -1 keeps upstream behaviour.

`train_link_prediction.py`, new lines 39-41:

- before: (none)
- after:
  `    # citecurv P5: destination node of every edge id (row 0 = padding), to read the direction indicator relative to the receiver`
  `    edge_dst_node_ids = np.zeros(edge_raw_features.shape[0], dtype=np.longlong); edge_dst_node_ids[full_data.edge_ids] = full_data.dst_node_ids`
  and one blank line
- reason: the attention module must know, for each neighbouring edge, whether the centre node is
  its destination; the model has no other access to edge endpoints.

`train_link_prediction.py` line 110 becomes lines 110-111 (the `MemoryModel(...)` call):

- before: `                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)`
- after: `                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device,`
  `                                           direction_feat_index=args.direction_feat_index, edge_dst_node_ids=edge_dst_node_ids)  # citecurv P5`
- reason: pass the switch and the map to the model.

`models/MemoryModel.py`:

- line 15, before: `                 dst_node_std_time_shift: float = 1.0, device: str = 'cpu'):`
  after: `                 dst_node_std_time_shift: float = 1.0, device: str = 'cpu', direction_feat_index: int = -1, edge_dst_node_ids: np.ndarray = None):  # citecurv P5`
  reason: new keyword arguments with upstream-neutral defaults.
- new line 48: `        self.direction_feat_index = direction_feat_index  # citecurv P5` (store the switch).
- line 84, before: `                                                            dropout=self.dropout)`
  after: `                                                            dropout=self.dropout, direction_feat_index=direction_feat_index, edge_dst_node_ids=edge_dst_node_ids)  # citecurv P5`
  reason: hand both to `GraphAttentionEmbedding`.
- line 158, before: `                                                                                                edge_ids=edge_ids)`
  after: `                                                                                                edge_ids=edge_ids, receiver_is_dst=True)  # citecurv P5`
  reason: the second `compute_new_node_raw_messages` call builds the messages received by the
  destination (cited) endpoint.
- line 214, before: `                                      node_interact_times: np.ndarray, edge_ids: np.ndarray):`
  after: `                                      node_interact_times: np.ndarray, edge_ids: np.ndarray, receiver_is_dst: bool = False):  # citecurv P5`
  reason: new argument, default keeps upstream behaviour.
- new lines 240-242:
  `        if receiver_is_dst and self.direction_feat_index >= 0:  # citecurv P5: the cited endpoint reads the indicator negated`
  `            edge_features = edge_features.clone()`
  `            edge_features[:, self.direction_feat_index] = -edge_features[:, self.direction_feat_index]`
  reason: role-relative direction in the memory message (the clone leaves the stored edge features untouched).
- line 556, before: `                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):`
  after: `                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, direction_feat_index: int = -1, edge_dst_node_ids: np.ndarray = None):  # citecurv P5`
  reason: same arguments for `GraphAttentionEmbedding`.
- new lines 582-583:
  `        self.direction_feat_index = direction_feat_index  # citecurv P5`
  `        self.edge_dst_node_ids = edge_dst_node_ids  # citecurv P5: ndarray (num_edges + 1, ), kept out of the state dict`
  reason: store them; a plain attribute, not a buffer, so checkpoints keep the upstream key set.
- new lines 657-661 (after `neighbor_edge_features = self.edge_raw_features[...]`):
  `            if self.direction_feat_index >= 0:  # citecurv P5: keys whose centre node is the cited endpoint read the indicator negated`
  `                centre_is_cited = torch.from_numpy(self.edge_dst_node_ids[neighbor_edge_ids] == node_ids[:, np.newaxis]).to(device)`
  `                neighbor_edge_features = neighbor_edge_features.clone()`
  `                neighbor_edge_features[..., self.direction_feat_index] = torch.where(centre_is_cited, -neighbor_edge_features[..., self.direction_feat_index],`
  `                                                                                     neighbor_edge_features[..., self.direction_feat_index])`
  reason: role-relative direction in the attention keys. Padded neighbour slots have edge id 0,
  whose destination entry is 0 and never equals a real node id, and whose feature row is zero.

Tests: `tests/test_tgn_infer.py::test_p5_memory_messages_keep_direction` and
`::test_p5_attention_keys_keep_direction` check +1/-1 with the switch on and +1/+1 with it off.

### P6: dataset names

`utils/load_configs.py` lines 14-15 become line 14:

- before:
  `    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia',`
  `                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts'])`
- after: `    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia')  # citecurv P6: upstream limited choices to the 14 benchmark names`
- reason: argparse rejected any converted citation dataset. Only the link-prediction parser is
  changed; the node-classification parser keeps its list (not used here).

### P7: seed offset

`utils/load_configs.py`, new line 51:

- before: (none)
- after: `    parser.add_argument('--seed_offset', type=int, default=0, help='citecurv P7: run r uses seed r + seed_offset')`

`train_link_prediction.py` lines 70 and 72:

- before: `        set_random_seed(seed=run)` / `        args.seed = run`
- after: `        set_random_seed(seed=run + args.seed_offset)  # citecurv P7` / `        args.seed = run + args.seed_offset  # citecurv P7`
- reason: upstream seeds run r with seed r, so seed 3 alone needs `--num_runs 4` and three wasted
  runs. With `--num_runs 1 --seed_offset s` one run uses seed s and saves under `TGN_seed{s}`.

### P8: a step updates only the memories it reads (opt-in)

Problem (measured 2026-11-12, see `tests/test_tgn_memory_scope.py` and PROGRESS.md): upstream
`compute_src_dst_node_temporal_embeddings` calls `get_updated_memories` over all nodes on every call
(twice per training step). That loops over every node id in Python (`MessageAggregator`), runs the
memory GRU, and later its backward pass, on every node with a pending message, and
`detach_memory_bank` visits every node after every step. Every node ever seen keeps one pending
message, so the time per step grows with the number of nodes seen: on the S1.7 pilot (309,425
nodes) from about 1.2 s to about 4 s within one epoch on a T4.

What P8 changes, only when `lazy_memory=True` (TGN only; JODIE and DyRep keep upstream):

- `MemoryModel.__init__`: new keyword `lazy_memory` (stored as `self.lazy_memory`, false unless the
  model is TGN), passed to `MemoryBank(track_undetached=...)`.
- `compute_src_dst_node_temporal_embeddings`: instead of `get_updated_memories(all nodes)`, uses
  `LazyUpdatedMemories(self)`, a new class (after `MemoryModel`). The embedding module reads the
  memory only as `node_memories[ids]`; indexing the lazy object returns, for those ids only, the
  stored memory (detached, as upstream's `node_memories.data.clone()`) with the GRU update applied
  to the ids that have a pending message, with upstream's aggregator (last message) and its
  "update to time in the past" assertion. It reads the message dictionary with `.get`, so it does
  not add an empty entry per node. The second return value (last-updated times) is `None`; only
  JODIE reads it.
- `MemoryBank`: new keyword `track_undetached`; `store_node_raw_messages` records the node ids it
  stored; `detach_memory_bank` then detaches only those, because every other message was detached
  by an earlier call. If the dictionary object was replaced since the last detach
  (`reload_memory_bank`, `EarlyStopping` loading a checkpoint), it visits every node as upstream.
- `utils/load_configs.py`: `--lazy_memory` (store_true, default off). `train_link_prediction.py`
  passes it to `MemoryModel`.
- `citecurv.tgn.train.TGN_DEFAULTS` sets `lazy_memory: True`; `citecurv.tgn.infer.infer_tgn` uses
  the run's setting from `resolved_args.json` (runs from before P8: upstream).

Why the results do not change: the loss depends on the memory only through the rows the embedding
module reads, and a row's GRU update depends only on that node's memory and message, so rows that
are never read change neither the forward values nor the gradients. The GRU now runs on fewer
rows, so the last bits of a matrix product can differ (float rounding). Tests:
`tests/test_tgn_memory_scope.py` compares several training steps (losses, embeddings, weights,
memories, pending messages, one and two layers) and an inference replay with the switch on and
off within 1e-5, checks that a step runs the GRU only on nodes it reads, and that no message keeps
a gradient graph after `detach_memory_bank`; `tests/test_tgn_infer.py::
test_patch_p8_changes_no_result_end_to_end` trains and scores the tiny stream both ways.

## Verification

- Unpatched failures reproduced on Python 3.12.3 / torch 2.15.0+cu132 / numpy 2.5.4 / pandas 3.1.0:
  P3 `TypeError: Population must be a sequence. For dicts or sets, use sorted(d).`;
  P4 `UnpicklingError: Weights only load failed`. P1 and P6 are argument/assert failures by
  construction (64- and 768-wide features; dataset name `synthcite`).
- With P1-P7: `scripts/smoke_tgn.py` trains stock TGN (P5 off) and the direction-aware variant (P5 on)
  for 3 epochs each on a synthetic citation stream, CPU only; results in `results/smoke_tgn.json`.
- No other incompatibility showed up in the TGN training, validation, test, checkpoint save/load
  and inference paths on that stack.

## Known issues left unpatched (paths citecurv does not use)

- `preprocess_data/preprocess_data.py` imports `distutils` (removed in Python 3.12). citecurv writes
  the processed files directly with `citecurv.tgn.convert.to_jodie`.
- `evaluate_link_prediction.py` and `evaluate_models_utils.py` build `MemoryModel` without the P5
  arguments and seed runs with `set_random_seed(run)`: evaluating a P5 checkpoint with that script
  runs the upstream message layout. citecurv evaluates through `citecurv.tgn.infer` instead.
- `train_node_classification.py` / `evaluate_node_classification.py`: not used, not checked.
- Scaling: upstream `compute_src_dst_node_temporal_embeddings` updates the memory of every node
  with a pending message in every batch. Measured on the S1.7 pilot and fixed by P8 (opt-in).

## Where S1.8 (text-initialised memory) would go

S1.8 is skipped by default (IMPLEMENTATION_PLAN.md S1.8): TGN here already adds the static node
features to the memory before attention (`models/MemoryModel.py` line 615,
`node_features = node_memories[...] + self.node_raw_features[...]`), so SPECTER2 vectors reach every
embedding. If the pilot argues for it, Patch P2 is a few lines:

- `models/MemoryModel.py` line 63 creates `MemoryBank(num_nodes, memory_dim)`; pass
  `node_raw_features` (or a projection to `memory_dim`) into it.
- `models/MemoryModel.py` lines 329-336, `MemoryBank.__init_memory_bank__`: replace
  `self.node_memories.data.zero_()` (line 334) with a copy of those vectors. This method runs at
  construction, at the start of every training epoch (`train_link_prediction.py` line 158), and in
  `citecurv.tgn.infer` before the replay, so one change covers all three.
- Variant "at first appearance": in `MemoryUpdater.update_memories` (line 439) and
  `get_updated_memories` (line 465), take nodes whose `node_last_updated_times` is still 0 and start
  them from the text vector instead of zero before the GRU step.
- Memory width equals node-feature width (`memory_dim = node_feat_dim`, line 53), so a 768-wide
  copy needs no shape change; the `memory_projection_dim` knob in `configs/default.yaml` would add a
  linear projection there.

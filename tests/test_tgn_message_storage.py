"""GPU memory on the full run, patch P9 (third_party/DyGLib/PATCHES.md): a stored message owns its storage.

Found 2026-11-23 in the full run's rehearsal on a T4: GPU memory grew by about 0.85 MB per training
step from the first step on, and the run stopped with CUDA out of memory at step 14,700 of 20,868.
Upstream DyGLib stores each node's raw message as a row of the batch's message tensor (a view), and
``detach_memory_bank`` replaces it with ``.detach()``, which is still a view: one kept message keeps
the whole batch tensor alive. A node's messages are cleared only when the node appears again, and
most cited works appear once, so every batch tensor stayed on the GPU. On the pilot (2,121 steps)
that was about 1.8 GB and went unnoticed.

Spec under test:
- After training steps (positive pass, backward, detach) and after an inference replay under
  ``torch.no_grad``, every stored message tensor's storage holds exactly that message
  (``untyped_storage().nbytes() == numel * element_size``), so dropping the batch frees it.
- The stored values are those of the batch's message rows: the training trace of
  ``test_tgn_memory_scope`` (lazy memory on and off agree) still holds with the patch.

Torch tests skip without torch (pytest.importorskip).
"""

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_tgn_memory_scope import batches, build  # noqa: E402  (same synthetic stream and model)


def _owns_storage(m):
    return m.untyped_storage().nbytes() == m.numel() * m.element_size()


def _stored(bank):
    return [m for v in bank.node_raw_messages.values() for m, _ in v]


@pytest.mark.parametrize("lazy", [False, True])
def test_stored_messages_own_their_storage_after_training_steps(lazy):
    model, ev = build(lazy)
    backbone, predictor = model[0], model[1]
    model.train()
    backbone.memory_bank.__init_memory_bank__()
    for b_src, b_dst, b_t, b_eid, b_neg in batches(ev):
        ps, pd = backbone.compute_src_dst_node_temporal_embeddings(b_src, b_dst, b_t, b_eid, edges_are_positive=True,
                                                                   num_neighbors=3)
        predictor(input_1=ps, input_2=pd).sum().backward()
        backbone.memory_bank.detach_memory_bank()
    stored = _stored(backbone.memory_bank)
    assert len(stored) > 20
    assert all(_owns_storage(m) for m in stored)


@pytest.mark.parametrize("lazy", [False, True])
def test_stored_messages_own_their_storage_in_the_inference_replay(lazy):
    model, ev = build(lazy)
    backbone = model[0]
    model.eval()
    backbone.memory_bank.__init_memory_bank__()
    with torch.no_grad():
        for b_src, b_dst, b_t, b_eid, b_neg in batches(ev, size=33):
            backbone.compute_src_dst_node_temporal_embeddings(b_src, b_dst, b_t, b_eid, edges_are_positive=True,
                                                              num_neighbors=3)
    stored = _stored(backbone.memory_bank)
    assert len(stored) > 20
    assert all(_owns_storage(m) for m in stored)


def test_stored_message_equals_its_row_of_the_batch():
    model, ev = build(lazy=True)
    backbone = model[0]
    model.eval()
    backbone.memory_bank.__init_memory_bank__()
    src, dst, ts, eid = ev
    b = slice(0, 20)
    with torch.no_grad():
        u, msgs = backbone.compute_new_node_raw_messages(src_node_ids=src[b], dst_node_ids=dst[b],
                                                         dst_node_embeddings=None, node_interact_times=ts[b],
                                                         edge_ids=eid[b])
        src_mem = backbone.memory_bank.get_memories(node_ids=src[b])
        dst_mem = backbone.memory_bank.get_memories(node_ids=dst[b])
        dt = torch.from_numpy(ts[b]).float() - backbone.memory_bank.node_last_updated_times[torch.from_numpy(src[b])]
        tf = backbone.time_encoder(dt.unsqueeze(dim=1)).reshape(len(src[b]), -1)
        full = torch.cat([src_mem, dst_mem, tf, backbone.edge_raw_features[torch.from_numpy(eid[b])]], dim=1)
    seen = {}
    for i, node in enumerate(src[b]):
        k = seen.get(int(node), 0)
        m, t = msgs[node][k]
        seen[int(node)] = k + 1
        assert _owns_storage(m)
        assert torch.equal(m, full[i]) and t == ts[b][i]
    assert set(int(x) for x in u) == set(int(x) for x in np.unique(src[b]))


# --- patch P10: a node keeps only its last pending message --------------------------------------
# With P9 the GPU memory still grew by about 0.55 MB per step on the full run (T4, 2026-11-23): a
# citing paper with 25 references gets 25 messages in its step and keeps all of them until it
# appears again, which for most papers is never. The only reader, MessageAggregator.aggregate_messages
# (also behind P8's LazyUpdatedMemories), takes the last message of a node and only asks whether the
# list is empty, so keeping the last message alone changes no value.

def _train_trace(model, ev):
    backbone, predictor = model[0], model[1]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    backbone.memory_bank.__init_memory_bank__()
    torch.manual_seed(3)
    trace = []
    for b_src, b_dst, b_t, b_eid, b_neg in batches(ev):
        ns, nd = backbone.compute_src_dst_node_temporal_embeddings(b_src, b_neg, b_t, None, edges_are_positive=False,
                                                                   num_neighbors=3)
        ps, pd = backbone.compute_src_dst_node_temporal_embeddings(b_src, b_dst, b_t, b_eid, edges_are_positive=True,
                                                                   num_neighbors=3)
        pos = predictor(input_1=ps, input_2=pd).squeeze(-1).sigmoid()
        neg = predictor(input_1=ns, input_2=nd).squeeze(-1).sigmoid()
        loss = torch.nn.BCELoss()(torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        backbone.memory_bank.detach_memory_bank()
        trace.append((loss.item(), torch.cat([ns, nd, ps, pd]).detach().clone()))
    return trace


@pytest.mark.parametrize("lazy", [False, True])
def test_a_node_keeps_at_most_one_pending_message(lazy):
    model, ev = build(lazy)
    _train_trace(model, ev)
    lengths = [len(v) for v in model[0].memory_bank.node_raw_messages.values()]
    assert max(lengths) == 1
    src, _, _, _ = ev
    assert np.bincount(src).max() > 1   # the stream has papers with several references in one step


@pytest.mark.parametrize("lazy", [False, True])
def test_keeping_the_last_message_changes_no_result(lazy, monkeypatch):
    model, ev = build(lazy)
    new = _train_trace(model, ev)
    MemoryBank = sys.modules["models.MemoryModel"].MemoryBank   # loaded by build()

    def keep_all(self, node_ids, new_node_raw_messages):   # upstream store_node_raw_messages (plus P8's tracking)
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])
        if self.track_undetached:
            self.undetached_node_ids.update(node_ids)

    monkeypatch.setattr(MemoryBank, "store_node_raw_messages", keep_all)
    model_ref, ev = build(lazy)
    ref = _train_trace(model_ref, ev)
    assert max(len(v) for v in model_ref[0].memory_bank.node_raw_messages.values()) > 1   # the reference kept them all
    assert len(new) == len(ref) > 5
    for (l_new, e_new), (l_ref, e_ref) in zip(new, ref):
        assert l_new == l_ref
        assert torch.equal(e_new, e_ref)
    for (k, p_new), (_, p_ref) in zip(model.state_dict().items(), model_ref.state_dict().items()):
        assert torch.equal(p_new, p_ref), k


# --- patch P11: the memory-bank backups live in CPU memory ---------------------------------------
# The first full training run (T4, 2026-11-23) stopped with CUDA out of memory after epoch 1's
# validation pass, in reload_memory_bank: DyGLib keeps two backups of the whole memory bank
# (memories and every pending message) around the validation passes, and upstream keeps them on the
# GPU, so the live bank and two copies (about 4.6 GB each for 1.8 million nodes) did not fit.

def _cuda_model(lazy=True, d=8):
    """build() from test_tgn_memory_scope, on the GPU (DyGLib keeps its feature tensors on the device
    it is built for, so the model is built there rather than moved)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU: the backups must leave GPU memory")
    from test_tgn_memory_scope import _dyglib, stream

    MemoryModel, MergeLayer, Data, get_neighbor_sampler = _dyglib()
    src, dst, ts, eid, n_nodes = stream()
    rng = np.random.default_rng(1)
    node_feat = rng.normal(size=(n_nodes, d)).astype(np.float32)
    node_feat[0] = 0
    edge_feat = np.ones((len(src) + 1, 1), dtype=np.float32)
    edge_feat[0] = 0
    edge_dst = np.zeros(len(src) + 1, dtype=np.longlong)
    edge_dst[eid] = dst
    data = Data(src_node_ids=src, dst_node_ids=dst, node_interact_times=ts, edge_ids=eid, labels=np.zeros(len(src)))
    sampler = get_neighbor_sampler(data, sample_neighbor_strategy="recent")
    torch.manual_seed(0)
    backbone = MemoryModel(node_raw_features=node_feat, edge_raw_features=edge_feat, neighbor_sampler=sampler,
                           time_feat_dim=4, model_name="TGN", num_layers=1, num_heads=2, dropout=0.1, device="cuda",
                           direction_feat_index=0, edge_dst_node_ids=edge_dst, lazy_memory=lazy)
    model = torch.nn.Sequential(backbone, MergeLayer(input_dim1=d, input_dim2=d, hidden_dim=d, output_dim=1)).to("cuda")
    model.eval()
    return model, (src, dst, ts, eid)


def _fill(backbone, ev, steps=4, first=0):
    """Replay ``steps`` batches of 20 events, starting at batch ``first`` (time only moves forward)."""
    src, dst, ts, eid = ev
    with torch.no_grad():
        for s in range(20 * first, 20 * (first + steps), 20):
            backbone.compute_src_dst_node_temporal_embeddings(src[s:s + 20], dst[s:s + 20], ts[s:s + 20], eid[s:s + 20],
                                                              edges_are_positive=True, num_neighbors=3)


def _bank_state(bank):
    msgs = {int(k): [(m.detach().cpu().clone(), float(t)) for m, t in v] for k, v in bank.node_raw_messages.items() if v}
    return bank.node_memories.detach().cpu().clone(), bank.node_last_updated_times.detach().cpu().clone(), msgs


def test_backup_is_held_in_cpu_memory():
    model, ev = _cuda_model()
    bank = model[0].memory_bank
    bank.__init_memory_bank__()
    _fill(model[0], ev)
    mem, times, msgs = bank.backup_memory_bank()
    assert mem.device.type == "cpu" and times.device.type == "cpu"
    stored = [m for v in msgs.values() for m, _ in v]
    assert len(stored) > 10 and all(m.device.type == "cpu" for m in stored)


@pytest.mark.parametrize("on_gpu", [False, True])
def test_reload_restores_the_bank_exactly_on_its_device(on_gpu):
    if on_gpu:
        model, ev = _cuda_model()
    else:
        model, ev = build(lazy=True)
    backbone = model[0]
    bank = backbone.memory_bank
    bank.__init_memory_bank__()
    _fill(backbone, ev)
    before = _bank_state(bank)
    backup = bank.backup_memory_bank()
    _fill(backbone, ev, steps=4, first=4)              # the bank moves on, as in a validation pass
    bank.reload_memory_bank(backup)
    after = _bank_state(bank)
    assert torch.equal(after[0], before[0]) and torch.equal(after[1], before[1])
    assert after[2].keys() == before[2].keys()
    for k in before[2]:
        assert [t for _, t in after[2][k]] == [t for _, t in before[2][k]]
        assert all(torch.equal(a, b) for (a, _), (b, _) in zip(after[2][k], before[2][k]))
    device = bank.node_memories.device
    live = [m for v in bank.node_raw_messages.values() for m, _ in v]
    assert all(m.device == device and _owns_storage(m) for m in live)
    assert bank.node_memories.device == device and bank.node_last_updated_times.device == device

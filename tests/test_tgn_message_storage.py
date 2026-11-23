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

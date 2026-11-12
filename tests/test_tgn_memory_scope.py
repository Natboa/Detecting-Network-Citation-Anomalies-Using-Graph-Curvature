"""Training speed, patch P8 (third_party/DyGLib/PATCHES.md): a step updates only the memories it reads.

Spec under test:
- With ``lazy_memory=True`` a call of ``compute_src_dst_node_temporal_embeddings`` runs the memory
  GRU only on nodes whose memory the step reads (the batch's nodes and their sampled neighbours),
  not on every node with a pending message. Upstream (``lazy_memory=False``) runs it on all of them,
  so its cost grows with the number of nodes seen so far.
- Same results: on a synthetic citation stream, several training steps (negative pass, positive
  pass, loss, backward, Adam step, detach) give the same embeddings, losses, weights, memories and
  pending messages with the flag on and off (float tolerance 1e-5; the GRU runs on a different
  number of rows, so the last bits of a matrix product may differ). Also in eval mode under
  ``torch.no_grad`` (the inference replay).
- ``detach_memory_bank`` with the flag on detaches only messages stored since the previous detach;
  afterwards no stored message carries a gradient graph, exactly as upstream.

Torch tests skip without torch (pytest.importorskip). DyGLib is put on sys.path directly, like
``citecurv.tgn.train.import_dyglib`` does, so the file also runs where that module cannot be
imported (it imports ``resource``, which Windows lacks).
"""

import sys

import numpy as np
import pytest

from citecurv.config import REPO_ROOT

torch = pytest.importorskip("torch")

DYGLIB = REPO_ROOT / "third_party" / "DyGLib"


def _dyglib():
    if str(DYGLIB) not in sys.path:
        sys.path.insert(0, str(DYGLIB))
    from models.MemoryModel import MemoryModel
    from models.modules import MergeLayer
    from utils.DataLoader import Data
    from utils.utils import get_neighbor_sampler

    return MemoryModel, MergeLayer, Data, get_neighbor_sampler


def stream(n_citing=40, refs=6, n_old=120, seed=0):
    """SYNTHETIC citation stream: citing papers 1..n_citing in time order, each citing ``refs``
    distinct works drawn from older works and from earlier citing papers (so neighbours exist)."""
    rng = np.random.default_rng(seed)
    src, dst, ts = [], [], []
    for j in range(1, n_citing + 1):
        pool = list(range(n_citing + 1, n_citing + 1 + n_old)) + list(range(1, j))
        for c in rng.choice(pool, size=min(refs, len(pool)), replace=False):
            src.append(j); dst.append(int(c)); ts.append(float(j // 3))  # three papers per day: same-time ties
    src = np.asarray(src, dtype=np.longlong)
    dst = np.asarray(dst, dtype=np.longlong)
    ts = np.asarray(ts, dtype=np.float64)
    eid = np.arange(1, len(src) + 1, dtype=np.longlong)
    return src, dst, ts, eid, n_citing + n_old + 1


def build(lazy, d=8, dropout=0.1, num_layers=1, seed=0):
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
    torch.manual_seed(seed)
    backbone = MemoryModel(node_raw_features=node_feat, edge_raw_features=edge_feat, neighbor_sampler=sampler,
                           time_feat_dim=4, model_name="TGN", num_layers=num_layers, num_heads=2, dropout=dropout,
                           direction_feat_index=0, edge_dst_node_ids=edge_dst, lazy_memory=lazy)
    model = torch.nn.Sequential(backbone, MergeLayer(input_dim1=d, input_dim2=d, hidden_dim=d, output_dim=1))
    return model, (src, dst, ts, eid)


def batches(ev, size=20):
    src, dst, ts, eid = ev
    neg = np.random.default_rng(7).choice(np.unique(dst), size=len(src))
    for s in range(0, len(src), size):
        sl = slice(s, s + size)
        yield src[sl], dst[sl], ts[sl], eid[sl], neg[sl]


def train_steps(lazy, num_layers=1):
    """DyGLib's TGN training step (train_link_prediction.py), one pass over the stream."""
    model, ev = build(lazy, num_layers=num_layers)
    backbone, predictor = model[0], model[1]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCELoss()
    model.train()
    backbone.memory_bank.__init_memory_bank__()
    torch.manual_seed(3)  # dropout masks: same draws in both runs
    trace = []
    for b_src, b_dst, b_t, b_eid, b_neg in batches(ev):
        ns, nd = backbone.compute_src_dst_node_temporal_embeddings(b_src, b_neg, b_t, None, edges_are_positive=False,
                                                                   num_neighbors=3)
        ps, pd = backbone.compute_src_dst_node_temporal_embeddings(b_src, b_dst, b_t, b_eid, edges_are_positive=True,
                                                                   num_neighbors=3)
        pos = predictor(input_1=ps, input_2=pd).squeeze(-1).sigmoid()
        neg = predictor(input_1=ns, input_2=nd).squeeze(-1).sigmoid()
        loss = loss_fn(torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        backbone.memory_bank.detach_memory_bank()
        trace.append((loss.item(), torch.cat([ns, nd, ps, pd]).detach().clone()))
    return model, trace


def test_dyglib_parser_takes_the_switch(monkeypatch):
    _dyglib()
    from utils.load_configs import get_link_prediction_args

    monkeypatch.setattr(sys, "argv", ["train_link_prediction.py", "--model_name", "TGN"])
    assert get_link_prediction_args().lazy_memory is False  # upstream default
    monkeypatch.setattr(sys, "argv", ["train_link_prediction.py", "--model_name", "TGN", "--lazy_memory"])
    assert get_link_prediction_args().lazy_memory is True


def test_citecurv_training_turns_the_switch_on():
    from citecurv.tgn.train import TGN_DEFAULTS, _cli_args

    assert TGN_DEFAULTS["lazy_memory"] is True and "--lazy_memory" in _cli_args(TGN_DEFAULTS)


def _messages(bank):
    return {int(k): [(m.detach().clone(), float(t)) for m, t in v] for k, v in bank.node_raw_messages.items() if v}


@pytest.mark.parametrize("num_layers", [1, 2])
def test_training_steps_match_upstream(num_layers):
    ref_model, ref = train_steps(lazy=False, num_layers=num_layers)
    new_model, new = train_steps(lazy=True, num_layers=num_layers)
    assert len(ref) == len(new) > 5
    for (l_ref, e_ref), (l_new, e_new) in zip(ref, new):
        assert l_new == pytest.approx(l_ref, abs=1e-5)
        torch.testing.assert_close(e_new, e_ref, atol=1e-5, rtol=1e-5)
    for (k, p_ref), (_, p_new) in zip(ref_model.state_dict().items(), new_model.state_dict().items()):
        torch.testing.assert_close(p_new, p_ref, atol=1e-5, rtol=1e-5, msg=k)
    m_ref, m_new = _messages(ref_model[0].memory_bank), _messages(new_model[0].memory_bank)
    assert m_ref.keys() == m_new.keys()
    for k in m_ref:
        assert [t for _, t in m_ref[k]] == [t for _, t in m_new[k]]
        for (a, _), (b, _) in zip(m_ref[k], m_new[k]):
            torch.testing.assert_close(b, a, atol=1e-5, rtol=1e-5)


def test_inference_replay_matches_upstream():
    outs = []
    for lazy in (False, True):
        model, ev = build(lazy)
        backbone = model[0]
        model.eval()
        backbone.memory_bank.__init_memory_bank__()
        embs = []
        with torch.no_grad():
            for b_src, b_dst, b_t, b_eid, b_neg in batches(ev, size=33):
                embs += backbone.compute_src_dst_node_temporal_embeddings(b_src, b_neg, b_t, None,
                                                                          edges_are_positive=False, num_neighbors=3)
                embs += backbone.compute_src_dst_node_temporal_embeddings(b_src, b_dst, b_t, b_eid,
                                                                          edges_are_positive=True, num_neighbors=3)
        outs.append(torch.cat(embs))
    torch.testing.assert_close(outs[1], outs[0], atol=1e-5, rtol=1e-5)


def _gru_rows(backbone):
    rows = []
    backbone.memory_updater.memory_updater.register_forward_hook(lambda mod, inp, out: rows.append(inp[0].shape[0]))
    return rows


def test_step_runs_gru_only_on_nodes_it_reads():
    model, ev = build(lazy=True)
    backbone = model[0]
    model.eval()
    backbone.memory_bank.__init_memory_bank__()
    src, dst, ts, eid = ev
    cut = len(src) - 6
    with torch.no_grad():
        for s in range(0, cut, 20):  # fill the memory: many nodes get a pending message
            e = min(s + 20, cut)
            backbone.compute_src_dst_node_temporal_embeddings(src[s:e], dst[s:e], ts[s:e], eid[s:e],
                                                              edges_are_positive=True, num_neighbors=3)
        pending = sum(1 for v in backbone.memory_bank.node_raw_messages.values() if v)
        rows = _gru_rows(backbone)
        b = slice(cut, cut + 2)  # a step of two events
        backbone.compute_src_dst_node_temporal_embeddings(src[b], dst[b], ts[b], None, edges_are_positive=False,
                                                          num_neighbors=3)
    read = {int(x) for x in np.concatenate([src[b], dst[b]])}
    neigh, _, _ = backbone.embedding_module.neighbor_sampler.get_historical_neighbors(
        np.concatenate([src[b], dst[b]]), np.concatenate([ts[b], ts[b]]), num_neighbors=3)
    read |= {int(x) for x in neigh.ravel()}
    assert pending > 4 * len(read)  # the case is only telling if most pending nodes are not read
    assert sum(rows) <= len(read)


def test_step_cost_upstream_grows_with_pending_nodes():
    """Control: the same case with the flag off runs the GRU on every pending node (why P8 exists)."""
    model, ev = build(lazy=False)
    backbone = model[0]
    model.eval()
    backbone.memory_bank.__init_memory_bank__()
    src, dst, ts, eid = ev
    cut = len(src) - 6
    with torch.no_grad():
        for s in range(0, cut, 20):
            e = min(s + 20, cut)
            backbone.compute_src_dst_node_temporal_embeddings(src[s:e], dst[s:e], ts[s:e], eid[s:e],
                                                              edges_are_positive=True, num_neighbors=3)
        pending = sum(1 for v in backbone.memory_bank.node_raw_messages.values() if v)
        rows = _gru_rows(backbone)
        b = slice(cut, cut + 2)
        backbone.compute_src_dst_node_temporal_embeddings(src[b], dst[b], ts[b], None, edges_are_positive=False,
                                                          num_neighbors=3)
    assert sum(rows) == pending


def test_detach_leaves_no_gradient_graph_and_touches_only_new_messages():
    model, ev = build(lazy=True)
    backbone, predictor = model[0], model[1]
    model.train()
    backbone.memory_bank.__init_memory_bank__()
    seen_ids = []
    for b_src, b_dst, b_t, b_eid, b_neg in batches(ev):
        ps, pd = backbone.compute_src_dst_node_temporal_embeddings(b_src, b_dst, b_t, b_eid, edges_are_positive=True,
                                                                   num_neighbors=3)
        predictor(input_1=ps, input_2=pd).sum().backward()
        before = {int(k): [id(m) for m, _ in v] for k, v in backbone.memory_bank.node_raw_messages.items()}
        backbone.memory_bank.detach_memory_bank()
        after = backbone.memory_bank.node_raw_messages
        touched = {int(k) for k, v in after.items() if [id(m) for m, _ in v] != before.get(int(k))}
        seen_ids.append((touched, {int(x) for x in np.concatenate([b_src, b_dst])}))
        assert all(not m.requires_grad and m.grad_fn is None for v in after.values() for m, _ in v)
        assert not backbone.memory_bank.node_memories.requires_grad
    for touched, batch_nodes in seen_ids:
        assert touched <= batch_nodes

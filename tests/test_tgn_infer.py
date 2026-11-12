"""S1.7: TGN inference outputs, the memory leak hook, and vendored patch P5.

Spec under test:
- batch_boundaries: consecutive slices of the canonical event order; with align_to_timestamps a
  boundary never splits a timestamp group unless the group exceeds max_batch_size (hand cases +
  randomized property check). Pure numpy, runs in the main venv.
- P5 (third_party/DyGLib/PATCHES.md): with direction_feat_index=0 the citing endpoint's memory
  message carries +1 and the cited endpoint's -1, and attention keys whose centre node is the cited
  endpoint read -1; with the flag off both endpoints see the same +1 (direction erased).
- install_memory_leak_check: raises when a batch starts before a pending memory message; silent
  for in-order batches; remove() restores the method.
- train_tgn -> infer_tgn end to end on a tiny synthetic stream (subprocess training): contract
  columns, one row per event, probabilities in (0, 1), embedding shapes; same seed -> same scores
  within 1e-6 (NFR4); scores of events up to day T do not change when every later event is
  dropped from the input (no future information, NFR3).
Torch tests skip in the main venv (pytest.importorskip); run them with the torch venv.
"""

import json

import numpy as np
import pandas as pd
import pytest

from citecurv.tgn.convert import to_jodie
from citecurv.tgn.infer import batch_boundaries, paired_share


# --------------------------------------------------------------------------- pure helpers

def test_paired_share_counts_ties_as_half():
    # wins: event 0; ties: events 1 and 3; loss: event 2
    out = paired_share([0.9, 0.5, 0.5, 0.2], [0.1, 0.5, 0.6, 0.2])
    assert out == {"n": 4, "strict": 0.25, "ties": 0.5, "ties_as_half": 0.5, "without_ties": 0.5}


def test_paired_share_all_ties_has_no_tie_free_share():
    out = paired_share([0.3, 0.3], [0.3, 0.3])
    assert out["ties"] == 1.0 and out["ties_as_half"] == 0.5 and out["without_ties"] is None


def test_paired_share_rejects_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        paired_share([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        paired_share([], [])


def test_batch_boundaries_fixed_size():
    assert batch_boundaries(np.arange(7), 3, align_to_timestamps=False) == [(0, 3), (3, 6), (6, 7)]


TS = np.array([1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5])


def test_batch_boundaries_aligned_hand_case():
    assert batch_boundaries(TS, 3, max_batch_size=4) == [(0, 2), (2, 5), (5, 6), (6, 10), (10, 11)]


def test_batch_boundaries_splits_oversized_group():
    assert batch_boundaries(TS, 3, max_batch_size=3) == [(0, 2), (2, 5), (5, 6), (6, 9), (9, 10), (10, 11)]


def test_batch_boundaries_property():
    rng = np.random.default_rng(1)
    for _ in range(300):
        ts = np.sort(rng.integers(0, int(rng.integers(1, 40)), size=int(rng.integers(1, 300))))
        bs = int(rng.integers(1, 30))
        cap = bs * int(rng.integers(1, 4))
        b = batch_boundaries(ts, bs, max_batch_size=cap)
        assert b[0][0] == 0 and b[-1][1] == len(ts)
        assert all(e0 == s1 for (_, e0), (s1, _) in zip(b, b[1:]))
        group_size = {v: c for v, c in zip(*np.unique(ts, return_counts=True))}
        for s, e in b:
            assert 0 < e - s <= cap
            if s > 0 and ts[s - 1] == ts[s]:  # a split group: only allowed when it exceeds the cap
                assert group_size[ts[s]] > cap


# --------------------------------------------------------------------------- tiny synthetic stream

def tiny_stream(n_papers=60, seed=0, dim=8):
    rng = np.random.default_rng(seed)
    pub = np.sort(rng.integers(0, 120, size=n_papers)) + 18000
    vec = rng.normal(size=(n_papers, dim)).astype(np.float32)
    ids = np.array([f"P{j:03d}" for j in range(n_papers)])
    rows = []
    for j in range(n_papers):
        prior = int(np.searchsorted(pub, pub[j], side="left"))
        if prior == 0:
            continue
        for c in rng.choice(prior, size=min(prior, 1 + int(rng.poisson(3))), replace=False):
            rows.append((ids[j], ids[c], int(pub[j])))
    ev = pd.DataFrame(rows, columns=["src_id", "dst_id", "t"])
    ev.insert(0, "event_id", (rng.permutation(len(ev)) * 7 + 5).astype(np.int64))
    return ev, (ids, vec)


@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def trained(torch_mod, tmp_path_factory):
    from citecurv.tgn.train import train_tgn

    base = tmp_path_factory.mktemp("tgn")
    ev, emb = tiny_stream()
    to_jodie(ev, emb, base / "data", "tiny")
    metrics = train_tgn(base / "data", "tiny", epochs=1, seed=0, node_feat_dim=8, out_dir=base / "run",
                        leak_check=True, num_threads=1, batch_size=50, learning_rate=1e-3)
    return base, ev, metrics


# --------------------------------------------------------------------------- patch P5 and the leak hook

def build_model(torch_mod, direction_feat_index, events, d=4):
    """MemoryModel on hand-made events [(u, i, t)] with node/edge ids from 1; edge features all +1."""
    from citecurv.tgn.train import import_dyglib

    dg = import_dyglib()
    u = np.array([e[0] for e in events], dtype=np.longlong)
    i = np.array([e[1] for e in events], dtype=np.longlong)
    t = np.array([e[2] for e in events], dtype=np.float64)
    eid = np.arange(1, len(events) + 1, dtype=np.longlong)
    n_nodes = int(max(u.max(), i.max())) + 1
    node_feat = np.random.default_rng(0).normal(size=(n_nodes, d)).astype(np.float32)
    node_feat[0] = 0
    edge_feat = np.ones((len(events) + 1, 1), dtype=np.float32)
    edge_feat[0] = 0
    data = dg.Data(src_node_ids=u, dst_node_ids=i, node_interact_times=t, edge_ids=eid, labels=np.zeros(len(u)))
    sampler = dg.get_neighbor_sampler(data, sample_neighbor_strategy="recent")
    edge_dst = np.zeros(len(events) + 1, dtype=np.longlong)
    edge_dst[eid] = i
    torch_mod.manual_seed(0)
    model = dg.MemoryModel(node_raw_features=node_feat, edge_raw_features=edge_feat, neighbor_sampler=sampler,
                           time_feat_dim=4, model_name="TGN", num_layers=1, num_heads=2, dropout=0.0,
                           direction_feat_index=direction_feat_index, edge_dst_node_ids=edge_dst)
    model.eval()
    return dg, model, (u, i, t, eid)


@pytest.mark.parametrize("flag, cited_sees", [(0, -1.0), (-1, 1.0)])
def test_p5_memory_messages_keep_direction(torch_mod, flag, cited_sees):
    _, model, (u, i, t, eid) = build_model(torch_mod, flag, [(1, 2, 10.0)])
    with torch_mod.no_grad():
        model.compute_src_dst_node_temporal_embeddings(u, i, t, eid, edges_are_positive=True, num_neighbors=3)
    msgs = model.memory_bank.node_raw_messages
    assert float(msgs[1][-1][0][-1]) == 1.0  # citing endpoint (u = 1): last message column = direction
    assert float(msgs[2][-1][0][-1]) == cited_sees  # cited endpoint (i = 2)


@pytest.mark.parametrize("flag, expected", [(0, [1.0, -1.0]), (-1, [1.0, 1.0])])
def test_p5_attention_keys_keep_direction(torch_mod, flag, expected):
    # history: 1 cites 2 at t=10, 3 cites 1 at t=20; then embed node 1 at t=30:
    # its neighbours are 2 (node 1 was the citer, key +1) and 3 (node 1 was cited, key -1)
    dg, model, (u, i, t, eid) = build_model(torch_mod, flag, [(1, 2, 10.0), (3, 1, 20.0)])
    seen = {}
    layer = model.embedding_module.temporal_conv_layers[0]
    original = layer.forward

    def spy(**kw):
        seen["edge"] = kw["neighbor_node_edge_features"].detach().clone()
        seen["mask"] = kw["neighbor_masks"].copy()
        return original(**kw)

    layer.forward = spy
    with torch_mod.no_grad():
        model.embedding_module.compute_node_temporal_embeddings(
            node_memories=model.memory_bank.node_memories, node_ids=np.array([1]),
            node_interact_times=np.array([30.0]), current_layer_num=1, num_neighbors=3)
    real = seen["mask"][0] > 0  # padded neighbour slots have node id 0
    assert seen["edge"][0, real, 0].tolist() == expected


def test_leak_hook(torch_mod):
    from citecurv.tgn.train import MemoryLeakError, install_memory_leak_check

    dg, model, (u, i, t, eid) = build_model(torch_mod, 0, [(1, 2, 10.0), (3, 1, 20.0), (2, 3, 30.0)])
    original = dg.MemoryModel.compute_src_dst_node_temporal_embeddings
    hook = install_memory_leak_check(dg.MemoryModel)
    try:
        with torch_mod.no_grad():
            model.compute_src_dst_node_temporal_embeddings(u[:2], i[:2], t[:2], eid[:2], True, 3)  # t = 10, 20
            with pytest.raises(MemoryLeakError):  # a batch at t = 15 after messages at t = 20
                model.compute_src_dst_node_temporal_embeddings(u[2:], i[2:], np.array([15.0]), eid[2:], True, 3)
            assert hook.violations == 1
            model.compute_src_dst_node_temporal_embeddings(u[2:], i[2:], t[2:], eid[2:], True, 3)  # t = 30: fine
        assert hook.n_positive_batches == 2 and hook.max_margin > 0
    finally:
        hook.remove()
    assert dg.MemoryModel.compute_src_dst_node_temporal_embeddings is original


def _chain_events(n):
    """n events (k, k + 1) at t = k: every node ends with a pending message."""
    return [(k, k + 1, float(k)) for k in range(1, n + 1)]


@pytest.mark.parametrize("lazy", [False, True])
def test_leak_hook_work_per_batch_does_not_grow_with_pending_messages(torch_mod, lazy):
    """The hook reads the messages of the batch's own nodes, plus one full scan per new message dictionary."""
    from citecurv.tgn.train import install_memory_leak_check

    dg, model, (u, i, t, eid) = build_model(torch_mod, 0, _chain_events(300))
    model.lazy_memory = model.memory_bank.track_undetached = lazy
    hook = install_memory_leak_check(dg.MemoryModel)
    try:
        with torch_mod.no_grad():
            for s in range(0, 296, 4):
                model.compute_src_dst_node_temporal_embeddings(u[s:s + 4], i[s:s + 4], t[s:s + 4], eid[s:s + 4], True, 3)
            before = hook.n_pending_checked
            model.compute_src_dst_node_temporal_embeddings(u[296:], i[296:], t[296:], eid[296:], True, 3)
            per_batch = hook.n_pending_checked - before
    finally:
        hook.remove()
    pending = sum(len(v) for v in model.memory_bank.node_raw_messages.values())
    assert pending > 250 and per_batch <= 4 * 4  # at most a few messages per node of a 4-event batch
    assert hook.violations == 0 and hook.n_full_scans >= 1


def test_leak_hook_same_verdict_as_full_scan_across_reload(torch_mod):
    """Backup, reload (a new message dictionary, as DyGLib does for new-node validation) and a planted
    future message in a reloaded dictionary: the hook rescans and gives the full-scan verdict and margin."""
    from citecurv.tgn.train import MemoryLeakError, install_memory_leak_check

    dg, model, (u, i, t, eid) = build_model(torch_mod, 0, _chain_events(40))
    hook = install_memory_leak_check(dg.MemoryModel)
    bank = model.memory_bank
    try:
        with torch_mod.no_grad():
            for s in range(0, 20, 4):  # t = 1..20
                model.compute_src_dst_node_temporal_embeddings(u[s:s + 4], i[s:s + 4], t[s:s + 4], eid[s:s + 4], True, 3)
            backup = bank.backup_memory_bank()
            for s in range(20, 32, 4):  # t = 21..32
                model.compute_src_dst_node_temporal_embeddings(u[s:s + 4], i[s:s + 4], t[s:s + 4], eid[s:s + 4], True, 3)
            assert hook.max_margin == -1.0  # every batch starts one day after the latest pending message
            bank.reload_memory_bank(backup)
            # back to t = 21 with the t <= 20 state: legal, although messages at t = 32 were stored before
            model.compute_src_dst_node_temporal_embeddings(u[20:24], i[20:24], t[20:24], eid[20:24], False, 3)
            assert hook.violations == 0 and hook.max_margin == -1.0
            planted = bank.backup_memory_bank()
            node = next(k for k, v in planted[2].items() if v)
            planted[2][node][0] = (planted[2][node][0][0], np.float64(99.0))
            bank.reload_memory_bank(planted)
            with pytest.raises(MemoryLeakError):
                model.compute_src_dst_node_temporal_embeddings(u[20:24], i[20:24], t[20:24], eid[20:24], False, 3)
            assert hook.violations == 1
    finally:
        hook.remove()


# --------------------------------------------------------------------------- end to end

def test_train_outputs(trained):
    base, _, metrics = trained
    run = base / "run"
    for f in ("metrics.json", "resolved_args.json", "checkpoint/TGN_seed0.pkl"):
        assert (run / f).is_file()
    assert metrics["leak_check"]["enabled"] and metrics["leak_check"]["violations"] == 0
    assert metrics["leak_check"]["n_positive_batches"] > 0
    assert len(metrics["epochs"]) == 1 and np.isfinite(metrics["epochs"][0]["train_loss"])
    resolved = json.loads((run / "resolved_args.json").read_text())
    assert resolved["dyglib"]["seed"] == 0 and resolved["dyglib"]["direction_feat_index"] == 0
    assert resolved["dyglib"]["lazy_memory"] is True  # patch P8 on by default (TGN_DEFAULTS)


def test_patch_p8_changes_no_result_end_to_end(trained, tmp_path):
    """Training and scoring with patch P8 off (upstream memory update) give the same numbers."""
    from citecurv.tgn.infer import infer_tgn
    from citecurv.tgn.train import train_tgn

    base, _, metrics = trained
    off = train_tgn(base / "data", "tiny", epochs=1, seed=0, node_feat_dim=8, out_dir=tmp_path / "run_off",
                    leak_check=True, num_threads=1, batch_size=50, learning_rate=1e-3, lazy_memory=False)
    resolved = json.loads((tmp_path / "run_off" / "resolved_args.json").read_text())
    assert resolved["dyglib"]["lazy_memory"] is False
    for key in ("train_loss", "val_ap", "val_auc"):
        assert off["epochs"][0][key] == pytest.approx(metrics["epochs"][0][key], abs=2e-4)  # DyGLib logs 4 decimals
    assert off["test"]["ap"] == pytest.approx(metrics["test"]["ap"], abs=2e-4)
    on_scores, _ = infer_tgn(base / "run", base / "data", "tiny", out_dir=tmp_path / "on", batch_size=40)
    off_scores, _ = infer_tgn(tmp_path / "run_off", base / "data", "tiny", out_dir=tmp_path / "off", batch_size=40)
    np.testing.assert_allclose(on_scores.p_link, off_scores.p_link, atol=1e-5)
    assert json.loads((tmp_path / "on" / "tgn_infer_meta.json").read_text())["lazy_memory"] is True
    assert json.loads((tmp_path / "off" / "tgn_infer_meta.json").read_text())["lazy_memory"] is False


def test_infer_contract(trained, tmp_path):
    from citecurv.tgn.infer import infer_tgn

    base, ev, _ = trained
    scores, emb_path = infer_tgn(base / "run", base / "data", "tiny", out_dir=tmp_path, batch_size=40, leak_check=True)
    assert list(scores.columns) == ["event_id", "p_link", "neg_p_link"]
    assert len(scores) == len(ev) and set(scores.event_id) == set(ev.event_id)
    canon = ev.sort_values(["t", "event_id"]).event_id.tolist()
    assert scores.event_id.tolist() == canon
    for col in ("p_link", "neg_p_link"):
        assert ((scores[col] > 0) & (scores[col] < 1)).all()
    with np.load(emb_path) as z:
        assert z["event_id"].tolist() == canon
        assert z["z_src"].shape == (len(ev), 8) and z["z_dst"].shape == (len(ev), 8)
    meta = json.loads((tmp_path / "tgn_infer_meta.json").read_text())
    assert meta["leak_check"]["violations"] == 0


def test_infer_deterministic_across_training_runs(trained, tmp_path):
    from citecurv.tgn.infer import infer_tgn
    from citecurv.tgn.train import train_tgn

    base, _, _ = trained
    train_tgn(base / "data", "tiny", epochs=1, seed=0, node_feat_dim=8, out_dir=tmp_path / "run2",
              num_threads=1, batch_size=50, learning_rate=1e-3)
    a, ea = infer_tgn(base / "run", base / "data", "tiny", out_dir=tmp_path / "a", batch_size=40)
    b, eb = infer_tgn(tmp_path / "run2", base / "data", "tiny", out_dir=tmp_path / "b", batch_size=40)
    np.testing.assert_allclose(a.p_link, b.p_link, atol=1e-6)
    np.testing.assert_allclose(a.neg_p_link, b.neg_p_link, atol=1e-6)
    with np.load(ea) as za, np.load(eb) as zb:
        np.testing.assert_allclose(za["z_src"], zb["z_src"], atol=1e-6)


def test_infer_ignores_future_events(trained, tmp_path):
    from citecurv.tgn.infer import infer_tgn

    base, ev, _ = trained
    cut = int(np.quantile(ev.t, 0.6))
    prefix = ev[ev.t <= cut]
    _, emb = tiny_stream()
    to_jodie(prefix, emb, tmp_path / "prefix", "tiny")
    full, ef = infer_tgn(base / "run", base / "data", "tiny", out_dir=tmp_path / "f", batch_size=40)
    part, ep = infer_tgn(base / "run", tmp_path / "prefix", "tiny", out_dir=tmp_path / "p", batch_size=40)
    joined = part.merge(full, on="event_id", suffixes=("_p", "_f"))
    assert len(joined) == len(prefix)
    np.testing.assert_allclose(joined.p_link_p, joined.p_link_f, atol=1e-5)
    with np.load(ef) as zf, np.load(ep) as zp:
        k = len(prefix)
        assert zf["event_id"][:k].tolist() == zp["event_id"].tolist()
        np.testing.assert_allclose(zf["z_dst"][:k], zp["z_dst"], atol=1e-5)

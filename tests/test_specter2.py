"""S1.6: SPECTER2 encoding with injected fake tokenizer/model (no network, no GPU).

Spec under test (IMPLEMENTATION_PLAN.md S1.6, FR3, model card usage):
- text = title + tokenizer.sep_token + abstract when an abstract exists, title alone otherwise;
- tokenizer called with truncation at max_length=512 and padding;
- model in eval mode, gradients off; CLS pooling (last_hidden_state[:, 0, :]);
- returns float32 (N, d) in input order, independent of batch size, deterministic;
- load_specter2 loads allenai/specter2_base and the allenai/specter2 proximity adapter;
- write_embeddings_npz / load_embeddings_npz follow the embeddings.npz contract;
- encode_resumable reuses finished chunks (Colab checkpoint-resume).
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from citecurv.embed import specter2
from citecurv.embed.specter2 import (
    encode,
    encode_resumable,
    load_embeddings_npz,
    load_specter2,
    paper_text,
    write_embeddings_npz,
)

DIM = 768


class FakeTokenizer:
    sep_token = "[SEP]"

    def __init__(self):
        self.calls = []

    def __call__(self, texts, padding=False, truncation=False, max_length=None,
                 return_tensors=None, return_token_type_ids=None):
        texts = list(texts)
        self.calls.append({"texts": texts, "padding": padding, "truncation": truncation,
                           "max_length": max_length, "return_tensors": return_tensors})
        seqs = []
        for text in texts:
            words = text.replace(self.sep_token, f" {self.sep_token} ").split()
            toks = [1] + [2 + sum(map(ord, w)) % 997 for w in words]
            seqs.append(toks[:max_length] if truncation else toks)
        width = max(len(s) for s in seqs)
        ids = np.zeros((len(seqs), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
            mask[i, :len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeModel:
    """Position 0 = masked mean of token vectors (depends on the whole text,
    not on padding); other positions = raw token vectors."""

    def __init__(self, dim=DIM, seed=0):
        self.table = np.random.default_rng(seed).normal(size=(1000, dim))
        self.training = True
        self.batch_sizes = []

    def eval(self):
        self.training = False
        return self

    def __call__(self, input_ids, attention_mask, **kwargs):
        self.batch_sizes.append(len(input_ids))
        emb = self.table[input_ids]
        m = attention_mask[..., None]
        hidden = emb.copy()
        hidden[:, 0, :] = (emb * m).sum(axis=1) / m.sum(axis=1)
        return SimpleNamespace(last_hidden_state=hidden)


PAPERS = [
    ("W1", "Graph curvature", "Ricci curvature on directed graphs."),
    ("W2", "Citation cartels", None),
    ("W3", "Temporal networks", "Memory-based temporal graph networks for link prediction."),
    ("W4", "Short", ""),
    ("W5", "Another title", "   "),
    ("W6", "Bursts", "Citation bursts and their detection in time series of events."),
    ("W7", "Seventh", float("nan")),
]


def _encode(papers=PAPERS, **kw):
    tok, model = kw.pop("tokenizer", FakeTokenizer()), kw.pop("model", FakeModel())
    return encode(papers, tokenizer=tok, model=model, **kw), tok, model


def test_paper_text_sep_rule():
    assert paper_text("T", "A b", "[SEP]") == "T[SEP]A b"
    assert paper_text("T", None, "[SEP]") == "T"
    assert paper_text("T", "", "[SEP]") == "T"
    assert paper_text("T", "  ", "[SEP]") == "T"
    assert paper_text("T", float("nan"), "[SEP]") == "T"
    assert paper_text(None, "A", "</s>") == "</s>A"


def test_encode_sends_title_only_when_abstract_missing():
    _, tok, _ = _encode(batch_size=100)
    sent = set(tok.calls[0]["texts"])
    assert "Graph curvature[SEP]Ricci curvature on directed graphs." in sent
    assert {"Citation cartels", "Short", "Another title", "Seventh"} <= sent
    assert not any(t.endswith("[SEP]") for t in sent)


def test_shape_dtype_and_cls_pooling():
    vec, _, model = _encode()
    assert vec.shape == (len(PAPERS), DIM) and vec.dtype == np.float32
    tok = FakeTokenizer()
    one = tok([paper_text("Citation cartels", None, "[SEP]")], truncation=True, max_length=512)
    expected = model(**one).last_hidden_state[0, 0, :]
    np.testing.assert_allclose(vec[1], expected.astype(np.float32), rtol=0, atol=1e-6)


def test_tokenizer_truncates_at_512_and_pads():
    long_abstract = " ".join(f"w{i}" for i in range(2000))
    _, tok, _ = _encode([("L", "Long", long_abstract)])
    call = tok.calls[0]
    assert call["truncation"] is True and call["max_length"] == 512 and call["padding"]


def test_eval_mode_and_batching():
    vec1, _, model = _encode(batch_size=3)
    assert model.training is False
    assert model.batch_sizes == [3, 3, 1]
    vec2, _, _ = _encode(batch_size=1)
    vec3, _, _ = _encode(batch_size=64)
    np.testing.assert_allclose(vec1, vec2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(vec1, vec3, rtol=0, atol=1e-6)


def test_output_rows_follow_input_order():
    vec, _, _ = _encode(batch_size=2)
    for i, paper in enumerate(PAPERS):
        single, _, _ = _encode([paper])
        np.testing.assert_allclose(vec[i], single[0], rtol=0, atol=1e-6)


def test_deterministic_across_calls():
    a, _, _ = _encode()
    b, _, _ = _encode()
    np.testing.assert_array_equal(a, b)


def test_accepts_dataframe_and_dicts():
    df = pd.DataFrame(PAPERS, columns=["work_id", "title", "abstract"])
    dicts = [{"id": i, "title": t, "abstract": a} for i, t, a in PAPERS]
    ref, _, _ = _encode()
    np.testing.assert_array_equal(_encode(df)[0], ref)
    np.testing.assert_array_equal(_encode(dicts)[0], ref)


def test_empty_input():
    vec, _, _ = _encode([])
    assert vec.shape == (0, DIM) and vec.dtype == np.float32


def test_write_and_load_embeddings_npz(tmp_path):
    ids = ["W1", "W2", "W3"]
    vecs = np.random.default_rng(0).normal(size=(3, 8))
    path = tmp_path / "embeddings.npz"
    write_embeddings_npz(path, ids, vecs)
    with np.load(path) as z:  # no allow_pickle needed: plain unicode + float32
        assert z["ids"].dtype.kind == "U" and z["vectors"].dtype == np.float32
        assert list(z["ids"]) == ids and z["vectors"].shape == (3, 8)
    got_ids, got_vecs = load_embeddings_npz(path)
    np.testing.assert_allclose(got_vecs, vecs.astype(np.float32))
    with pytest.raises(ValueError):
        write_embeddings_npz(tmp_path / "x.npz", ids[:2], vecs)
    with pytest.raises(ValueError):
        write_embeddings_npz(tmp_path / "x.npz", ["a", "a", "b"], vecs)
    with pytest.raises(ValueError):
        write_embeddings_npz(tmp_path / "x.npz", ids, vecs[:, 0])


def test_encode_resumable_reuses_finished_chunks(tmp_path):
    tok, model = FakeTokenizer(), FakeModel()
    vec, n_new = encode_resumable(PAPERS, tmp_path, chunk_size=3, tokenizer=tok, model=model)
    assert n_new == len(PAPERS)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "part_00000.npz", "part_00001.npz", "part_00002.npz"]
    np.testing.assert_allclose(vec, _encode()[0], rtol=0, atol=1e-6)

    class Boom(FakeModel):
        def __call__(self, *a, **k):
            raise AssertionError("must not re-encode finished chunks")

    again, n_new = encode_resumable(PAPERS, tmp_path, chunk_size=3, tokenizer=tok, model=Boom())
    assert n_new == 0
    np.testing.assert_array_equal(again, vec)

    (tmp_path / "part_00001.npz").unlink()
    model2 = FakeModel()
    redo, n_new = encode_resumable(PAPERS, tmp_path, chunk_size=3, tokenizer=tok, model=model2)
    assert n_new == 3 and model2.batch_sizes == [3]
    np.testing.assert_array_equal(redo, vec)

    changed = [("X1", "t", None)] + PAPERS[1:]
    with pytest.raises(ValueError, match="different ids"):
        encode_resumable(changed, tmp_path, chunk_size=3, tokenizer=tok, model=model)


def test_load_specter2_uses_base_model_plus_proximity_adapter(monkeypatch):
    log = []

    class Tok:
        @staticmethod
        def from_pretrained(name):
            log.append(("tokenizer", name))
            return FakeTokenizer()

    class Model(FakeModel):
        @classmethod
        def from_pretrained(cls, name):
            log.append(("model", name))
            return cls()

        def load_adapter(self, name, **kw):
            log.append(("adapter", name, tuple(sorted(kw.items()))))

        def to(self, device):
            log.append(("to", device))
            return self

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tok))
    monkeypatch.setitem(sys.modules, "adapters", types.SimpleNamespace(AutoAdapterModel=Model))
    tok, model = load_specter2("cpu")
    assert ("tokenizer", "allenai/specter2_base") in log
    assert ("model", "allenai/specter2_base") in log
    assert ("adapter", "allenai/specter2",
            (("load_as", "specter2"), ("set_active", True), ("source", "hf"))) in log
    assert ("to", "cpu") in log and model.training is False


def test_load_specter2_reports_missing_stack(monkeypatch):
    monkeypatch.setitem(sys.modules, "adapters", None)  # import adapters -> ImportError
    with pytest.raises(ImportError, match="adapters"):
        load_specter2("cpu")
    with pytest.raises(ImportError):
        encode(PAPERS)  # no model injected -> lazy load -> same error


def test_torch_model_runs_without_grad_in_eval_mode():
    torch = pytest.importorskip("torch")

    class TorchTok(FakeTokenizer):
        def __call__(self, *a, **k):
            out = super().__call__(*a, **k)
            return {key: torch.as_tensor(v) for key, v in out.items()}

    class TorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(1000, 16)
            self.drop = torch.nn.Dropout(0.5)
            self.grad_seen = []

        def forward(self, input_ids, attention_mask, **kw):
            self.grad_seen.append(torch.is_grad_enabled())
            return SimpleNamespace(last_hidden_state=self.drop(self.emb(input_ids)))

    model = TorchModel()
    a = encode(PAPERS, tokenizer=TorchTok(), model=model, device="cpu", batch_size=2)
    b = encode(PAPERS, tokenizer=TorchTok(), model=model, device="cpu", batch_size=5)
    assert a.dtype == np.float32 and a.shape == (len(PAPERS), 16)
    assert not model.training and model.grad_seen and not any(model.grad_seen)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)  # dropout off -> deterministic


def test_module_constants():
    assert specter2.BASE_MODEL == "allenai/specter2_base"
    assert specter2.PROXIMITY_ADAPTER == "allenai/specter2"


def test_embed_notebook_is_valid_and_parses():
    nbformat = pytest.importorskip("nbformat")
    tm = pytest.importorskip("IPython.core.inputtransformer2")
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "notebooks" / "20_embed.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert code and all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    for src in code:
        ast.parse(tm.TransformerManager().transform_cell(src))
    joined = "\n".join(code)
    assert '"adapters==1.3.0" "transformers==4.57.6"' in joined  # exact embed pins
    assert "encode_resumable" in joined and "write_embeddings_npz" in joined
    assert "docs_per_min" in joined and "drive.mount" in joined

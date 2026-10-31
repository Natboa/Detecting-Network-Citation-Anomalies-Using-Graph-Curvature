"""SPECTER2 paper embeddings (task S1.6, FR3, decision L6).

Model: base model `allenai/specter2_base` with the `allenai/specter2` proximity
adapter loaded through the `adapters` package (naming trap: `allenai/specter2`
is the adapter, not a standalone model). Usage follows the SPECTER2 model card:
text = title + tokenizer.sep_token + abstract, truncation at 512 tokens, and the
first token of last_hidden_state as the embedding (CLS pooling). FR3 fallback:
a paper without an abstract is encoded from its title alone (no trailing SEP).

torch, transformers and adapters are imported lazily, so the package imports
in every stage environment. Embed pins: adapters==1.3.0 requires
transformers~=4.57.6, so the embed notebook installs exactly
`adapters==1.3.0 transformers==4.57.6` rather than the runtime's default
transformers (configs/pins.md). Tests inject fake tokenizer/model objects: any
tokenizer returning a mapping of arrays/tensors and any model whose output has
`last_hidden_state` works.

Fallback if the adapters/transformers pin cannot be installed (risk register,
plan section 10): SciNCL (Ostendorff et al., 2022), a plain-transformers model
used with the same text rule and CLS pooling. Load it with
`AutoTokenizer.from_pretrained(name)` and `AutoModel.from_pretrained(name)`
(hub id `malteos/scincl`; confirm on the hub before use) and pass both to
encode(model=..., tokenizer=...). Record the switch in PROGRESS.md, because the
vectors are not interchangeable with SPECTER2 vectors.

Leak note: an embedding depends only on the paper's own title and abstract,
which exist from its publication date; no graph state enters. SPECTER2 itself
was trained with a citation-informed objective, which docs/score_components.md
discusses as a limitation.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np

BASE_MODEL = "allenai/specter2_base"
PROXIMITY_ADAPTER = "allenai/specter2"
EMBED_DIM = 768


def load_specter2(device=None, base_model=BASE_MODEL, adapter=PROXIMITY_ADAPTER):
    """(tokenizer, model) for SPECTER2 proximity embeddings, in eval mode on device.

    Needs network access to the Hugging Face hub on first use (cached after)."""
    try:
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "load_specter2 needs the embed-stage stack: adapters==1.3.0 (pins transformers "
            "~4.57.x) and torch; see configs/pins.md") from e
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoAdapterModel.from_pretrained(base_model)
    model.load_adapter(adapter, source="hf", load_as="specter2", set_active=True)
    model.to(device or _default_device())
    model.eval()
    return tokenizer, model


def _default_device():
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_device(model):
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _no_grad():
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.no_grad()


def _clean(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return str(x)


def paper_text(title, abstract, sep_token) -> str:
    """title + sep_token + abstract when the abstract is non-blank, else the title alone (FR3)."""
    title = _clean(title) or ""
    abstract = _clean(abstract)
    if abstract is not None and abstract.strip():
        return title + sep_token + abstract
    return title


def _papers(papers):
    """-> (ids, titles, abstracts). Accepts a DataFrame (work_id or id, title,
    abstract columns), (id, title, abstract) tuples, or dicts with those keys."""
    if hasattr(papers, "columns"):
        n = len(papers)
        id_col = "work_id" if "work_id" in papers.columns else ("id" if "id" in papers.columns else None)
        ids = list(papers[id_col]) if id_col else [None] * n
        titles = list(papers["title"])
        abstracts = list(papers["abstract"]) if "abstract" in papers.columns else [None] * n
        return ids, titles, abstracts
    ids, titles, abstracts = [], [], []
    for p in papers:
        if isinstance(p, dict):
            ids.append(p.get("work_id", p.get("id")))
            titles.append(p.get("title"))
            abstracts.append(p.get("abstract"))
        else:
            i, t, a = p
            ids.append(i)
            titles.append(t)
            abstracts.append(a)
    return ids, titles, abstracts


def _to_numpy(h):
    if hasattr(h, "detach"):
        h = h.detach().float().cpu().numpy()
    return np.asarray(h, dtype=np.float32)


def encode(papers, batch_size=32, model=None, tokenizer=None, device=None, max_length=512) -> np.ndarray:
    """Embed papers -> float32 (N, d), d = 768 for SPECTER2, rows in input order.

    papers: DataFrame with title/abstract columns, (id, title, abstract) tuples
    or dicts. model/tokenizer default to load_specter2(device). Texts are sorted
    by length before batching (less padding; outputs are restored to input
    order), the model runs in eval mode without gradients, and the embedding is
    the first token of last_hidden_state. Deterministic for a fixed model and
    device; independent of batch_size up to float rounding of padded batches.
    """
    _, titles, abstracts = _papers(papers)
    if model is None or tokenizer is None:
        loaded_tok, loaded_model = load_specter2(device)
        tokenizer = loaded_tok if tokenizer is None else tokenizer
        model = loaded_model if model is None else model
    if hasattr(model, "eval"):
        model.eval()
    if device is None:
        device = _model_device(model)
    texts = [paper_text(t, a, tokenizer.sep_token) for t, a in zip(titles, abstracts)]
    n = len(texts)
    if n == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    order = sorted(range(n), key=lambda i: len(texts[i]))
    out = None
    with _no_grad():
        for s in range(0, n, int(batch_size)):
            idx = order[s:s + int(batch_size)]
            inputs = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt",
                               return_token_type_ids=False)
            if device is not None:
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            vec = _to_numpy(model(**inputs).last_hidden_state[:, 0, :])
            if out is None:
                out = np.empty((n, vec.shape[1]), dtype=np.float32)
            out[idx] = vec
    return out


def write_embeddings_npz(path, ids, vectors) -> None:
    """Write the embeddings.npz contract: ids (N,) unicode str, vectors (N, d) float32.
    The file is written exactly at `path` (no suffix added)."""
    ids = np.asarray([str(i) for i in ids], dtype=str)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or ids.ndim != 1 or len(ids) != vectors.shape[0]:
        raise ValueError(f"need ids (N,) and vectors (N, d); got {ids.shape} and {vectors.shape}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("ids must be unique")
    with open(path, "wb") as f:
        np.savez(f, ids=ids, vectors=vectors)


def load_embeddings_npz(path):
    """-> (ids, vectors) from an embeddings.npz file."""
    with np.load(path) as z:
        return z["ids"], z["vectors"]


def encode_resumable(papers, chunk_dir, chunk_size=20_000, **encode_kwargs):
    """Encode in chunks saved as chunk_dir/part_{i:05d}.npz; finished parts are
    reused, so a killed Colab session resumes where it stopped (plan 9.1).

    Returns (vectors, n_encoded_now). A part whose stored ids differ from the
    current chunk raises ValueError (stale directory). The model is loaded at
    most once, and only when some part is missing."""
    ids, titles, abstracts = _papers(papers)
    if any(i is None for i in ids):
        raise ValueError("encode_resumable needs paper ids (work_id/id)")
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parts, n_new = [], 0
    for c, s in enumerate(range(0, len(ids), int(chunk_size))):
        e = s + int(chunk_size)
        cid = np.asarray([str(i) for i in ids[s:e]], dtype=str)
        part = chunk_dir / f"part_{c:05d}.npz"
        if part.exists():
            old_ids, old_vec = load_embeddings_npz(part)
            if not np.array_equal(old_ids, cid):
                raise ValueError(f"{part} holds different ids than chunk {c}; use a fresh chunk_dir")
            parts.append(old_vec)
            continue
        if encode_kwargs.get("model") is None or encode_kwargs.get("tokenizer") is None:
            tok, model = load_specter2(encode_kwargs.get("device"))
            if encode_kwargs.get("tokenizer") is None:
                encode_kwargs["tokenizer"] = tok
            if encode_kwargs.get("model") is None:
                encode_kwargs["model"] = model
        vec = encode(list(zip(cid, titles[s:e], abstracts[s:e])), **encode_kwargs)
        tmp = chunk_dir / f"part_{c:05d}.tmp"
        write_embeddings_npz(tmp, cid, vec)
        os.replace(tmp, part)
        parts.append(vec)
        n_new += len(cid)
    if not parts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32), 0
    return np.vstack(parts), n_new

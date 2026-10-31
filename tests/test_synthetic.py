"""Synthetic citation stream generator (stand-in data before S4.0; plan S4.5 / 9.4).

Spec under test (citecurv.data.synthetic):
- generate_citation_stream(...) returns works / events / authorships / embeddings in the
  CONTRACTS.md shapes; works carry `field` and `venue_id`; events carry event_id, src_id,
  dst_id, t with t = publication date of src (days since 1970-01-01).
- Every reference points strictly backwards in time: pub_date(dst) < pub_date(src).
- No self-loops and no duplicate (src, dst) pairs; canonical order (t, event_id).
- Fully deterministic under seed; a different seed gives a different stream.
- Realism knobs act in the documented direction: yearly growth, cross-field rate,
  self-citation rate, preferential attachment, ageing, heavy-tailed reference lists.
- Embeddings carry field signal: mean intra-field cosine > mean inter-field cosine, and
  cited pairs are more similar than random same-field pairs (subtopic affinity).
- profile(stream) summarises the stream; paper-level reciprocity is 0 by construction.
- The committed integration fixture tests/fixtures/mini_stream/ equals make_mini_fixture().
"""

import numpy as np
import pandas as pd
import pytest

from citecurv.data.synthetic import (
    MINI_FIXTURE_DIR,
    generate_citation_stream,
    load_fixture,
    make_mini_fixture,
    profile,
    write_fixture,
)

SMALL = dict(n_works=1500, n_fields=4, years=(2010, 2016), refs_mean=10, emb_dim=16)
WORK_COLUMNS = [
    "work_id", "doi_norm", "title", "abstract", "pub_date",
    "concepts", "venue_id", "is_retracted_oa", "field",
]


def days(pub_dates):
    return (pd.to_datetime(pd.Series(pub_dates)) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()


@pytest.fixture(scope="module")
def small():
    return generate_citation_stream(seed=0, **SMALL)


def _pub_days(stream):
    w = stream["works"]
    return dict(zip(w["work_id"], days(w["pub_date"])))


def test_schema(small):
    works, events, auth = small["works"], small["events"], small["authorships"]
    assert list(works.columns) == WORK_COLUMNS
    assert len(works) == SMALL["n_works"]
    assert works["work_id"].is_unique
    assert list(events.columns) == ["event_id", "src_id", "dst_id", "t"]
    assert events["event_id"].dtype == np.int64 and events["t"].dtype == np.int64
    assert events["event_id"].is_unique
    assert list(auth.columns) == ["work_id", "author_id"]
    assert set(auth["work_id"]) == set(works["work_id"])  # every work has >= 1 author
    ids, vectors = small["embeddings"]
    assert vectors.dtype == np.float32 and vectors.shape == (len(works), SMALL["emb_dim"])
    assert list(ids) == list(works["work_id"])
    assert not works["is_retracted_oa"].any()
    assert works["field"].nunique() == SMALL["n_fields"]


def test_canonical_order_and_ids(small):
    ev = small["events"]
    order = ev.sort_values(["t", "event_id"]).index
    assert (order == ev.index).all()
    assert (ev["event_id"].to_numpy() == np.arange(len(ev))).all()


def test_t_is_src_pub_date_and_refs_strictly_earlier(small):
    pub = _pub_days(small)
    ev = small["events"]
    assert (ev["t"].to_numpy() == ev["src_id"].map(pub).to_numpy()).all()
    assert (ev["dst_id"].map(pub).to_numpy() < ev["t"].to_numpy()).all()


def test_no_self_loops_no_duplicate_pairs(small):
    ev = small["events"]
    assert (ev["src_id"] != ev["dst_id"]).all()
    assert not ev.duplicated(["src_id", "dst_id"]).any()


def test_deterministic_and_seed_sensitive(small):
    again = generate_citation_stream(seed=0, **SMALL)
    for key in ("works", "events", "authorships"):
        pd.testing.assert_frame_equal(small[key], again[key])
    assert np.array_equal(small["embeddings"][1], again["embeddings"][1])
    other = generate_citation_stream(seed=1, **SMALL)
    assert not small["events"][["src_id", "dst_id"]].equals(other["events"][["src_id", "dst_id"]])


def test_unknown_knob_rejected():
    with pytest.raises(TypeError):
        generate_citation_stream(n_works=50, not_a_knob=1)


def test_yearly_growth():
    def per_year(stream):
        return pd.to_datetime(stream["works"]["pub_date"]).dt.year.value_counts().sort_index()

    grown = per_year(generate_citation_stream(seed=0, growth_rate=0.15, **SMALL))
    flat = per_year(generate_citation_stream(seed=0, growth_rate=0.0, **SMALL))
    assert list(grown.index) == list(range(2010, 2017))  # years are inclusive
    assert grown.iloc[-1] > 2 * grown.iloc[0]  # 1.15**6 = 2.3
    assert abs(flat.iloc[-1] - flat.iloc[0]) <= 2  # equal shares up to rounding


def _cross_field_share(stream):
    field = dict(zip(stream["works"]["work_id"], stream["works"]["field"]))
    ev = stream["events"]
    return float((ev["src_id"].map(field) != ev["dst_id"].map(field)).mean())


def test_cross_field_and_noise_knobs():
    closed = generate_citation_stream(seed=0, cross_field_rate=0.0, noise_rate=0.0, **SMALL)
    assert _cross_field_share(closed) == 0.0
    open_ = generate_citation_stream(seed=0, cross_field_rate=0.3, noise_rate=0.0, **SMALL)
    assert 0.2 < _cross_field_share(open_) < 0.4
    noisy = generate_citation_stream(seed=0, cross_field_rate=0.0, noise_rate=0.2, **SMALL)
    # uniform noise lands in another field with probability about (1 - own field share)
    assert 0.05 < _cross_field_share(noisy) < 0.2


def test_self_citation_knob():
    off = profile(generate_citation_stream(seed=0, self_cite_rate=0.0, **SMALL))
    on = profile(generate_citation_stream(seed=0, self_cite_rate=0.3, **SMALL))
    assert on["self_citation_rate"] > off["self_citation_rate"] + 0.1
    # knob off: only incidental author overlap (larger in small streams: small author pools)
    assert off["self_citation_rate"] < 0.10


def test_preferential_attachment_concentrates_citations():
    pa = profile(generate_citation_stream(seed=0, pa_exponent=1.0, **SMALL))
    flat = profile(generate_citation_stream(seed=0, pa_exponent=0.0, **SMALL))
    assert pa["in_degree"]["gini"] > flat["in_degree"]["gini"] + 0.05
    assert pa["in_degree"]["max"] > flat["in_degree"]["max"]


def test_ageing_knob_shifts_citation_age():
    fast = profile(generate_citation_stream(seed=0, ageing_decay_years=1.0, **SMALL))
    slow = profile(generate_citation_stream(seed=0, ageing_decay_years=20.0, **SMALL))
    assert fast["citation_age_years"]["median"] < slow["citation_age_years"]["median"]


def test_reference_lengths_heavy_tail():
    stream = generate_citation_stream(seed=0, **{**SMALL, "n_works": 3000, "years": (2000, 2016)})
    refs = stream["events"].groupby("src_id").size()
    late = refs[refs.index.isin(stream["works"]["work_id"].iloc[1500:])]  # past the start-up years
    assert late.quantile(0.99) > 2.5 * late.median()
    assert late.max() > 3 * late.mean()


def test_venues_belong_to_one_field(small):
    w = small["works"]
    assert (w.groupby("venue_id")["field"].nunique() == 1).all()
    assert w["venue_id"].nunique() <= SMALL["n_fields"] * 6


def _cos(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def test_embedding_field_separation(small):
    ids, vec = small["embeddings"]
    field = small["works"]["field"].to_numpy()
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(ids), size=(2, 20000))
    keep = i != j
    i, j = i[keep], j[keep]
    cos = _cos(vec[i], vec[j])
    same = field[i] == field[j]
    assert cos[same].mean() > cos[~same].mean() + 0.2


def test_cited_pairs_more_similar_than_random_same_field():
    """Subtopic affinity: within each field, cited pairs beat random pairs on cosine."""
    stream = generate_citation_stream(seed=0, **{**SMALL, "emb_dim": 64})
    ids, vec = stream["embeddings"]
    row = {w: k for k, w in enumerate(ids)}
    field = stream["works"]["field"].to_numpy()
    ev = stream["events"]
    s = ev["src_id"].map(row).to_numpy()
    d = ev["dst_id"].map(row).to_numpy()
    rng = np.random.default_rng(1)
    i, j = rng.integers(0, len(ids), size=(2, 40000))
    diffs, weights = [], []
    for f in np.unique(field):
        cited = (field[s] == f) & (field[d] == f)
        rand = (field[i] == f) & (field[j] == f) & (i != j)
        if cited.sum() < 50 or rand.sum() < 50:
            continue
        diffs.append(_cos(vec[s[cited]], vec[d[cited]]).mean()
                     - _cos(vec[i[rand]], vec[j[rand]]).mean())
        weights.append(cited.sum())
    assert np.average(diffs, weights=weights) > 0.02


def test_profile_fields(small):
    p = profile(small)
    assert p["synthetic"] is True
    assert p["n_works"] == SMALL["n_works"] and p["n_events"] == len(small["events"])
    assert p["reciprocity_paper"] == 0.0  # a citation points back in time: never mutual
    assert 0.0 <= p["reciprocity_author"] <= 1.0
    assert 0.0 <= p["self_citation_rate"] <= 1.0
    assert abs(sum(p["field_mix"]["works_share"].values()) - 1.0) < 1e-3  # 4-decimal rounding
    assert 0.0 < p["field_mix"]["within_field_event_share"] <= 1.0
    assert p["refs_per_paper"]["mean"] > 0
    assert p["embedding_cosine"]["intra_field"] > p["embedding_cosine"]["inter_field"]


# --- integration fixture (plan 9.4) ---------------------------------------------------

def test_mini_fixture_shape():
    fx = make_mini_fixture()
    ev, labels, cartels = fx["events"], fx["event_labels"], fx["cartels"]
    assert 150 <= len(ev) <= 260
    assert list(labels.columns) == ["event_id", "label"]
    assert (labels["event_id"].to_numpy() == ev["event_id"].to_numpy()).all()
    assert cartels["cartel_id"].nunique() == 1
    pos = labels.loc[labels["label"] == 1, "event_id"]
    assert len(pos) >= 3
    t_min = fx["meta"]["t_min"]
    assert (ev.set_index("event_id").loc[pos, "t"] >= t_min).all()


def test_fixture_roundtrip(tmp_path):
    fx = make_mini_fixture()
    write_fixture(fx, tmp_path)
    back = load_fixture(tmp_path)
    _assert_fixture_equal(back, fx)


def test_committed_fixture_equals_regenerated():
    assert MINI_FIXTURE_DIR.is_dir(), "run: python -m citecurv.data.synthetic --write-mini-fixture"
    _assert_fixture_equal(load_fixture(MINI_FIXTURE_DIR), make_mini_fixture())


def _assert_fixture_equal(a, b):
    for key in ("works", "events", "authorships", "event_labels", "cartels"):
        pd.testing.assert_frame_equal(a[key], b[key], check_dtype=True, obj=key)
    assert list(a["embeddings"][0]) == list(b["embeddings"][0])
    np.testing.assert_allclose(a["embeddings"][1], b["embeddings"][1], rtol=0, atol=1e-6)
    assert a["meta"] == b["meta"]

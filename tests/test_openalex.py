"""S1.1: pull harness — pagination, retry/backoff, cost ledger, parquet contracts.

No network: pages are injected through the _pager seam; the pyalex client is
only constructed in the default pager (covered by the key-guard test, which
fails before any request).
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from citecurv.data.openalex import ApiKeyMissing, PullStats, pull_works

CFG = {
    "data": {
        "page_size": 100,
        "mailto": None,
        "max_retries": 3,
        "backoff_base_s": 0.0,
    }
}

W1 = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1/A",
    "title": "Paper one",
    "abstract_inverted_index": {"Deep": [0], "learning": [1]},
    "publication_date": "2020-01-01",
    "concepts": [{"id": "https://openalex.org/C41008148"}],
    "primary_location": {"source": {"id": "https://openalex.org/S123"}},
    "is_retracted": False,
    "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W3"],
}
W4_NO_DATE = {
    "id": "https://openalex.org/W4",
    "referenced_works": ["https://openalex.org/W5"],
}
W6 = {"id": "https://openalex.org/W6", "publication_date": "1970-01-02"}


class FakeHTTP(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class FlakyPager:
    """Raises the given error on the first __next__, then serves one page."""

    def __init__(self, page, status):
        self.page, self.status, self.calls, self.served = page, status, 0, False

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls == 1:
            raise FakeHTTP(self.status)
        if self.served:
            raise StopIteration
        self.served = True
        return self.page


def test_writes_contracted_parquet_schemas(tmp_path):
    stats = pull_works({}, tmp_path, _pager=[[W1], [W4_NO_DATE, W6]], config=CFG)

    assert isinstance(stats, PullStats)
    assert (stats.works, stats.edges, stats.pages, stats.works_no_date) == (3, 2, 2, 1)
    assert stats.est_cost_usd == pytest.approx(2 * 0.0001)

    works = pd.read_parquet(tmp_path / "works.parquet")
    edges = pd.read_parquet(tmp_path / "edges.parquet")
    assert list(works.columns) == [
        "work_id", "doi_norm", "title", "abstract", "pub_date",
        "concepts", "venue_id", "is_retracted_oa",
    ]
    assert list(edges.columns) == ["src_id", "dst_id", "t", "doi_pair_key"]

    w1 = works[works.work_id == "W1"].iloc[0]
    assert w1.doi_norm == "10.1/a"
    assert w1.abstract == "Deep learning"
    assert w1.venue_id == "S123"
    assert list(w1.concepts) == ["C41008148"]

    assert set(edges.dst_id) == {"W2", "W3"}          # W4's ref dropped: no timestamp
    assert set(edges.t) == {18262}                     # days(2020-01-01)
    assert set(edges.doi_pair_key) == {"W1|W2", "W1|W3"}


def test_retries_retryable_status_then_succeeds(tmp_path):
    pager = FlakyPager([W6], status=429)
    stats = pull_works({}, tmp_path, _pager=pager, config=CFG)
    assert (stats.pages, stats.works) == (1, 1)
    assert pager.calls == 3  # fail, page, StopIteration


def test_non_retryable_status_raises_immediately(tmp_path):
    pager = FlakyPager([W6], status=403)
    with pytest.raises(FakeHTTP):
        pull_works({}, tmp_path, _pager=pager, config=CFG)
    assert pager.calls == 1


def test_max_works_stops_early(tmp_path):
    stats = pull_works({}, tmp_path, _pager=[[W1, W6, W6]], config=CFG, max_works=2)
    assert stats.works == 2
    assert len(pd.read_parquet(tmp_path / "works.parquet")) == 2


def test_missing_api_key_refuses_keyless_pull(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    with pytest.raises(ApiKeyMissing):
        pull_works({}, tmp_path, config=CFG)

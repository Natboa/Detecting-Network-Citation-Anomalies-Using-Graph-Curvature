"""Cited works outside the pulled set: which they are, their text, and how many events are cold.

Spec under test (citecurv.data.cited):
- cited_only_ids: sorted unique dst_id values of the events that are not a work_id of works.
- prior_citation_counts: per event, the number of events of the same cited work with a strictly
  earlier t (same-day events do not count), in the row order of the input.
- cold_start_summary: share of events with fewer than k prior citations, overall and for positives.
- pull_works_by_ids: one request per batch of at most 100 ids, each id asked exactly once; rows
  work_id, title (display_name fallback), abstract (rebuilt), pub_date; only requested ids are
  kept, once each; ids OpenAlex does not return are counted as missing; finished parts on disk are
  reused, so a rerun after a failure asks only for the parts that are missing; a part folder that
  belongs to another id list is refused.
- resolve_id_filter: the first candidate filter name whose probe returns exactly the sample ids.
"""

import numpy as np
import pandas as pd
import pytest

from citecurv.data.cited import (
    cited_only_ids,
    cold_start_summary,
    prior_citation_counts,
    pull_works_by_ids,
    resolve_id_filter,
)


def _events():
    # W1 and W2 are pulled works; X*, Y* are cited only
    return pd.DataFrame({
        "event_id": [0, 1, 2, 3, 4, 5, 6],
        "src_id": ["W1", "W1", "W2", "W2", "W2", "W1", "W2"],
        "dst_id": ["X1", "W2", "X1", "Y1", "X1", "X1", "X1"],
        "t": [10, 10, 11, 11, 11, 12, 13],
    })


def test_cited_only_ids_are_the_dsts_missing_from_works():
    works = pd.DataFrame({"work_id": ["W1", "W2"]})
    assert list(cited_only_ids(_events(), works)) == ["X1", "Y1"]


def test_prior_citation_counts_are_strict_and_keep_row_order():
    # X1 is cited at t = 10, 11, 11, 12, 13 -> priors 0, 1, 1, 3, 4; W2 and Y1 once -> 0
    assert list(prior_citation_counts(_events())) == [0, 0, 1, 0, 1, 3, 4]
    shuffled = _events().iloc[[6, 2, 0, 5, 4, 1, 3]]
    assert list(prior_citation_counts(shuffled)) == [4, 1, 0, 3, 1, 0, 0]


def test_cold_start_summary_overall_and_for_positives():
    labels = pd.DataFrame({"event_id": [0, 1, 2, 3, 4, 5, 6], "label": [0, 0, 0, 1, 0, 1, 1]})
    out = cold_start_summary(_events(), k=3, event_labels=labels)
    # priors 0,0,1,0,1,3,4 -> cold (< 3): events 0..4 = 5 of 7; positives 3, 5, 6 -> cold: only event 3
    assert out == {"k": 3, "n_events": 7, "n_cold": 5, "cold_share": 5 / 7,
                   "n_positive": 3, "n_positive_cold": 1, "positive_cold_share": 1 / 3}
    assert cold_start_summary(_events(), k=1)["n_cold"] == 3
    assert "n_positive" not in cold_start_summary(_events(), k=3)


def _work(i, title="T", inv=None, date="2015-01-02", display=None):
    return {"id": f"https://openalex.org/{i}", "title": title, "display_name": display,
            "abstract_inverted_index": inv, "publication_date": date}


class FakeFetch:
    """Answers a batch with one record per known id; records every batch it was asked for."""

    def __init__(self, known, fail_on_call=None, extra=None):
        self.known, self.fail_on_call, self.extra = known, fail_on_call, extra or []
        self.batches = []

    def __call__(self, ids):
        if self.fail_on_call is not None and len(self.batches) == self.fail_on_call:
            raise RuntimeError("boom")
        self.batches.append(list(ids))
        return {"results": [self.known[i] for i in ids if i in self.known] + self.extra,
                "meta": {"cost_usd": 0.0001}}


def test_pull_by_ids_rows_batches_and_stats(tmp_path):
    ids = [f"W{i:03d}" for i in range(7)]
    known = {i: _work(i, title=f"title {i}") for i in ids if i != "W003"}
    known["W001"] = _work("W001", title=None, display="shown name", inv={"world": [1], "hello": [0]})
    fetch = FakeFetch(known, extra=[_work("W999"), _work("W000", title="duplicate")])
    out = tmp_path / "cited_works.parquet"
    said = []
    stats = pull_works_by_ids(reversed(ids + ["W002"]), out, batch_size=3, ids_per_part=6, _fetch=fetch,
                              log=said.append)
    assert len(said) == 2 and "part 2 of 2" in said[-1]            # one progress line per fetched part
    assert [len(b) for b in fetch.batches] == [3, 3, 1]
    assert sorted(i for b in fetch.batches for i in b) == ids          # each id once, duplicates dropped
    df = pd.read_parquet(out)
    assert list(df.columns) == ["work_id", "title", "abstract", "pub_date"]
    assert list(df.work_id) == [i for i in ids if i != "W003"]            # sorted, requested ids only, once each
    row = df.set_index("work_id").loc["W001"]
    assert row.title == "shown name" and row.abstract == "hello world" and row.pub_date == "2015-01-02"
    assert df.set_index("work_id").loc["W000", "title"] == "title W000"   # the first record of an id wins
    assert stats["n_ids"] == 7 and stats["n_found"] == 6 and stats["n_missing"] == 1
    # W999 is never asked for (3 calls); the extra W000 record is a duplicate in call 1 and unasked in calls 2, 3
    assert stats["calls"] == 3 and stats["n_with_abstract"] == 1
    assert stats["n_other_id"] == 5 and stats["n_duplicate"] == 1
    assert stats["cost_usd_reported"] == pytest.approx(0.0003)


def test_pull_by_ids_resumes_from_finished_parts(tmp_path):
    ids = [f"W{i:03d}" for i in range(10)]
    known = {i: _work(i) for i in ids}
    out = tmp_path / "cited_works.parquet"
    failing = FakeFetch(known, fail_on_call=3)     # part 0 = calls 0, 1 (4 ids); the failure hits part 1
    with pytest.raises(RuntimeError):
        pull_works_by_ids(ids, out, batch_size=2, ids_per_part=4, _fetch=failing, config={"data": {"max_retries": 0}})
    assert not out.exists()
    fetch = FakeFetch(known)
    stats = pull_works_by_ids(ids, out, batch_size=2, ids_per_part=4, _fetch=fetch)
    assert sorted(i for b in fetch.batches for i in b) == ids[4:]          # part 0 was not asked again
    assert stats["n_found"] == 10 and stats["parts_reused"] == 1 and stats["calls"] == 3
    assert list(pd.read_parquet(out).work_id) == ids


def test_pull_by_ids_refuses_a_part_folder_of_another_id_list(tmp_path):
    known = {f"W{i}": _work(f"W{i}") for i in range(4)}
    out = tmp_path / "cited_works.parquet"
    pull_works_by_ids(["W0", "W1"], out, _fetch=FakeFetch(known), keep_parts=True)
    with pytest.raises(ValueError, match="different id list"):
        pull_works_by_ids(["W0", "W2"], out, _fetch=FakeFetch(known))


def test_pull_by_ids_rejects_bad_sizes(tmp_path):
    with pytest.raises(ValueError):
        pull_works_by_ids(["W0"], tmp_path / "o.parquet", batch_size=101, _fetch=FakeFetch({}))
    with pytest.raises(ValueError):
        pull_works_by_ids([], tmp_path / "o.parquet", _fetch=FakeFetch({}))


def test_resolve_id_filter_picks_the_first_name_that_returns_the_sample():
    asked = []

    def fetch_for(name):
        def fetch(ids):
            asked.append(name)
            if name == "openalex":
                raise RuntimeError("400 invalid filter")
            if name == "ids.openalex":
                return {"results": [_work(ids[0])]}       # one of two: not good enough
            return {"results": [_work(i) for i in ids]}
        return fetch

    assert resolve_id_filter(["W1", "W2"], _fetch_for=fetch_for) == "openalex_id"
    assert asked == ["openalex", "ids.openalex", "openalex_id"]
    with pytest.raises(RuntimeError, match="no id filter"):
        resolve_id_filter(["W1", "W2"], candidates=("openalex",), _fetch_for=fetch_for)


def test_prior_counts_match_a_brute_force_count():
    rng = np.random.default_rng(3)
    ev = pd.DataFrame({"event_id": np.arange(400), "src_id": "S",
                       "dst_id": rng.integers(0, 25, 400).astype(str), "t": rng.integers(0, 30, 400)})
    brute = [int(((ev.dst_id == d) & (ev.t < t)).sum()) for d, t in zip(ev.dst_id, ev.t)]
    assert list(prior_citation_counts(ev)) == brute

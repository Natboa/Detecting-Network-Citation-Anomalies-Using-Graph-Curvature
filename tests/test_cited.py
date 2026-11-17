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


# ---- reference lists of cited works (what="references"), added 2026-11-12 after the S2.2 pilot finding:
# the Forman screen needs out_deg_t(cited) = the cited work's reference count, and the Ollivier/coupling
# part needs the reference ids; neither exists for works outside the pulled set.

def _ref_work(i, refs, count=None, date="2015-01-02"):
    return {"id": f"https://openalex.org/{i}", "publication_date": date,
            "referenced_works": [f"https://openalex.org/{r}" for r in refs],
            "referenced_works_count": len(refs) if count is None else count}


def test_pull_references_rows_and_stats(tmp_path):
    from citecurv.data.cited import REFERENCE_COLUMNS, REFERENCE_SELECT

    assert {"id", "referenced_works", "referenced_works_count", "publication_date"} <= set(REFERENCE_SELECT)
    ids = ["W1", "W2", "W3", "W4"]
    known = {"W1": _ref_work("W1", ["R1", "R2", "R3"]),
             "W2": _ref_work("W2", []),                       # a work with no references
             "W3": _ref_work("W3", ["R1"], count=5)}          # count and list disagree: both are kept
    out = tmp_path / "cited_refs.parquet"
    stats = pull_works_by_ids(ids, out, what="references", batch_size=2, _fetch=FakeFetch(known))
    df = pd.read_parquet(out)
    assert list(df.columns) == REFERENCE_COLUMNS == ["work_id", "pub_date", "n_refs", "ref_ids"]
    assert list(df.work_id) == ["W1", "W2", "W3"]
    by = df.set_index("work_id")
    assert list(by.loc["W1", "ref_ids"]) == ["R1", "R2", "R3"] and int(by.loc["W1", "n_refs"]) == 3
    assert len(by.loc["W2", "ref_ids"]) == 0 and int(by.loc["W2", "n_refs"]) == 0
    assert list(by.loc["W3", "ref_ids"]) == ["R1"] and int(by.loc["W3", "n_refs"]) == 5
    assert by.loc["W1", "pub_date"] == "2015-01-02"
    assert stats["n_ids"] == 4 and stats["n_found"] == 3 and stats["n_missing"] == 1 and stats["calls"] == 2
    assert stats["n_with_refs"] == 2 and stats["n_ref_ids"] == 4 and stats["n_count_mismatch"] == 1
    assert stats["what"] == "references" and "n_with_abstract" not in stats


def test_pull_references_resumes_and_keeps_empty_parts(tmp_path):
    # part 1 holds only a work without references: its list column must survive the part round trip
    ids = ["W1", "W2", "W3"]
    known = {"W1": _ref_work("W1", ["R1"]), "W2": _ref_work("W2", ["R2", "R3"]), "W3": _ref_work("W3", [])}
    out = tmp_path / "cited_refs.parquet"
    failing = FakeFetch(known, fail_on_call=1)
    with pytest.raises(RuntimeError):
        pull_works_by_ids(ids, out, what="references", batch_size=2, ids_per_part=2, _fetch=failing,
                          config={"data": {"max_retries": 0}})
    fetch = FakeFetch(known)
    stats = pull_works_by_ids(ids, out, what="references", batch_size=2, ids_per_part=2, _fetch=fetch)
    assert fetch.batches == [["W3"]] and stats["parts_reused"] == 1
    df = pd.read_parquet(out).set_index("work_id")
    assert [len(x) for x in df["ref_ids"]] == [1, 2, 0] and stats["n_ref_ids"] == 3


def test_pull_text_is_the_default_and_unknown_what_is_refused(tmp_path):
    known = {"W0": _work("W0", title="t")}
    stats = pull_works_by_ids(["W0"], tmp_path / "a.parquet", _fetch=FakeFetch(known))
    assert stats["what"] == "text" and "n_with_abstract" in stats
    with pytest.raises(ValueError, match="what"):
        pull_works_by_ids(["W0"], tmp_path / "b.parquet", what="citations", _fetch=FakeFetch(known))


# ---- reference lists as context rows of the event table (2026-11-13)

def _refs_frame():
    return pd.DataFrame({
        "work_id": ["X1", "X2", "X3", "W1"],
        "pub_date": ["1970-01-11", None, "1970-01-21", "1970-01-05"],
        "n_refs": [2, 1, 0, 1],
        "ref_ids": [["R1", "R2"], ["R1"], [], ["R9"]],
    })


def test_reference_events_are_one_row_per_reference_at_the_citing_works_publication_day():
    from citecurv.data.cited import reference_events

    out, stats = reference_events(_refs_frame(), exclude_src=["W1"], start_event_id=100)
    # X1 published on day 10 cites R1 and R2; X2 has no date (dropped); X3 has no references; W1 is excluded
    assert list(out.columns) == ["event_id", "src_id", "dst_id", "t"]
    assert out.to_dict("list") == {"event_id": [100, 101], "src_id": ["X1", "X1"], "dst_id": ["R1", "R2"], "t": [10, 10]}
    assert out["t"].dtype == np.int64 and out["event_id"].dtype == np.int64
    assert stats == {"n_works": 4, "n_excluded": 1, "n_no_date": 1, "n_without_refs": 1, "n_rows": 2,
                     "n_ref_ids_dropped_no_date": 1}


def test_reference_events_default_ids_start_at_zero_and_bad_dates_are_dropped():
    from citecurv.data.cited import reference_events

    refs = pd.DataFrame({"work_id": ["A", "B"], "pub_date": ["not a date", "1970-01-02"], "n_refs": [1, 1],
                         "ref_ids": [["R1"], ["R2"]]})
    out, stats = reference_events(refs)
    assert out.to_dict("list") == {"event_id": [0], "src_id": ["B"], "dst_id": ["R2"], "t": [1]}
    assert stats["n_no_date"] == 1 and stats["n_rows"] == 1


def test_reference_events_make_the_forman_screen_see_the_cited_works_references():
    # event W1 -> X1 at day 20; X1 (published day 10) has 2 references. Without context F = 2 - 0 - 0 = 2;
    # with the reference rows as context F = 2 - 0 - 2 = 0. A reference row dated after the event must not count.
    from citecurv.curvature.screen import forman_screen
    from citecurv.data.cited import reference_events

    events = pd.DataFrame({"event_id": [0, 1], "src_id": ["W1", "W1"], "dst_id": ["X1", "X3"], "t": [20, 20]})
    refs = _refs_frame().assign(ref_ids=[["R1", "R2"], ["R1"], ["R5"], ["R9"]])   # X3 published day 20 + 1
    refs.loc[refs.work_id == "X3", "pub_date"] = "1970-01-22"
    ctx, _ = reference_events(refs, exclude_src=["W1"], start_event_id=int(events.event_id.max()) + 1)
    alone = forman_screen(events).set_index("event_id")["forman"]
    both = forman_screen(pd.concat([events, ctx], ignore_index=True)).set_index("event_id")["forman"]
    assert alone.loc[0] == 2 and both.loc[0] == 0
    assert both.loc[1] == 2                      # X3's reference row is at day 21, after the event at day 20


# ---- a finished pull is not downloaded again (2026-11-16). The parts folder is removed after the merge,
# so before this a rerun with the output already on disk asked OpenAlex for every id a second time.

def test_pull_by_ids_rerun_with_the_output_on_disk_makes_no_call(tmp_path):
    import json

    ids = [f"W{i:03d}" for i in range(5)]
    known = {i: _work(i) for i in ids if i != "W002"}
    out = tmp_path / "cited_works.parquet"
    first = pull_works_by_ids(ids, out, batch_size=2, _fetch=FakeFetch(known))
    assert first["calls"] == 3 and first["already_complete"] is False
    assert not (tmp_path / "cited_works_parts").exists()
    before = out.read_bytes()
    again = FakeFetch(known)
    stats = pull_works_by_ids(reversed(ids), out, batch_size=2, _fetch=again)
    assert again.batches == []
    assert stats["already_complete"] is True and stats["calls"] == 0 and stats["cost_usd_reported"] is None
    for key in ("n_ids", "n_found", "n_missing", "n_with_abstract", "n_no_title", "what"):
        assert stats[key] == first[key]
    assert out.read_bytes() == before and not (tmp_path / "cited_works_parts").exists()
    # the stats file keeps the record of the run that paid for the download
    on_disk = json.loads((tmp_path / "cited_works_stats.json").read_text(encoding="utf-8"))
    assert on_disk["calls"] == 3 and on_disk["cost_usd_reported"] == pytest.approx(0.0003)


def test_a_finished_pull_builds_no_api_client(tmp_path, monkeypatch):
    import citecurv.data.cited as cited

    known = {"W0": _work("W0"), "W1": _work("W1")}
    out = tmp_path / "cited_works.parquet"
    pull_works_by_ids(["W0", "W1"], out, _fetch=FakeFetch(known))

    def no_client(*args, **kwargs):
        raise AssertionError("a client was built for a finished pull")

    monkeypatch.setattr(cited, "_client_fetch_for", no_client)
    stats = pull_works_by_ids(["W0", "W1"], out, config={"data": {}})   # no key needed, no network touched
    assert stats["already_complete"] is True and stats["n_found"] == 2


def _text_and_refs(i):
    return {**_work(i), **_ref_work(i, ["R1"])}


def test_pull_by_ids_fetches_again_for_another_id_list_what_or_select(tmp_path):
    from citecurv.data.cited import REFERENCE_COLUMNS

    known = {f"W{i}": _text_and_refs(f"W{i}") for i in range(4)}
    out = tmp_path / "o.parquet"
    pull_works_by_ids(["W0", "W1"], out, _fetch=FakeFetch(known))
    other_ids = FakeFetch(known)
    stats = pull_works_by_ids(["W0", "W2"], out, _fetch=other_ids)
    assert other_ids.batches == [["W0", "W2"]] and stats["already_complete"] is False
    assert list(pd.read_parquet(out).work_id) == ["W0", "W2"]
    other_select = FakeFetch(known)
    pull_works_by_ids(["W0", "W2"], out, select=("id", "title"), _fetch=other_select)
    assert other_select.batches == [["W0", "W2"]]
    other_what = FakeFetch(known)
    pull_works_by_ids(["W0", "W2"], out, what="references", _fetch=other_what)
    assert other_what.batches == [["W0", "W2"]]
    assert list(pd.read_parquet(out).columns) == REFERENCE_COLUMNS


def test_pull_by_ids_fetches_again_when_the_file_is_not_the_one_the_stats_describe(tmp_path):
    import json

    known = {f"W{i}": _work(f"W{i}") for i in range(3)}
    out = tmp_path / "o.parquet"
    stats_file = tmp_path / "o_stats.json"
    pull_works_by_ids(["W0", "W1", "W2"], out, _fetch=FakeFetch(known))
    # another file under the same name (here: fewer rows than the stats say)
    pd.DataFrame({"work_id": ["W0"], "title": ["t"], "abstract": ["a"], "pub_date": ["2015-01-02"]}).to_parquet(out, index=False)
    replaced = FakeFetch(known)
    stats = pull_works_by_ids(["W0", "W1", "W2"], out, _fetch=replaced)
    assert replaced.batches == [["W0", "W1", "W2"]] and stats["already_complete"] is False
    assert list(pd.read_parquet(out).work_id) == ["W0", "W1", "W2"]
    # a stats file written before the id hash was stored (the pilot's files) proves nothing about the id list
    old = json.loads(stats_file.read_text(encoding="utf-8"))
    del old["ids_sha256"]
    stats_file.write_text(json.dumps(old), encoding="utf-8")
    old_style = FakeFetch(known)
    pull_works_by_ids(["W0", "W1", "W2"], out, _fetch=old_style)
    assert old_style.batches == [["W0", "W1", "W2"]]
    # an unreadable stats file is treated the same way
    stats_file.write_text("{not json", encoding="utf-8")
    broken = FakeFetch(known)
    pull_works_by_ids(["W0", "W1", "W2"], out, _fetch=broken)
    assert broken.batches == [["W0", "W1", "W2"]]


# ---- text and reference lists in one request (what="text_and_references"), 2026-11-17. The full pull
# asks for both about every cited work outside the slice; two separate pulls cost two calls per 100 ids
# of a daily allowance of 10,000 calls, one joint pull costs one.

def _plain(series):
    """Cell values as plain Python: missing -> None, a list cell -> list."""
    return [list(x) if isinstance(x, (list, np.ndarray)) else (None if pd.isna(x) else x) for x in series]


def _joint_known():
    known = {"W1": {**_work("W1", title="first", inv={"world": [1], "hello": [0]}), **_ref_work("W1", ["R1", "R2"])},
             "W2": {**_work("W2", title=None, display="shown name"), **_ref_work("W2", [])},     # no abstract, no references
             "W3": {**_work("W3", title="third", date="2016-05-06"), **_ref_work("W3", ["R1"], count=5, date="2016-05-06")}}
    return known


def test_pull_text_and_references_in_one_request_per_batch(tmp_path):
    from citecurv.data.cited import DEFAULT_SELECT, JOINT_COLUMNS, JOINT_SELECT, REFERENCE_SELECT

    assert set(JOINT_SELECT) == set(DEFAULT_SELECT) | set(REFERENCE_SELECT)
    assert len(JOINT_SELECT) == len(set(JOINT_SELECT))                 # a field is asked for once
    ids = ["W1", "W2", "W3", "W4"]
    fetch = FakeFetch(_joint_known())
    out = tmp_path / "cited_joint.parquet"
    stats = pull_works_by_ids(ids, out, what="text_and_references", batch_size=2, _fetch=fetch)
    assert [len(b) for b in fetch.batches] == [2, 2] and stats["calls"] == 2      # as many calls as one pull alone
    df = pd.read_parquet(out)
    assert list(df.columns) == JOINT_COLUMNS == ["work_id", "title", "abstract", "pub_date", "n_refs", "ref_ids"]
    assert list(df.work_id) == ["W1", "W2", "W3"]
    assert _plain(df["title"]) == ["first", "shown name", "third"]
    assert _plain(df["abstract"]) == ["hello world", None, None]
    assert _plain(df["pub_date"]) == ["2015-01-02", "2015-01-02", "2016-05-06"]
    assert [int(n) for n in df["n_refs"]] == [2, 0, 5]
    assert _plain(df["ref_ids"]) == [["R1", "R2"], [], ["R1"]]
    assert stats["what"] == "text_and_references" and stats["select"] == list(JOINT_SELECT)
    assert stats["n_ids"] == 4 and stats["n_found"] == 3 and stats["n_missing"] == 1
    assert stats["n_with_abstract"] == 1 and stats["n_no_title"] == 0             # the text pull's counts
    assert stats["n_with_refs"] == 2 and stats["n_ref_ids"] == 3 and stats["n_count_mismatch"] == 1   # the reference pull's


def test_the_joint_file_holds_what_the_two_separate_pulls_hold(tmp_path):
    from citecurv.data.cited import COLUMNS, REFERENCE_COLUMNS

    ids, known = ["W1", "W2", "W3", "W4"], _joint_known()
    pull_works_by_ids(ids, tmp_path / "text.parquet", batch_size=2, _fetch=FakeFetch(known))
    pull_works_by_ids(ids, tmp_path / "refs.parquet", what="references", batch_size=2, _fetch=FakeFetch(known))
    pull_works_by_ids(ids, tmp_path / "joint.parquet", what="text_and_references", batch_size=2, _fetch=FakeFetch(known))
    text, refs, joint = (pd.read_parquet(tmp_path / f"{n}.parquet") for n in ("text", "refs", "joint"))
    for col in COLUMNS:
        assert _plain(joint[col]) == _plain(text[col]), col
    for col in REFERENCE_COLUMNS:
        assert _plain(joint[col]) == _plain(refs[col]), col
    # the reader of reference lists takes the joint frame as it is
    from citecurv.data.cited import reference_events

    assert reference_events(joint)[0].equals(reference_events(refs)[0])


def test_pull_text_and_references_resumes_and_keeps_a_part_without_text_or_references(tmp_path):
    # part 1 holds only W3 with no abstract and no references: its columns must survive the part round trip
    known = _joint_known()
    known["W3"] = {**_work("W3", title="third"), **_ref_work("W3", [])}
    out = tmp_path / "cited_joint.parquet"
    failing = FakeFetch(known, fail_on_call=1)
    with pytest.raises(RuntimeError):
        pull_works_by_ids(["W1", "W2", "W3"], out, what="text_and_references", batch_size=2, ids_per_part=2,
                          _fetch=failing, config={"data": {"max_retries": 0}})
    fetch = FakeFetch(known)
    stats = pull_works_by_ids(["W1", "W2", "W3"], out, what="text_and_references", batch_size=2, ids_per_part=2,
                              _fetch=fetch)
    assert fetch.batches == [["W3"]] and stats["parts_reused"] == 1
    df = pd.read_parquet(out)
    assert _plain(df["abstract"]) == ["hello world", None, None]
    assert _plain(df["ref_ids"]) == [["R1", "R2"], [], []] and stats["n_ref_ids"] == 2


def test_a_finished_joint_pull_is_not_fetched_again_and_is_not_taken_for_a_text_pull(tmp_path):
    known = _joint_known()
    out = tmp_path / "o.parquet"
    pull_works_by_ids(["W1", "W2"], out, what="text_and_references", _fetch=FakeFetch(known))
    again = FakeFetch(known)
    stats = pull_works_by_ids(["W1", "W2"], out, what="text_and_references", _fetch=again)
    assert again.batches == [] and stats["already_complete"] is True
    as_text = FakeFetch(known)
    pull_works_by_ids(["W1", "W2"], out, _fetch=as_text)
    assert as_text.batches == [["W1", "W2"]]
    assert list(pd.read_parquet(out).columns) == ["work_id", "title", "abstract", "pub_date"]


def test_a_part_folder_of_a_text_pull_is_refused_for_a_joint_pull(tmp_path):
    known = _joint_known()
    out = tmp_path / "o.parquet"
    pull_works_by_ids(["W1", "W2"], out, _fetch=FakeFetch(known), keep_parts=True)
    (tmp_path / "o.parquet").unlink()                                  # only the text parts are left
    with pytest.raises(ValueError, match="different id list"):
        pull_works_by_ids(["W1", "W2"], out, what="text_and_references", _fetch=FakeFetch(known))


# --------------------------------------------------------------------------- the merge, one part at a time
# 2026-11-17: the full run has 312 parts of 5,000 cited works (1,555,150 ids) and a Colab machine with
# 12.67 GB of memory. Loading every part before writing the output may not fit, so the output is written
# part by part: one row group per part that holds rows.

class ReversedFetch(FakeFetch):
    """Answers like FakeFetch with the records of a batch in reverse order."""

    def __call__(self, ids):
        resp = super().__call__(ids)
        return {**resp, "results": resp["results"][::-1]}


def test_the_joint_output_is_written_one_part_at_a_time_and_stays_sorted(tmp_path):
    import json

    import pyarrow.parquet as pq

    ids = [f"W{i}" for i in range(1, 6)]                                 # parts of 2 ids: [W1 W2] [W3 W4] [W5]
    known = {i: {**_work(i, title=f"title {i}", inv={"text": [0]} if i in ("W1", "W4") else None),
                 **_ref_work(i, ["R1"] * int(i[1:]) if i != "W2" else [], count=3)} for i in ids}
    out = tmp_path / "cited_joint.parquet"
    stats = pull_works_by_ids(ids, out, what="text_and_references", batch_size=2, ids_per_part=2,
                              _fetch=ReversedFetch(known))
    meta = pq.ParquetFile(out).metadata
    assert meta.num_row_groups == 3 and [meta.row_group(g).num_rows for g in range(3)] == [2, 2, 1]
    df = pd.read_parquet(out)
    assert list(df.work_id) == ids                                       # sorted inside a part and across parts
    assert _plain(df["abstract"]) == ["text", None, None, "text", None]
    assert [len(x) for x in df["ref_ids"]] == [1, 0, 3, 4, 5]
    # the counts are the sums over the parts
    assert stats["n_found"] == 5 and stats["n_missing"] == 0 and stats["parts"] == 3
    assert stats["n_with_abstract"] == 2 and stats["n_no_title"] == 0
    assert stats["n_with_refs"] == 4 and stats["n_ref_ids"] == 13 and stats["n_count_mismatch"] == 4
    assert json.loads((tmp_path / "cited_joint_stats.json").read_text(encoding="utf-8")) == stats
    assert not (tmp_path / "cited_joint_parts").exists() and not (tmp_path / "cited_joint.parquet.tmp").exists()


def test_a_text_output_written_part_by_part_keeps_a_part_with_no_abstract_and_a_part_with_no_rows(tmp_path):
    import pyarrow.parquet as pq

    ids = [f"W{i}" for i in range(1, 7)]                                 # parts: [W1 W2] [W3 W4] [W5 W6]
    known = {"W1": _work("W1", title="one"), "W2": _work("W2", title="two"),          # part 0: no abstract at all
             "W5": _work("W5", title="five", inv={"world": [1], "hello": [0]}),       # part 1: no rows
             "W6": _work("W6", title=None)}
    out = tmp_path / "cited_works.parquet"
    stats = pull_works_by_ids(ids, out, batch_size=2, ids_per_part=2, _fetch=ReversedFetch(known))
    meta = pq.ParquetFile(out).metadata
    assert meta.num_rows == 4 and meta.num_row_groups == 2               # the part with no rows adds no row group
    df = pd.read_parquet(out)
    assert list(df.columns) == ["work_id", "title", "abstract", "pub_date"]
    assert list(df.work_id) == ["W1", "W2", "W5", "W6"]
    assert _plain(df["title"]) == ["one", "two", "five", None]
    assert _plain(df["abstract"]) == [None, None, "hello world", None]
    assert stats["n_found"] == 4 and stats["n_missing"] == 2
    assert stats["n_with_abstract"] == 1 and stats["n_no_title"] == 1


def test_an_output_with_no_found_work_is_an_empty_file_with_the_columns(tmp_path):
    out = tmp_path / "cited_joint.parquet"
    stats = pull_works_by_ids(["W1", "W2", "W3"], out, what="text_and_references", ids_per_part=2,
                              _fetch=FakeFetch({}))
    df = pd.read_parquet(out)
    assert len(df) == 0 and list(df.columns) == ["work_id", "title", "abstract", "pub_date", "n_refs", "ref_ids"]
    assert stats["n_found"] == 0 and stats["n_missing"] == 3 and stats["n_ref_ids"] == 0

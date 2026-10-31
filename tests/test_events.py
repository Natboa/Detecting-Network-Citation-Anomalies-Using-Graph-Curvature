"""Edges -> contract events frame, and event labels from Retraction Watch (CONTRACTS.md "Events", "Labels").

Spec under test:
- build_events turns the S1.1 edges frame (src_id, dst_id, t, doi_pair_key) into the events frame
  (event_id, src_id, dst_id, t) with int64 event_id and t. event_id is assigned after sorting by
  (t, src_id, dst_id), so it does not depend on the row order of the pull (pages, year buckets).
- Duplicate (src_id, dst_id) pairs keep the first row in that sort order (the earliest t); dedupe=False keeps
  every row. Self-citation loops (src_id == dst_id) are kept.
- write_events / read_events round-trip and refuse frames that break the contract.
- event_labels_from_rwdb marks an event positive when its citing or its cited paper is a labelled paper
  (normalised-DOI join through rwdb.join_labels; "loose" also matches version suffixes); optional
  family / nature filters apply to load_labels-style frames.
All expected values are hand-computed from the small fixtures below.
"""

import numpy as np
import pandas as pd
import pytest

from citecurv.data.events import (
    EVENT_COLUMNS,
    build_events,
    event_labels_from_rwdb,
    read_events,
    validate_events,
    write_events,
)


def _edges(rows):
    df = pd.DataFrame(rows, columns=["src_id", "dst_id", "t"])
    df["t"] = df["t"].astype("int64")
    df["doi_pair_key"] = df["src_id"] + "|" + df["dst_id"]
    return df


# Pull order as OpenAlex might return it: newest citing work first, references in list order.
EDGES = _edges([
    ("W5", "W2", 20),
    ("W5", "W1", 20),
    ("W3", "W1", 10),
    ("W4", "W3", 15),
    ("W4", "W1", 15),
    ("W4", "W4", 15),   # self-citation loop
    ("W3", "W2", 10),
])
# Sorted by (t, src_id, dst_id):
EXPECTED = [
    (0, "W3", "W1", 10),
    (1, "W3", "W2", 10),
    (2, "W4", "W1", 15),
    (3, "W4", "W3", 15),
    (4, "W4", "W4", 15),
    (5, "W5", "W1", 20),
    (6, "W5", "W2", 20),
]


def test_build_events_contract_columns_dtypes_and_hand_order():
    ev = build_events(EDGES)
    assert list(ev.columns) == EVENT_COLUMNS == ["event_id", "src_id", "dst_id", "t"]
    assert ev["event_id"].dtype == np.int64 and ev["t"].dtype == np.int64
    assert list(ev.itertuples(index=False, name=None)) == EXPECTED
    assert ev.index.equals(pd.RangeIndex(len(ev)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_event_ids_do_not_depend_on_pull_order(seed):
    shuffled = EDGES.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pd.testing.assert_frame_equal(build_events(shuffled), build_events(EDGES))


def test_year_buckets_concatenated_in_any_order_give_the_same_ids():
    a, b = EDGES[EDGES["t"] < 15], EDGES[EDGES["t"] >= 15]
    pd.testing.assert_frame_equal(build_events(pd.concat([b, a])), build_events(pd.concat([a, b])))


def test_duplicate_pairs_keep_first_in_sort_order():
    # W4 -> W1 twice: once with a later date (a record re-read with another date), once exactly repeated
    dup = pd.concat([EDGES, _edges([("W4", "W1", 18), ("W4", "W1", 15)])], ignore_index=True)
    ev = build_events(dup)
    assert list(ev.itertuples(index=False, name=None)) == EXPECTED
    assert ev.attrs["n_duplicates_dropped"] == 2
    kept = build_events(dup, dedupe=False)
    assert len(kept) == len(dup)
    w4w1 = kept[(kept["src_id"] == "W4") & (kept["dst_id"] == "W1")]
    assert w4w1["t"].tolist() == [15, 15, 18] and w4w1["event_id"].tolist() == [2, 3, 6]
    pd.testing.assert_frame_equal(build_events(dup.iloc[::-1], dedupe=False), kept)


def test_self_loops_are_kept_and_counted():
    ev = build_events(EDGES)
    loops = ev[ev["src_id"] == ev["dst_id"]]
    assert loops["event_id"].tolist() == [4]
    assert ev.attrs["n_self_loops"] == 1


def test_build_events_accepts_edges_without_doi_pair_key_and_empty_input():
    ev = build_events(EDGES[["src_id", "dst_id", "t"]])
    assert len(ev) == 7
    empty = build_events(EDGES.iloc[:0])
    assert list(empty.columns) == EVENT_COLUMNS and len(empty) == 0
    assert empty["event_id"].dtype == np.int64 and empty["t"].dtype == np.int64


def test_build_events_refuses_missing_columns_and_nulls():
    with pytest.raises(ValueError, match="missing"):
        build_events(EDGES.drop(columns="t"))
    bad = EDGES.astype({"dst_id": object}).copy()
    bad.loc[0, "dst_id"] = None
    with pytest.raises(ValueError, match="null"):
        build_events(bad)


def test_write_read_round_trip(tmp_path):
    ev = build_events(EDGES)
    path = write_events(ev.sample(frac=1.0, random_state=0), tmp_path / "run" / "events.parquet")
    back = read_events(path)
    assert list(back.columns) == EVENT_COLUMNS
    assert back["event_id"].tolist() == list(range(7))       # written in canonical (t, event_id) order
    pd.testing.assert_frame_equal(back, ev, check_dtype=False)
    assert back["event_id"].dtype == np.int64 and back["t"].dtype == np.int64


def test_read_events_refuses_broken_files(tmp_path):
    ev = build_events(EDGES)
    cases = {
        "missing": ev.drop(columns="t"),
        "duplicate": ev.assign(event_id=0),
        "integer": ev.assign(t=ev["t"].astype(float) + 0.5),
        "null": ev.assign(src_id=[None] + ev["src_id"].tolist()[1:]),
    }
    for i, (msg, df) in enumerate(cases.items()):
        p = tmp_path / f"bad{i}.parquet"
        df.to_parquet(p, index=False)
        with pytest.raises(ValueError, match=msg):
            read_events(p)
        with pytest.raises(ValueError, match=msg):
            validate_events(df)


# ---------------------------------------------------------------- labels

WORKS = pd.DataFrame({
    "work_id": ["W1", "W2", "W3", "W4", "W5"],
    "doi_norm": ["10.1/a", "10.1/b.v2", None, "10.1/d", "10.1/e"],
})
# load_labels-style frame: one row per (paper, reason)
LABELS = pd.DataFrame({
    "doi_norm": ["10.1/a", "10.1/b", "10.1/e", "10.1/d"],
    "reason": ["Concerns/Issues about Referencing/Attributions", "Concerns/Issues about Referencing/Attributions",
               "Paper Mill", "Concerns/Issues about Referencing/Attributions"],
    "retraction_date": ["1/1/2024"] * 4,
    "retraction_nature": ["Retraction", "Retraction", "Retraction", "Expression of concern"],
})


def _labels_by_id(ev, lab):
    return dict(zip(ev.set_index("event_id").loc[lab["event_id"], "src_id"] + ">"
                    + ev.set_index("event_id").loc[lab["event_id"], "dst_id"], lab["label"]))


def test_event_labels_strict_src_or_dst_positive():
    ev = build_events(EDGES)
    # unfiltered labels: labelled papers W1 (strict), W4, W5; W2 only matches loosely (".v2")
    lab = event_labels_from_rwdb(ev, WORKS, LABELS, mode="strict")
    assert list(lab.columns) == ["event_id", "label"]
    assert lab["event_id"].tolist() == ev["event_id"].tolist()
    assert lab["label"].dtype == np.int64
    assert _labels_by_id(ev, lab) == {
        "W3>W1": 1, "W3>W2": 0, "W4>W1": 1, "W4>W3": 1, "W4>W4": 1, "W5>W1": 1, "W5>W2": 1}
    assert lab.attrs["stats"]["labeled_works"] == 3
    assert lab.attrs["stats"]["positive_events"] == 6


def test_event_labels_family_and_nature_filters_and_loose_mode():
    ev = build_events(EDGES)
    lab = event_labels_from_rwdb(ev, WORKS, LABELS, mode="strict",
                                 families=["referencing/attributions"], natures=["Retraction"])
    # only W1 survives both filters (W2 needs loose, W4 is an expression of concern, W5 is Paper Mill)
    assert _labels_by_id(ev, lab) == {
        "W3>W1": 1, "W3>W2": 0, "W4>W1": 1, "W4>W3": 0, "W4>W4": 0, "W5>W1": 1, "W5>W2": 0}
    loose = event_labels_from_rwdb(ev, WORKS, LABELS, mode="loose",
                                   families=["referencing/attributions"], natures=["Retraction"])
    assert _labels_by_id(ev, loose) == {
        "W3>W1": 1, "W3>W2": 1, "W4>W1": 1, "W4>W3": 0, "W4>W4": 0, "W5>W1": 1, "W5>W2": 1}


def test_event_labels_cited_paper_outside_works_is_negative_and_filters_need_columns():
    ev = build_events(_edges([("W3", "W9", 10)]))      # W9: referenced, never pulled, so no DOI known
    lab = event_labels_from_rwdb(ev, WORKS, LABELS)
    assert lab["label"].tolist() == [0]
    contract_labels = LABELS[["doi_norm", "retraction_date"]].assign(reason_family="ref_attr")
    with pytest.raises(ValueError, match="retraction_nature"):
        event_labels_from_rwdb(ev, WORKS, contract_labels, natures=["Retraction"])
    with pytest.raises(ValueError, match="mode"):
        event_labels_from_rwdb(ev, WORKS, LABELS, mode="fuzzy")

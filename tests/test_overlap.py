"""Reference overlap of a citation event at its own time (citecurv.curvature.overlap).

For e = (u -> v, t) on the graph of every other row with ts <= t:
    n_refs_dst = |succ(v)|, n_shared = |succ(u) & succ(v)|, overlap = n_shared / n_refs_dst.
The brute-force reference below filters the table row by row, an independent code path.
"""

import numpy as np
import pandas as pd
import pytest

from citecurv.curvature.overlap import OVERLAP_COLUMNS, reference_overlap


def _frame(rows, start_id=0):
    """rows: (src, dst, t) triples -> event frame with ids start_id, start_id + 1, ..."""
    return pd.DataFrame(
        [{"event_id": start_id + i, "src_id": s, "dst_id": d, "t": t} for i, (s, d, t) in enumerate(rows)],
        columns=["event_id", "src_id", "dst_id", "t"],
    )


def _brute(events, context=None):
    rows = events if context is None or len(context) == 0 else pd.concat([events, context], ignore_index=True)
    out = []
    for e in events.itertuples(index=False):
        g = rows[(rows["t"] <= e.t) & (rows["event_id"] != e.event_id)]
        succ_v = set(g.loc[g["src_id"] == e.dst_id, "dst_id"])
        succ_u = set(g.loc[g["src_id"] == e.src_id, "dst_id"])
        if e.src_id == e.dst_id:
            n_shared, share = 0, np.nan
        else:
            n_shared = len(succ_u & succ_v)
            share = n_shared / len(succ_v) if succ_v else np.nan
        out.append({"event_id": e.event_id, "src_id": e.src_id, "dst_id": e.dst_id, "t": e.t,
                    "n_refs_dst": len(succ_v), "n_shared": n_shared, "overlap": share})
    res = pd.DataFrame(out, columns=OVERLAP_COLUMNS)
    return res.sort_values(["t", "event_id"], kind="mergesort").reset_index(drop=True)


def _stream(seed, n_nodes, n_events, n_context, t_max):
    """Random events and context rows with duplicates, self-loops and ties in time."""
    rng = np.random.default_rng(seed)
    nodes = [f"W{i}" for i in range(n_nodes)]

    def draw(n, start):
        return pd.DataFrame({
            "event_id": np.arange(start, start + n),
            "src_id": rng.choice(nodes, size=n),
            "dst_id": rng.choice(nodes, size=n),
            "t": rng.integers(0, t_max + 1, size=n),
        })

    events = draw(n_events, 0)
    context = draw(n_context, n_events + 100)
    return events.sample(frac=1.0, random_state=seed).reset_index(drop=True), context


def _row(res, event_id):
    return res.set_index("event_id").loc[event_id]


def test_share_of_the_cited_works_references_that_the_citing_work_also_cites():
    events = _frame([("u", "v", 10), ("u", "a", 10), ("u", "b", 10)])
    context = _frame([("v", "a", 5), ("v", "c", 5)], start_id=100)
    res = reference_overlap(events, context=context)
    r = _row(res, 0)
    assert (r["n_refs_dst"], r["n_shared"], r["overlap"]) == (2, 1, 0.5)


def test_only_event_rows_come_back_in_canonical_order_with_the_overlap_columns():
    events = _frame([("u", "v", 10), ("x", "v", 3), ("u", "a", 10)])
    context = _frame([("v", "a", 1)], start_id=100)
    res = reference_overlap(events.iloc[::-1], context=context)
    assert list(res.columns) == OVERLAP_COLUMNS
    assert res["event_id"].tolist() == [1, 0, 2]
    assert res["n_refs_dst"].dtype == np.int64 and res["n_shared"].dtype == np.int64


def test_a_cited_work_without_a_reference_at_t_has_no_overlap_value():
    events = _frame([("u", "v", 10), ("u", "a", 10)])
    context = _frame([("v", "a", 11)], start_id=100)  # v's list is dated after the event
    r = _row(reference_overlap(events, context=context), 0)
    assert r["n_refs_dst"] == 0 and r["n_shared"] == 0 and np.isnan(r["overlap"])


def test_a_reference_of_the_cited_work_dated_after_the_event_is_not_read():
    events = _frame([("u", "v", 10), ("u", "a", 10), ("u", "c", 10)])
    context = _frame([("v", "a", 5), ("v", "c", 12)], start_id=100)
    r = _row(reference_overlap(events, context=context), 0)
    assert (r["n_refs_dst"], r["n_shared"], r["overlap"]) == (1, 1, 1.0)


def test_a_reference_of_the_citing_work_dated_after_the_event_is_not_shared():
    events = _frame([("u", "v", 10), ("u", "a", 20)])
    context = _frame([("v", "a", 5)], start_id=100)
    r = _row(reference_overlap(events, context=context), 0)
    assert (r["n_refs_dst"], r["n_shared"], r["overlap"]) == (1, 0, 0.0)


def test_duplicate_rows_count_a_reference_once():
    events = _frame([("u", "v", 10), ("u", "a", 10), ("u", "a", 10)])
    context = _frame([("v", "a", 5), ("v", "a", 6), ("v", "c", 5)], start_id=100)
    r = _row(reference_overlap(events, context=context), 0)
    assert (r["n_refs_dst"], r["n_shared"], r["overlap"]) == (2, 1, 0.5)


def test_the_scored_row_itself_is_not_a_shared_reference_when_the_cited_work_cites_itself():
    # v -> v makes v a reference of v; u -> v is the scored row and is removed, so v is not shared
    events = _frame([("u", "v", 10)])
    context = _frame([("v", "v", 5), ("v", "a", 5)], start_id=100)
    r = _row(reference_overlap(events, context=context), 0)
    assert (r["n_refs_dst"], r["n_shared"], r["overlap"]) == (2, 0, 0.0)


def test_another_row_of_the_same_pair_keeps_the_cited_work_among_the_citing_works_references():
    events = _frame([("u", "v", 10), ("u", "v", 10)])
    context = _frame([("v", "v", 5), ("v", "a", 5)], start_id=100)
    res = reference_overlap(events, context=context)
    assert res["n_shared"].tolist() == [1, 1] and res["overlap"].tolist() == [0.5, 0.5]


def test_a_later_row_of_the_same_pair_does_not_count_for_the_earlier_one():
    events = _frame([("u", "v", 10), ("u", "v", 11)])
    context = _frame([("v", "v", 5)], start_id=100)
    res = reference_overlap(events, context=context)
    assert res["n_shared"].tolist() == [0, 1]


def test_a_self_loop_row_has_no_overlap_value_and_does_not_count_itself():
    events = _frame([("v", "v", 10), ("v", "a", 10)])
    r = _row(reference_overlap(events), 0)
    assert r["n_refs_dst"] == 1 and r["n_shared"] == 0 and np.isnan(r["overlap"])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(6, 60, 60, 4), (15, 80, 120, 30), (40, 120, 40, 8)])
def test_matches_brute_force_on_random_streams(seed, shape):
    events, context = _stream(seed, *shape)
    got = reference_overlap(events, context=context)
    pd.testing.assert_frame_equal(got, _brute(events, context), check_dtype=False)


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_brute_force_without_context_rows(seed):
    events, _ = _stream(seed, 8, 90, 0, 6)
    pd.testing.assert_frame_equal(reference_overlap(events), _brute(events), check_dtype=False)


def test_the_random_streams_hold_shared_references_unknown_rows_and_self_loops():
    # the brute-force comparison would prove little on streams where nothing is shared
    events, context = _stream(0, 6, 60, 60, 4)
    ref = _brute(events, context)
    assert (ref["n_shared"] > 0).any() and ref["overlap"].isna().any()
    assert (events["src_id"] == events["dst_id"]).any()
    sparse = _brute(*_stream(0, 40, 120, 40, 8))
    assert (sparse["n_refs_dst"] == 0).any()


@pytest.mark.parametrize("chunk_pairs", [1, 7, 1000])
def test_the_chunk_size_does_not_change_the_result(chunk_pairs):
    events, context = _stream(5, 10, 100, 100, 6)
    whole = reference_overlap(events, context=context)
    assert len(whole) == 100 and (whole["n_shared"] > 0).any()
    pd.testing.assert_frame_equal(reference_overlap(events, context=context, chunk_pairs=chunk_pairs), whole)


def test_context_rows_must_not_reuse_an_event_id():
    events = _frame([("u", "v", 10)])
    with pytest.raises(ValueError, match="event_id"):
        reference_overlap(events, context=_frame([("v", "a", 5)], start_id=0))


def test_an_empty_event_table_gives_an_empty_frame():
    res = reference_overlap(_frame([]))
    assert list(res.columns) == OVERLAP_COLUMNS and len(res) == 0

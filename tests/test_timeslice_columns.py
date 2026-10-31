"""EventStore input formats (CONTRACTS.md "Events"): besides (src, dst, t) tuples
and src/dst/t DataFrames, EventStore accepts the contract's events frame with
src_id/dst_id columns (extra columns such as event_id are ignored)."""

import pandas as pd
import pytest

from citecurv.curvature.timeslice import EventStore
from test_timeslice import HAND_STREAM


def test_src_id_dst_id_frame_equals_tuple_input():
    df = pd.DataFrame(
        [(i + 100, s, d, t) for i, (s, d, t) in enumerate(HAND_STREAM)],
        columns=["event_id", "src_id", "dst_id", "t"],
    )
    a, b = EventStore(HAND_STREAM), EventStore(df.sample(frac=1.0, random_state=0))
    for s, d, t in HAND_STREAM:
        for radius in (1, 2):
            ga, gb = a.subgraph_at(s, d, t, radius), b.subgraph_at(s, d, t, radius)
            assert set(ga.nodes) == set(gb.nodes) and set(ga.edges) == set(gb.edges)
    assert a.degrees_at(10) == b.degrees_at(10)


def test_legacy_src_dst_frame_still_works():
    df = pd.DataFrame(HAND_STREAM, columns=["src", "dst", "t"])
    assert EventStore(df).degrees_at(10) == EventStore(HAND_STREAM).degrees_at(10)


def test_frame_without_endpoint_columns_is_rejected():
    with pytest.raises(ValueError, match="src_id"):
        EventStore(pd.DataFrame({"a": [1], "b": [2], "t": [3]}))

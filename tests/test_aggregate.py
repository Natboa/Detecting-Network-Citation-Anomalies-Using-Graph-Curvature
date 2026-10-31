"""S3.3: scores file, per-paper / per-author rollups (FR8), filtered export (FR9).

Spec under test (IMPLEMENTATION_PLAN.md S3.3, CONTRACTS.md "Scores file"):
- config_hash: first 12 hex chars of sha256 over canonical JSON (sorted keys).
- write_scores validates the scores.parquet columns, stamps model_variant,
  seed, config_hash (NFR10) and round-trips through read_scores.
- paper_rollup: per paper, mean of its top-n edge totals, count of totals
  above a threshold, and event count; role picks cited (dst), citing (src) or
  both endpoints; cold-start events are excluded by default.
- author_rollup: mean of an author's top-n paper scores, plus n_papers.
- export_csv filters by time window / field / venue / minimum total and writes
  a CSV that reads back to the same rows.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from citecurv.scoring.aggregate import (
    SCORES_COLUMNS,
    author_rollup,
    config_hash,
    export_csv,
    paper_rollup,
    read_scores,
    write_scores,
)

COMPONENT_COLS = ["s_embed", "s_geom", "s_temp", "s_curvdyn"]


def scores_fixture():
    rows = [
        # event_id, src, dst, t, total, cold_start
        (1, "A", "X", 100, 3.0, False),
        (2, "B", "X", 110, 1.0, False),
        (3, "C", "X", 120, 2.0, False),
        (4, "A", "Y", 130, 0.5, False),
        (5, "B", "Y", 140, 5.0, True),
        (6, "C", "Z", 150, 1.5, False),
    ]
    df = pd.DataFrame(rows, columns=["event_id", "src_id", "dst_id", "t", "total", "cold_start"])
    for i, c in enumerate(COMPONENT_COLS):
        df[c] = df["total"] * (i + 1) / 10
    return df[["event_id", "src_id", "dst_id", "t", *COMPONENT_COLS, "total", "cold_start"]]


def test_config_hash_is_canonical():
    a = {"seed": 1, "scoring": {"cold_start_k": 3, "weights": [0.25, 0.75]}}
    b = {"scoring": {"weights": (0.25, 0.75), "cold_start_k": np.int64(3)}, "seed": np.int64(1)}
    h = config_hash(a)
    assert h == config_hash(b)
    canonical = json.dumps(
        {"scoring": {"cold_start_k": 3, "weights": [0.25, 0.75]}, "seed": 1},
        sort_keys=True, separators=(",", ":"),
    )
    assert h == hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    assert len(h) == 12 and int(h, 16) >= 0
    assert config_hash({**a, "seed": 2}) != h


def test_write_and_read_scores_round_trip(tmp_path):
    df = scores_fixture()
    df["z_s_embed"] = 0.0  # extra breakdown columns are kept after the contract ones
    path = write_scores(df, tmp_path / "run" / "scores.parquet", model_variant="full",
                        seed=3, config_hash="abc123def456")
    back = read_scores(path)
    assert list(back.columns) == SCORES_COLUMNS + ["z_s_embed"]
    assert len(back) == len(df)
    assert (back["model_variant"] == "full").all() and (back["seed"] == 3).all()
    assert (back["config_hash"] == "abc123def456").all()
    assert back["event_id"].dtype == np.int64 and back["t"].dtype == np.int64
    assert back["cold_start"].dtype == bool
    assert np.array_equal(back["total"].to_numpy(), df["total"].to_numpy())


def test_write_scores_rejects_missing_columns_and_bad_rows(tmp_path):
    df = scores_fixture()
    with pytest.raises(ValueError, match="s_temp"):
        write_scores(df.drop(columns=["s_temp"]), tmp_path / "a.parquet", "full", 0, "h")
    dup = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        write_scores(dup, tmp_path / "b.parquet", "full", 0, "h")
    with pytest.raises(ValueError, match="model_variant"):
        write_scores(df.assign(model_variant="other"), tmp_path / "c.parquet", "full", 0, "h")


def test_read_scores_validates(tmp_path):
    path = tmp_path / "bad.parquet"
    scores_fixture().drop(columns=["total"]).to_parquet(path)
    with pytest.raises(ValueError, match="total"):
        read_scores(path)


def _as_records(df):
    return [tuple(r) for r in df.itertuples(index=False)]


def test_paper_rollup_cited_hand_case():
    out = paper_rollup(scores_fixture(), role="cited", top_n=2, threshold=1.0)
    assert list(out.columns) == ["work_id", "mean_top_n", "n_above", "n_events"]
    assert _as_records(out) == [("X", 2.5, 2, 3), ("Z", 1.5, 1, 1), ("Y", 0.5, 0, 1)]


def test_paper_rollup_keeps_cold_start_when_asked():
    out = paper_rollup(scores_fixture(), role="cited", top_n=2, threshold=1.0, exclude_cold_start=False)
    assert _as_records(out) == [("Y", 2.75, 1, 2), ("X", 2.5, 2, 3), ("Z", 1.5, 1, 1)]


def test_paper_rollup_citing_ties_sorted_by_work_id():
    out = paper_rollup(scores_fixture(), role="citing", top_n=2, threshold=1.0)
    # threshold is strict: B's single total 1.0 is not above 1.0
    assert _as_records(out) == [("A", 1.75, 1, 2), ("C", 1.75, 2, 2), ("B", 1.0, 0, 1)]


def test_paper_rollup_both_counts_self_citation_once():
    df = scores_fixture()
    loop = df.iloc[[0]].assign(event_id=7, src_id="W", dst_id="W", total=4.0)
    out = paper_rollup(pd.concat([df, loop]), role="both", top_n=2, threshold=None)
    rec = {r.work_id: (r.mean_top_n, r.n_events) for r in out.itertuples()}
    assert rec == {
        "A": (1.75, 2), "B": (1.0, 1), "C": (1.75, 2),
        "X": (2.5, 3), "Y": (0.5, 1), "Z": (1.5, 1), "W": (4.0, 1),
    }
    assert out["n_above"].isna().all()  # no threshold given


def test_paper_rollup_brute_force():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "event_id": np.arange(n),
        "src_id": [f"p{i}" for i in rng.integers(0, 40, n)],
        "dst_id": [f"p{i}" for i in rng.integers(0, 40, n)],
        "total": np.round(rng.normal(size=n), 2),
        "cold_start": rng.random(n) < 0.1,
    })
    out = paper_rollup(df, role="cited", top_n=5, threshold=0.5)
    kept = df[~df.cold_start]
    for r in out.itertuples():
        totals = sorted(kept.loc[kept.dst_id == r.work_id, "total"], reverse=True)
        assert r.mean_top_n == pytest.approx(np.mean(totals[:5]))
        assert r.n_above == sum(t > 0.5 for t in totals)
        assert r.n_events == len(totals)
    assert set(out.work_id) == set(kept.dst_id)
    assert out["mean_top_n"].is_monotonic_decreasing


def test_paper_rollup_rejects_bad_role():
    with pytest.raises(ValueError, match="role"):
        paper_rollup(scores_fixture(), role="author")


def test_author_rollup_hand_case():
    papers = paper_rollup(scores_fixture(), role="cited", top_n=2, threshold=1.0)  # X 2.5, Z 1.5, Y 0.5
    authorships = pd.DataFrame({
        "work_id": ["X", "Y", "Z", "X", "Q", "X"],
        "author_id": ["au1", "au1", "au1", "au2", "au3", "au2"],  # (X, au2) duplicated; Q unscored
    })
    out = author_rollup(papers, authorships, top_n=2)
    assert list(out.columns) == ["author_id", "mean_top_n", "n_papers"]
    assert _as_records(out) == [("au2", 2.5, 1), ("au1", 2.0, 3)]


def _works():
    return pd.DataFrame({
        "work_id": ["A", "B", "C", "X", "Y", "Z"],
        "venue_id": ["V1", "V2", "V1", "V3", None, "V2"],
        "field": ["Medicine", "Biology", "Medicine", "Physics", "Biology", "Physics"],
        "concepts": [["oncology"], [], ["oncology", "genetics"], ["optics"], [], ["optics"]],
    })


def test_export_csv_filters_and_round_trips(tmp_path):
    path = tmp_path / "out" / "export.csv"
    n = export_csv(scores_fixture(), path, t_from=105, t_to=150, min_total=1.0)
    back = pd.read_csv(path)
    assert n == len(back) == 4  # events 2, 3, 5, 6 (event 1 too early, 4 below 1.0)
    assert list(back["event_id"]) == [5, 3, 6, 2]  # sorted by total, descending
    assert list(back.columns[:5]) == ["event_id", "src_id", "dst_id", "t", "date"]
    assert back.loc[back.event_id == 5, "date"].item() == "1970-05-21"  # day 140
    expected = scores_fixture().set_index("event_id").loc[[5, 3, 6, 2]].reset_index()
    for col in expected.columns:
        assert back[col].tolist() == expected[col].tolist(), col


def _exported_ids(path):
    return sorted(pd.read_csv(path)["event_id"])


def test_export_csv_iso_dates_field_venue_and_endpoint(tmp_path):
    # events: 1 A->X, 2 B->X, 3 C->X, 4 A->Y, 5 B->Y, 6 C->Z
    df, works = scores_fixture(), _works()
    p = tmp_path / "e.csv"
    assert export_csv(df, p, t_from="1970-04-11", t_to="1970-04-21") == 2  # days 100..110
    assert _exported_ids(p) == [1, 2]
    assert export_csv(df, p, works=works, fields=["Biology"]) == 3  # B or Y on either end
    assert _exported_ids(p) == [2, 4, 5]
    assert export_csv(df, p, works=works, fields=["oncology"]) == 4  # concept of A and C
    assert _exported_ids(p) == [1, 3, 4, 6]
    assert export_csv(df, p, works=works, venues=["V2"]) == 3  # B, Z
    assert _exported_ids(p) == [2, 5, 6]
    assert export_csv(df, p, works=works, venues=["V2"], endpoint="dst") == 1
    assert export_csv(df, p, works=works, venues=["V2"], endpoint="src") == 2
    assert export_csv(df, p, works=works, venues=["V1"], endpoint="src", min_total=2.0) == 2
    assert _exported_ids(p) == [1, 3]


def test_export_csv_needs_works_for_field_filters(tmp_path):
    with pytest.raises(ValueError, match="works"):
        export_csv(scores_fixture(), tmp_path / "x.csv", fields=["Biology"])
    with pytest.raises(ValueError, match="endpoint"):
        export_csv(scores_fixture(), tmp_path / "x.csv", works=_works(), venues=["V1"], endpoint="any")


def test_export_csv_reads_a_scores_file(tmp_path):
    path = write_scores(scores_fixture(), tmp_path / "scores.parquet", "full", 0, "h")
    n = export_csv(path, tmp_path / "all.csv")
    back = pd.read_csv(tmp_path / "all.csv")
    assert n == 6 and list(back.columns[-3:]) == ["model_variant", "seed", "config_hash"]

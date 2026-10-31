"""S0.6: Retraction Watch CSV parsing — reason explosion, DOI normalization, family filter."""

import pytest

from citecurv.data.rwdb import load_labels, parse_reasons

FIXTURE = """RetractionDOI,OriginalPaperDOI,Reason,RetractionDate,RetractionNature
10.9/r1,https://doi.org/10.1/AAA,+Concerns/Issues about Referencing/Attributions;+Investigation by Journal/Publisher,1/2/2020,Retraction
10.9/r2,10.2/bbb,+Paper Mill,3/4/2021,Retraction
10.9/r3,,+Plagiarism of Article,5/6/2021,Retraction
10.9/r4,DOI:10.3/CCC,+Fake Peer Review;+Paper Mill,7/8/2022,Retraction
10.9/r5,10.4/ddd,,9/9/2022,Expression of concern
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "rwdb.csv"
    p.write_text(FIXTURE, encoding="utf-8")
    return p


def test_parse_reasons_splits_and_strips():
    raw = "+Concerns/Issues about Referencing/Attributions;+Paper Mill;"
    assert parse_reasons(raw) == [
        "Concerns/Issues about Referencing/Attributions",
        "Paper Mill",
    ]


def test_load_labels_explodes_reasons_and_normalizes_dois(csv_path):
    df = load_labels(csv_path)
    assert len(df) == 6  # 2 + 1 + 1 + 2; the reasonless row drops
    assert set(df.columns) == {"doi_norm", "reason", "retraction_date", "retraction_nature"}
    assert "10.1/aaa" in set(df.doi_norm)
    assert "10.3/ccc" in set(df.doi_norm)  # DOI: prefix stripped
    assert df[df.doi_norm.isna()].reason.tolist() == ["Plagiarism of Article"]


def test_family_filter_by_substring(csv_path):
    df = load_labels(csv_path, families=["referencing/attributions"])
    assert len(df) == 1
    assert df.iloc[0].doi_norm == "10.1/aaa"


def test_schema_guard_raises_on_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("A,B\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RWDB"):
        load_labels(bad)


# ---------- S1.2: works <-> labels join (strict vs loose) ----------
# Spec: strict = equality of normalized DOIs; loose = additionally strip a trailing
# version suffix (".v<digits>") on both sides before comparing. The joined frame keeps
# every work row once, in order; stats report both counts so the delta can be recorded.

import pandas as pd  # noqa: E402

from citecurv.data.rwdb import join_labels  # noqa: E402

WORKS = pd.DataFrame({
    "work_id": ["W1", "W2", "W3", "W4", "W5", "W6"],
    "doi_norm": ["10.1/a", "10.1/b.v2", "10.1/c", None, "10.1/x.v3", "10.1/y"],
})
LABELS = pd.DataFrame({
    "doi_norm": ["10.1/a", "10.1/a", "10.1/b", "10.1/c.v1", "10.1/z", None],
    "reason": ["R1", "R2", "R1", "R1", "R1", "R1"],
})


def test_join_strict_counts_exact_doi_matches_only():
    joined, stats = join_labels(WORKS, LABELS, mode="strict")
    assert joined["work_id"].tolist() == WORKS["work_id"].tolist()   # no row duplication
    assert joined["label"].tolist() == [1, 0, 0, 0, 0, 0]
    assert joined.loc[0, "label_doi"] == "10.1/a" and joined.loc[0, "match"] == "strict"
    assert stats == {
        "mode": "strict", "works": 6, "works_with_doi": 5, "label_papers": 4,
        "strict_matches": 1, "loose_matches": 3, "loose_only": 2, "labeled_works": 1,
    }


def test_join_loose_strips_version_suffix_on_both_sides():
    joined, stats = join_labels(WORKS, LABELS, mode="loose")
    assert joined["label"].tolist() == [1, 1, 1, 0, 0, 0]
    assert joined["match"].tolist()[:3] == ["strict", "loose", "loose"]
    assert joined.loc[1, "label_doi"] == "10.1/b"
    assert joined.loc[2, "label_doi"] == "10.1/c.v1"
    assert stats["labeled_works"] == 3 and stats["loose_only"] == 2


def test_join_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        join_labels(WORKS, LABELS, mode="fuzzy")


# ---------- snapshot refresh delta ----------
# Spec: compare two RWDB snapshots by Record ID; blank rows are counted separately and
# never treated as records; family membership changes are reported per record and per DOI.

from citecurv.data.rwdb import snapshot_delta  # noqa: E402

HEAD = ("Record ID,OriginalPaperDOI,Reason,RetractionDate,RetractionNature,"
        "OriginalPaperDate,RetractionDOI\n")
OLD = HEAD + (
    "1,10.1/a,+Concerns/Issues about Referencing/Attributions,1/1/2023 0:00,Retraction,1/1/2021 0:00,r1\n"
    "2,10.1/b,+Paper Mill,1/1/2023 0:00,Retraction,1/1/2022 0:00,r2\n"
    "3,10.1/c,+Plagiarism of Text,1/1/2023 0:00,Retraction,1/1/2022 0:00,r3\n"
    ",,,,,,\n"
)
NEW = HEAD + (
    "1,10.1/a,Concerns/Issues about Referencing/Attributions;,1/1/2023 0:00,Retraction,1/1/2021 0:00,r1\n"
    "2,10.1/b,Paper Mill;Concerns/Issues about Referencing/Attributions;,1/1/2023 0:00,Retraction,1/1/2022 0:00,r2\n"
    "4,10.1/d,Concerns/Issues about Referencing/Attributions;,1/1/2026 0:00,Retraction,6/1/2024 0:00,r4\n"
)


def test_snapshot_delta_counts_added_removed_and_family_changes(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    old.write_text(OLD, encoding="utf-8")
    new.write_text(NEW, encoding="utf-8")
    d = snapshot_delta(old, new)
    assert (d["records_old"], d["records_new"]) == (3, 3)
    assert (d["blank_rows_old"], d["blank_rows_new"]) == (1, 0)
    assert (d["records_added"], d["records_removed"]) == (1, 1)
    assert d["records_reason_changed"] == 1        # r2 gained RA; r1 is a format-only change
    assert (d["family_gained"], d["family_lost"]) == (1, 0)
    assert (d["family_dois_old"], d["family_dois_new"]) == (1, 3)
    assert (d["family_dois_added"], d["family_dois_removed"]) == (2, 0)
    assert d["family_dois_added_by_orig_year"] == {2022: 1, 2024: 1}

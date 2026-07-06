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

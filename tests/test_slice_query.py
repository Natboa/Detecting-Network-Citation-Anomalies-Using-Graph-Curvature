"""S1.3: label-density analysis and slice measurement (IMPLEMENTATION_PLAN.md task S1.3).

Spec under test:
- parse_subjects splits the RWDB Subject field (semicolon-joined, e.g.
  "(BLS) Biology - Cellular;(HSC) Medicine - Oncology;") into clean, de-duplicated
  subject strings; subject_group extracts the parenthesised group code.
- parse_rwdb_date reads "M/D/YYYY H:MM" and maps the 1753/1756 placeholder dates to NaT.
- prepare_records gives one row per RWDB record (blank rows dropped) with parsed lists,
  years and one boolean column per label family.
- rwdb_density counts distinct DOIs ("papers") per group, per family, after a
  RetractionNature filter; list-valued columns are exploded.
- candidate_table applies a candidate's RWDB selectors and reports totals, test-window
  totals, per-year counts, concentration, and the S1.3 label criteria.
- openalex_count_queries / label_doi_queries build the cheap count calls;
  measure_slice runs them through an injectable pager (no network here);
  write_slice_yaml writes a config that round-trips.
All expected values are hand-counted from the fixture below.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from citecurv.data.openalex import ApiKeyMissing
from citecurv.data.slice_query import (
    candidate_table,
    cooccurrence,
    filter_to_string,
    label_doi_queries,
    mass_events,
    measure_slice,
    openalex_count_queries,
    parse_rwdb_date,
    parse_subjects,
    prepare_records,
    rwdb_density,
    subject_group,
    write_slice_yaml,
)

RA = "Concerns/Issues about Referencing/Attributions"
COLS = [
    "Record ID", "Subject", "Journal", "Publisher", "Country", "RetractionDate",
    "RetractionDOI", "OriginalPaperDate", "OriginalPaperDOI", "RetractionNature", "Reason",
]
ROWS = [
    # r1: CS + DS, two countries, RA + Paper Mill, 2021 original
    ["1", "(B/T) Computer Science;(B/T) Data Science;", "J1", "P1", "China;India",
     "6/21/2023 0:00", "10.9/r1", "3/1/2021 0:00", "10.1/A", "Retraction",
     f"{RA};Paper Mill;"],
    # r2: CS only, old '+' reason format, same publisher + retraction date as r1
    ["2", "(B/T) Computer Science;", "J1", "P1", "China",
     "6/21/2023 0:00", "10.9/r2", "5/5/2022 0:00", "https://doi.org/10.1/b", "Retraction",
     f"+{RA};+Compromised Peer Review"],
    # r3: oncology + cancer biology, RA only
    ["3", "(HSC) Medicine - Oncology;(BLS) Biology - Cancer;", "J2", "P2", "Iran",
     "2/2/2024 0:00", "10.9/r3", "1/1/2022 0:00", "10.1/c", "Retraction", f"{RA};"],
    # r4: same paper as r3, earlier expression of concern (excluded by the nature filter)
    ["4", "(HSC) Medicine - Oncology;", "J2", "P2", "Iran",
     "1/1/2023 0:00", "10.9/r4", "1/1/2022 0:00", "10.1/c", "Expression of concern", f"{RA};"],
    # r5: oncology, Paper Mill only (broadened family, not RA)
    ["5", "(HSC) Medicine - Oncology;", "J3", "P2", "United States",
     "8/8/2020 0:00", "10.9/r5", "7/7/2019 0:00", "10.1/d", "Retraction", "Paper Mill;"],
    # r6: no DOI -> never a joinable paper
    ["6", "(B/T) Computer Science;", "J1", "P1", "China",
     "6/21/2023 0:00", "10.9/r6", "2/2/2023 0:00", None, "Retraction", f"{RA};"],
    # r7: reinstatement, trailing space in the subject literal
    ["7", "(PHY) Mathematics ;", "J4", "P3", "India",
     "1/1/2025 0:00", "10.9/r7", "1/1/2023 0:00", "10.1/e", "Reinstatement", f"{RA};"],
    # r8: fully blank trailing row (the 2026-07-06 snapshot has 181 of these)
    [None] * len(COLS),
]


@pytest.fixture
def raw():
    return pd.DataFrame(ROWS, columns=COLS)


@pytest.fixture
def recs(raw):
    return prepare_records(raw)


# ---------- parsing ----------

def test_parse_subjects_splits_strips_and_dedupes():
    raw = "(BLS) Biology - Cellular;(HSC) Medicine - Oncology;"
    assert parse_subjects(raw) == ["(BLS) Biology - Cellular", "(HSC) Medicine - Oncology"]
    assert parse_subjects("(PHY) Mathematics ;(PHY)  Physics;(PHY) Mathematics;") == [
        "(PHY) Mathematics", "(PHY) Physics",
    ]
    assert parse_subjects(None) == []
    assert parse_subjects(float("nan")) == []
    assert parse_subjects("") == []


def test_subject_group_code():
    assert subject_group("(B/T) Computer Science") == "B/T"
    assert subject_group("(HSC) Medicine - Oncology") == "HSC"
    assert subject_group("no code here") is None


def test_parse_rwdb_date_formats_and_placeholders():
    assert parse_rwdb_date("3/10/2026 0:00") == pd.Timestamp("2026-03-10")
    assert parse_rwdb_date("12/31/2021") == pd.Timestamp("2021-12-31")
    assert pd.isna(parse_rwdb_date("1/1/1753 12:00:00 AM"))   # SQL-minimum placeholder
    assert pd.isna(parse_rwdb_date("6/24/1756 12:00:00 AM"))
    assert pd.isna(parse_rwdb_date(None))
    assert pd.isna(parse_rwdb_date("garbage"))


def test_prepare_records_columns_and_families(recs):
    assert len(recs) == 7  # blank row dropped
    r1 = recs[recs.record_id == "1"].iloc[0]
    assert r1.doi_norm == "10.1/a"
    assert r1.subjects == ["(B/T) Computer Science", "(B/T) Data Science"]
    assert r1.subject_groups == ["B/T"]
    assert r1.countries == ["China", "India"]
    assert (r1.orig_year, r1.ret_year) == (2021, 2023)
    assert bool(r1.fam_ref_attr) and bool(r1.fam_paper_mill) and bool(r1.fam_broadened)
    r2 = recs[recs.record_id == "2"].iloc[0]
    assert r2.doi_norm == "10.1/b"
    assert bool(r2.fam_peer_review) and not bool(r2.fam_paper_mill)
    r5 = recs[recs.record_id == "5"].iloc[0]
    assert not bool(r5.fam_ref_attr) and bool(r5.fam_broadened)
    assert recs[recs.record_id == "7"].iloc[0].subjects == ["(PHY) Mathematics"]


# ---------- density ----------

def _as_dict(df, key, col):
    return dict(zip(df[key], df[col]))


def test_density_by_subject_counts_distinct_retracted_papers(recs):
    d = rwdb_density(recs, "subjects", families=("ref_attr", "broadened"))
    assert _as_dict(d, "subjects", "ref_attr_papers") == {
        "(B/T) Computer Science": 2,      # a, b (r6 has no DOI)
        "(B/T) Data Science": 1,          # a
        "(HSC) Medicine - Oncology": 1,   # c (r4 is an EoC; r5 not RA)
        "(BLS) Biology - Cancer": 1,      # c
    }
    assert _as_dict(d, "subjects", "broadened_papers")["(HSC) Medicine - Oncology"] == 2
    assert _as_dict(d, "subjects", "all_papers")["(HSC) Medicine - Oncology"] == 2
    assert d.iloc[0]["subjects"] == "(B/T) Computer Science"   # sorted by first family


def test_density_by_group_year_country(recs):
    g = rwdb_density(recs, "subject_groups", families=("ref_attr", "broadened"))
    assert _as_dict(g, "subject_groups", "ref_attr_papers") == {"B/T": 2, "HSC": 1, "BLS": 1}
    assert _as_dict(g, "subject_groups", "broadened_papers")["HSC"] == 2

    y = rwdb_density(recs, "orig_year", families=("ref_attr", "broadened"))
    assert _as_dict(y, "orig_year", "ref_attr_papers") == {2021: 1, 2022: 2, 2019: 0}
    assert _as_dict(y, "orig_year", "broadened_papers") == {2021: 1, 2022: 2, 2019: 1}

    c = rwdb_density(recs, "countries", families=("ref_attr",))
    assert _as_dict(c, "countries", "ref_attr_papers") == {
        "China": 2, "India": 1, "Iran": 1, "United States": 0,
    }


def test_density_nature_filter_off_counts_every_nature(recs):
    d = rwdb_density(recs, "subjects", families=("ref_attr",), natures=None)
    assert _as_dict(d, "subjects", "ref_attr_papers")["(PHY) Mathematics"] == 1  # reinstatement
    assert _as_dict(d, "subjects", "ref_attr_papers")["(HSC) Medicine - Oncology"] == 1  # c once


def test_density_two_keys(recs):
    d = rwdb_density(recs, ["publisher", "orig_year"], families=("ref_attr",))
    got = {(p, y): n for p, y, n in d[["publisher", "orig_year", "ref_attr_papers"]].itertuples(index=False)}
    assert got[("P1", 2021)] == 1 and got[("P1", 2022)] == 1 and got[("P2", 2022)] == 1


def test_mass_events_and_cooccurrence(recs):
    ev = mass_events(recs, family="ref_attr", min_size=1)
    top = ev.iloc[0]
    assert (top.publisher, str(top.ret_date.date()), top.papers) == ("P1", "2023-06-21", 2)
    co = cooccurrence(recs, family="ref_attr")
    got = _as_dict(co, "reason", "papers")
    assert got[RA] == 3
    assert got["Paper Mill"] == 1 and got["Compromised Peer Review"] == 1
    assert _as_dict(co, "reason", "share")["Paper Mill"] == pytest.approx(1 / 3)


# ---------- candidates ----------

CAND_CS = {
    "name": "cs",
    "subjects": ["(B/T) Computer Science", "(B/T) Data Science"],
    "years": [2020, 2023],
    "test_years": [2022, 2023],
    "openalex": {"filter": {"primary_topic": {"subfield": {"id": "1702"}}}},
}
CAND_ONC = {
    "name": "onc",
    "subject_groups": ["HSC"],
    "years": [2018, 2022],
    "test_years": [2022, 2022],
    "openalex": {"filter": {"primary_topic": {"subfield": {"id": "2730"}}}},
}


def test_candidate_table_totals_test_window_and_criteria(recs):
    t = candidate_table(recs, [CAND_CS, CAND_ONC], families=("ref_attr", "broadened"),
                        min_labeled=2, min_labeled_test=1).set_index("name")
    cs, onc = t.loc["cs"], t.loc["onc"]
    assert (cs.ref_attr_papers, cs.ref_attr_test_papers) == (2, 1)
    assert (cs.broadened_papers, cs.broadened_test_papers) == (2, 1)
    assert (cs.ref_attr_2020, cs.ref_attr_2021, cs.ref_attr_2022, cs.ref_attr_2023) == (0, 1, 1, 0)
    assert cs.top_publisher == "P1" and cs.top_publisher_share == pytest.approx(1.0)
    assert cs.top_event_share == pytest.approx(1.0)
    assert bool(cs.passes_label_criteria)
    assert (onc.ref_attr_papers, onc.ref_attr_test_papers) == (1, 1)
    assert (onc.broadened_papers, onc.broadened_test_papers) == (2, 1)
    assert not bool(onc.passes_label_criteria)


def test_candidate_publisher_restriction(recs):
    cand = dict(CAND_ONC, name="onc_p2_j3", journals=["J3"])
    t = candidate_table(recs, [cand], families=("broadened",)).iloc[0]
    assert t.broadened_papers == 1   # only d (J3)


# ---------- OpenAlex queries ----------

def test_filter_to_string_flattens_and_rejects_lists():
    f = {"primary_topic": {"subfield": {"id": "1702|1705"}}, "publication_year": "2016-2023"}
    assert filter_to_string(f) == "primary_topic.subfield.id:1702|1705,publication_year:2016-2023"
    assert filter_to_string({"is_retracted": True}) == "is_retracted:true"
    with pytest.raises(TypeError):
        filter_to_string({"doi": ["10.1/a", "10.1/b"]})  # pyalex would AND a list


def test_openalex_count_queries_shape():
    qs = {q["name"]: q for q in openalex_count_queries(CAND_CS)}
    assert set(qs) >= {"works_by_year", "with_refs_by_year", "retracted_oa_by_year",
                       "refcount_sample", "works_by_subfield", "works_by_topic"}
    assert qs["works_by_topic"]["group_by"] == "primary_topic.id"
    base = qs["works_by_year"]
    assert base["filter"]["publication_year"] == "2020-2023"
    assert base["filter"]["primary_topic"] == {"subfield": {"id": "1702"}}
    assert base["group_by"] == "publication_year"
    assert qs["with_refs_by_year"]["filter"]["has_references"] is True   # documented filter
    assert qs["retracted_oa_by_year"]["filter"]["is_retracted"] is True
    s = qs["refcount_sample"]
    assert s["sample"] == 100 and s["seed"] == 0 and s["per_page"] == 100   # per_page 200 deprecated
    assert "referenced_works" in s["select"].split(",")
    assert all("concepts" not in q["url"] for q in qs.values())          # topics only
    assert "group_by=publication_year" in base["url"]
    assert "filter=primary_topic.subfield.id:1702,publication_year:2020-2023" in base["url"]
    assert CAND_CS["openalex"]["filter"] == {"primary_topic": {"subfield": {"id": "1702"}}}  # not mutated


def test_unresolved_placeholder_filter_is_refused():
    cand = dict(CAND_CS, openalex={"filter": {"primary_location": {"source": {"id": "<resolve live>"}}}})
    with pytest.raises(ValueError, match="unresolved"):
        openalex_count_queries(cand)


def test_shortlist_is_well_formed_and_queryable():
    from citecurv.data.slice_query import SHORTLIST

    assert 3 <= len(SHORTLIST) <= 5
    assert len({c["name"] for c in SHORTLIST}) == len(SHORTLIST)
    for c in SHORTLIST:
        lo, hi = c["years"]
        tlo, thi = c["test_years"]
        assert lo <= tlo <= thi <= hi
        assert c.get("subjects") or c.get("subject_groups") or c.get("journals")
        if not c["openalex"].get("needs_resolution"):
            assert openalex_count_queries(c)[0]["filter"]["publication_year"] == f"{lo}-{hi}"


def test_label_doi_queries_batches_and_skips_unsafe_dois():
    dois = ["10.1/a", "10.1/b", "10.1/c", "10.1/x,y", "10.1/p|q"]
    qs = label_doi_queries(CAND_CS, dois, batch_size=2)
    assert [q["n_dois"] for q in qs] == [2, 1]
    assert qs[0]["filter"]["doi"] == "https://doi.org/10.1/a|https://doi.org/10.1/b"
    assert qs[0]["filter"]["publication_year"] == "2020-2023"
    assert qs[0]["group_by"] == "publication_year"


# ---------- measure_slice ----------

def _fake_pager(calls):
    by_name = {
        "works_by_year": {"group_by": [
            {"key": "2020", "count": 100}, {"key": "2021", "count": 150},
            {"key": "2022", "count": 200}, {"key": "2023", "count": 250}]},
        "with_refs_by_year": {"group_by": [
            {"key": "2020", "count": 80}, {"key": "2021", "count": 120},
            {"key": "2022", "count": 180}, {"key": "2023", "count": 220}]},
        "retracted_oa_by_year": {"group_by": [{"key": "2022", "count": 3}]},
        "refcount_sample": {"results": [   # one record carries the count field, the rest the list
            {"referenced_works": ["W"] * 10}, {"referenced_works_count": 30},
            {"referenced_works": []}, {"referenced_works": ["W"] * 20}]},
        "works_by_subfield": {"group_by": [
            {"key": "https://openalex.org/subfields/1702", "key_display_name": "AI", "count": 700}]},
        "works_by_topic": {"group_by": [
            {"key": "https://openalex.org/T10", "count": 400}, {"key": "https://openalex.org/T11", "count": 300}]},
    }

    def pager(query):
        calls.append(query["name"])
        if query["name"].startswith("labeled_by_year"):
            return {"meta": {}, "group_by": [{"key": "2022", "count": 1}]}
        return {"meta": {}, **by_name[query["name"]]}

    return pager


CFG = {"data": {"page_size": 100, "mailto": None, "max_retries": 2, "backoff_base_s": 0.0}}


def test_measure_slice_aggregates_counts_and_criteria():
    calls = []
    m = measure_slice(CAND_CS, pager=_fake_pager(calls), dois=["10.1/a", "10.1/b", "10.1/c"],
                      config=CFG, batch_size=2, works_target=700, works_tol=0.5,
                      min_edges=10_000, min_labeled=2, min_labeled_test=3)
    assert m["works_by_year"] == {2020: 100, 2021: 150, 2022: 200, 2023: 250}
    assert m["works_total"] == 700
    assert m["with_refs_share"] == pytest.approx(600 / 700)
    assert m["retracted_oa_total"] == 3
    assert m["mean_refs_sample"] == pytest.approx(15.0) and m["sample_n"] == 4
    assert m["edges_est"] == 700 * 15
    assert m["labeled_by_year"] == {2022: 2}       # one per batch, two batches
    assert (m["labeled_total"], m["labeled_test"]) == (2, 2)
    assert m["dois_queried"] == 3 and m["dois_skipped"] == 0
    assert m["pull_pages_est"] == 7 and m["pull_cost_usd_est"] == pytest.approx(0.0007)
    assert m["calls"] == 8 and m["measure_cost_usd_est"] == pytest.approx(0.0008)
    assert m["works_by_subfield"] == {"1702": 700}
    assert m["works_by_topic"] == {"T10": 400, "T11": 300}
    assert m["criteria"] == {"works": True, "edges": True, "labeled": True,
                             "labeled_test": False, "budget": True, "daily_cap": True}
    assert not m["passes"]
    assert len(calls) == 8


class FakeHTTP(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


def test_measure_slice_retries_429():
    calls, inner = [], _fake_pager([])
    state = {"failed": False}

    def flaky(query):
        if not state["failed"]:
            state["failed"] = True
            raise FakeHTTP(429)
        calls.append(query["name"])
        return inner(query)

    m = measure_slice(CAND_CS, pager=flaky, config=CFG)
    assert m["works_total"] == 700 and m["labeled_total"] is None   # no DOIs given
    assert len(calls) == 6


def test_measure_slice_refuses_keyless(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    with pytest.raises(ApiKeyMissing):
        measure_slice(CAND_CS, config=CFG)


def test_write_density_tables_writes_traceable_files(recs, tmp_path):
    import json

    from citecurv.data.slice_query import write_density_tables

    paths = write_density_tables(recs, tmp_path, tag="fixture", candidates=[CAND_CS, CAND_ONC],
                                 families=("ref_attr", "broadened"), event_min=1)
    names = {p.name for p in paths}
    assert "rwdb_density_fixture_summary.json" in names
    assert "rwdb_density_fixture_candidates.csv" in names
    assert all(p.stat().st_size < 200_000 for p in paths)
    nat = pd.read_csv(tmp_path / "rwdb_density_fixture_natures.csv")   # every nature, unfiltered
    assert dict(zip(nat.nature, nat.ref_attr_papers)) == {
        "Retraction": 3, "Expression of concern": 1, "Reinstatement": 1,
    }
    s = json.loads((tmp_path / "rwdb_density_fixture_summary.json").read_text(encoding="utf-8"))
    assert s["natures"] == ["Retraction"]
    conc = s["concentration"]["ref_attr"]
    assert conc["papers"] == 3
    assert conc["top_publishers"][0] == {"publisher": "P1", "papers": 2, "share": pytest.approx(2 / 3)}
    assert conc["events_ge_min"] == {"min_size": 1, "events": 2, "papers": 3, "share": 1.0}
    assert [c["name"] for c in s["candidates"]] == ["cs", "onc"]
    assert s["candidates"][0]["openalex_queries"][0]["name"] == "works_by_year"
    subj = pd.read_csv(tmp_path / "rwdb_density_fixture_subjects.csv")
    assert dict(zip(subj.subjects, subj.ref_attr_papers))["(B/T) Computer Science"] == 2


def test_write_slice_yaml_round_trips(tmp_path):
    m = measure_slice(CAND_CS, pager=_fake_pager([]), dois=["10.1/a"], config=CFG)
    p = write_slice_yaml(tmp_path / "slice.yaml", CAND_CS, m,
                         rwdb_snapshot={"date": "2026-07-06", "sha256": "ae6e"})
    text = p.read_text(encoding="utf-8")
    assert "PROVISIONAL(G3)" in text
    s = yaml.safe_load(text)["slice"]
    assert s["name"] == "cs"
    assert s["openalex_filter"] == {"primary_topic": {"subfield": {"id": "1702"}},
                                    "publication_year": "2020-2023"}
    assert s["filter_string"] == "primary_topic.subfield.id:1702,publication_year:2020-2023"
    assert s["years"] == [2020, 2023] and s["test_years"] == [2022, 2023]
    assert s["expected"]["works"] == 700 and s["expected"]["edges_est"] == 10500
    assert s["rwdb_snapshot"]["date"] == "2026-07-06"
    assert s["expected"]["labeled_papers"] == 1
    oa = s["openalex"]   # provenance the integrator asked for
    assert oa["corpus"] == "core"                       # no include_xpac
    assert "2026-10-06" in oa["topic_assignments"]       # topics re-tagged by OpenAlex
    assert oa["measured_on"] == m["measured_on"]


# ---------- labeled-DOI lookup (topic-level label density in one pass) ----------

def test_doi_lookup_queries_batch_and_cap():
    from citecurv.data.slice_query import doi_lookup_queries

    qs = doi_lookup_queries(["10.1/a", "10.1/b", "10.1/c", "10.1/x,y"], batch_size=2)
    assert [q["n_dois"] for q in qs] == [2, 1]
    assert qs[0]["filter"] == {"doi": "https://doi.org/10.1/a|https://doi.org/10.1/b"}
    assert qs[0]["per_page"] == 100 and qs[0]["group_by"] is None
    assert {"id", "doi", "publication_year", "primary_topic"} <= set(qs[0]["select"].split(","))
    with pytest.raises(ValueError, match="100"):
        doi_lookup_queries(["10.1/a"], batch_size=101)   # documented OR limit


def test_lookup_labeled_works_flattens_topic_hierarchy():
    from citecurv.data.slice_query import lookup_labeled_works

    def pager(query):
        if "10.1/a" in query["filter"]["doi"]:
            return {"meta": {}, "results": [
                {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/A", "publication_year": 2022,
                 "primary_topic": {"id": "https://openalex.org/T10", "subfield": {"id": 1702},
                                   "field": {"id": "https://openalex.org/fields/17"}},
                 "primary_location": {"source": {"id": "https://openalex.org/S9"}}},
                {"id": "https://openalex.org/W2", "doi": "https://doi.org/10.1/b",
                 "publication_year": 2021, "primary_topic": None, "primary_location": None}]}
        return {"meta": {}, "results": []}

    df, stats = lookup_labeled_works(["10.1/a", "10.1/b", "10.1/c"], pager=pager, config=CFG, batch_size=2)
    assert df["work_id"].tolist() == ["W1", "W2"]
    assert df["doi_norm"].tolist() == ["10.1/a", "10.1/b"]
    w1 = df.iloc[0]
    assert (w1.topic_id, w1.subfield_id, w1.field_id, w1.source_id) == ("T10", "1702", "17", "S9")
    assert df.iloc[1].subfield_id is None
    assert stats == {"dois_queried": 3, "works_returned": 2, "dois_found": 2, "calls": 2}


# ---------- topic-level narrowing ----------

def test_labeled_by_topic_counts_distinct_dois_in_years():
    from citecurv.data.slice_query import labeled_by_topic

    df = pd.DataFrame({
        "doi_norm": ["10.1/a", "10.1/b", "10.1/b", "10.1/c", "10.1/d"],
        "publication_year": [2022, 2021, 2021, 2016, 2023],
        "topic_id": ["T1", "T1", "T1", "T1", None],
    })
    assert labeled_by_topic(df, years=[2018, 2023]) == {"T1": 2}   # c out of range, d no topic


def test_pick_topics_greedy_by_label_density():
    from citecurv.data.slice_query import pick_topics

    works = {"T1": 100, "T2": 200, "T3": 50, "T4": 400, "T5": 10}
    labeled = {"T1": 10, "T2": 10, "T3": 1, "T4": 40}            # T5 has no labels: never picked
    got = pick_topics(works, labeled, target_works=500, max_works=750)
    assert got == {"topics": ["T4", "T1"], "works": 500, "labeled": 50}   # density tie -> more labels first
    got = pick_topics(works, labeled, target_works=600, max_works=650)
    assert got == {"topics": ["T4", "T1", "T3"], "works": 550, "labeled": 51}  # T2 would overshoot


def test_topic_candidate_keeps_rwdb_side_and_swaps_filter():
    from citecurv.data.slice_query import topic_candidate

    c = topic_candidate(CAND_CS, ["T4", "T1"])
    assert c["name"] == "cs_topics" and c["subjects"] == CAND_CS["subjects"]
    assert c["openalex"]["filter"] == {"primary_topic": {"id": "T4|T1"}}
    assert openalex_count_queries(c)[0]["filter"]["publication_year"] == "2020-2023"
    assert CAND_CS["openalex"]["filter"] == {"primary_topic": {"subfield": {"id": "1702"}}}


# ---------- default fetch path: the S1.1 OpenAlex client (key modes, BudgetExhausted) ----------
# Spec: _default_fetch builds its requests with citecurv.data.openalex.make_client, so the key goes
# in the Bearer header or the api_key parameter by key mode, a missing key refuses before any
# request, and a 429 whose /rate-limit check shows a spent budget raises BudgetExhausted (no retry).
# HTTP comes from a fake session; no network.

from urllib.parse import parse_qs, urlsplit  # noqa: E402

from citecurv.data.openalex import BudgetExhausted  # noqa: E402
from citecurv.data.slice_query import _default_fetch, lookup_labeled_works  # noqa: E402

KEY = "sk-test-SECRET-9876"
FETCH_CFG = {"data": {**CFG["data"], "max_requests_per_s": 0}}


class _Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def json(self):
        return self._payload


class _Session:
    """Records (url, headers); answers through a callable(url) -> _Resp."""

    def __init__(self, answer):
        self.answer, self.requests = answer, []

    def get(self, url, headers=None, timeout=None):
        self.requests.append((url, dict(headers or {})))
        return self.answer(url)


def _rate_limit_payload(remaining):
    return {"rate_limit": {"credits_limit": 10000, "credits_used": 10000 - remaining,
                           "credits_remaining": remaining, "resets_in_seconds": 3600}}


def test_default_fetch_bearer_mode_group_by_query(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    monkeypatch.delenv("OPENALEX_KEY_MODE", raising=False)
    s = _Session(lambda url: _Resp(200, {"meta": {"count": 3}, "group_by": [{"key": "2022", "count": 3}]}))
    fetch = _default_fetch(FETCH_CFG["data"], session=s)
    q = openalex_count_queries(CAND_CS)[0]                       # works_by_year
    assert fetch(q) == {"meta": {"count": 3}, "group_by": [{"key": "2022", "count": 3}]}
    url, headers = s.requests[0]
    assert headers["Authorization"] == f"Bearer {KEY}" and KEY not in url
    parts = urlsplit(url)
    assert parts.path == "/works"
    qs = parse_qs(parts.query)
    assert qs["group_by"] == ["publication_year"]
    assert qs["filter"] == [filter_to_string(q["filter"])]
    assert callable(fetch.rate_limit)


def test_default_fetch_param_mode_sample_select_page(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    monkeypatch.setenv("OPENALEX_KEY_MODE", "param")
    s = _Session(lambda url: _Resp(200, {"meta": {}, "results": [{"id": "https://openalex.org/W1"}]}))
    fetch = _default_fetch(FETCH_CFG["data"], session=s)
    q = next(x for x in openalex_count_queries(CAND_CS, sample_n=40, seed=7) if x["name"] == "refcount_sample")
    assert fetch(q)["results"] == [{"id": "https://openalex.org/W1"}]
    url, headers = s.requests[0]
    assert "Authorization" not in headers
    qs = parse_qs(urlsplit(url).query)
    assert qs["api_key"] == [KEY]
    assert (qs["sample"], qs["seed"], qs["per-page"]) == (["40"], ["7"], ["40"])
    assert qs["select"] == ["id,publication_year,referenced_works"]


def test_default_fetch_refuses_keyless_before_any_request(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    s = _Session(lambda url: _Resp(200, {}))
    with pytest.raises(ApiKeyMissing):
        _default_fetch(FETCH_CFG["data"], session=s)
    assert s.requests == []


def test_measure_slice_spent_budget_raises_budget_exhausted_without_retry(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)

    def answer(url):
        if "/rate-limit" in url:
            return _Resp(200, _rate_limit_payload(0))
        return _Resp(429, {"error": "Rate limit exceeded"})

    s = _Session(answer)
    with pytest.raises(BudgetExhausted) as info:
        measure_slice(CAND_CS, pager=_default_fetch(FETCH_CFG["data"], session=s), config=FETCH_CFG)
    assert [("/rate-limit" in u) for u, _ in s.requests] == [False, True]   # one call, one check, no retry
    assert KEY not in str(info.value)


def test_lookup_labeled_works_retries_a_too_fast_429(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", KEY)
    state = {"n": 0}
    work = {"id": "https://openalex.org/W7", "doi": "https://doi.org/10.1/A", "publication_year": 2022,
            "primary_topic": {"id": "https://openalex.org/T10",
                              "subfield": {"id": "https://openalex.org/subfields/1702"},
                              "field": {"id": "https://openalex.org/fields/17"}},
            "primary_location": {"source": {"id": "https://openalex.org/S5"}}}

    def answer(url):
        if "/rate-limit" in url:
            return _Resp(200, _rate_limit_payload(500))
        state["n"] += 1
        if state["n"] == 1:
            return _Resp(429, {"error": "Too many requests"})
        return _Resp(200, {"meta": {}, "results": [work]})

    s = _Session(answer)
    df, stats = lookup_labeled_works(["10.1/a"], pager=_default_fetch(FETCH_CFG["data"], session=s),
                                     config=FETCH_CFG)
    assert df["work_id"].tolist() == ["W7"] and df["subfield_id"].tolist() == ["1702"]
    assert stats == {"dois_queried": 1, "works_returned": 1, "dois_found": 1, "calls": 1}
    assert [("/rate-limit" in u) for u, _ in s.requests] == [False, True, False]
